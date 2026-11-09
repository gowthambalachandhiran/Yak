#Metrics.py
"""Accuracy and calibration metrics, numpy only.

Scores come out of the model per slot. A slot is a position in that example's label list, so slot scores are
mapped back to class probabilities with slot_class before anything is measured.
"""
import numpy as np


def softmax(scores, temperature=1.0):
    """Row-wise softmax over slots. -inf entries get probability 0."""
    s = np.asarray(scores, dtype=np.float64) / temperature
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def slots_to_class_probs(slot_probs, slot_class, num_classes):
    """[N, M] slot probabilities and [N, M] class ids (-1 = unused slot) -> [N, num_classes].
    Classes that were not in an example's label list get probability 0."""
    n = slot_probs.shape[0]
    out = np.zeros((n, num_classes), dtype=np.float64)
    rows, cols = np.nonzero(slot_class >= 0)
    np.add.at(out, (rows, slot_class[rows, cols]), slot_probs[rows, cols])
    return out


def accuracy(probs, true_class):
    return float((probs.argmax(axis=1) == true_class).mean())


def brier(probs, true_class):
    """Mean over examples of sum_k (p_k - y_k)^2. 0 is perfect, 2 is the worst."""
    y = np.zeros_like(probs)
    y[np.arange(len(true_class)), true_class] = 1.0
    return float(((probs - y) ** 2).sum(axis=1).mean())


def nll(probs, true_class, eps=1e-12):
    p = probs[np.arange(len(true_class)), true_class]
    return float(-np.log(np.clip(p, eps, 1.0)).mean())


def ece(probs, true_class, n_bins=15):
    """Expected calibration error on the top-1 confidence."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == true_class).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def fit_temperature(scores, target_slot, grid=None):
    """Pick T that minimises NLL of softmax(scores / T) on validation slot scores [N, M]."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 60)) if grid is None else grid
    best_t, best = 1.0, float("inf")
    for t in grid:
        p = softmax(scores, t)
        loss = -np.log(np.clip(p[np.arange(len(target_slot)), target_slot], 1e-12, 1.0)).mean()
        if loss < best:
            best_t, best = float(t), float(loss)
    return best_t


def report(probs, true_class):
    return {"acc": accuracy(probs, true_class), "brier": brier(probs, true_class),
            "nll": nll(probs, true_class), "ece": ece(probs, true_class)}


def oos_report(probs, true_class, oos_index):
    """Split a task that has an out-of-scope class (CLINC) into in-scope and out-of-scope behaviour.

    in_scope_acc:         accuracy on rows whose true class is a real intent (predicting oos counts as wrong)
    oos_recall:           share of true out-of-scope rows that were predicted out-of-scope
    oos_precision:        share of out-of-scope predictions that really were out-of-scope
    in_scope_flagged_oos: share of real-intent rows wrongly predicted out-of-scope
    oos_share:            share of out-of-scope rows in the data
    """
    pred = np.asarray(probs).argmax(axis=1)
    true_class = np.asarray(true_class)
    is_oos, ins, said_oos = true_class == oos_index, true_class != oos_index, pred == oos_index

    def frac(num, den):
        return float(num / den) if den else None

    return {
        "in_scope_acc": frac((pred[ins] == true_class[ins]).sum(), ins.sum()),
        "oos_recall": frac(said_oos[is_oos].sum(), is_oos.sum()),
        "oos_precision": frac(is_oos[said_oos].sum(), said_oos.sum()),
        "in_scope_flagged_oos": frac(said_oos[ins].sum(), ins.sum()),
        "oos_share": float(is_oos.mean()),
    }