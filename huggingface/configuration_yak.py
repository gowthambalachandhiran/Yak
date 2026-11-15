from transformers import PretrainedConfig

class YakConfig(PretrainedConfig):
    model_type = "yak"

    def __init__(self, vocab_size=30000, d_model=128, nhead=4, num_layers=4,
                 max_seq_len=2048, dropout=0.1, pad_token_id=0,
                 cls_token_id=2, sep_token_id=3, label_vocabularies=None,
                 held_out=None, **kwargs):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=cls_token_id,
                         eos_token_id=sep_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hidden_size = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.max_position_embeddings = max_seq_len
        self.dropout = dropout
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.label_vocabularies = label_vocabularies or {}
        self.held_out = held_out or {}
        self.architectures = ["YakForLabelScoring"]
