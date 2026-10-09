import sys
from pathlib import Path

# The tokeniser package lives at the project root, not under src/.
# Ensure the project root is on sys.path so that
#   from tokeniser.bpe import ...
# works during test collection.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))