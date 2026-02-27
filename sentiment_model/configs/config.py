# Data
DATA_PATH = "data/raw/imdb_50k.csv"
TEXT_COL = "review"
LABEL_COL = "sentiment"

# Vocabulary
MAX_VOCAB_SIZE = 30000
MAX_SEQ_LEN = 300

# Training
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3

# Model
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 1

# Device
DEVICE = "cuda"  # will auto-fallback in code
