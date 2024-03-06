import torch
import string

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EPOCHS = 100000000
BATCH_SIZE = 512
LEARNING_RATE = 1e-5

LETTERS = [ch for ch in string.ascii_letters]
NUMBERS = [num for num in string.digits]
SYMBOLS = ["!", "?", ".", ":", ";", "'", "&", " "]
START_TOKEN = '\N{Start of Text}'
END_TOKEN = '\N{End of Text}'
PAD_TOKEN = '\N{Substitute}'

CHARS = ["\x00", PAD_TOKEN, START_TOKEN, END_TOKEN] + LETTERS + NUMBERS + SYMBOLS
CHAR_TO_INDEX = {c:i for i, c in enumerate(CHARS)}

