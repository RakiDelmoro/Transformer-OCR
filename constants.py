import torch
import string

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = (32, 800)

NUM_EPOCHS = 100000000
BATCH_SIZE = 64
LEARNING_RATE = 1e-5

LETTERS = [ch for ch in string.ascii_letters]
NUMBERS = [num for num in string.digits]
SYMBOLS = ["!", "?", ".", ":", ";", "'", "&", " "]
START_TOKEN = '\N{Start of Text}'
END_TOKEN = '\N{End of Text}'
PAD_TOKEN = '\N{Substitute}'

CHARS = ["\x00", PAD_TOKEN, START_TOKEN, END_TOKEN] + LETTERS + NUMBERS + SYMBOLS

MAX_PHRASE_LENGTH = 95
NUM_PHRASE_LENGTH = 80
INFERENCE_PHRASE_LENGTH = 5
