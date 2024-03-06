from datasets.stream_dataset import StreamDataset
from main import main
from constants import INFERENCE_PHRASE_LENGTH, NUM_PHRASE_LENGTH, MAX_PHRASE_LENGTH
TRAINING_DATASET_SIZE = 64*100
VALIDATION_DATASET_SIZE = 64*10
INFERENCE_DATASET_SIZE = 5

training_dataset = StreamDataset(NUM_PHRASE_LENGTH, False)
validation_dataset = StreamDataset(NUM_PHRASE_LENGTH, False)
inference_dataset = StreamDataset(INFERENCE_PHRASE_LENGTH, False)

main(training_dataset, validation_dataset, enumerate(inference_dataset), TRAINING_DATASET_SIZE, VALIDATION_DATASET_SIZE, INFERENCE_DATASET_SIZE, False, True)
