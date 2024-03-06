from torch.utils.data import DataLoader
from constants import BATCH_SIZE, DEVICE, LEARNING_RATE, NUM_EPOCHS
import os
import torch
import utils as main_utils

from model.transformer_ocr import TransformerOcr
from datasets.data_utils import decode_for_print
from utils import load_checkpoint

# File where to save model checkpoint and fully trained
MODEL_CHECKPOINT_FOLDER = "ModelCheckpoints"
FULLY_TRAINED_MODEL_FILE = "model.pth"

MODEL = TransformerOcr().to(DEVICE)

# Hyperparameters for the model
LOSS_FN = torch.nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95))

def main(training_dataset, validation_dataset, inference_iterable, training_data_length, validation_data_length,
         inference_data_length, shuffle, use_checkpoint):

    training_loader = DataLoader(dataset=training_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=shuffle)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=shuffle)

    train_loader_length = training_data_length // training_loader.batch_size
    validation_loader_length = validation_data_length // validation_loader.batch_size

    start_epoch = 1
    if use_checkpoint:
        if os.path.exists(MODEL_CHECKPOINT_FOLDER):
            list_of_checkpoints = os.listdir(MODEL_CHECKPOINT_FOLDER)
            if len(list_of_checkpoints) == 0:
                print(f"No checkpoint file yet.")
            else:
                loaded_epoch, loaded_loss, checkpoint = load_checkpoint(model_checkpoint_folder=MODEL_CHECKPOINT_FOLDER, model=MODEL,
                                                            optimizer=OPTIMIZER)
                print(f"loaded checkpoint file from {checkpoint} have EPOCH: {loaded_epoch} with a loss: {loaded_loss}")
                start_epoch = loaded_epoch + 1

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loss = main_utils.train_model(train_dataset=training_loader, model=MODEL, optimizer=OPTIMIZER,
                                            dataloader_length=train_loader_length, encoder_trainer=False)
        
        val_loss = main_utils.evaluate_model(validation_dataset=validation_loader, model=MODEL,
                                             dataloader_length=validation_loader_length, encoder_trainer=False)
        
        print(f"EPOCH: {epoch} Training loss: {train_loss}, Validation loss: {val_loss}")
         
        for _ in range(inference_data_length):
            predicted, expected = main_utils.generate_token_prediction(next(inference_iterable)[1], MODEL)
            print(f"Predicted: {decode_for_print(predicted[0])} Expected: {decode_for_print(expected)}")

        if use_checkpoint:
            print("Saving do not turn off!")
            checkpoint = main_utils.save_checkpoint(epoch=epoch, model=MODEL, optimizer=OPTIMIZER,
                                                    loss=train_loss)

            torch.save(checkpoint, f"./ModelCheckpoints/model_checkpoint-{epoch%5}.tar", _use_new_zipfile_serialization=False)
            print("Done saving")
    
    torch.save(MODEL.state_dict(), FULLY_TRAINED_MODEL_FILE)
