import os
import random
import torch
import Levenshtein
from tqdm import tqdm

import statistics as S
from einops import rearrange
import numpy as np
from constants import DEVICE, START_TOKEN, END_TOKEN, INFERENCE_PHRASE_LENGTH
from datasets.data_utils import decode_for_print, char_to_index

# CONSTANTS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
START_TOKEN = '\N{Start of Text}'
END_TOKEN = '\N{End of Text}'
PAD_TOKEN = '\N{Substitute}'

def train_model(train_dataset, model, optimizer, dataloader_length,
                encoder_trainer):
    
    model.train()
    print("TRAINING--->>>")

    loop = tqdm(enumerate(train_dataset), total=dataloader_length, leave=False)
    for i, each in loop:
        image = each['image'].to(DEVICE)
        expected_target = each['expected'].to(DEVICE)
        
        train_loss = 0.0
        running_loss = 0.0

        if encoder_trainer:
            _, loss, _ = model(image, expected_target)
        
        else:
            _, loss, _= model(image, expected_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_loss = running_loss / dataloader_length

        if i >= dataloader_length: break

    return train_loss

def evaluate_model(validation_dataset, model, dataloader_length,
                   encoder_trainer):

    print("EVALUATING--->>>")
    model.eval()

    pred_and_expected_list = []
    batch_distance_accumulator = []
    running_loss = 0.0
    val_loss = 0.0
    
    loop = tqdm(enumerate(validation_dataset), total=dataloader_length, leave=False)
    with torch.no_grad():
        for i, each in loop:
            image = each["image"].to(DEVICE)
            expected_target = each["expected"].to(DEVICE)

            if encoder_trainer:
                _, loss, _ = model(image, expected_target)
            
            else:
                logits, loss, expected_output = model(image, expected_target)

                batch_distance = each_batch_distance(logits, expected_output)
                batch_distance_accumulator.extend(batch_distance)
                
                predicted_and_expected = model_and_target_string(logits, expected_output)
                pred_and_expected_list.append(predicted_and_expected)

                percentile_10 = np.percentile(batch_distance_accumulator, 10)
                percentile_90 = np.percentile(batch_distance_accumulator, 90)

            running_loss += loss.item()
            val_loss = running_loss / dataloader_length

            if i >= dataloader_length: break

        for each in pred_and_expected_list:
            print(f"Predicted: {each[0]} Target: {each[1]}")

        print(f"Percentile 10: {percentile_10}, Percentile 90: {percentile_90}")
        print(f"Average: {S.fmean(batch_distance_accumulator)}, Minimum: {min(batch_distance_accumulator)}, Maximum: {max(batch_distance_accumulator)}")
    
    return val_loss

def save_checkpoint(epoch, model, optimizer, loss):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }

    return checkpoint

def get_latest_modified_file(model_checkpoint_folder, list_of_checkpoint_files): 
    latest_modified_checkpoint_file = ""
    for each in range(len(list_of_checkpoint_files)):
        checkpoint_file = list_of_checkpoint_files[each]
    
        if latest_modified_checkpoint_file == "":
            latest_modified_checkpoint_file += checkpoint_file
        
        checkpoint_mod_time = os.path.getmtime(os.path.join(model_checkpoint_folder, checkpoint_file))
        latest_checkpoint_mod_time = os.path.getmtime(os.path.join(model_checkpoint_folder, latest_modified_checkpoint_file))
        
        if checkpoint_mod_time < latest_checkpoint_mod_time:
            latest_modified_checkpoint_file.replace(latest_modified_checkpoint_file, checkpoint_file)

    return latest_modified_checkpoint_file

def load_checkpoint(model_checkpoint_folder, model, optimizer):
    list_of_file_checkpoints = os.listdir(model_checkpoint_folder)
    for _ in range(len(list_of_file_checkpoints)):
        try:
            latest_checkpoint_file = get_latest_modified_file(model_checkpoint_folder, list_of_file_checkpoints)
            checkpoint = torch.load(os.path.join(model_checkpoint_folder, latest_checkpoint_file))
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            load_epoch = checkpoint["epoch"]
            load_loss = checkpoint["loss"]
            print(f"Successfully load checkpoint from file: {latest_checkpoint_file}")
            break
        except RuntimeError:
            print(f"Failed to load checkpoint from file: {latest_checkpoint_file}")
            list_of_file_checkpoints.remove(latest_checkpoint_file)
            continue

    return load_epoch, load_loss, latest_checkpoint_file

def model_and_target_string(batched_model_logits, batched_expected):
    model_pred = batched_model_logits.data
    highest_idx = model_pred.topk(1)[1].squeeze(-1)
    model_prediction = highest_idx.cpu().numpy()

    batch_iter = batched_model_logits.shape[0]

    for i in range(batch_iter):
        predicted_sequence = model_prediction[i]
        expected_sequence = batched_expected[i]
        model_pred_str, target_str = decode_for_print(predicted_sequence), decode_for_print(expected_sequence)

        if i == batch_iter-1:
            return (model_pred_str, target_str)

def normalize_levenshtein_distance(model_output, expected):
    model_pred = model_output.data
    high_idx = model_pred.topk(1)[1].squeeze(-1)
    model_pred = high_idx.cpu().numpy()
    
    model_output_str, expected_str = decode_for_print(model_pred), decode_for_print(expected)
    distance = Levenshtein.distance(model_output_str, expected_str)

    max_length = max(len(model_output_str), len(expected_str))
    similarity = float(max_length - distance) / float(max_length)

    return similarity

def each_batch_distance(batch_model_output, batch_expected):
    each_distance = []
    batch_iter = batch_model_output.shape[0]
    for each in range(batch_iter):
        each_model_and_expected = batch_model_output[each], batch_expected[each]
        distance = normalize_levenshtein_distance(each_model_and_expected[0], each_model_and_expected[1])
        each_distance.append(distance)

    return each_distance

def generate_square_mask(size):
    mask = (torch.triu(torch.ones((size, size), device=DEVICE)) == 1).transpose(1, 0)
    mask = mask.float().masked_fill(mask == 0, float("-inf")
                                     ).masked_fill(mask == 1, float(0.0)).to(DEVICE)
    return mask

def generate_token_prediction(model_input, model, max_length=INFERENCE_PHRASE_LENGTH+3):
    image = model_input['image'].unsqueeze(0).to("cuda")
    expected = model_input['expected'].to("cuda")
    model.eval()
    
    memory = model.encode(image).to("cuda")
    predicted = torch.tensor([char_to_index[START_TOKEN]], device="cuda", dtype=torch.long).unsqueeze(0)  #torch.ones(1, 1).fill_(char_to_index[START_TOKEN]).type(torch.long).to("cuda")
    for _ in range(max_length-1):
        start_masked = generate_square_mask(predicted.size(1))
        out = model.decode(predicted, memory, start_masked)
        prob = out[:, -1]
        _, next_char_predicted = torch.max(prob, dim=1)
        next_char_predicted = next_char_predicted.item()

        predicted = torch.cat((predicted, torch.tensor([next_char_predicted], device=DEVICE).unsqueeze(0)), dim=1)

        if next_char_predicted == char_to_index[END_TOKEN]:
            break

    return predicted, expected
