import torch
from torch.utils.data import IterableDataset
from datasets.data_utils import generate_random_text, text_to_image, text_to_tensor_and_pad_with_zeros, label_to_tensor

class StreamDataset(IterableDataset):
    
    def __init__(self, phrase_length, encoder_trainer):
        self.phrase_length = phrase_length
        self.encoder_trainer = encoder_trainer
    
    def generate(self):
        while True:
            
            if self.encoder_trainer:
                text = generate_random_text(length_of_char=self.phrase_length)
                target = label_to_tensor(text)
                image = torch.from_numpy(text_to_image(text)[1])
                training_sample = {"image": image, "expected": target}
                
                yield training_sample

            else:
                text = generate_random_text(length_of_char=self.phrase_length)
                target = text_to_tensor_and_pad_with_zeros(text)
                image = torch.from_numpy(text_to_image(text)[1])
                training_sample = {"image": image, "expected": target}

                yield training_sample

    def __iter__(self):
        return iter(self.generate())