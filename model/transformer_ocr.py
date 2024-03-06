import torch
import torch.nn as nn
from torch import Tensor

from model.deit_encoder import DeitEncoder
# from model.beit_encoder import BeitEncoder
from model.roberta_decoder import RobertaModel

from model.configurations import EncoderConfig, DecoderConfig
from model.utils import generate_square_mask

class TransformerOcr(nn.Module):
    def __init__(self, encoder=DeitEncoder, decoder=RobertaModel, enc_config=EncoderConfig, dec_config=DecoderConfig):
        super(TransformerOcr, self).__init__()
        self.encoder = encoder(enc_config)
        self.decoder = decoder(dec_config)

        self.encoder_embedding = enc_config.embedding_dimension
        self.decoder_embedding = dec_config.embedding_dimension

        self.enc_to_dec_embedding = nn.Linear(self.encoder_embedding, self.decoder_embedding)

    def forward(self, image: torch.Tensor, target_idx: torch.Tensor):
        encoded_image = self.encoder(image)
        memory = self.enc_to_dec_embedding(encoded_image)

        target_idx_length = target_idx.shape[1]
        target_attention_mask = generate_square_mask(target_idx_length)

        model_output = self.decoder(target_idx, target_idx, target_attention_mask,
                                    memory, None)
        return model_output
    
    def encode(self, image: torch.Tensor):
        encoded_image = self.encoder(image)
        memory = self.enc_to_dec_embedding(encoded_image)

        return memory
    
    def decode(self, input_ids: Tensor, memory: Tensor, target_mask: Tensor):
        return self.decoder(input_ids, None, target_mask, memory, None)

# model = TransformerOcr()
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Number of parameters: {total_params}")
