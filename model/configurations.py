import torch.nn as nn
from model.constants import CHARS, PAD_TOKEN, CHAR_TO_INDEX

# CONSTANTS
IMAGE_SIZE = (32, 800)
PATCH_SIZE = (IMAGE_SIZE[0], 32)
NUM_CLASSES = 128
ENCODER_EMB = 1024
NUM_ENCODER_LAYER = 24
NUM_ENC_ATTN_HEADS = 16
MLP_DIMENSION = ENCODER_EMB*4
ENC_DROPOUT = 0.1

DECODER_EMB = 1024
DIM_FEED_FORWARD = DECODER_EMB*4
DEC_DROPOUT = 0.1
NUM_DECODER_LAYER = 24
NUM_DEC_ATTN_HEADS = 16

class DecoderConfig:
    vocab_size=len(CHARS)
    embedding_dimension=DECODER_EMB
    num_decoder_layers=NUM_DECODER_LAYER
    num_attention_heads=NUM_DEC_ATTN_HEADS
    intermediate_size=DIM_FEED_FORWARD
    embedding_act=nn.GELU()
    embedding_dropout=0.1
    attention_dropout=0.1
    max_sequence_length=150
    layer_norm_eps=1e-12
    pad_token_id=CHAR_TO_INDEX[PAD_TOKEN]
    add_cross_attention=True
    is_decoder=True

class EncoderConfig:
    image_size=IMAGE_SIZE
    patch_size=(image_size[0], image_size[0])
    num_classes=128
    embedding_dimension=ENCODER_EMB
    num_encoder_layers=NUM_ENCODER_LAYER
    num_attention_heads=NUM_ENC_ATTN_HEADS
    intermediate_size=MLP_DIMENSION
    encoder_dropout=ENC_DROPOUT
