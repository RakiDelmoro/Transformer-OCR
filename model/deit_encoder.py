import torch
import torch.nn as nn
import math

from einops.layers.torch import Rearrange
from model.configurations import EncoderConfig


class DeitEmbeddings(nn.Module):
    """
        Construct the CLS token, distillation token, and patch embeddings
    """
    def __init__(self, config=EncoderConfig):
        super().__init__()

        image_h, image_w = config.image_size
        patch_h, patch_w = config.patch_size

        number_of_patches = (image_h // patch_h) * (image_w // patch_w)
        patch_embedding_dimension = patch_h * patch_w

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (h p1) (w p2) -> b (h w) (p1 p2)', p1=patch_h, p2=patch_w),
            nn.Linear(patch_embedding_dimension, config.embedding_dimension)
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embedding_dimension))
        self.distillation_token = nn.Parameter(torch.zeros(1, 1, config.embedding_dimension))
        self.position_embeddings = nn.Parameter(torch.zeros(1, number_of_patches + 2, config.embedding_dimension))
        self.dropout = nn.Dropout(config.encoder_dropout)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        embeddings = self.to_patch_embedding(image)
        batch_size, _, _ = embeddings.shape

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        distillation_tokens = self.distillation_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, distillation_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings
    
class DeitSelfAttention(nn.Module):
    def __init__(self, config=EncoderConfig):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.atttention_embedding_dim = int(config.embedding_dimension // config.num_attention_heads)
        self.combine_embedding_size = self.num_heads * self.atttention_embedding_dim

        self.query = nn.Linear(config.embedding_dimension, self.combine_embedding_size)
        self.key = nn.Linear(config.embedding_dimension, self.combine_embedding_size)
        self.value = nn.Linear(config.embedding_dimension, self.combine_embedding_size)

        self.dropout = nn.Dropout(config.encoder_dropout)

    def transpose_for_attn_scores(self, x: torch.Tensor) -> torch.Tensor:
        # batch | patches | attn_heads | attention_dimension
        new_x_shape = x.shape[:-1] + (self.num_heads, self.atttention_embedding_dim)
        x = x.view(new_x_shape)

        # batch | attn_heads | patches | attention_dimension
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states: torch.Tensor):

        query_layer = self.transpose_for_attn_scores(self.query(hidden_states))
        key_layer = self.transpose_for_attn_scores(self.key(hidden_states))
        value_layer = self.transpose_for_attn_scores(self.value(hidden_states))

        # Take the dot product of "query" and "key" to the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-2, -1))

        attention_scores = attention_scores / math.sqrt(self.atttention_embedding_dim)

        # normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.shape[:-2] + (self.combine_embedding_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer

class DeitSelfAttetionOutput(nn.Module):
    def __init__(self, config=EncoderConfig):
        super().__init__()

        self.dense = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.dropout = nn.Dropout(config.encoder_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

class DeitAttention(nn.Module):
    def __init__(self, config=EncoderConfig):
        super().__init__()

        self.attention = DeitSelfAttention(config)
        self.output = DeitSelfAttetionOutput(config)

    def forward(self, hidden_states: torch.Tensor):
        self_attention = self.attention(hidden_states)
        attention_output = self.output(self_attention)
        
        return attention_output

class DeitFeedForward(nn.Module):
    def __init__(self, config=EncoderConfig):
        super().__init__()

        self.dense = nn.Linear(config.embedding_dimension, config.intermediate_size)
        self.activation_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        return hidden_states
    
class DeitOutput(nn.Module):
    def __init__(self, config=EncoderConfig):
        super().__init__()

        self.dense = nn.Linear(config.intermediate_size, config.embedding_dimension)
        self.dropout = nn.Dropout(config.encoder_dropout)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states    

class DeitLayer(nn.Module):
    def __init__(self, config=EncoderConfig):
        super().__init__()

        self.attention = DeitAttention(config)
        self.ff_layer = DeitFeedForward(config)
        self.output = DeitOutput(config)
        self.layer_norm_before = nn.LayerNorm(config.embedding_dimension)
        self.layer_norm_after = nn.LayerNorm(config.embedding_dimension)

    def forward(self, hidden_states: torch.Tensor):
        self_attention_outputs = self.attention(self.layer_norm_before(hidden_states))
        
        # first residual connection
        hidden_states = self_attention_outputs + hidden_states
        # Apply Layer Norm
        layer_output = self.layer_norm_after(hidden_states)
        # Feed Forward
        layer_output = self.ff_layer(layer_output)
        # second residual connection
        layer_output = self.output(layer_output, hidden_states)

        return layer_output
    
class DeitEncoder(nn.Module):
    def __init__(self, config=EncoderConfig):
        super().__init__()
        self.config = config
        self.embeddings = DeitEmbeddings(config)
        self.encoder_layer = DeitLayer(config)
        
        self.encoder_layer = nn.ModuleList([self.encoder_layer for _ in range(config.num_encoder_layers)])

    def forward(self, image: torch.Tensor):
        hidden_states = self.embeddings(image)
        
        for layer_module in self.encoder_layer:
            layer_outputs = layer_module(hidden_states)

        return layer_outputs

class DeitModel(nn.Module):
    def __init__(self, config=EncoderConfig):
        super().__init__()
        self.num_classes = config.num_classes
        self.embed_dim = config.embedding_dimension
        self.encoder_layer = DeitEncoder(config.image_size, config.patch_size, config.embedding_dimension, config.num_encoder_layer,
                                         config.intermediate_size, config.encoder_dropout, config.encoder_dropout,
                                         config.num_attention_heads, config.num_classes)
        
        self.layer_norm = nn.LayerNorm(config.embedding_dimension, config.embedding_dimension)
        self.pooler = nn.Sequential(
            nn.Linear(config.embedding_dimension, config.embedding_dimension),
            nn.Tanh()
        )

        self.classier = nn.Linear(config.embedding_dimension, self.num_classes)

    def forward(self, image: torch.Tensor, expected: torch.Tensor):
        
        encoder_output = self.encoder_layer(image)

        sequence_output = self.layer_norm(encoder_output)
        logits = self.classier(sequence_output[:, 0, :])
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, expected.view(-1))

        return logits, loss

