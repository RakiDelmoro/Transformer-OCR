import torch
import torch.nn as nn
from model.configurations import DecoderConfig
import math

def create_position_ids_from_input_ids(input_ids: torch.Tensor, padding_idx, past_key_values_length=0):
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx

class RobertaEmbeddings(nn.Module):

    def __init__(self, config=DecoderConfig):
        super().__init__()
        self.character_embeddings = nn.Embedding(config.vocab_size, config.embedding_dimension,
                                                 config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_sequence_length, config.embedding_dimension)
        
        self.layer_norm = nn.LayerNorm(config.embedding_dimension, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.embedding_dropout)
        
        self.register_buffer("position_ids", torch.arange(config.max_sequence_length).expand((1, -1)), persistent=False)

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_sequence_length, config.embedding_dimension, padding_idx=self.padding_idx
        )

    def forward(self, input_ids, past_key_value_length=0):

        input_embedding = self.character_embeddings(input_ids)
        position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_value_length)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embedding + position_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
    
class RobertaSelfAttention(nn.Module):
    def __init__(self, config=DecoderConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_embedding_dim = int(config.embedding_dimension / config.num_attention_heads)
        self.combine_embedding_dim = self.num_attention_heads * self.attention_embedding_dim

        self.query = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.key = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.value = nn.Linear(config.embedding_dimension, config.embedding_dimension)
    
        self.dropout = nn.Dropout(config.attention_dropout)

        self.max_position_embeddings = config.max_sequence_length
        self.distance_embedding = nn.Embedding(2 * config.max_sequence_length - 1, self.attention_embedding_dim)

    def transpose_for_scores(self, x: torch.Tensor):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_embedding_dim)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask):
        
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # Take the dot product between query and key to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-2, -1))

        query_length, key_length = query_layer.shape[2], key_layer.shape[2]
        position_ids_left = torch.arange(query_length, dtype=torch.long,
                                            device=hidden_states.device).view(-1, 1)
        position_ids_right = torch.arange(key_length, dtype=torch.long,
                                            device=hidden_states.device).view(1, -1)
        distance = position_ids_left - position_ids_right
        position_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
        positional_embedding = position_embedding.type(query_layer.dtype)

        relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
        relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding) 
        attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        
        attention_scores = attention_scores / math.sqrt(self.attention_embedding_dim)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer = context_layer.size()[:-2] + (self.combine_embedding_dim,)
        context_layer = context_layer.view(new_context_layer)

        return context_layer

class RobertaSelfAttentionOutput(nn.Module):
    def __init__(self, config=DecoderConfig):
        super().__init__()
        self.dense = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.layer_norm = nn.LayerNorm(config.embedding_dimension, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.embedding_dropout)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        
        return hidden_states
    
class RobertaAttention(nn.Module):
    def __init__(self, config=DecoderConfig):
        super().__init__()
        self.self_attn = RobertaSelfAttention(config)
        self.output = RobertaSelfAttentionOutput(config)

    def forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask):
        
        self_outputs = self.self_attn(hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        
        return attention_output
    
class RobertaFeedForward(nn.Module):
    def __init__(self, config=DecoderConfig):
        super().__init__()
        self.dense = nn.Linear(config.embedding_dimension, config.intermediate_size)
        self.act_fnc = config.embedding_act

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fnc(hidden_states)

        return hidden_states

class RobertaOutput(nn.Module):
    def __init__(self, config=DecoderConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.embedding_dimension)
        self.layer_norm = nn.LayerNorm(config.embedding_dimension, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.embedding_dropout)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)

        return hidden_states

class RobertLayer(nn.Module):
    def __init__(self, config=DecoderConfig):
        super().__init__()
        self.attention = RobertaAttention(config)
        self.cross_attention = RobertaAttention(config)
        
        self.intermediate = RobertaFeedForward(config)
        self.output = RobertaOutput(config)

    def forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask):
        
        self_attention_outputs = self.attention(
        hidden_states, attention_mask, None, None)

        if encoder_hidden_states is not None:
            cross_attention_outputs = self.cross_attention(self_attention_outputs, attention_mask,
                                                           encoder_hidden_states, encoder_attention_mask)
        
        layer_output = self.intermediate(cross_attention_outputs)

        return self.output(layer_output, cross_attention_outputs)

class RobertaDecoder(nn.Module):
    def __init__(self, config=DecoderConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RobertLayer(config) for _ in range(config.num_decoder_layers)])

    def forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask):
        
        if encoder_hidden_states is not None:
            encoder_batch_size, encoder_seq_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_seq_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device="cuda")
    
        for each_layer in self.layer:
            layer_outputs = each_layer(hidden_states, attention_mask,
                                       encoder_hidden_states, encoder_attention_mask)
            
        return layer_outputs
    
class RobertaModel(nn.Module):
    def __init__(self, config=DecoderConfig):
        super().__init__()
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.decoder = RobertaDecoder(config)

        self.output_head = nn.Sequential(
            nn.Linear(config.embedding_dimension, config.embedding_dimension),
            nn.GELU(),
            nn.LayerNorm(config.embedding_dimension, eps=config.layer_norm_eps),
        )

        self.output = nn.Linear(config.embedding_dimension, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, input_ids, expected_ids, attention_mask, encoder_hidden_states, encoder_attention_mask):

        batch, seq_length = input_ids.size()
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch, seq_length)), device=device)

        embedding_output = self.embeddings(input_ids)

        decoder_output = self.decoder(embedding_output, attention_mask,
                                      encoder_hidden_states, encoder_attention_mask)

        prediction_score = self.output_head(decoder_output)
        logits = self.output(prediction_score)

        model_loss = None
        if expected_ids is not None:
            labels = expected_ids.to(prediction_score.device)
            shifted_logits = logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_func = nn.CrossEntropyLoss()
            model_loss = loss_func(shifted_logits.view(-1, self.config.vocab_size), labels.view(-1))
            return shifted_logits, model_loss, labels
        
        return logits
