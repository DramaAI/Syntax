from dataclasses import dataclass
import copy
import json
import os

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import math
import logging

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'syntax-base-bert': "[location of weights...]",
}
BERT_CONFIG_NAME = 'bert_config.json'
PT_WEIGHTS_PATH = './weights/model/syntax-bert-base.pt'


@dataclass
class BertConfig:
    vocab_size: int = 28996 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    max_positional_embeddings=512
    layer_norm_eps=1e-12
    type_vocab_size=2
    n_embd: int = 768
    n_layer: int= 12
    n_heads: int=12
    dropout: float = 0.1
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


    @classmethod
    def from_dict(cls, json_object):
        config = BertConfig(vocab_size=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config
    
    def to_dict(self) -> dict:
        output = copy.deepcopy(self.__dict__)
        return output
    
    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def __repr__(self) -> str:
        return str(self.to_json_string())

def download_weights(cls):
    """
        Function will only run once given the weights are not downloaded.
        host the weights on hugginface and allow user to download the weights
        on to there local machine in location so then the weight could then be loaded
        onto the syntax-bert.
    """
    ...
    

class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class SyntaxBertEmbedding(nn.Module):

    def __init__(self, config : BertConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=0)
        self.position_embeddings=nn.Embedding(num_embeddings=config.max_positional_embeddings, embedding_dim=config.n_embd)
        self.token_type_embeddings = nn.Embedding(num_embeddings=config.type_vocab_size, embedding_dim=config.n_embd)
        self.LayerNorm = nn.LayerNorm((config.n_embd, ), eps=config.layer_norm_eps, elementwise_affine=True)
        self.dropout = nn.Dropout(p=config.dropout)
    
    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

         

class SyntaxBertSelfAttention(nn.Module):

    def __init__(self, config : BertConfig) -> None:
        super(SyntaxBertSelfAttention, self).__init__()

        
        self.num_attn_heads = config.n_layer
        self.attn_head_size = config.n_embd // config.n_layer
        self.all_head_size = self.num_attn_heads * self.attn_head_size

        self.query = nn.Linear(in_features=config.n_embd, out_features=config.n_embd, bias=config.bias)
        self.key   = nn.Linear(in_features=config.n_embd, out_features=config.n_embd, bias=config.bias)
        self.value = nn.Linear(in_features=config.n_embd, out_features=config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(p=config.dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attn_heads, self.attn_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_state : Tensor, attention_mask : Tensor):
        q = self.query(hidden_state) 
        k = self.key(hidden_state)
        v = self.value(hidden_state)

        q_l = self.transpose_for_scores(q)
        k_l = self.transpose_for_scores(k)
        v_l = self.transpose_for_scores(v)

        attention_scores = q_l @ k_l.transpose(-1, -2) / math.sqrt(self.num_attn_heads)

        attention_scores += attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = attention_probs @ v_l
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)      
       
        return context_layer


class SyntaxSelfOutput(nn.Module):
    
    def __init__(self, config : BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(in_features=config.n_embd, out_features=config.n_embd, bias=config.bias)
        self.LayerNorm = nn.LayerNorm((config.n_embd,), eps=config.layer_norm_eps, elementwise_affine=True)
        self.dropout = nn.Dropout(p=config.dropout)
    
    def forward(self, hidden_state : Tensor, input : Tensor):
        return  self.LayerNorm(self.dropout(self.dense(hidden_state)) + input)

class SyntaxBertAttention(nn.Module):

    def __init__(self, config : BertConfig) -> None:
        super().__init__()
        self.self = SyntaxBertSelfAttention(config)
        self.output = SyntaxSelfOutput(config)
    
    def forward(self, input : Tensor, mask : Tensor):
        output = self.self(input, mask)
        return self.output(output, input)


class SyntaxBertIntermediate(nn.Module):
    
    def __init__(self, config : BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(in_features=config.n_embd, out_features=config.n_embd * 4)
        self.intermediate_act_fn = F.gelu
    
    def forward(self, input : Tensor):
        return self.intermediate_act_fn(self.dense(input))
  
class SyntaxBertOutput(nn.Module):
    
    def __init__(self, config : BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(in_features=config.n_embd * 4, out_features=config.n_embd)
        self.LayerNorm = nn.LayerNorm((config.n_embd,), eps=1e-12, elementwise_affine=True)
        self.dropout = nn.Dropout(p=config.dropout)
    
    def forward(self, hidden_states : Tensor, input : Tensor):
        return self.LayerNorm(\
                              self.dropout(\
                                            self.dense(hidden_states)\
                                          ) + input\
                             )

class SyntaxBertLayer(nn.Module):

    def __init__(self, config : BertConfig) -> None:
        super().__init__()
        self.attention = SyntaxBertAttention(config)
        self.intermediate = SyntaxBertIntermediate(config)
        self.output = SyntaxBertOutput(config)

    def forward(self, hidden_state  : Tensor, mask : Tensor):
        attention_output = self.attention(hidden_state, mask)
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)
    

class SyntaxBertEncoder(nn.Module):

    def __init__(self, config : BertConfig) -> None:
        super().__init__()

        self.layer = nn.ModuleList([
            SyntaxBertLayer(config) for _ in range(config.n_layer)
        ])
    
    def forward(self, hidden_states, mask, output_all_encoded_layer=True):
        all_encoder_layers = [0] *  (len(self.layer) if output_all_encoded_layer else 1)
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, mask)
            if output_all_encoded_layer:
                all_encoder_layers[i] = hidden_states
        if not output_all_encoded_layer:
            all_encoder_layers[0] = hidden_states
        return all_encoder_layers
    
class SyntaxBertPredictionHeadTransform(nn.Module):

    def __init__(self, config : BertConfig) -> None:
        super().__init__()
        
        self.transform_act_fn =  F.gelu
        self.dense =  nn.Linear(in_features=config.n_embd, out_features=config.n_embd, bias=config.bias)
        self.LayerNorm = nn.LayerNorm((config.n_embd,), eps=1e-12, elementwise_affine=True)

    def forward(self, input : Tensor):
        input = self.transform_act_fn(self.dense(input))
        return self.LayerNorm(input)


class SyntaxBertLMPredictionHead(nn.Module):

    def __init__(self, config : BertConfig) -> None:
        super().__init__()

        self.transform = SyntaxBertPredictionHeadTransform(config)
        self.decoder = nn.Linear(in_features=config.n_embd, out_features=config.vocab_size, bias=config.bias)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, input : Tensor):
        hidden_states = self.transform(input)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class SyntaxOnlyMLMHead(nn.Module):

    def __init__(self, config : BertConfig) -> None:
        super().__init__()
        self.predictions = SyntaxBertLMPredictionHead(config)

    def forward(self, input : Tensor):
        return self.predictions(input)
        
        
class SyntaxBertModel(nn.Module):

    def __init__(self, config : BertConfig) -> None:
        super().__init__()
        self.embeddings=SyntaxBertEmbedding(config)
        self.encoder=SyntaxBertEncoder(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=(next(self.parameters()).dtype))
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding,
                                      extended_attention_mask,
                                      output_all_encoded_layers)

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        
        return encoded_layers 

class SyntaxBert(nn.Module):

    def __init__(self, config : BertConfig) -> None:
        super().__init__()
        self.config : BertConfig = config
        self.bert = SyntaxBertModel(config)
        self.cls = SyntaxOnlyMLMHead(config)

    @classmethod
    def load_local_weights(cls, config : BertConfig, filepath : str):
        assert os.path.isfile(filepath), f'File Path Error: {filepath} is not a path to a pt file'
        try:
            logger.info("loading pytorch file (pt) from filepath...")
            temp = torch.load(filepath)
            model = cls(config)
            model.load_state_dict(temp.state_dict())
            logger.info("loaded weights local weights onto model...")
        except Exception as e:
            logger.warn("Something went wrong when loading weights onto the syntax-bert...")
            raise e

        return model
    

    def total_parameters(self, include_embedding=True):

        prams = 0
        for name, param in self.named_parameters():
            if not include_embedding  and"embedding" in name:
                 continue
            else:
                prams += param.numel()

        
        print(f"{prams // 1_000_000} Million")    

    def forward(self, inputs_ids : Tensor, token_type_ids=None, mask=None):
        sequence_output = self.bert(inputs_ids, token_type_ids, mask, output_all_encoded_layers=False)
        pred = self.cls(sequence_output)
        return pred
    