import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel

class attn_config(BaseModel):
    """
        Configuration for the attn_module.

    Args:
        embed_dim: The list of embedding dimensions for each attention layer.
        num_heads: The list of number of heads for each attention layer.
        dropout: The list of dropout probabilities for each attention layer.
        input_dim: The dimension of the input.
        dict_dim: The dimension of the dictionary.
        synonym_head: The type of head to use for the synonym probabilities. Can be either "linear" or "softmax".
    """
    embed_dim: int = None
    num_heads: list = [None, None]
    dropout: list = [None, None]
    input_dim: int
    dict_dim: int
    synonym_head: str
    replace_head: str
    



# Batch version
class attn_module(nn.Module):
    """
    An attention module class for processing batches of input sequences using multi-head attention mechanisms.
    
    This class assumes that the input is batched and structured as (batch_size, num_tokens, num_features).
    It contains two series of linear transformations to generate queries, keys, and values for two separate
    attention mechanisms. The results from these attention mechanisms are then used to compute probabilities
    for replacing tokens and choosing synonyms from a dictionary of possible tokens.

    Attributes:
        q1, k1, v1 (nn.Linear): Linear layers for generating queries, keys, and values for the first attention mechanism.
        q2, k2, v2 (nn.Linear): Linear layers for generating queries, keys, and values for the second attention mechanism.
        proj1, proj2, proj3 (nn.Linear): Linear projection layers for transforming attention outputs.
        attn1, attn2 (nn.MultiheadAttention): Multi-head attention mechanisms.

    Args:
        config (object): Configuration object with attributes such as input_dim, embed_dim, num_heads, dropout, dict_dim,
                         replace_head, and synonym_head which control various aspects of the module.
    """    

    def __init__(self, config):
        super(attn_module, self).__init__()
        
        # Fully connected layers
        self.q1 = nn.Linear(config.input_dim, config.embed_dim)
        self.v1 = nn.Linear(config.input_dim, config.embed_dim)
        self.k1 = nn.Linear(config.input_dim, config.embed_dim)
        self.q2 = nn.Linear(config.input_dim, config.embed_dim)
        self.v2 = nn.Linear(config.input_dim, config.embed_dim)
        self.k2 = nn.Linear(config.input_dim, config.embed_dim)
        
        self.proj1 = nn.Linear(config.embed_dim, config.embed_dim)
        self.proj2 = nn.Linear(config.embed_dim, 1)
        self.proj3 = nn.Linear(config.embed_dim, config.dict_dim)
        
        # Attention layers with batch_first=True
        self.attn1 = nn.MultiheadAttention(config.embed_dim, config.num_heads[0], 
                                           dropout=config.dropout[0], batch_first=True)
        self.attn2 = nn.MultiheadAttention(config.embed_dim, config.num_heads[1], 
                                           dropout=config.dropout[1], batch_first=True)
        
        self.synonym_head = config.synonym_head
        self.replace_head = config.replace_head
        
        self.apply(self._xavier_initialization)

    def _xavier_initialization(self, module):
        """ author: Andrej karpathy
            NanoGPT:https://github.com/karpathy/nanoGPT/blob/master/model.py 
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias) 
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, input):
        """
        Forward pass for processing input sequences with attention mechanisms.

        Args:
            input (torch.Tensor): A batch of input sequences with shape (batch_size, num_tokens, num_features).
        
        Returns:
            torch.Tensor: A tensor with shape (batch_size, num_tokens, 1) containing the probability of
                          replacing each token (replace_probs).
            torch.Tensor: A tensor with shape (batch_size, num_tokens, dict_dim) containing the softmax
                          probabilities of each token being replaced with another token from the dictionary
                          (synonym_probs). That is, the entry synonym_probs[i][j] is the probability distribution
                          vector over all the words in the dictionary, for token j in batch i.
        """
        
        # Queries, Values, and Keys for both attention modules
        query1, value1, key1 = self.q1(input), self.v1(input), self.k1(input)
        query2, value2, key2 = self.q2(input), self.v2(input), self.k2(input)
        
        # Attention 1
        attn_out1, _ = self.attn1(query1, key1, value1)
        
        # Attention 2
        attn_out2, _ = self.attn2(query2, key2, value2)
        
        # Project attn_out2 to match dimensions for addition
        attn_out2_proj1 = self.proj1(attn_out2)
        
        # Compute replacement probabilities
        replace_probs = self.proj2(attn_out1 + attn_out2_proj1)
        if self.replace_head == "sigmoid":
            replace_probs = torch.sigmoid(replace_probs)
        elif self.replace_head == "linear":
            pass
    
        # Compute synonym probabilities
        synonym_probs = self.proj3(attn_out2)
        if self.synonym_head == "softmax":
            synonym_probs = F.softmax(synonym_probs, dim=-1)
        elif self.synonym_head == "linear":
            pass
        
        return replace_probs, synonym_probs

