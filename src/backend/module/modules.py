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
    
    embed_dim: list = [None, None]
    num_heads: list = [None, None]
    dropout: list = [None, None]
    input_dim: int
    dict_dim: int
    synonym_head: str
    replace_head: str
    


class attn_module(nn.Module):
    def __init__(self, config):
        super(attn_module, self).__init__()
        
        # Fully connected layers
        self.q1 = nn.Linear(config.input_dim, config.embed_dim[0])
        self.v1 = nn.Linear(config.input_dim, config.embed_dim[0])
        self.k1 = nn.Linear(config.input_dim, config.embed_dim[0])
        self.q2 = nn.Linear(config.input_dim, config.embed_dim[1])
        self.v2 = nn.Linear(config.input_dim, config.embed_dim[1])
        self.k2 = nn.Linear(config.input_dim, config.embed_dim[1])
        
        
        self.proj1 = nn.Linear(config.embed_dim[1], config.embed_dim[0])
        self.proj2 = nn.Linear(config.embed_dim[0], 1)
        self.proj3 = nn.Linear(config.embed_dim[1], config.dict_dim)
        
        # Attention layers
        self.attn1 = nn.MultiheadAttention(config.embed_dim[0], config.num_heads[0], config.dropout[0])
        self.attn2 = nn.MultiheadAttention(config.embed_dim[1], config.num_heads[1], config.dropout[1])

    def forward(self, input, config):
        """
           Forward pass for the attention module.

        Args:
            input (torch tensor): Input tensor of shape (num_tokens, num_features) 

        Returns:
            reaplce_probs (torch tensor): Tensor of shape (num_tokens, 1) containing the probability of replacing each token with a synonym. Depends on
                                            config.replace_head which has options "linear" and "sigmoid". If "sigmoid" then probabilities are returned,
                                            if "linear" then sigmoid is not applied which is useful for the torch.nn.BCELoss() as it applies sigmoid internally.
            synonym_probs (torch tensor): Tensor of shape (num_tokens, dict_size) containing the softmax probability of replacing each token with another token.
                                          Depends on config.synonym_head which has options "linear" and "softmax". If "softmax" then probabilities are returned,
                                          if "linear" then softmax is not applied which is useful for the torch.nn.CrossEntropyLoss() as it applies softmax internally.
        """
        
        # Queries, Values, and Keys
        query1, query2, value1, value2, key1, key2 = self.q1(input), self.q2(input), self.v1(input), self.v2(input), self.k1(input), self.k2(input)
        
        # Attention 1
        attn_out1, _ = self.attn1(query1, key1, value1, need_weights=False)
        
        # Attention 2.
        attn_out2, _ = self.attn2(query2, key2, value2, need_weights=False)
        
        # Project attention output 2 to correct dimension 
        attn_out2_proj1 = self.proj1(attn_out2)
        
        # Replacement probabilities
        # Sum attention outputs project to 1 dimension
        replace_probs = self.proj2(attn_out1 + attn_out2_proj1)
        if config.replace_head == "linear":
            replace_probs = replace_probs
        elif config.replace_head == "sigmoid":
            replace_probs = F.sigmoid(replace_probs)
        
        # Synonym probabilities
        # Project attention outputs to dictionary dimension
        attn_out2_proj3 = self.proj3(attn_out2)
        
        if config.synonym_head == "linear":
            synonym_probs = attn_out2_proj3
        elif config.synonym_head == "softmax":
            synonym_probs = F.softmax(attn_out2_proj3, dim=0)
        
        return replace_probs, synonym_probs




# Test
config = attn_config(embed_dim=[512, 512], num_heads=[2,2], dropout=[0.0, 0.0], input_dim=256, dict_dim=20000, synonym_head="linear", replace_head="linear")

# Input must be of shape (num_tokens, num_features)
input = torch.randn(100, 256)

attn_mech = attn_module(config)

a1, a2 = attn_mech(input, config)

print(f"Replacement Probabilities Shape: {a1.shape}")
print(f"Synonym Probabilities Shape: {a2.shape}")
print(f"Replacement Probabilities: {a1}")
print(f"Synonym Probabilities: {a2}")
