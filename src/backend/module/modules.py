import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
import unittest

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
        self.q1 = nn.Linear(config.input_dim, config.embed_dim[0])
        self.v1 = nn.Linear(config.input_dim, config.embed_dim[0])
        self.k1 = nn.Linear(config.input_dim, config.embed_dim[0])
        self.q2 = nn.Linear(config.input_dim, config.embed_dim[1])
        self.v2 = nn.Linear(config.input_dim, config.embed_dim[1])
        self.k2 = nn.Linear(config.input_dim, config.embed_dim[1])
        
        self.proj1 = nn.Linear(config.embed_dim[1], config.embed_dim[0])
        self.proj2 = nn.Linear(config.embed_dim[0], 1)
        self.proj3 = nn.Linear(config.embed_dim[1], config.dict_dim)
        
        # Attention layers with batch_first=True
        self.attn1 = nn.MultiheadAttention(config.embed_dim[0], config.num_heads[0], 
                                           dropout=config.dropout[0], batch_first=True)
        self.attn2 = nn.MultiheadAttention(config.embed_dim[1], config.num_heads[1], 
                                           dropout=config.dropout[1], batch_first=True)
        
        self.synonym_head = config.synonym_head
        self.replace_head = config.replace_head

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



if __name__ == '__main__':

    # Unit tests
    config = attn_config(embed_dim=[512, 512], num_heads=[2, 2], dropout=[0.0, 0.0], 
                        input_dim=256, dict_dim=3, synonym_head="softmax", replace_head="sigmoid")

    # Input must now be of shape (batch_size, num_tokens, num_features)
    # Let's assume a batch_size of 10 for this example
    batch_size = 2
    num_tokens = 3
    input_features = 256

    # Generating random input to simulate a batch of sequences
    input = torch.randn(batch_size, num_tokens, input_features)

    # Instantiate the attention module with the given configuration
    attn_mech = attn_module(config)

    # Forward pass through the attention mechanism
    # Note that config is no longer passed as an argument to the forward method
    replace_probs, synonym_probs = attn_mech(input)

    # Print out shapes and values
    print(f"Replacement Probabilities Shape: {replace_probs.shape}")  # Expected: (batch_size, num_tokens, 1)
    print(f"Synonym Probabilities Shape: {synonym_probs.shape}")      # Expected: (batch_size, num_tokens, dict_dim)
    print(f"Replacement probabilities for each token in Batch 2: {replace_probs[1]}")           # Expected: A vector of probabilities for each token in the batch. There are three words, so three probabilities.
    print(f"Replacement probability Batch 1, Token 1: {replace_probs[0][0]}")        # Expected: Value between 0 and 1
    print(f"Synonym Probability Distribution for the Batch 1, Token 2: {synonym_probs[0][1]}") # Expected: A vector of probabilities for each word in the dictionary. There are three words, so three probabilities.
    print(f"Synonym Probabilities Sum-to-1 Constraint for Token 1: {torch.sum(synonym_probs[0][0])}") # Expected : Sum to 1 constraint for the softmax probabilities, for the first token

