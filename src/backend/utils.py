import torch
import os

FILEPATH = os.path.dirname(os.path.realpath(__file__))
LOCAL_BASE_MODEL = os.path.join(FILEPATH, "data", 'weights', 'bert-base-uncased.pt')
LOCAL_TOKENIZER  = os.path.join(FILEPATH, "data", 'tokenizer', 'tokenizer.pt')

__all__ = ["load_base_model", "load_tokenizer"]

def load_base_model():
    model = None 
    if not os.path.isfile(LOCAL_BASE_MODEL):
        from transformers import BertForMaskedLM
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        torch.save(model, LOCAL_BASE_MODEL)
    else:
        model = torch.load(LOCAL_BASE_MODEL)
    return model
    

def load_tokenizer():
    token = None
    if not os.path.isfile(LOCAL_TOKENIZER):
        from transformers import AutoTokenizer
        token = AutoTokenizer.from_pretrained('bert-base-uncased')
        torch.save(token, LOCAL_TOKENIZER)
    else:
        token = torch.load(LOCAL_TOKENIZER)
    return token

