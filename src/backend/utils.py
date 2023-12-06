import torch
import os
import re
import pandas as pd

__all__ = ["load_base_model", "load_tokenizer", "process_data"]

FILEPATH = os.path.dirname(os.path.realpath(__file__))
LOCAL_BASE_MODEL = os.path.join(FILEPATH, "data", 'weights', 'bert-base-uncased.pt')
LOCAL_TOKENIZER  = os.path.join(FILEPATH, "data", 'tokenizer', 'tokenizer.pt')

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



def ASSERTION_DATAFRAME_COLUMN(columns):
    columns = tuple(columns)
    DATASET_COLUMNS_WITH_ID = ("ID", "Input", "Output")
    DATASET_COLUMNS = ("Input", "Output")
    return DATASET_COLUMNS == columns or DATASET_COLUMNS_WITH_ID == columns


def preprocessed(df, tokenizer, vocab=30522, context_length=512, flag="~_~"):
    """"""
    
    assert isinstance(df, pd.DataFrame), "pre-process precondition needs to be in a DataFrame"
    assert ASSERTION_DATAFRAME_COLUMN(df.columns)

    # parse string to convert into a list
    df["Output"] = df["Output"].apply(lambda x : x.split(',') if isinstance(x, str) else x)

    # regular expression that obtains flag and associated word
    pattern = rf"{flag}(\w+)"

    # if ID exist within the column then remove it
    if "ID" in df.columns:
        df.drop("ID", axis=1, inplace=True)

    inputs, outputs, masks = [], [], []
    for i, sentence in enumerate(df["Input"]):
        replace = re.findall(pattern, sentence) # find all replaced tokens
        replacement : list = df["Output"][i]    # index the associated replacement list
        
        # replace tokens should equal replacement
        if len(replacement) != len(replace):
            continue

        # construct the replacement dictionary
        replacement_dict = {replace : replacement[i] for i, replace in enumerate(replace)}   
        mask_replacement_dict = {replace : tokenizer.mask_token for i, replace in enumerate(replace)}   

        # TODO Not ideal in initializing each iteration 
        def replaced(match):
            word = match.group(1)
            return replacement_dict.get(word, match.group(0))
        
        output_text = re.sub(pattern, replaced, sentence)
        mask_text = re.sub(pattern, lambda x :  mask_replacement_dict.get(x.group(1), x.group(0)), sentence)
        input_text = sentence.replace(flag, "")
        
        inputs.append(input_text)
        outputs.append(output_text)
        masks.append(mask_text)

    input_tokens = tokenizer(inputs, padding='max_length', max_length=context_length, return_tensors="pt", return_token_type_ids=False)
    output_tokens = tokenizer(outputs, padding='max_length', max_length=context_length, return_tensors="pt", return_token_type_ids=False)
    masked_tokens = tokenizer(masks, padding='max_length', max_length=context_length, return_tensors="pt", return_token_type_ids=False, return_attention_mask=False)
    masked_tokens = (masked_tokens["input_ids"] == tokenizer.mask_token_id)
    prob = torch.nn.functional.one_hot(output_tokens["input_ids"], num_classes=vocab)
    
    return input_tokens, output_tokens, masked_tokens, prob