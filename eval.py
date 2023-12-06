import torch

import pandas as pd

from src.backend.module import modules, nn
from src.backend.utils import load_base_model, load_tokenizer, preprocessed
from src.backend.train import evaluation

import os
import argparse


FILEPATH = os.path.dirname(os.path.realpath(__file__))
LOCAL_BASE_MODEL = os.path.join(FILEPATH, "src", "backend","data", 'weights', 'bert-base-uncased.pt')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='evaluation script with command-line arguments')

    parser.add_argument('-b', '--batch_size', type=int, default=7, help='Batch size for training')
    parser.add_argument('-t', '--threshold', type=float, default=0.6, help='hyperparameter threshold (0,1)')
    parser.add_argument('-f', '--file_path', type=str, required=True, help='Path to the input file')
    parser.add_argument('-c', '--checkpoint', type=str, required=False, help="Load model checkpoint")

    args = parser.parse_args()

    df = pd.read_csv(args.file_path)
    n_embd = 768
    dict_dim = 30522


    if not os.path.isfile(LOCAL_BASE_MODEL):
        load_base_model()
    
    print("[INFO] Initialization Bert model")
    model = nn.SyntaxBert.load_local_weights(nn.BertConfig)
    tokenizer = load_tokenizer()
    
    inputs, labels, _, _ = preprocessed(df, tokenizer)

    config = modules.attn_config(embed_dim=n_embd, 
                                num_heads=[2, 2], 
                                dropout=[0.1, 0.1], 
                                input_dim=n_embd, 
                                dict_dim=dict_dim, 
                                synonym_head="softmax", 
                                replace_head="sigmoid")    
    
    
    print("[INFO] Initialization Head")
    head = modules.attn_module(config=config)
    if args.checkpoint is not None:
        head.load_state_dict(torch.load(args.checkpoint))

    output = evaluation(
                        model=model,
                        head=head,
                        test=inputs,
                        batch_size=args.batch_size,
                        threshold=args.threshold
                      )
    
    output = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    pd.DataFrame({
        "ID" : range(len(output)),
        "OUTPUT" : output
    }).to_csv("output.csv", index=False)

    print(f'{" END PROGRAM ":#^100}')