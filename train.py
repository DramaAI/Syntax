import torch
from torch.optim import Adam

import pandas as pd

from src.backend.module import loss as Loss, modules, nn
from src.backend.utils import load_base_model, load_tokenizer, preprocessed
from src.backend.train import training

import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

FILEPATH = os.path.dirname(os.path.realpath(__file__))
LOCAL_BASE_MODEL = os.path.join(FILEPATH, "src", "backend","data", 'weights', 'bert-base-uncased.pt')



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Training script with command-line arguments')

    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('-b', '--batch_size', type=int, default=7, help='Batch size for training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('-f', '--files', type=list, required=False, default=['data/xav_training_examples.csv'], help='Path to the input file')
    parser.add_argument('-c', '--checkpoint', type=str, required=False, help="Load model checkpoint")
    parser.add_argument('-fl', '--flag', type=int, required=False, default=5, help="Each epoch display loss")


    args = parser.parse_args()

    df = None
    if len(args.files) == 1:
        df = pd.read_csv(args.files[0])
    elif len(args.files) > 1:
        temp = [0]*len(args.files)
        for i, file in enumerate(args.files):
            temp[i] = pd.read_csv(file)
        df = pd.concat(temp) 
    else:
        raise FileNotFoundError("You muse provide a file or files to run this program")
    
    # hyperparameters from bert-base-model
    n_embd = 768     # embedding space per token
    dict_dim = 30522 # vocabulary size

    if not os.path.isfile(LOCAL_BASE_MODEL):
        load_base_model()


    print("[INFO] Initialization Bert model")
    bert = nn.SyntaxBert.load_local_weights(nn.BertConfig)
    
    # bert tokenizer
    tokenizer = load_tokenizer()
    
    # config for attention head
    config = modules.attn_config(embed_dim=n_embd, 
                                num_heads=[2, 2], 
                                dropout=[0.1, 0.1], 
                                input_dim=n_embd, 
                                dict_dim=dict_dim, 
                                synonym_head="linear", 
                                replace_head="linear")    

    print("[INFO] Initialization Head")
    head = modules.attn_module(config=config)

    model = nn.Syntax(bert=bert, head=head)

    if args.checkpoint is not None:
        print("[INFO] Loading checkpoint weights...")
        model.load_state_dict(torch.load(args.checkpoint))


    # optimizer initialization 
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    # Custom loss function
    loss_fn = Loss.JointCrossEntropy(head_type="linear")

    # Preprocess the dataframe
    inputs, labels, replacement, synonyms = preprocessed(df, tokenizer)

    print(f"[INFO] Training Initialization | epoch: {args.epochs:^10} | batch: {args.batch_size:^10} | learning rate: {args.learning_rate:^10} |")

    # Training process
    losses = training(
          model=model,
          train=(inputs,replacement,synonyms),
          optimizer=optimizer,
          loss_fn=loss_fn,
          batch_size=args.batch_size,
          epoch=args.epochs,
          flag=args.flag
    )


    print(f'{" END PROGRAM ":#^100}')