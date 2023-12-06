import gradio as gr
import os
import argparse
from src.backend.module.nn import Syntax, SyntaxBert, BertConfig
from src.backend.module.modules import attn_config, attn_module
from src.backend.utils import load_tokenizer
import torch

MODEL = None
TOKENIZER = load_tokenizer()
EMBD = 768
DICT_DIM = 30522

def is_pt_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower() == '.pt'


def demo(prompt, temperature):
    if prompt is None:
        return ""
    

    inputs = TOKENIZER(prompt, return_tensors="pt", return_token_type_ids=False)
    print(inputs)
    logits_r, logits_s = MODEL(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    replacement_mask = logits_r > temperature
    indices = torch.nonzero(replacement_mask)

    if len(indices) == 0:
        return ""
    new_tokens = logits_s[0, indices[0, 1]].argmax(dim=-1)
    inputs["input_ids"][0, indices[0, 1]] = new_tokens
    return TOKENIZER.decode(inputs["input_ids"][0], skip_special_tokens=True)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='evaluation script with command-line arguments')

    parser.add_argument('-c', '--checkpoint', type=str, required=False, default=None, help="Load model checkpoint")
    parser.add_argument('-d', '--device', type=str, required=False, default="cpu", help="where to allocate space for model" )

    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint) and is_pt_file(args.checkpoint):
        raise FileNotFoundError(f"{args.checkpoint} is either not a pt file or an existing file")
    
    bert = SyntaxBert.load_local_weights(BertConfig)
    tokenizer = load_tokenizer()
    
    config = attn_config(embed_dim=EMBD, 
                         num_heads=[2, 2], 
                         dropout=[0.1, 0.1], 
                         input_dim=EMBD, 
                         dict_dim=DICT_DIM, 
                         synonym_head="softmax", 
                         replace_head="sigmoid")    

    
    print("[INFO] Initialization Model")
    head = attn_module(config=config)
    MODEL = Syntax(bert=bert, head=head)
    MODEL.load_state_dict(torch.load(args.checkpoint, map_location=torch.device(args.device)))
    
    syntax = gr.Interface(
    fn=demo,
    inputs=[gr.TextArea(lines=10), gr.Slider(minimum=0.1, maximum=1.0, label="Temperature")],
    outputs=gr.TextArea(line=10),
    live=True,
    title="Syntax",
    description="Enter a prompt, adjust temperature, to adjust text in live real time.",
    )

    syntax.launch()



   
