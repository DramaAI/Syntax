import torch
import torch.nn as nn

import os
import sys

FILEPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FILEPATH, '..'))

from module import nn, modules

def training(model        : nn.SyntaxBert, 
             head         : modules.attn_module,
             X            : torch.Tensor,
             replacements : torch.Tensor,
             synonyms     : torch.Tensor,
             optimizer    : any,
             loss_fn      : any,
             batch_size   : int=16,
             epoch        : int=2,
             **kwarg):
    
    
    # pre-train process =========================================
    flag = ( 10 
             if epoch > 50 
             else kwarg.get("flag", 1)  )
    model_name = type(model).__name__
    total_dataset = len(X)
    # off load forward and back propagation to the cuda kernel
    device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
             )


    head.to(device)
    model.to(device)

    # freeze Bert Weights
    # 
    for param in model.parameters():
        param.required_grad = False


    avg_loss = []
    # train process ============================================
    for i in range(epoch):
        losses = []
        for batch in range(0, total_dataset, batch_size):
        
            # for each batch zero grad 
            optimizer.zero_grad()

            # index input
            x = X[batch:batch+batch_size, ...].to(device)
                
            # forward pass
            _, hidden_layer = model(x)
            logits_r, logits_s = head(hidden_layer.to(device))
        

            # access batch replacements, and synonyms
            syn_y = synonyms[batch:batch+batch_size, ...].float().to(device)
            rep_y = replacements[batch:batch+batch_size, ...].float().unsqueeze(-1).to(device)
            

            # Compute the loss and its gradients
            loss = loss_fn(logits_s.to(device), logits_r.to(device), syn_y, rep_y)
            loss.backward()

            # Adjust learning weights
            optimizer.step()
            losses.append(loss.item())
            
        avg_loss.append(sum(losses)/len(losses)) 

        if i % flag == 0:
            print(f"[INFO] |{f'model: {model_name:^5}':^20}|{f'epoch: {i:^5}':^20}|{f'avg loss: {avg_loss[i]:^5.4f}':^20}|")
    
    return avg_loss
    # ==========================================================


