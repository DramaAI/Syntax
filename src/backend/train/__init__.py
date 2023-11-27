import torch
import torch.nn as nn

import os
import sys

FILEPATH = os.path.dirname(os.path.realpath(__file__))
LOCAL_HEAD_MODEL = os.path.join(FILEPATH, "..", "data", 'weights', "head", 'bert-base-attention-head.pt')
sys.path.append(os.path.join(FILEPATH, '..'))


from module import nn, modules

def evaluation( model        : nn.SyntaxBert, 
                head         : modules.attn_module,
                test         : torch.Tensor,
                batch_size   : int=16,
                threshold    : float=0.6):
    
    X_test = test
    total_dataset = len(X)
    device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
             )


    head.to(device)
    model.to(device)

    head.replace_head = "sigmoid"
    head.synonym_head == "softmax"

    with torch.no_grad():
        for batch in range(0, total_dataset, batch_size):
            
                # index input
                x = X_test[batch:batch+batch_size, ...].to(device)
                    
                # forward pass
                _, hidden_layer = model(x)
                logits_r, logits_s = head(hidden_layer.to(device))

                replacement_mask = logits_r > threshold
                indices = torch.nonzero(replacement_mask)

                new_tokens = logits_s[indices[:, 0], indices[:, 1]].argmax(axis=-1)
                x[indices[:, 0], indices[:, 1]] = new_tokens

    return X_test     

                
               
def training(model        : nn.SyntaxBert, 
             head         : modules.attn_module,
             train        : tuple[torch.Tensor, torch.Tensor, torch.Tensor],
             optimizer    : any,
             loss_fn      : any,
             batch_size   : int=16,
             epoch        : int=2,
             **kwarg):
    
    X_train, y_train_replacement, y_train_synonyms = train
    
    # pre-train process =========================================
    flag = ( 10 
             if epoch > 50 
             else kwarg.get("flag", 1)  )
    model_name = type(model).__name__
    total_dataset = len(X_train)
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
            syn_y = y_train_synonyms[batch:batch+batch_size, ...].float().to(device)
            rep_y = y_train_replacement[batch:batch+batch_size, ...].float().unsqueeze(-1).to(device)
            

            # Compute the loss and its gradients
            loss = loss_fn(logits_s.to(device), logits_r.to(device), syn_y, rep_y)
            loss.backward()

            # Adjust learning weights
            optimizer.step()
            losses.append(loss.item())
            
        avg_loss.append(sum(losses)/len(losses)) 

        if i % flag == 0:
            print(f"[INFO] |{f'model: {model_name:^5}':^20}|{f'epoch: {i:^5}':^20}|{f'avg loss: {loss.item():^5.4f}':^20}|")
    
    if kwarg.get("save", True):
         torch.save(head.state_dict(), LOCAL_HEAD_MODEL)

    return avg_loss
    # ==========================================================


