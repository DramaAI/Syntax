import torch
import torch.nn as nn

import os
import sys

FILEPATH = os.path.dirname(os.path.realpath(__file__))
CHECKPOINT_MODEL = lambda x : os.path.join(FILEPATH, "..", "data", 'weights', 'checkpoint', f'syntax-base-bert_{x}.pt')
LOCAL_MODEL = os.path.join(FILEPATH, "..", "data", 'weights', 'syntax-base-bert.pt')
sys.path.append(os.path.join(FILEPATH, '..'))


from module import nn

def evaluation( model        : nn.SyntaxBert, 
                test         : torch.Tensor,
                batch_size   : int=16,
                threshold    : float=0.6):
    
    X_test = test
    total_dataset = len(test)
    device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
             )

    model.to(device)

    model.head.replace_head = "sigmoid"
    model.head.synonym_head = "softmax"


    with torch.no_grad():
        for batch in range(0, total_dataset, batch_size):
            
                # index input
                inputs_ids = X_test["input_ids"][batch:batch+batch_size, ...].to(device)
                attn_mask  = X_test["attention_mask"][batch:batch+batch_size, ...].to(device)
                    
                # forward pass
                logits_r, logits_s = model(inputs_ids, attn_mask)

                replacement_mask = logits_r > threshold
                indices = torch.nonzero(replacement_mask)


                new_tokens = logits_s[indices[:, 0], indices[:, 1]].argmax(dim=-1)
                inputs_ids[indices[:, 0], indices[:, 1]] = new_tokens
                X_test["input_ids"][batch:batch+batch_size, ...] = inputs_ids.cpu()

    return X_test["input_ids"] 

                
               
def training(model        : nn.SyntaxBert, 
             train        : tuple[dict, torch.Tensor, torch.Tensor],
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


    model.to(device)

    avg_loss = []
    # train process ============================================
    for i in range(epoch):
        losses = []
        for batch in range(0, total_dataset, batch_size):

            # batched input
            input_ids = X_train["input_ids"][batch:batch+batch_size, ...].to(device)
            attn_mask = X_train["attention_mask"][batch:batch+batch_size, ...].to(device)
            
            # forward pass
            logits_r, logits_s = model(input_ids=input_ids, attention_mask=attn_mask)
        

            # access batch replacements, and synonyms
            syn_y = y_train_synonyms[batch:batch+batch_size, ...].float().to(device)
            rep_y = y_train_replacement[batch:batch+batch_size, ...].float().unsqueeze(-1).to(device)
            

            # Compute the loss and its gradients
            optimizer.zero_grad()
            loss = loss_fn(logits_s.to(device), logits_r.to(device), syn_y, rep_y)
            
            if torch.isnan(loss).any():
              # print("[WARNING] nan loss was found when training, ignoring loss, update and continuing")
              continue


            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        avg_loss.append(sum(losses)/len(losses)) 

        if i % flag == 0:
            print(f"[INFO] |{f'model: {model_name:^5}':^20}|{f'epoch: {i:^5}':^20}|{f'avg loss: {avg_loss[i]:^5.4f}':^20}|")
            torch.save(model.state_dict(), CHECKPOINT_MODEL(i))
    
    if kwarg.get("save", True):
         torch.save(model.state_dict(), LOCAL_MODEL)

    return avg_loss
    # ==========================================================


