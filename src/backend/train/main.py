import torch
import torch.nn as nn

def training(model        : nn.SyntaxBert, 
             head         : any,
             X            : torch.Tensor,
             replacements : torch.Tensor,
             synonyms     : torch.Tensor,
             optimizer    : any,
             loss_fn      : any,
             batch_size   : int=16,
             epoch        : int=2) -> None:
    
    
    # pre-train process =========================================
    flag = 10 if epoch > 50 else 1
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

    # assign head to mps/gpu/cpu for training
    head.to(device)

    # freeze Bert Weights
    for param in model.parameters():
        param.required_grad = False

    # train process ============================================
    for i in range(epoch):
        losses = []
        for batch in range(0, total_dataset, batch_size):
        
            x = X[batch:batch+batch_size, ...]
            
            syn_y = synonyms[batch:batch+batch_size, ...].float()
            rep_y = replacements[batch:batch+batch_size, ...].float()
            
            # # for each batch zero grad 
            optimizer.zero_grad()
            
            _, hidden_layer = model(x)[1]
            logits_r, logits_s = head(hidden_layer.to(device))
        
            # Compute the loss and its gradients
            loss = loss_fn(logits_s, logits_r, syn_y.float(), rep_y.float())
            loss.backward()

            # Adjust learning weights
            optimizer.step()
            if i % flag == 0:
                losses.append(loss)
                
        if i % flag == 0:
            print(f"[INFO] |{f'model: {model_name:<5}':^10}|{f'epoch: {i:<5}':^10}|{f'loss: {sum(losses)/len(losses):<5}':^10}|")
    
    # ==========================================================


