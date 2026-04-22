
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
import torch.nn.functional as F



def collate_fn(prompts, labels):

    # Find the maximum sequence length in the batch
    max_seq_len = max(prompt.size(1) for prompt in prompts)

    # Initialize a tensor to hold the batched prompts
    batch_size = len(prompts)
    dtype = prompts[0].dtype
    device = prompts[0].device  # Assuming all prompts are on the same device
    prompts_padded = torch.zeros(batch_size, 1, max_seq_len, dtype=dtype)

    # Pad each prompt to the maximum sequence length
    for i, prompt in enumerate(prompts):
        seq_len = prompt.size(1)
        prompts_padded[i, :, :seq_len] = prompt

    # Stack labels into a tensor
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    return prompts_padded, labels


def get_last_non_padded_token_rep(hidden_states, attention_mask):
    """
    Get the last non-padded token's representation for each sequence in the batch.
    """
    # Find the length of each sequence by summing the attention mask (1 for real tokens, 0 for padding)
    lengths = attention_mask.squeeze().sum(dim=1).long()

    # Index the last non-padded token for each sequence
    batch_size, max_seq_len, hidden_size = hidden_states.size()
    last_token_reps = torch.stack([hidden_states[i, lengths[i]-1, :] for i in range(batch_size)])

    return last_token_reps


def get_ex_data(model, prompts, labels, batch_size, centroids, sinkhorn, num_selected_data, cls_dist, args):
    
    all_embeddings = []
    all_labels = []
    num_samples = len(prompts)

    with torch.no_grad():
        with autocast(dtype=torch.float16):
            for batch_start in tqdm(range(0, num_samples, batch_size)):
                    batch_prompts = prompts[batch_start: batch_start + batch_size]
                    batch_labels = labels[batch_start: batch_start + batch_size]
                    batch_prompts, batch_labels = collate_fn(batch_prompts,batch_labels)
                    attention_mask = (batch_prompts != 0).half()
                    batch_prompts = batch_prompts.cuda()
                    batch_labels = batch_labels.cuda()
                    attention_mask = attention_mask.to(batch_prompts.device)
                    all_labels.append(batch_labels.cpu().numpy())

                    output = model(batch_prompts.squeeze(), attention_mask=attention_mask.squeeze(),  output_hidden_states=True)
                    hidden_states = output.hidden_states

                    hidden_states = torch.stack(hidden_states, dim=0).squeeze()
                    last_layer_hidden_state = hidden_states[-1]  

                    last_token_rep = get_last_non_padded_token_rep(last_layer_hidden_state, attention_mask.squeeze())  
                    all_embeddings.append(last_token_rep)

            all_embeddings = F.normalize(torch.concat(all_embeddings),p=2,dim=-1)
   
            pseudo_label = sinkhorn(all_embeddings, centroids)
            
            selected_indices = compute_entropy(all_embeddings, centroids, pseudo_label, num_selected_data, cls_dist, args)
            
            selected_labels_soft = pseudo_label[selected_indices]
      

    return selected_indices, selected_labels_soft


def compute_ot_loss_cos(last_token_rep, centroids, pseudo_label, batch_size, args):
    
    last_token_rep = F.normalize(last_token_rep, p=2, dim=-1)
    
    centroids = F.normalize(centroids, p=2, dim=-1)

    similarities = torch.matmul(last_token_rep, centroids.T)  

    similarities = similarities / args.cos_temp
    
    pt = F.softmax(similarities, dim=-1)  
    
    ot_loss = -torch.sum(pseudo_label * torch.log(pt + 1e-8)) / pseudo_label.shape[0]
    
    return ot_loss, similarities


def compute_entropy(last_token_rep, centroids, pseudo_label, k, cls_dist, args):
    

    last_token_rep = F.normalize(last_token_rep, p=2, dim=-1)
    
    centroids = F.normalize(centroids, p=2, dim=-1)

    similarities = torch.matmul(last_token_rep, centroids.T)  

    similarities = similarities / args.cos_temp
    
    pt = F.softmax(similarities, dim=-1)  
    
    ce = - (pseudo_label * torch.log(pt + 1e-8))
    
    pseudo_label_hard = torch.argmax(pt,dim=1) 
    
    # * Added for preventing severe cases
    # Class-wise data selection: Select pseudo-labeled unlabeled data in proportion to the class distribution of the exemplar set. 
    
    cls0_num = k*cls_dist[0]
    cls1_num = k*cls_dist[1]
    
    cls_0_indices = (pseudo_label_hard == 0).nonzero(as_tuple=True)[0]
    cls_1_indices = (pseudo_label_hard == 1).nonzero(as_tuple=True)[0]

    ce = torch.sum(ce, dim=1)
    
    ce_class_0 = ce[cls_0_indices]
    ce_class_1 = ce[cls_1_indices]
    
    if len(ce_class_0) < cls0_num or len(ce_class_1) < cls1_num: # Fallback to top-k across all classes
        
        _, top_k_indices = torch.topk(ce, k, largest=False, sorted=True)
        
    else:
        
        top_0_indices = cls_0_indices[torch.topk(ce_class_0, int(cls0_num), largest=False, sorted=True).indices]  
        top_1_indices = cls_1_indices[torch.topk(ce_class_1, int(cls1_num), largest=False, sorted=True).indices]  
        top_k_indices = torch.cat((top_0_indices, top_1_indices))
        
    return top_k_indices  


def update_centroids_ema(centroids, last_token_rep, pseudo_label, args):

    last_token_rep_norm = F.normalize(last_token_rep, p=2, dim=1)
    
    centroids= F.normalize(centroids, p=2, dim=1)
    
    weighted_sum = torch.matmul(pseudo_label.T, last_token_rep_norm)  
    
    # Normalize the weighted sums to get the new centroids
    pseudo_label_sum = pseudo_label.sum(dim=0).unsqueeze(1) + 1e-8  
    new_centroids_batch = weighted_sum / pseudo_label_sum  
    
    # EMA update for centroids
    updated_centroids = F.normalize(args.ema_decay * centroids + (1 - args.ema_decay) * new_centroids_batch, p=2, dim=1)
    
    return updated_centroids

def update_centroids_ema_hard(centroids, last_token_rep, pseudo_label, args):
    
    last_token_rep_norm = F.normalize(last_token_rep, p=2, dim=1)
    
    centroids = F.normalize(centroids, p=2, dim=1)
    
    max_indices = torch.argmax(pseudo_label, dim=1)
    
    discrete_labels = torch.zeros_like(pseudo_label)
    
    discrete_labels[torch.arange(pseudo_label.size(0)), max_indices] = 1
    
    weighted_sum = torch.matmul(discrete_labels.T.float(), last_token_rep_norm)  
    
    pseudo_label_sum = discrete_labels.sum(dim=0).unsqueeze(1) + 1e-8  
    
    new_centroids_batch = weighted_sum / pseudo_label_sum  
    
    # EMA update for centroids
    updated_centroids = F.normalize(args.ema_decay * centroids + (1 - args.ema_decay) * new_centroids_batch, p=2, dim=-1)
    
    return updated_centroids