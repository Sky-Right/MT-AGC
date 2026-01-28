import torch
import torch.nn as nn
import torch.nn.functional as F

class my_model(nn.Module):
    def __init__(self, dims, sm_fea_s_list,cluster_num=7):
        super(my_model, self).__init__()
        self.layers1 = nn.Linear(dims[0], dims[1])
        self.layers2 = nn.Linear(dims[0], dims[1])
        num_features = len(sm_fea_s_list) + 1
        self.num_features = num_features
        self.alpha = nn.Parameter(torch.rand(num_features))
        self.semantic_projection = nn.Linear(dims[1], cluster_num)

    def forward(self, x_l, is_train=True, sigma=0.01):
        out1 = self.layers1(x_l)
        out2 = self.layers2(x_l)
        
        out1 = F.normalize(out1, dim=1, p=2)
        out2 = F.normalize(out2, dim=1, p=2)
        
        if is_train:
            out3 = out2 + torch.normal(0, torch.ones_like(out2) * sigma).to(out2.device)
        else:
            out3 = out2
            
        return out1, out2, out3

    def get_semantic_labels(self, embeddings):
        logits = self.semantic_projection(embeddings)
        semantic_labels = F.softmax(logits, dim=1)
        return semantic_labels
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2, z2_s):
        sim_z1_z2 = F.cosine_similarity(z1, z2)
        sim_z1_z2_s = F.cosine_similarity(z1, z2_s)
        sim_z2_z2_s = F.cosine_similarity(z2, z2_s)

        loss = -torch.mean(torch.log(torch.exp(sim_z1_z2 / self.temperature) /
                                     (torch.exp(sim_z1_z2_s / self.temperature) + 
                                      torch.exp(sim_z2_z2_s / self.temperature) + 
                                      torch.exp(sim_z1_z2 / self.temperature))))

        return loss

class LabelMatchingLoss(nn.Module):
    def __init__(self):
        super(LabelMatchingLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, semantic_labels, pseudo_labels):
        return self.ce_loss(semantic_labels, pseudo_labels)
