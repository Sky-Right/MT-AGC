from utils import *
from tqdm import tqdm
from torch import optim
from model import my_model, ContrastiveLoss, LabelMatchingLoss
import torch.nn.functional as F
import numpy as np
import torch

print("Using {} dataset on {}".format(args.dataset, args.device))
file = open("result_topk.csv", "a+")
print(f"Dataset: {args.dataset} (Top-K Enhanced + Contrastive)", file=file)
file.close()
k_value = 10

if args.dataset == 'cora':
    args.cluster_num = 7
    args.gnnlayers = 10
    args.global_layers = 1
    args.lr = 1e-3
    args.dims = [1200]
    args.sigma = 0.01
    args.gama = 0.5
    k_value = 10
    args.stage1_epochs = 200
    args.label_match_weight = 0.5

elif args.dataset == 'citeseer':
    args.cluster_num = 6
    args.gnnlayers = 6
    args.global_layers = 1
    args.lr = 2e-4
    args.dims = [1000]
    args.sigma = 0.15
    args.gama = 0.1
    k_value = 5
    args.stage1_epochs = 300
    args.label_match_weight = 0.8

elif args.dataset == 'amap':
    args.cluster_num = 8
    args.gnnlayers = 12
    args.global_layers = 1
    args.lr = 1e-3
    args.dims = [800]
    args.sigma = 0.1
    args.gama = 0.5
    k_value = 12
    args.stage1_epochs = 300
    args.label_match_weight = 1.0

elif args.dataset == 'bat':
    args.cluster_num = 4
    args.gnnlayers = 78
    args.global_layers = 1
    args.lr = 4e-3
    args.dims = [800]
    args.sigma = 0.001
    args.gama = 1.0
    k_value = 3
    args.stage1_epochs = 200
    args.label_match_weight = 0.5

elif args.dataset == 'eat':
    args.cluster_num = 4
    args.gnnlayers = 100
    args.global_layers = 3
    args.lr = 2e-3
    args.dims = [800]
    args.sigma = 0.01
    args.gama = 0.5
    k_value = 20
    args.stage1_epochs = 300
    args.label_match_weight=1.0

elif args.dataset == 'uat':
    args.cluster_num = 4
    args.gnnlayers = 4
    args.global_layers = 1
    args.lr = 5e-4
    args.dims = [600]
    args.sigma = 0.01
    args.gama = 1.0
    k_value = 15
    args.stage1_epochs = 300
    args.label_match_weight = 0.5

X, y, A = load_graph_data(args.dataset, show_details=False)
features = X
true_labels = y
adj = sp.csr_matrix(A)
adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
adj.eliminate_zeros()

print(f'Preprocessing... Layers={args.gnnlayers}, Top-K={k_value}, Global_layers={args.global_layers}')
adj_norm_s, sm_fea_s_list ,adj_with_global= preprocess_graph(features, adj, args.gnnlayers,args.global_layers, norm='sym', renorm=True, k=k_value)
adj_1st = adj_with_global.toarray()
acc_list = []
nmi_list = []
ari_list = []
f1_list = []

for seed in range(10):
    setup_seed(seed)

    model = my_model([features.shape[1]] + args.dims, sm_fea_s_list, cluster_num=args.cluster_num)
    inx_l = calculate_weighted_features(model.alpha, sm_fea_s_list)
    best_acc, best_nmi, best_ari, best_f1, prediect_labels = clustering(inx_l, true_labels, args.cluster_num)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    model = model.to(args.device)
    inx_l = inx_l.to(args.device)
    target = torch.FloatTensor(adj_1st).to(args.device)
    
    cc_loss_fn = ContrastiveLoss(temperature=1.0)
    lm_loss_fn = LabelMatchingLoss()
    
    for epoch in tqdm(range(args.epochs), desc=f"Seed {seed}"):
        model.train()
        
        inx_l = calculate_weighted_features(model.alpha, sm_fea_s_list)

        z1, z2, z2_s = model(inx_l, is_train=True, sigma=args.sigma)
        z = (z1 + z2) / 2

        cc_loss = cc_loss_fn(z1, z2, z2_s)

        S = (z1 @ z2_s.T) / 2
        c_loss = F.mse_loss(S, target)

        if args.use_pseudo_label and epoch >= args.stage1_epochs:
            with torch.no_grad():
                predict_labels, _ = kmeans(X=z, num_clusters=args.cluster_num,
                                           distance="euclidean", device=args.device)

                high_conf_mask, high_conf_labels = select_high_confidence_samples(
                    z, predict_labels, args.cluster_num,
                    confidence_ratio=args.confidence_ratio
                )

                high_conf_labels_tensor = high_conf_labels
                high_conf_mask_tensor = high_conf_mask

            semantic_labels_1 = model.get_semantic_labels(z1)
            semantic_labels_2 = model.get_semantic_labels(z2)

            if high_conf_mask_tensor.sum() > 0:
                lm_loss_1 = lm_loss_fn(
                    semantic_labels_1[high_conf_mask_tensor],
                    high_conf_labels_tensor[high_conf_mask_tensor]
                )
                lm_loss_2 = lm_loss_fn(
                    semantic_labels_2[high_conf_mask_tensor],
                    high_conf_labels_tensor[high_conf_mask_tensor]
                )
                lm_loss = (lm_loss_1 + lm_loss_2) / 2
            else:
                lm_loss = torch.tensor(0.0).to(args.device)

            loss = cc_loss + args.gama * c_loss + args.label_match_weight * lm_loss
        else:
            loss = args.gama * c_loss + cc_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 4 == 0:
            model.eval()
            z1, z2, _ = model(inx_l, is_train=True, sigma=args.sigma)
            hidden_emb = (z1 + z2) / 2
            acc, nmi, ari, f1, predict_labels = clustering(hidden_emb, true_labels, args.cluster_num)
            
            if acc >= best_acc:
                best_acc = acc
                best_nmi = nmi
                best_ari = ari
                best_f1 = f1


    tqdm.write(f'Seed {seed} Best - ACC: {best_acc:.2f}, NMI: {best_nmi:.2f}, ARI: {best_ari:.2f}, F1: {best_f1:.2f}')
    
    file = open("result_topk.csv", "a+")
    print(best_acc, best_nmi, best_ari, best_f1, file=file)
    file.close()
    
    acc_list.append(best_acc)
    nmi_list.append(best_nmi)
    ari_list.append(best_ari)
    f1_list.append(best_f1)

acc_list = np.array(acc_list)
nmi_list = np.array(nmi_list)
ari_list = np.array(ari_list)
f1_list = np.array(f1_list)

file = open("result_topk.csv", "a+")
print(f"Set: {args.dataset}, TopK={k_value}", file=file)
print(round(acc_list.mean(), 2), round(acc_list.std(), 2), file=file)
print(round(nmi_list.mean(), 2), round(nmi_list.std(), 2), file=file)
print(round(ari_list.mean(), 2), round(ari_list.std(), 2), file=file)
print(round(f1_list.mean(), 2), round(f1_list.std(), 2), file=file)
file.close()

print("------------------------------------------------")
print(f" Top-K Results for {args.dataset} stage:{args.stage1_epochs}:")
print(f"ACC: {round(acc_list.mean(), 2)} +/- {round(acc_list.std(), 2)}")
print(f"NMI: {round(nmi_list.mean(), 2)} +/- {round(nmi_list.std(), 2)}")
print(f"ARI: {round(ari_list.mean(), 2)} +/- {round(ari_list.std(), 2)}")
print(f"F1 : {round(f1_list.mean(), 2)} +/- {round(f1_list.std(), 2)}")
print("------------------------------------------------")
