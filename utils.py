import torch
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn import metrics
from munkres import Munkres
from kmeans_gpu import kmeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics.pairwise import cosine_similarity
from opt import args

def load_data(dataset):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()
            objects.append(cur_data)
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(np.array(features.todense()))
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    return adj, features, np.argmax(labels, 1)

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def get_knn_graph(features, k=10):
    sims = cosine_similarity(features)
    np.fill_diagonal(sims, 0)

    num_nodes = features.shape[0]
    knn_indices = np.argsort(sims, axis=1)[:, -k:]

    row = np.arange(num_nodes).repeat(k)
    col = knn_indices.flatten()
    data = np.ones(num_nodes * k)

    adj_knn = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

    adj_knn = adj_knn + adj_knn.T.multiply(adj_knn.T > adj_knn) - adj_knn.multiply(adj_knn.T > adj_knn)
    return adj_knn


def preprocess_graph(features, adj, layer, global_layers,norm='sym', renorm=True, k=10):
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj

    rowsum = np.array(adj_.sum(1))
    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized

    reg = [1] * (layer)
    adjs = []
    sm_fea_s_list = []

    sm_fea_s = sp.csr_matrix(features).toarray()
    sm_fea_s_list.append(torch.FloatTensor(sm_fea_s))

    for i in range(len(reg)):
        adjs.append(ident - (reg[i] * laplacian))
    for a in adjs:
        sm_fea_s = a.dot(sm_fea_s)
        sm_fea_s_list.append(torch.FloatTensor(sm_fea_s))

    print(f"Constructing Top-{k} KNN Graph for Global Enhancement...")
    adj_knn = get_knn_graph(features, k=k)

    adj_knn = adj_knn + ident
    rowsum_knn = np.array(adj_knn.sum(1))
    d_inv_sqrt_knn = sp.diags(np.power(rowsum_knn, -0.5).flatten())
    adj_knn_norm = adj_knn.dot(d_inv_sqrt_knn).transpose().dot(d_inv_sqrt_knn).tocoo()
    adj_csr = adj.tocsr()
    adj_knn_csr = adj_knn.tocsr()
    ident = sp.eye(adj.shape[0], format='csr')

    adj_with_global = adj_csr + adj_knn_csr + ident

    adj_with_global.data = np.ones_like(adj_with_global.data)
    rowsum_fused = np.array(adj_with_global.sum(1))
    d_inv_sqrt_fused = sp.diags(
        np.power(rowsum_fused, -0.5).flatten()
    )

    adj_fused_norm = adj_with_global.dot(d_inv_sqrt_fused) \
        .transpose() \
        .dot(d_inv_sqrt_fused) \
        .tocoo()

    sm_fea_global = adj_fused_norm.dot(sp.csr_matrix(features).toarray())

    sm_fea_s_list.append(torch.FloatTensor(sm_fea_global))

    for i in range(global_layers - 1):
        sm_fea_global = adj_fused_norm.dot(sm_fea_s_list[-1])
        sm_fea_s_list.append(torch.FloatTensor(sm_fea_global))

    return adjs, sm_fea_s_list , adj_with_global

def calculate_weighted_features(alpha, sm_fea_s_list):
    sm_fea_s_list_l = torch.zeros_like(sm_fea_s_list[0]).to(args.device)
    alpha = alpha.to(args.device)
    alpha_sum = sum(alpha)
    for i in range(len(sm_fea_s_list)):
        sm_fea_s = torch.FloatTensor(sm_fea_s_list[i]).to(args.device)
        sm_fea_s_weighted = alpha[i] * sm_fea_s
        sm_fea_s_list_l += sm_fea_s_weighted
    sm_fea_s_list_l /= alpha_sum
    return sm_fea_s_list_l

def laplacian(adj):
    rowsum = np.array(adj.sum(1))
    degree_mat = sp.diags(rowsum.flatten())
    lap = degree_mat - adj
    return torch.FloatTensor(lap.toarray())

def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro

def eva(y_true, y_pred, show_details=True):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    if show_details:
        print(':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
              ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1

def load_graph_data(dataset_name, show_details=False):
    load_path = "dataset/" + dataset_name + "/" + dataset_name
    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_label.npy", allow_pickle=True)
    adj = np.load(load_path+"_adj.npy", allow_pickle=True)
    if show_details:
        print("Dataset details: ", dataset_name)
    return feat, label, adj

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def clustering(feature, true_labels, cluster_num):
    # 确保传入 Tensor
    predict_labels, _ = kmeans(X=feature, num_clusters=cluster_num, distance="euclidean", device=args.device)
    acc, nmi, ari, f1 = eva(true_labels, predict_labels.numpy(), show_details=False)
    return round(100 * acc, 2), round(100 * nmi, 2), round(100 * ari, 2), round(100 * f1, 2), predict_labels.numpy()


def select_high_confidence_samples(embeddings, pseudo_labels, cluster_num, confidence_ratio=0.7):

    import torch

    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.FloatTensor(embeddings)

    device = embeddings.device  # 保持在GPU

    if not isinstance(pseudo_labels, torch.Tensor):
        pseudo_labels = torch.LongTensor(pseudo_labels).to(device)
    else:
        pseudo_labels = pseudo_labels.to(device)

    N = embeddings.shape[0]

    cluster_centers = torch.zeros(cluster_num, embeddings.shape[1], device=device)
    for c in range(cluster_num):
        mask = (pseudo_labels == c)
        if mask.sum() > 0:
            cluster_centers[c] = embeddings[mask].mean(dim=0)

    assigned_centers = cluster_centers[pseudo_labels]
    distances = torch.norm(embeddings - assigned_centers, dim=1)

    high_conf_mask = torch.zeros(N, dtype=torch.bool, device=device)

    for c in range(cluster_num):
        cluster_mask = (pseudo_labels == c)
        cluster_indices = torch.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            continue

        cluster_distances = distances[cluster_indices]
        num_select = max(1, int(len(cluster_indices) * confidence_ratio))
        _, top_k_indices = torch.topk(cluster_distances, num_select, largest=False)

        selected_indices = cluster_indices[top_k_indices]
        high_conf_mask[selected_indices] = True

    high_conf_labels = pseudo_labels.clone()
    high_conf_labels[~high_conf_mask] = -1

    return high_conf_mask, high_conf_labels  # 返回GPU tensor

def get_semantic_labels(embeddings, cluster_num):
    import torch
    import torch.nn as nn

    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.FloatTensor(embeddings)

    embed_dim = embeddings.shape[1]
    projection = torch.nn.Linear(embed_dim, cluster_num)
    projection = projection.to(embeddings.device)

    logits = projection(embeddings)
    semantic_labels = torch.softmax(logits, dim=1)

    return semantic_labels
