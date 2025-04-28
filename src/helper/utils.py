
import yaml
import pickle as pkl
import os
import torch
import requests
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LogNorm
import torch_geometric.utils as utils
from models.nn import EstimateAdj, Rewire_GCN
import torch.optim as optim
import torch.nn.functional as F
def load_yaml(file_path):
    
    with open(file_path, 'r') as file:
        try:
            yaml_dict = yaml.safe_load(file)
            return yaml_dict
        except yaml.YAMLError as e:
            print(f"Error while parsing YAML file: {e}")


def pkl_and_write(obj, path):
    directory = os.path.dirname(path)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, 'wb') as f:
        pkl.dump(obj, f)
    return path

def read_and_unpkl(path):
    with open(path, 'rb') as f:
        res = pkl.load(f)
    return res 


def replace_tensor_values(tensor, mapping):
    # Create an empty tensor with the same shape as the original tensor
    new_tensor = torch.zeros_like(tensor)
    new_tensor -= 1

    # Loop over the mapping and replace the values
    for k, v in mapping.items():
        mask = tensor == k  # create a mask where the tensor value equals the current key
        new_tensor[mask] = v 
    return new_tensor


def neighbors(edge_index, node_id):
    row, col = edge_index 
    match_idx = torch.where(row == node_id)[0]
    neigh_nodes = col[match_idx]
    return neigh_nodes.tolist()

def get_one_hop_neighbors(data_obj, sampled_test_node_idxs, sample_num = -1):
    ## if sample_nodes == -1, all test nodes within test masks will be considered
    neighbor_dict = {}
    for center_node_idx in sampled_test_node_idxs:
        center_node_idx = center_node_idx.item()
        neighbor_dict[center_node_idx] = neighbors(data_obj.edge_index, center_node_idx)
    return neighbor_dict

def get_two_hop_neighbors_no_multiplication(data_obj, sampled_test_node_idxs, sample_num = -1):
    neighbor_dict = {}
    # for center_node_idx in sampled_test_node_idxs:
    one_hop_neighbor_dict = get_one_hop_neighbors(data_obj, sampled_test_node_idxs)
    for key, value in one_hop_neighbor_dict.items():
        this_key_neigh = []
        second_hop_neighbor_dict = get_one_hop_neighbors(data_obj, torch.IntTensor(value))
        second_hop_neighbors = set(itertools.chain.from_iterable(second_hop_neighbor_dict.values()))
        second_hop_neighbors.discard(key)
        neighbor_dict[key] = sorted(list(second_hop_neighbors))
    return neighbor_dict


def get_sampled_nodes(data_obj, sample_num = -1):
    train_mask = data_obj.train_masks[0]
    # val_mask = data_obj.val_masks[0]
    test_mask = data_obj.test_masks[0]
    all_idxs = torch.arange(data_obj.x.shape[0])
    test_node_idxs = all_idxs[test_mask]
    train_node_idxs = all_idxs[train_mask]
    # val_node_idxs = all_idxs[val_mask]
    if sample_num == -1:
        sampled_test_node_idxs = test_node_idxs
    else:
        sampled_test_node_idxs = test_node_idxs[torch.randperm(test_node_idxs.shape[0])[:sample_num]]
    return sampled_test_node_idxs, train_node_idxs


def query_arxiv_classify_api(title, abstract, url = "http://export.arxiv.org/api/classify"):
    text = title + abstract
    data = {
        "text": text
    }
    r = requests.post(url, data = data)
    return r



def plot_multiple_curve(methods_acc, method_name, output_path):
    markers = ["*-r", "v-b", "o-c", "^-m", "<-y", ">-k"]
    method1_acc = methods_acc[0]
    epochs = range(1, len(method1_acc) + 1)
    plt.figure(figsize=(10,6))

    # Plotting each method's accuracy over epochs
    for i, acc in enumerate(methods_acc):
        plt.plot(epochs, acc, markers[i], label=method_name[i])

    # Adding labels and a legend
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(output_path)


def noise_transition_matrix(pred, gt, output_path, x_axis_labels = None, y_axis_labels = None, cbar = True):
    plt.figure(figsize=(12, 12))
    pred_list = pred.tolist()
    gt_list = gt.tolist()
    num_class = gt.max().item() + 1
    transition_matrix = np.zeros((num_class, num_class))
    num_of_occurence = np.bincount(gt_list)
    for x, y in zip(pred_list, gt_list):
        transition_matrix[y][x] += 1
    # pal = sns.color_palette("crest", as_cmap=True)
    cmap_pal = sns.color_palette("crest", as_cmap=True)
    # sns.set_palette(pal)
    transition_matrix /= num_of_occurence[:, np.newaxis]
    if x_axis_labels is None or y_axis_labels is None:
        ax = sns.heatmap(transition_matrix, vmin=0, vmax=1,  annot=True, fmt=".2f", norm = LogNorm(), cmap = cmap_pal, cbar = cbar, annot_kws={"size": 20}, square = True, cbar_kws={'shrink': 0.5})
    else:
        ax = sns.heatmap(transition_matrix, vmin=0, vmax=1, annot=True, fmt=".2f", norm = LogNorm(), xticklabels=x_axis_labels, yticklabels=y_axis_labels, cmap = cmap_pal, cbar = cbar, annot_kws={"size": 20}, square = True, cbar_kws={'shrink': 0.5})
    plt.tight_layout()
    # Adjust x-axis label properties
    plt.xticks(fontsize=40, fontweight='bold')

    # Adjust y-axis label properties
    plt.yticks(fontsize=40, fontweight='bold')
    # cbar = ax.collections[0].colorbar
    # # And an example of setting custom ticks:
    # cbar.set_ticks([0, 1])
    plt.savefig(output_path)
    plt.clf()




def delete_non_tensor_attributes(data):
    for attr_name in data.keys:
        if not isinstance(data[attr_name], torch.Tensor):
            delattr(data, attr_name)
    return data


def find_selected_node_attribute(data, inconfident_indices, confident_indices, hops=1, dataset='cora'):
    '''
    input the node index, analyze the node attributes, including the k-hop neighbor
    label distribution, etc.
    '''
    from torch_geometric.utils import k_hop_subgraph
    average_inconfi_homo = 0
    average_inconfi_heter = 0
    average_confi_homo = 0
    average_confi_heter = 0

    for node in inconfident_indices:
        if hops > 1:
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(node.item(), hops, data.edge_index, relabel_nodes=True)
            prev_subset, prev_edge_index, prev_mapping, prev_edge_mask = k_hop_subgraph(node.item(), hops-1,data.edge_index, relabel_nodes=True)
            mask = ~torch.isin(subset, prev_subset)
            subset = subset[mask]
        else:
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(node.item(), hops, data.edge_index, relabel_nodes=True)
        same_label_num = 0
        diff_label_num = 0
        for neighbor in subset:
            if neighbor != node:
                if data.backup_y[node] == data.backup_y[neighbor]:
                    same_label_num += 1
                else:
                    diff_label_num += 1
       # print(f"in inconfident node set, node {node} has totally {same_label_num+diff_label_num} neighbor nodes, {same_label_num} same labels, {diff_label_num} different labels")
        if same_label_num+diff_label_num > 0:
            average_inconfi_homo += same_label_num/(same_label_num+diff_label_num)
            average_inconfi_heter += diff_label_num/(same_label_num+diff_label_num)
    average_inconfi_homo /= inconfident_indices.shape[0]
    average_inconfi_heter /= inconfident_indices.shape[0]

    for node in confident_indices:
        if hops > 1:
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(node.item(), hops, data.edge_index, relabel_nodes=True)
            prev_subset, prev_edge_index, prev_mapping, prev_edge_mask = k_hop_subgraph(node.item(), hops-1,data.edge_index, relabel_nodes=True)
            mask = ~torch.isin(subset, prev_subset)
            subset = subset[mask]
        else:
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(node.item(), hops, data.edge_index, relabel_nodes=True)
        same_label_num = 0
        diff_label_num = 0
        for neighbor in subset:
            if neighbor != node:
                if data.backup_y[node] == data.backup_y[neighbor]:
                    same_label_num += 1
                else:
                    diff_label_num += 1
       # print(f"in confident node set, node {node} has totally {same_label_num+diff_label_num} neighbor nodes, {same_label_num} same labels, {diff_label_num} different labels")
        if same_label_num+diff_label_num > 0:
            average_confi_homo += same_label_num/(same_label_num+diff_label_num)
            average_confi_heter += diff_label_num/(same_label_num+diff_label_num)
    average_confi_homo /= confident_indices.shape[0]
    average_confi_heter /= confident_indices.shape[0]
   # print(f"{dataset}'s {hops} hop, {confident_indices.shape[0]} most confident node's homo is {average_confi_homo}, {inconfident_indices.shape[0]} most inconfident node's homo is {average_inconfi_homo}")
    return average_confi_homo, average_inconfi_homo

def paint_homo(cur_confi_homo_list, cur_inconfi_homo_list, stage, dataset, model):
    import matplotlib.pyplot as plt
    import numpy as np

    # 生成示例数据
    x = [i for i in range(len(cur_confi_homo_list))]
    # 创建绘图对象
    plt.figure(figsize=(10, 6))

    # 绘制多条折线图
    
    plt.plot(x, cur_confi_homo_list, label='confident', color='blue', linewidth=2)
    plt.plot(x, cur_inconfi_homo_list, label='inconfident', color='red', linestyle='--', linewidth=2)
    for i, node in enumerate(cur_confi_homo_list):
        plt.text(x[i], node + 0.05, f'{node:.2f}', ha='center', va='bottom')
    for i, node in enumerate(cur_inconfi_homo_list):
        plt.text(x[i], node + 0.05, f'{node:.2f}', ha='center', va='bottom')
    # 设置轴的范围
    plt.xlim(0, 2)
    plt.ylim(0,1)

    # 添加图例
    plt.legend(loc='upper right')

    # 添加标题和标签
    plt.title(f'{dataset} stage {stage} confident and inconfident result')
    plt.xlabel('hop')
    plt.ylabel('average homo')

    # 添加网格
    plt.grid(True)



def find_similar_node_attribute(data, args, ensemble_ca_matrix, most_inconfident_indices, dataset = 'cora', stage=1, model='GCN'):
    '''
    for each node in the inconfident node set, find the most similar nodes in the labeled budget 
    and calculate the homophily 
    '''
    label_idxs = torch.where(data.train_mask)[0].detach().cpu()
    neighbor_size = args.neighbor_size
    neighbor_idxs = ensemble_ca_matrix[most_inconfident_indices][:, label_idxs]
    ordered_neighbor_idxs = torch.argsort(neighbor_idxs, dim=1, descending=True)
    negative_train_indices = []
    avg_inconfident_homo = 0
    for i, node in enumerate (most_inconfident_indices):
        
        neighbors = ordered_neighbor_idxs[i]
        same_label_num = 0
        diff_label_num = 0
        for neighbor in neighbors[:neighbor_size]:
            if data.backup_y[node] == data.backup_y[neighbor]:
                same_label_num += 1
            else:
                diff_label_num += 1
        print(f"Node {node} has {same_label_num} same labels, {diff_label_num} diffrent labels, ratio is {same_label_num/(same_label_num+diff_label_num)}")

        avg_inconfident_homo += same_label_num/(same_label_num+diff_label_num)
    
    avg_inconfident_homo /= len(most_inconfident_indices)
    print(f"At stage {stage}, {model} on {dataset} neighbor homophly is {avg_inconfident_homo}")


def paint_new_fig_analysis1(true_label, before_postprocess, after_postprocess, strategy = "", dataset="", label_acc = 0):
    # note that the true label shows the ratio of after_possprocess index
    # 设置柱状图的位置和宽度
    bar_width = 0.20
    index = np.arange(len(true_label))
    base_path = ''
    # 创建一个画布
    plt.figure(figsize=(10, 8))

    # 绘制柱状图
    
    bars = plt.bar(index + bar_width, true_label, bar_width, label='True label', color='red' )
    for idx, bar in enumerate(bars):
        yval = round(true_label[idx],2 )
        height = round(bar.get_height(),2 )
        plt.text(bar.get_x() + bar.get_width()/2, height, yval, ha='center', va='bottom' ,fontsize=8 )
    bars = plt.bar(index + 2 * bar_width, before_postprocess, bar_width, label=strategy, color='blue')
    for idx, bar in enumerate(bars):
        yval = round(before_postprocess[idx],2 )
        height = round(bar.get_height(),2 )
        plt.text(bar.get_x() + bar.get_width()/2, height, yval, ha='center', va='bottom' ,fontsize=8)
    bars = plt.bar(index + 3 * bar_width, after_postprocess, bar_width, label='PostProcess', color='green')
    for idx, bar in enumerate(bars):
        yval = round(after_postprocess[idx],2 )
        height = round(bar.get_height(),2 )
        plt.text(bar.get_x() + bar.get_width()/2, height, yval, ha='center', va='bottom', fontsize=8)
    # 添加标题和坐标轴标签
    plt.xlabel('Class')
    plt.ylabel('Ratio')
    plt.xticks(index + bar_width*2, [f'{i+1}' for i in range(len(true_label))])
    plt.title(f"label acc is {label_acc:.2f}, true std:{np.std(true_label):.2f}, before std:{np.std(before_postprocess):.2f}, after std:{np.std(after_postprocess):.2f}")
    # 添加图例
    plt.legend(fontsize='small')

    # 显示图表
    plt.savefig(f"{base_path}/imgs/label_bias/{dataset}_{strategy}.png")
    plt.close()
    
# def get_train_edge(edge_index, features, n_p, idx_train):
#     '''
#     obtain the candidate edge between labeled nodes and unlabeled nodes based on cosine sim
#     n_p is the top n_p labeled nodes similar with unlabeled nodes
#     '''

#     if n_p == 0:
#         return None

#     poten_edges = []
#     if n_p > len(idx_train) or n_p < 0:
#         for i in range(len(features)):
#             indices = set(idx_train)
#             indices = indices - set(edge_index[1,edge_index[0]==i])
#             for j in indices:
#                 pair = [i, j]
#                 poten_edges.append(pair)
#     else:
#         for i in range(len(features)):
#             sim = torch.div(torch.matmul(features[i],features[idx_train].T), features[i].norm()*features[idx_train].norm(dim=1))
#             _,rank = sim.topk(n_p)
#             if rank.max() < len(features) and rank.min() >= 0:
#                 indices = idx_train[rank.cpu().numpy()]
#                 indices = set(indices)
#             else:
#                 indices = set()
#             indices = indices - set(edge_index[1,edge_index[0]==i])
#             for j in indices:
#                 pair = [i, j]
#                 poten_edges.append(pair)
#     poten_edges = torch.as_tensor(poten_edges).T
#     poten_edges = utils.to_undirected(poten_edges,len(features)).to(self.device)

#     return poten_edges
# def get_model_edge(pred):

#     idx_add = self.idx_unlabel[(pred.max(dim=1)[0][self.idx_unlabel] > self.args.p_u)]

#     row = self.idx_unlabel.repeat(len(idx_add))
#     col = idx_add.repeat(len(self.idx_unlabel),1).T.flatten()
#     mask = (row!=col)
#     unlabel_edge_index = torch.stack([row[mask],col[mask]], dim=0)

#     return unlabel_edge_index, idx_add
# def train_EstimateAdj(data, args):
#     predictor = NRGCN(nfeat=data.x.shape[1],
#                          nhid=args.hidden_dim,
#                          nclass=data.y.max().item() + 1,
#                          self_loop=True,
#                          dropout=args.dropout, device=args.device).to(args.device)
#     estimator = EstimateAdj(data.x.shape[0], args, args.device).to(args.device)
#     pred_edge_index = get_train_edge(data.edge_index, data.x, -1, data.train_mask)
#     optimizer = optim.Adam(list(estimator.parameters())+ list(predictor.parameters()),
#                                lr=args.lr, weight_decay=args.weight_decay)
#     best_pred = None
#     for epoch in range(args.epochs):
#         estimator.train()
#         predictor.train()
#         optimizer.zero_grad()
#         representations, rec_loss = estimator(data.edge_index, data.x)
#         predictor_weights = estimator.get_estimated_weights(pred_edge_index, representations)
#         pred_edge_index = torch.cat([data.edge_index, pred_edge_index], dim=1)
#         predictor_weights = torch.cat([torch.ones(data.edge_index.shape[1]), predictor_weights])
#         log_pred = predictor(data.x, pred_edge_index, representations)
#         if best_pred is None:
#             pred = F.softmax(log_pred, dim=1).detach()
#             best_pred = pred
def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow, sparsecol),1)
    sparsedata=torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))

def dirichlet_energy(x, edge_index, predictor_weights=None):
    """Return the Dirichlet energy of x given the graph described by edge_index.

    Parameters
    ----------
    x : torch.Tensor
        node features
    edge_index : torch.Tensor
        edge index

    Returns
    -------
    torch.Tensor
        Dirichlet energy
    """
    from torch_geometric.utils import get_laplacian
    if predictor_weights is None:
        predictor_weights = torch.ones(edge_index.shape[1]).to(x.device)
    lap = get_laplacian(edge_index, edge_weight=predictor_weights, normalization=None,num_nodes=x.shape[0])
    lap = torch.sparse_coo_tensor(lap[0], lap[1])
    try:
        dirichlet = x.t().double() @ lap.double() @ x.double()
    except Exception as e:
        import pdb; pdb.set_trace()
        
    dirichlet_trace = torch.trace(dirichlet)
    return dirichlet_trace, lap

def harmonicity(x, edge_index):
    """Return the harmonicity of x given the graph described by edge_index.

    Parameters
    ----------
    x : torch.Tensor
        node features
    edge_index : torch.Tensor
        edge index

    Returns
    -------
    torch.Tensor
        harmonicity
    """
    from torch_geometric.utils import get_laplacian
    lap = get_laplacian(edge_index, normalization='rw', num_nodes=x.shape[0])
    lap = torch.sparse_coo_tensor(lap[0], lap[1])
    harmonicity = (lap.double() @ x)**2
    harmonicity = harmonicity ** (1/2)
    class_num = x.shape[1]
    one_tensor = torch.ones(class_num).to(x.device)
    return harmonicity @ one_tensor.double()

def entrophy_confidence(y):
    y_log = torch.log(y + 1e-9)
    
    # 计算熵：每个 Y[i, j] 乘以其对数值，然后对所有类别求和
    entropy = -torch.sum(y * y_log, dim=1)
    return entropy
def measure_confidence(y, edge_index, confidence_num = 10, inconfidence_num = 10,filtered_indices = None,  max_confidence_num = 40, max_inconfidence_num = 40, ):
    '''
    input: 
    y: prediction of the GNN model
    output: most confident and least confident node index
    '''
    
    y = F.softmax(y, dim=1)
    y_log = torch.log(y + 1e-9)
    
    entropy = -torch.sum(y * y_log, dim=1)
    harm = harmonicity(y.double(), edge_index)
    # high entropy and high harmonicity, low confidence
    # high entropy and low harmonicity, should believe GNN prediction
    # low entropy and high harmonicity, should seek LLM help
    # low entropy and low harmonicity, high confidence
    approximated_zero = 1e-9
    approximated_one = 1 - approximated_zero
    harmonicity_confident_indices = torch.argsort(harm[filtered_indices], descending=False)
    harmonicity_inconfident_indices = torch.argsort(harm[filtered_indices], descending=True)
    sorted_confident_harm_filtered = harm[filtered_indices][harmonicity_confident_indices]
    sorted_inconfident_harm_filtered = harm[filtered_indices][harmonicity_inconfident_indices]
    entropy_confident_indices = torch.argsort(entropy[filtered_indices], descending=False)
    entropy_inconfident_indices = torch.argsort(entropy[filtered_indices], descending=True)
    sorted_confident_entropy_filtered = entropy[filtered_indices][entropy_confident_indices]
    sorted_inconfident_entropy_filtered = entropy[filtered_indices][entropy_inconfident_indices]
    harmonicity_inconfident_interval = sorted_inconfident_harm_filtered.shape[0] - torch.searchsorted(sorted_confident_harm_filtered, approximated_one).item()
    harmonicity_confident_interval = torch.searchsorted(sorted_confident_harm_filtered, approximated_zero).item()
    update_confidence_num = max(confidence_num, harmonicity_confident_interval)
    update_inconfidence_num = max(inconfidence_num, harmonicity_inconfident_interval)
    most_confident_indices = list(set(harmonicity_confident_indices[:update_confidence_num].tolist()) & set(entropy_confident_indices[:max_confidence_num].tolist()))
    most_inconfident_indices = list(set(harmonicity_inconfident_indices[:update_inconfidence_num].tolist()) & set(entropy_inconfident_indices[:max_inconfidence_num].tolist()))
    harmonicity_confident_indices = harmonicity_confident_indices[~torch.isin(harmonicity_confident_indices, torch.tensor(most_confident_indices).to(harmonicity_confident_indices.device))]
    entropy_confident_indices = entropy_confident_indices[~torch.isin(entropy_confident_indices, torch.tensor(most_confident_indices).to(entropy_confident_indices.device))]
    harmonicity_inconfident_indices = harmonicity_inconfident_indices[~torch.isin(harmonicity_inconfident_indices, torch.tensor(most_inconfident_indices).to(harmonicity_inconfident_indices.device))]
    entropy_inconfident_indices = entropy_inconfident_indices[~torch.isin(entropy_inconfident_indices, torch.tensor(most_inconfident_indices).to(entropy_inconfident_indices.device))]
    remain_confident_number = max_confidence_num - len(most_confident_indices)
    remain_inconfident_number = max_inconfidence_num - len(most_inconfident_indices)
    if remain_confident_number > 0 :
        confident_indices = harmonicity_confident_indices.tolist()[:int(remain_confident_number/2)] + entropy_confident_indices.tolist()[:int(remain_confident_number/2)]
        most_confident_indices = most_confident_indices + confident_indices
    else:
        most_confident_indices = most_confident_indices[:max_confidence_num]
    if remain_inconfident_number > 0:
        inconfident_indices = harmonicity_inconfident_indices.tolist()[:int(remain_inconfident_number/2)] + entropy_inconfident_indices.tolist()[:int(remain_inconfident_number/2)]
        most_inconfident_indices = most_inconfident_indices + inconfident_indices
    else:
        most_inconfident_indices = most_inconfident_indices[:max_inconfidence_num]

  
    return most_confident_indices, most_inconfident_indices



def rank_and_correct_v2(llm_confidence, update_pred, inconfident_indices, data, llm_confidence_score=80):
    '''
    rank the inconfident indices based on the llm confidence and confidence score
    lower than the harmonicity score and entrophy score rank
    '''
    
    pred_labels = torch.argmax(update_pred, dim=1)
    update_value, update_label = torch.max(update_pred[inconfident_indices], dim=-1)
    update_indices = torch.argsort(update_value, descending=True)
    llm_confidence_rank = np.flip(np.argsort(llm_confidence))
    for idx, node in enumerate(inconfident_indices):
        node = node.item()
        llm_rank = np.where(llm_confidence_rank == idx)[0].item()
        update_rank = torch.where(update_indices == idx)[0].item()
        # if llm_confidence[idx] < 90 and llm_confidence[idx] < update_value[idx]*100:
        #     data.y[node] = update_label[idx]
        
        if llm_rank > update_rank or llm_confidence[idx] < llm_confidence_score:
            data.y[node] = update_label[idx]