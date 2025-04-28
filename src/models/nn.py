
from torch_geometric.nn.conv import GCNConv, SAGEConv
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import torch.nn as nn
from torch_geometric.nn import LabelPropagation
from torch_geometric.nn.models import GAT
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv, APPNP
from torch_geometric.nn import GATConv as PYGGATConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import models.rev.memgcn as memgcn
from models.rev.rev_layer import SharedDropout
import copy
import tqdm
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
import torch_geometric.utils as utils
import time
from torch.cuda.amp import autocast
from torch_sparse import SparseTensor
import numpy as np
import torch.optim as optim
from copy import deepcopy

BIG_CONSTANT = 1e8


def get_model(args):
    if args.model_name == 'MLP':
        return UniversalMLP(args.num_layers, args.input_dim, args.hidden_dimension, args.num_classes, args.dropout, args.norm, args.return_embeds)
    elif args.model_name == 'GCN':
        return GCN(args.num_layers, args.input_dim, args.hidden_dimension, args.num_classes, args.dropout, args.norm)
    elif args.model_name == 'SAGE':
        return SAGE(args.num_layers, args.input_dim, args.hidden_dimension, args.num_classes, args.dropout, args.norm)
    elif args.model_name == 'S_model':
        return GCN(args.num_layers, args.input_dim, args.hidden_dimension, args.num_classes, args.dropout, args.norm)
    elif args.model_name == 'MLP2':
        return DeepMLP(args.input_dim, args.num_classes)
    elif args.model_name == 'LP':
        return LP(args.num_layers, args.alpha)
    elif args.model_name == 'BSAGE':
        return BSAGE(args.input_dim, args.hidden_dimension, args.num_classes, args.num_layers, args.dropout)
    elif args.model_name == 'GAT':
        return GAT2(args.input_dim, args.hidden_dimension, args.num_layers, args.num_classes, args.dropout, args.dropout, args.num_of_heads, args.num_of_out_heads, args.norm)
    elif args.model_name == 'AdjGCN':
        return AdjGCN(args.input_dim, args.hidden_dimension, args.num_classes, args.num_layers, args.dropout)
    elif args.model_name == 'AdjSAGE':
        return AdjSAGE(args.input_dim, args.hidden_dimension, args.num_classes, args.num_layers, args.dropout)
    elif args.model_name == 'GCNII':
        return GCNII(args.input_dim, args.hidden_dimension, args.num_classes, args.num_layers, args.gcn2_alpha, args.theta, args.shared_weights, args.dropout, args.device)
    elif args.model_name == 'APPNP':
        return APPNP_Net(args.input_dim, args.hidden_dimension, args.num_classes, args.dropout, args.appnp_alpha, args.num_layers)
    elif args.model_name == 'LINK':
        return LINK(args.num_node, args.num_classes)
    elif args.model_name == 'LINKX':
        return LINKX(args.input_dim, args.hidden_dimension, args.num_classes, args.num_layers, args.num_node, args.dropout)


class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
            x = data.graph['node_feat']
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
class LINKX(nn.Module):	
    """ our LINKX method with skip connections 
        a = MLP_1(A), x = MLP_2(X), MLP_3(sigma(W_1[a, x] + a + x))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5, cache=False, inner_activation=False, inner_dropout=False, init_layers_A=1, init_layers_X=1):
        super(LINKX, self).__init__()	
        self.mlpA = MLP(num_nodes, hidden_channels, hidden_channels, init_layers_A, dropout=0)
        self.mlpX = MLP(in_channels, hidden_channels, hidden_channels, init_layers_X, dropout=0)
        self.W = nn.Linear(2*hidden_channels, hidden_channels)
        self.mlp_final = MLP(hidden_channels, hidden_channels, out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.A = None
        self.inner_activation = inner_activation
        self.inner_dropout = inner_dropout

    def reset_parameters(self):	
        self.mlpA.reset_parameters()	
        self.mlpX.reset_parameters()
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()	

    def forward(self, data):	
        m = self.num_nodes
        row, col = data.edge_index
        row = row-row.min()
        A = SparseTensor(row=row, col=col,	
                 sparse_sizes=(m, self.num_nodes)
                        ).to_torch_sparse_coo_tensor()
        # A = -A
        xA = self.mlpA(A, input_tensor=True)
        
        xX = self.mlpX(data.x, input_tensor=True)
        x = torch.cat((xA, xX), axis=-1)
        x = self.W(x)
        if self.inner_dropout:
            x = F.dropout(x)
        if self.inner_activation:
            x = F.relu(x)
        x = F.relu(x + xA + xX)
        x = self.mlp_final(x, input_tensor=True)

        return x

class LINK(nn.Module):
    """ logistic regression on adjacency matrix """
    
    def __init__(self, num_nodes, out_channels):
        super(LINK, self).__init__()
        self.W = nn.Linear(num_nodes, out_channels)
        self.num_nodes = num_nodes

    def reset_parameters(self):
        self.W.reset_parameters()
        
    def forward(self, data):
        N = self.num_nodes
        edge_index = data.edge_index
        if isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            row = row-row.min() # for sampling
            A = SparseTensor(row=row, col=col, sparse_sizes=(N, self.num_nodes)).to_torch_sparse_coo_tensor()
        elif isinstance(edge_index, SparseTensor):
            A = edge_index.to_torch_sparse_coo_tensor()
        logits = self.W(A)
        return logits

class Rewire_GCN(nn.Module):
    """ 2 Layer Graph Convolutional Network.
    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    dropout : float
        dropout rate for GCN
    lr : float
        learning rate for GCN
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN. When `with_relu` is True, `weight_decay` will be set to 0.
    with_relu : bool
        whether to use relu activation function. If False, GCN will be linearized.
    with_bias: bool
        whether to include bias term in GCN weights.
    device: str
        'cpu' or 'cuda'.
    Examples
    --------
	We can first load dataset and then train GCN.
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    >>> gcn = gcn.to('cpu')
    >>> gcn.fit(features, adj, labels, idx_train) # train without earlystopping
    >>> gcn.fit(features, adj, labels, idx_train, idx_val, patience=30) # train with earlystopping
    """

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, self_loop=True ,device=None,relu='relu'):

        super(Rewire_GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1 = GCNConv(nfeat, nhid, bias=with_bias,add_self_loops=self_loop)
        self.gc2 = GCNConv(nhid, nclass, bias=with_bias,add_self_loops=self_loop)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.relu = relu
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None
        

    def forward(self, x, edge_index, edge_weight):
        if self.with_relu:
            if self.relu == 'relu':
                x = F.relu(self.gc1(x, edge_index,edge_weight))
            elif self.relu == 'selu':
                x = F.selu(self.gc1(x, edge_index,edge_weight))
        else:
            x = self.gc1(x, edge_index,edge_weight)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index,edge_weight)
        return x

    def initialize(self):
        """Initialize parameters of GCN.
        """
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
    def accuracy(self, output, labels):
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
    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False, **kwargs):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.
        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        normalize : bool
            whether to normalize the input adjacency matrix.
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """

        if initialize:
            self.initialize()

        self.edge_index, self.edge_weight = from_scipy_sparse_matrix(adj)
        self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.float().to(self.device)

        if sp.issparse(features):
            features = utils.sparse_mx_to_torch_sparse_tensor(features).to_dense().float()
        else:
            features = torch.FloatTensor(np.array(features))
        self.features = features.to(self.device)
        self.labels = torch.LongTensor(np.array(labels)).to(self.device)


        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            acc_val = self.accuracy(output[idx_val], labels[idx_val])

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)


    def test(self, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.forward(self.features, self.edge_index,self.edge_weight)
        loss_test = F.cross_entropy(output[idx_test], self.labels[idx_test])
        acc_test = self.accuracy(output[idx_test], self.labels[idx_test])
        
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return output

# class S_Model(torch.nn.Module):
class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, nfea, args, idx_train ,device='cuda'):
        super(EstimateAdj, self).__init__()
        
        self.estimator = Rewire_GCN(nfea, args.edge_hidden, args.edge_hidden,dropout=0.0,device=device)
        self.device = device
        self.args = args
        self.representations = 0

    def forward(self, edge_index, features):

        representations = self.estimator(features,edge_index,\
                                        torch.ones([edge_index.shape[1]]).to(self.device).float())
        rec_loss = self.reconstruct_loss(edge_index, representations)

        return representations,rec_loss
    
    def get_estimated_weights(self, edge_index, representations):

        x0 = representations[edge_index[0]]
        x1 = representations[edge_index[1]]
        output = torch.sum(torch.mul(x0.clone(),x1.clone()),dim=1)
        
        estimated_weights = F.relu(output).clone()
        estimated_weights[estimated_weights < self.args.t_small] = 0.0

        return estimated_weights
    
    def reconstruct_loss(self, edge_index, representations):
        
        num_nodes = representations.shape[0]
        randn = utils.negative_sampling(edge_index,num_nodes=num_nodes, num_neg_samples=self.args.n_n*num_nodes)
        randn = randn[:,randn[0]<randn[1]]

        edge_index = edge_index[:, edge_index[0]<edge_index[1]]
        neg0 = representations[randn[0]]
        neg1 = representations[randn[1]]
        neg = torch.sum(torch.mul(neg0,neg1),dim=1)

        pos0 = representations[edge_index[0]]
        pos1 = representations[edge_index[1]]
        pos = torch.sum(torch.mul(pos0,pos1),dim=1)

        rec_loss = (F.mse_loss(neg,torch.zeros_like(neg), reduction='sum') \
                    + F.mse_loss(pos, torch.ones_like(pos), reduction='sum')) \
                    * num_nodes/(randn.shape[1] + edge_index.shape[1]) 

        return rec_loss
class APPNP_Net(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dprate=.0, dropout=.5, K=10, alpha=.1, num_layers=3):
        super(APPNP_Net, self).__init__()
        
        self.mlp = MLP(in_channels= in_channels, hidden_channels=hidden_channels,out_channels= out_channels, num_layers=num_layers, dropout=dropout)
        self.prop1 = APPNP(K, alpha)

        self.dprate = dprate
        self.dropout = dropout
        
    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, data):
        edge_index = data.edge_index
        x = self.mlp(data.x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return x
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return x




class GCNII(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, alpha, theta, shared_weights=True, dropout=0.5, device='cuda:0'):
        super(GCNII, self).__init__()
        self.device = device
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        
        self.bns = nn.ModuleList()
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.dropout = dropout
        self.reset_parameters()
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.x
        n = data.x.shape[0]
        edge_index = data.edge_index
        edge_weight = None
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm( 
                edge_index, edge_weight, n, False, dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False, dtype=x.dtype)
            edge_weight=None
            adj_t = edge_index
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = self.bns[i](x)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)
        return x
    
class GCNII_predictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, alpha, theta, shared_weights=True, dropout=0.5):
        super(GCNII_predictor, self).__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        
        self.bns = nn.ModuleList()
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.dropout = dropout
        self.reset_parameters()
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
       
        n = x.shape[0]
        edge_weight = None
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm( 
                edge_index, edge_weight, n, False, dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False, dtype=x.dtype)
            edge_weight=None
            adj_t = edge_index
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = self.bns[i](x)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)
        return x
    


class GAT2(torch.nn.Module):
    def __init__(self, num_feat, hidden_dimension, num_layers, num_class, dropout, attn_drop, num_of_heads = 1, num_of_out_heads = 1, norm = None):
        super().__init__()
        self.layers = []
        self.bns = []
        if num_layers == 1:
            self.conv1 = PYGGATConv(num_feat, hidden_dimension, num_of_heads, concat = False, dropout=attn_drop)
        else:
            self.conv1 = PYGGATConv(num_feat, hidden_dimension, num_of_heads, concat = True, dropout=attn_drop)
            self.bns.append(torch.nn.BatchNorm1d(hidden_dimension * num_of_heads))
        self.layers.append(self.conv1)
        for _ in range(num_layers - 2):
            self.layers.append(
                PYGGATConv(hidden_dimension * num_of_heads, hidden_dimension, num_of_heads, concat = True, dropout = dropout)
            )
            self.bns.append(torch.nn.BatchNorm1d(hidden_dimension * num_of_heads))

        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        if num_layers > 1:
            self.layers.append(PYGGATConv(hidden_dimension * num_of_heads, num_class, heads=num_of_out_heads,
                             concat=False, dropout=attn_drop).cuda())
        self.layers = torch.nn.ModuleList(self.layers)
        self.bns = torch.nn.ModuleList(self.bns)
        self.norm = norm 
        self.num_layers = num_layers
        self.with_bn = True if self.norm == 'BatchNorm' else False
        self.dropout = dropout
        self.reset_parameters()
    def reset_parameters(self):
        """Resets the parameters of the model."""
        for layer in self.layers:
            layer.reset_parameters()  # 重置 GAT 层的参数
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()  # 重置 BatchNorm 层的参数
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.layers[i](x, edge_index)
            if i != self.num_layers - 1:
                if self.with_bn:
                    x = self.bns[i](x)
                x = F.elu(x)
        return x

class GAT2_predictor(torch.nn.Module):
    def __init__(self, num_feat, hidden_dimension, num_layers, num_class, dropout, attn_drop, num_of_heads = 1, num_of_out_heads = 1, norm = None):
        super().__init__()
        self.layers = []
        self.bns = []
        if num_layers == 1:
            self.conv1 = PYGGATConv(num_feat, hidden_dimension, num_of_heads, concat = False, dropout=attn_drop)
        else:
            self.conv1 = PYGGATConv(num_feat, hidden_dimension, num_of_heads, concat = True, dropout=attn_drop)
            self.bns.append(torch.nn.BatchNorm1d(hidden_dimension * num_of_heads))
        self.layers.append(self.conv1)
        for _ in range(num_layers - 2):
            self.layers.append(
                PYGGATConv(hidden_dimension * num_of_heads, hidden_dimension, num_of_heads, concat = True, dropout = dropout)
            )
            self.bns.append(torch.nn.BatchNorm1d(hidden_dimension * num_of_heads))

        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        if num_layers > 1:
            self.layers.append(PYGGATConv(hidden_dimension * num_of_heads, num_class, heads=num_of_out_heads,
                             concat=False, dropout=attn_drop).cuda())
        self.layers = torch.nn.ModuleList(self.layers)
        self.bns = torch.nn.ModuleList(self.bns)
        self.norm = norm 
        self.num_layers = num_layers
        self.with_bn = True if self.norm == 'BatchNorm' else False
        self.dropout = dropout
        self.reset_parameters()
    def reset_parameters(self):
        """Resets the parameters of the model."""
        for layer in self.layers:
            layer.reset_parameters()  # 重置 GAT 层的参数
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()  # 重置 BatchNorm 层的参数
    def forward(self, x, edge_index, edge_weight=None):
        for i in range(self.num_layers):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.layers[i](x, edge_index)
            if i != self.num_layers - 1:
                if self.with_bn:
                    x = self.bns[i](x)
                x = F.elu(x)
        return x


class GATWrapper(torch.nn.Module):
    def __init__(self, in_size, hidden_size, num_layers, out_size, dropout):
        super().__init__()
        self.gat = GAT(in_size, hidden_size, num_layers, out_size, dropout)
    
    def forward(self, data):
        x, edge_index= data.x, data.edge_index
        return self.gat(x, edge_index)



class UniversalMLP(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dimension, num_classes, dropout, norm=None, return_embeds = False) -> None:
        super().__init__()
        from torch_geometric.nn.models import MLP
        hidden_dimensions = [hidden_dimension] * (num_layers - 1)
        self.hidden_dimensions = [input_dim] + hidden_dimensions + [num_classes]
        self.mlp = MLP(channel_list=self.hidden_dimensions, dropout=dropout, norm=norm)
        self.return_embeds = False

    def forward(self, data):
        x = data.x
        return self.mlp(x)
    
    def inference(self, x_all, subgraph_loader, device):
        xs = []
        for batch in tqdm.tqdm(subgraph_loader):
            edge_index, n_id, size = batch.edge_index, batch.n_id, batch.batch_size
            edge_index = edge_index.to(device)
            # import ipdb; ipdb.set_trace()
            x = x_all[n_id][:batch.batch_size].to(device)
            x = self.mlp(x)
            xs.append(x.cpu())
        x_all = torch.cat(xs, dim=0)
        return x_all

class UniversalMLP_edge(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dimension, num_classes, dropout, norm=None, return_embeds = False) -> None:
        super().__init__()
        from torch_geometric.nn.models import MLP
        hidden_dimensions = [hidden_dimension] * (num_layers - 1)
        self.hidden_dimensions = [input_dim] + hidden_dimensions + [num_classes]
        self.mlp = MLP(channel_list=self.hidden_dimensions, dropout=dropout, norm=norm)
        self.return_embeds = False

    def forward(self, x):
        
        return self.mlp(x)
    
    def inference(self, x_all, subgraph_loader, device):
        xs = []
        for batch in tqdm.tqdm(subgraph_loader):
            edge_index, n_id, size = batch.edge_index, batch.n_id, batch.batch_size
            edge_index = edge_index.to(device)
            # import ipdb; ipdb.set_trace()
            x = x_all[n_id][:batch.batch_size].to(device)
            x = self.mlp(x)
            xs.append(x.cpu())
        x_all = torch.cat(xs, dim=0)
        return x_all

class DeepMLP(torch.nn.Module):
    def __init__(self, in_size, out_size) -> None:
        super().__init__()
        from torch_geometric.nn.models import MLP
        self.mlp = nn.Sequential(nn.Linear(in_size, 1024),
                nn.SELU(),
                nn.Dropout(0.5),
                nn.LayerNorm(1024),
                nn.Linear(1024, 512),
                nn.SELU(),
                nn.Dropout(0.5),
                nn.LayerNorm(512),
                nn.Linear(512, out_size),
        )
    
    def forward(self, data):
        x = data.x
        return self.mlp(x)


class DeepMLP_edge(torch.nn.Module):
    def __init__(self, in_size, out_size) -> None:
        super().__init__()
        from torch_geometric.nn.models import MLP
        self.mlp = nn.Sequential(nn.Linear(in_size, 1024),
                nn.SELU(),
                nn.Dropout(0.5),
                nn.LayerNorm(1024),
                nn.Linear(1024, 512),
                nn.SELU(),
                nn.Dropout(0.5),
                nn.LayerNorm(512),
                nn.Linear(512, out_size),
        )
    def reset_parameters(self):
        pass
    def forward(self, x):
        return self.mlp(x)


class GCN(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dimension, num_classes, dropout, norm=None) -> None:
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        if num_layers == 1:
            self.convs.append(GCNConv(input_dim, num_classes, cached=False,
                             normalize=True))
        else:
            self.convs.append(GCNConv(input_dim, hidden_dimension, cached=False,
                             normalize=True))
            if norm:
                self.norms.append(torch.nn.BatchNorm1d(hidden_dimension))
            else:
                self.norms.append(torch.nn.Identity())

            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dimension, hidden_dimension, cached=False,
                             normalize=True))
                if norm:
                    self.norms.append(torch.nn.BatchNorm1d(hidden_dimension))
                else:
                    self.norms.append(torch.nn.Identity())

            self.convs.append(GCNConv(hidden_dimension, num_classes, cached=False, normalize=True))

    def forward(self, data):
        x, edge_index, edge_weight= data.x, data.edge_index, data.edge_weight
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            if edge_weight != None:
                x = self.convs[i](x, edge_index, edge_weight)
            else:
                x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = self.norms[i](x)
                x = F.relu(x)
        return x
class GCN_predcitor(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dimension, num_classes, dropout, norm=None) -> None:
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        if num_layers == 1:
            self.convs.append(GCNConv(input_dim, num_classes, cached=False,
                             normalize=True))
        else:
            self.convs.append(GCNConv(input_dim, hidden_dimension, cached=False,
                             normalize=True))
            if norm:
                self.norms.append(torch.nn.BatchNorm1d(hidden_dimension))
            else:
                self.norms.append(torch.nn.Identity())

            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dimension, hidden_dimension, cached=False,
                             normalize=True))
                if norm:
                    self.norms.append(torch.nn.BatchNorm1d(hidden_dimension))
                else:
                    self.norms.append(torch.nn.Identity())

            self.convs.append(GCNConv(hidden_dimension, num_classes, cached=False, normalize=True))

    def forward(self, x, edge_index, edge_weight=None):
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            if edge_weight != None:
                x = self.convs[i](x, edge_index, edge_weight)
            else:
                x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = self.norms[i](x)
                x = F.relu(x)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dimension, num_classes, dropout, norm=None) -> None:
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        if num_layers == 1:
            self.convs.append(SAGEConv(input_dim, num_classes, cached=False,
                             normalize=True))
        else:
            self.convs.append(SAGEConv(input_dim, hidden_dimension, cached=False,
                             normalize=True))
            if norm:
                self.norms.append(torch.nn.BatchNorm1d(hidden_dimension))
            else:
                self.norms.append(torch.nn.Identity())

            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dimension, hidden_dimension, cached=False,
                             normalize=True))
                self.norms.append(torch.nn.BatchNorm1d(hidden_dimension))

            self.convs.append(SAGEConv(hidden_dimension, num_classes, cached=False, normalize=True))

    def forward(self, data):
        x, edge_index, edge_weight= data.x, data.edge_index, data.edge_weight
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            if edge_weight != None:
                x = self.convs[i](x, edge_index, edge_weight)
            else:
                x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = self.norms[i](x)
                x = F.relu(x)
        return x



class LP(torch.nn.Module):
    def __init__(self, num_layers, alpha) -> None:
        super().__init__()
        self.lp = LabelPropagation(num_layers, alpha)
    
    def forward(self, data):
        y= data.y
        train_mask = data.train_mask
        return self.lp(y, data.adj_t, train_mask)


def sbert(device):
    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/localscratch/czk/huggingface', device=device).to(device)
    return model 


def mpnet(device):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', cache_folder='/localscratch/czk/huggingface', device=device).to(device)
    return model 



class BSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = None
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return x

    def inference(self, x_all, subgraph_loader, device):
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                edge_index, n_id, size = batch.edge_index, batch.n_id, batch.batch_size
                edge_index = edge_index.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)
        return x_all




class ElementWiseLinear(nn.Module):
    def __init__(self, size, weight=True, bias=True, inplace=False):
        super().__init__()
        if weight:
            self.weight = nn.Parameter(torch.Tensor(size))
        else:
            self.weight = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(size))
        else:
            self.bias = None
        self.inplace = inplace

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.inplace:
            if self.weight is not None:
                x.mul_(self.weight)
            if self.bias is not None:
                x.add_(self.bias)
        else:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        return x


class GATConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads=1,
        feat_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        use_attn_dst=True,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        use_symmetric_norm=False,
    ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._use_symmetric_norm = use_symmetric_norm
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        if use_attn_dst:
            self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        else:
            self.register_buffer("attn_r", None)
        self.feat_drop = nn.Dropout(feat_drop)
        assert feat_drop == 0.0 # not implemented
        self.attn_drop = nn.Dropout(attn_drop)
        assert attn_drop == 0.0 # not implemented
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        if isinstance(self.attn_r, nn.Parameter):
            nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, perm=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src, feat_dst = h_src, h_dst
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = self.feat_drop(feat)
                feat_src = h_src
                feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    h_dst = h_src[: graph.number_of_dst_nodes()]
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                else:
                    h_dst = h_src
                    feat_dst = feat_src

            if self._use_symmetric_norm:
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            if self.attn_r is not None:
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.dstdata.update({"er": er})
                graph.apply_edges(fn.u_add_v("el", "er", "e"))
            else:
                graph.apply_edges(fn.copy_u("el", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))

            if self.training and self.edge_drop > 0:
                if perm is None:
                    perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                graph.edata["a"] = torch.zeros_like(e)
                graph.edata["a"][eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
            else:
                graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            if self._use_symmetric_norm:
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, 0.5)
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval

            # activation
            if self._activation is not None:
                rst = self._activation(rst)
            return rst


class RevGATBlock(nn.Module):
    def __init__(
        self,
        node_feats,
        edge_feats,
        edge_emb,
        out_feats,
        n_heads=1,
        attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        residual=True,
        activation=None,
        use_attn_dst=True,
        allow_zero_in_degree=True,
        use_symmetric_norm=False,
    ):
        super(RevGATBlock, self).__init__()

        self.norm = nn.BatchNorm1d(n_heads * out_feats)
        self.conv = GATConv(
                        node_feats,
                        out_feats,
                        num_heads=n_heads,
                        attn_drop=attn_drop,
                        edge_drop=edge_drop,
                        negative_slope=negative_slope,
                        residual=residual,
                        activation=activation,
                        use_attn_dst=use_attn_dst,
                        allow_zero_in_degree=allow_zero_in_degree,
                        use_symmetric_norm=use_symmetric_norm,
                    )
        self.dropout = SharedDropout()
        if edge_emb > 0:
            self.edge_encoder = nn.Linear(edge_feats, edge_emb)
        else:
            self.edge_encoder = None

    def forward(self, x, graph, dropout_mask=None, perm=None, efeat=None):
        if perm is not None:
            perm = perm.squeeze()
        out = self.norm(x)
        out = F.relu(out, inplace=True)
        if isinstance(self.dropout, SharedDropout):
            self.dropout.set_mask(dropout_mask)
        out = self.dropout(out)

        if self.edge_encoder is not None:
            if efeat is None:
                efeat = graph.edata["feat"]
            efeat_emb = self.edge_encoder(efeat)
            efeat_emb = F.relu(efeat_emb, inplace=True)
        else:
            efeat_emb = None

        out = self.conv(graph, out, perm).flatten(1, -1)
        return out


class RevGAT(nn.Module):
    def __init__(
        self,
        in_feats,
        n_classes,
        n_hidden,
        n_layers,
        n_heads,
        activation,
        dropout=0.0,
        input_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        use_attn_dst=True,
        use_symmetric_norm=False,
        group=2,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.num_heads = n_heads
        self.group = group

        self.convs = nn.ModuleList()
        self.norm = nn.BatchNorm1d(n_heads * n_hidden)

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            num_heads = n_heads if i < n_layers - 1 else 1
            out_channels = n_heads

            if i == 0 or i == n_layers -1:
                self.convs.append(
                    GATConv(
                        in_hidden,
                        out_hidden,
                        num_heads=num_heads,
                        attn_drop=attn_drop,
                        edge_drop=edge_drop,
                        use_attn_dst=use_attn_dst,
                        use_symmetric_norm=use_symmetric_norm,
                        residual=True,
                    )
                )
            else:
                Fms = nn.ModuleList()
                fm = RevGATBlock(
                    in_hidden // group,
                    0,
                    0,
                    out_hidden // group,
                    n_heads=num_heads,
                    attn_drop=attn_drop,
                    edge_drop=edge_drop,
                    use_attn_dst=use_attn_dst,
                    use_symmetric_norm=use_symmetric_norm,
                    residual=True,
                )
                for i in range(self.group):
                    if i == 0:
                        Fms.append(fm)
                    else:
                        Fms.append(copy.deepcopy(fm))

                invertible_module = memgcn.GroupAdditiveCoupling(Fms,
                                                                 group=self.group)

                conv = memgcn.InvertibleModuleWrapper(fn=invertible_module,
                                                      keep_input=False)

                self.convs.append(conv)

        self.bias_last = ElementWiseLinear(n_classes, weight=False, bias=True, inplace=True)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = dropout
        self.dp_last = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        h = self.input_drop(h)

        self.perms = []
        for i in range(self.n_layers):
            perm = torch.randperm(graph.number_of_edges(),
                                  device=graph.device)
            self.perms.append(perm)

        h = self.convs[0](graph, h, self.perms[0]).flatten(1, -1)

        m = torch.zeros_like(h).bernoulli_(1 - self.dropout)
        mask = m.requires_grad_(False) / (1 - self.dropout)

        for i in range(1, self.n_layers-1):
            graph.requires_grad = False
            perm = torch.stack([self.perms[i]]*self.group, dim=1)
            h = self.convs[i](h, graph, mask, perm)

        h = self.norm(h)
        h = self.activation(h, inplace=True)
        h = self.dp_last(h)
        h = self.convs[-1](graph, h, self.perms[-1])

        h = h.mean(1)
        h = self.bias_last(h)

        return h
               
class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        # input_dim is the number of input features.
        # We have one output, hence the "1" in the second argument.
        self.linear = nn.Linear(input_dim, 1)  # This includes the bias by default

    def forward(self, x):
        return self.linear(x)



class AdjGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(AdjGCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x = data.x
        adj_t = data.adj_t
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)
    

class AdjSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(AdjSAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    # @autocast()
    def forward(self, data):
        x = data.x
        adj_t = data.adj_t
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)
