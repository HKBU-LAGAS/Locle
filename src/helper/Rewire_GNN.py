#%%
import time
import numpy as np
from copy import deepcopy
from ogb.nodeproppred import Evaluator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
import torch_geometric.utils as utils
import scipy.sparse as sp
from models.nn import Rewire_GCN, DeepMLP, UniversalMLP, DeepMLP_edge, UniversalMLP_edge, GCN_predcitor, GAT2_predictor, GCNII_predictor
from helper.utils import accuracy,sparse_mx_to_torch_sparse_tensor, dirichlet_energy, harmonicity, entrophy_confidence
from helper.train_utils import seed_everything, train
class Rewire_GNN:
    def __init__(self, args, device, confident_indices, inconfident_indices):

        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_acc_pred_val = 0
        self.best_pred = None
        self.best_graph = None
        self.best_model_index = None
        self.weights = None
        self.estimator = None
        self.model = None
        self.pred_edge_index = None
        self.confident_indices = confident_indices.to(self.device)
        self.inconfident_indices = inconfident_indices.to(self.device)

    def fit(self, data, prev_res=None):
        seed_everything(self.args.seed)
        from torch_geometric.utils import homophily
        before_edge_homo = homophily(data.edge_index, data.backup_y)
        before_node_homo = homophily(data.edge_index, data.backup_y, method='node')
        before_edge_insensitive_homo = homophily(data.edge_index, data.backup_y, method='edge_insensitive')
        features = data.x
        edge_index = data.edge_index
        labels = data.y
        self.best_epoch = 0
        idx_train = np.array(torch.where(data.train_mask)[0].cpu().detach() )
        idx_val = np.array(torch.where(data.test_mask)[0].cpu().detach())
        labels[idx_val] = data.backup_y[idx_val]
        
        args = self.args
        features = features.to(self.device)
        labels = torch.LongTensor(np.array(labels.cpu().detach())).to(self.device)
        self.features = features
        self.labels = labels
        self.edge_num = edge_index.shape[1]
        self.edge_index = edge_index
        
        self.idx_unlabel = torch.LongTensor(list(set(range(features.shape[0])) - set(idx_train))).to(self.device)
        if prev_res is not None:
            features = prev_res
            self.features = features
       
        if self.args.model_name == 'GCN':
            self.predictor = GCN_predcitor(num_layers=2, input_dim=data.x.shape[1],hidden_dimension = args.hidden_dimension, num_classes=features.shape[1], dropout=args.dropout, norm = args.norm).to(self.device)
        elif self.args.model_name == 'GAT':
            self.predictor = GAT2_predictor(args.input_dim, args.hidden_dimension, args.num_layers, args.num_classes, args.dropout, args.dropout, args.num_of_heads, args.num_of_out_heads, args.norm).to(self.device)
        elif self.args.model_name == 'GCNII':
            self.predictor = GCNII_predictor(args.input_dim, args.hidden_dimension, features.shape[1], args.num_layers, args.gcn2_alpha, args.theta, args.shared_weights, args.dropout).to(self.device)
        
        self.model = Rewire_GCN(nfeat=features.shape[1],
                        nhid=self.args.hidden_dimension,
                        nclass=labels.max().item() + 1,
                        self_loop=True,
                        dropout=self.args.dropout, device=self.device).to(self.device)
        if prev_res is not None:
            self.estimator = EstimateAdj(prev_res.shape[1], args, idx_train ,edge_num = self.edge_num,data=data,device=self.device).to(self.device)
           
        else:
            self.estimator = EstimateAdj(features.shape[1], args, idx_train ,edge_num = self.edge_num,data=data,device=self.device).to(self.device)

        self.pred_edge_index = self.get_train_edge(edge_index,features, args.n_p ,idx_train)

        self.predictor_weights = torch.ones([self.edge_index.shape[1]],device=self.device)

        self.optimizer_est = optim.Adam(list(self.estimator.parameters()),
                               lr=args.lc_lr, weight_decay=args.weight_decay)
        self.optimizer_pre = optim.Adam(list(self.predictor.parameters()),
                               lr=args.lc_lr, weight_decay=args.weight_decay)
        # Train model
        t_total = time.time()

        features = features.requires_grad_()
        self.train_est(features,  self.edge_index,self.pred_edge_index, idx_train, idx_val, prev_res, args)
        features = features.clone().detach().requires_grad_(True)
        for epoch in range(args.label_correction_epoch):
            self.acc_update = self.train(epoch, data.x,  self.edge_index,self.pred_edge_index,idx_train, idx_val, prev_res, args)
            

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        print("picking the best model according to validation performance")
        self.predictor.load_state_dict(self.predictor_model_weigths)

        print("=====validation set accuracy=======")
        test_res = self.test(idx_val, orig_feature=data.x)
        print("===================================")

       

        return self.edge_index, self.best_pred, test_res, self.acc_update
    def train_est(self, features, edge_index,pred_edge_index, idx_train, idx_val, prev_embedding = None, args=None):
        best_rec_loss = 1000000
        best_epoch = 0
        for epoch in range(30):
            args = self.args

            t = time.time()
            self.estimator.estimator.train()
            self.optimizer_est.zero_grad()
            # obtain representations and rec loss of the estimator
            representations, predictor_weights,total_edge_index, edge_index = self.estimator(self.edge_index,pred_edge_index, self.predictor_weights,prev_embedding, epoch=epoch, args=args)
            
            rec_loss = self.estimator.reconstruct_loss(edge_index, features, args=args, epoch=epoch)
            rec_loss.backward()
            torch.cuda.empty_cache()
            self.optimizer_est.step()

            if best_rec_loss > rec_loss:
                best_rec_loss = rec_loss
                self.predictor_weights = predictor_weights

                self.edge_index = edge_index
                best_epoch = epoch
      

        
    def train(self, epoch, features, edge_index,pred_edge_index, idx_train, idx_val, prev_embedding = None, args=None):
        args = self.args
        t = time.time()
        self.predictor.train()
        self.optimizer_pre.zero_grad()
      
        log_pred = self.predictor(features,self.edge_index)

        y_pred = log_pred.argmax(dim=-1, keepdim=True)
        y = self.labels.unsqueeze(dim=1)
        evaluator = Evaluator(name='ogbn-arxiv')
        acc = evaluator.eval({
            'y_true': y[idx_val],
            'y_pred': y_pred[idx_val],
        },        )
      
        if self.best_pred == None:
            
            pred = F.softmax(log_pred,dim=1).detach()
            self.best_pred = pred
         
        else:
            pred = self.best_pred

       
        eps = 1e-8
      
        loss_pred = F.cross_entropy(log_pred[idx_train], self.labels[idx_train])
        loss_pred.backward()
      
        self.optimizer_pre.step()
      
        self.predictor.eval()
        pred = F.softmax(self.predictor(features,self.edge_index),dim=1)
     
        acc_pred_val = accuracy(pred[idx_val], self.labels[idx_val])
    
        if acc_pred_val > self.best_acc_pred_val:
            self.best_acc_pred_val = acc_pred_val
            self.best_pred_graph = self.predictor_weights.detach()
            self.best_pred = pred.detach()
            self.best_model_index = self.edge_index
            self.best_epoch = epoch
            self.predictor_model_weigths = deepcopy(self.predictor.state_dict())


   
        return acc
    def test(self, idx_test, orig_feature = None):
        """Evaluate the performance of ProGNN on test set
        """
        features = self.features
        labels = self.labels

        self.predictor.eval()
        if orig_feature is not None:
            features = orig_feature
        output = self.predictor(features, self.edge_index)
        y_pred = output.argmax(dim=-1, keepdim=True)
        y = labels.unsqueeze(dim=1)
        evaluator = Evaluator(name='ogbn-arxiv')
        acc = evaluator.eval({
            'y_true': y[idx_test],
            'y_pred': y_pred[idx_test],
        },        )
        loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
      

        return acc
    
    def get_train_edge(self, edge_index, features, n_p, idx_train):
        '''
        obtain the candidate edge between labeled nodes and unlabeled nodes based on cosine sim
        n_p is the top n_p labeled nodes similar with unlabeled nodes
        '''

        if n_p == 0:
            return None
        poten_edges = []
        if n_p > len(idx_train) or n_p < 0:
            for i in range(len(features)):
                indices = set(idx_train)
                indices = indices - set(edge_index[1,edge_index[0]==i])
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        else:
            for i in range(len(features)):
                sim = torch.div(torch.matmul(features[i],features[idx_train].T), features[i].norm()*features[idx_train].norm(dim=1))
                _,rank = sim.topk(n_p)
                if rank.max() < len(features) and rank.min() >= 0:
                    indices = idx_train[rank.cpu().numpy()]
                    indices = set(indices)
                else:
                    indices = set()
                indices = indices - set(edge_index[1,edge_index[0]==i])
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        poten_edges = torch.as_tensor(poten_edges).T
        poten_edges = utils.to_undirected(poten_edges,len(features)).to(self.device)

        return poten_edges
   
    def get_model_edge(self, pred):
        idx_add = self.idx_unlabel[(pred.max(dim=1)[0][self.idx_unlabel] > self.args.p_u)]

        row = self.idx_unlabel.repeat(len(idx_add))
        col = idx_add.repeat(len(self.idx_unlabel),1).T.flatten()
        mask = (row!=col)
        unlabel_edge_index = torch.stack([row[mask],col[mask]], dim=0)

        return unlabel_edge_index, idx_add
                        
#%%
class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, nfea, args, idx_train ,edge_num, data=None, device='cuda'):
        super(EstimateAdj, self).__init__()
        self.estimator = DeepMLP_edge(nfea, args.edge_hidden)
        self.device = device
        self.args = args
        self.representations = 0
        self.data = data
        self.edge_num = edge_num
    def forward(self, edge_index, pred_edge_index,predictor_weights,features, data=None, epoch=0, step = 5, args=None):
        orig_edge_size = edge_index.shape[1]
        candidate_edge_size = pred_edge_index.shape[1]
        max_edge_size = features.shape[0] * args.max_edge
        total_edge_index = torch.cat([edge_index, pred_edge_index], dim=1)
        representations = self.estimator(features)
        predictor_weights = self.get_estimated_weigths(total_edge_index, representations).to(self.device)
        if epoch % args.edge_change_epoch == 0 and epoch > 0:
            max_candidiate_edge_size = int(max_edge_size - orig_edge_size*(1-args.orig_edge_change_ratio))
            
            orig_edge_indices = torch.argsort(predictor_weights[:orig_edge_size], )[:int(orig_edge_size*args.orig_edge_change_ratio)]
            pred_edge_indices = torch.argsort(predictor_weights[orig_edge_size:], descending=True)[:int(candidate_edge_size*args.candidate_edge_change_ratio)] + orig_edge_size
            edge_index_indices = torch.tensor(range(orig_edge_size)).to(self.device)
            mask = ~torch.isin(edge_index_indices, orig_edge_indices)
            change_indices = torch.cat([orig_edge_indices, pred_edge_indices], dim=0)
            change_size = min(max(int(orig_edge_size*args.orig_edge_change_ratio), max_candidiate_edge_size), len(change_indices))
            changed_indices = torch.argsort(predictor_weights[change_indices], descending=True)[:change_size]
            edge_index = torch.cat([edge_index[:, edge_index_indices[mask]], total_edge_index[:, change_indices[changed_indices]]], dim=1)
            predictor_weights = predictor_weights[torch.cat([edge_index_indices[mask], change_indices[changed_indices]])]
            edge_index = utils.to_undirected(edge_index, num_nodes=features.shape[0]).to(self.device)
        else:
            predictor_weights = predictor_weights[:orig_edge_size]
        return representations, predictor_weights, total_edge_index, edge_index
    
    def get_estimated_weigths(self, edge_index, representations):
        x0 = representations[edge_index[0]]
        x1 = representations[edge_index[1]]
        output = torch.sum(torch.mul(x0,x1),dim=1)
        torch.cuda.empty_cache()
        estimated_weights = F.relu(output).clone()


        return estimated_weights
    import torch.nn.functional as F

  

    def reconstruct_loss(self, edge_index, representations, predictor_weights=None, args=None, epoch=0):
        
     
        dirichlet_loss, lap = dirichlet_energy(representations, edge_index) 
        if dirichlet_loss < 0:
            import pdb; pdb.set_trace()

        final_loss = (dirichlet_loss*args.edge_loss_alpha + (torch.norm(lap, p='fro').requires_grad_()**2)*args.edge_loss_beta)*representations.shape[0] / edge_index.shape[1]
        return final_loss

