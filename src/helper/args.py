import argparse


def replace_args_with_dict_values(args, dictionary):
    for key, value in dictionary.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args

def get_command_line_args():
     parser = argparse.ArgumentParser(description='LLM Graph')
     parser.add_argument('--device', type=str, default='cpu')
     parser.add_argument('--dataset', default='', type=str)
     parser.add_argument('--normalize', default=0, type=int)
     parser.add_argument('--epochs', type=int, default=30)
     parser.add_argument('--early_stopping', type=int, default=10)
     parser.add_argument('--model_name', type=str, default='MLP')
     parser.add_argument('--norm', type=str, default=None)
     parser.add_argument('--main_seed_num', type=int, default=5)
     parser.add_argument('--sweep_seed_num', type=int, default=5)
     parser.add_argument('--return_embeds', type=int, default=1)
     parser.add_argument('--lr', type=float, default=0.01)
     parser.add_argument('--weight_decay', type=float, default=5e-4)
     parser.add_argument('--num_split', type=int, default=1)
     parser.add_argument('--sweep_split', type=int, default=1)
     parser.add_argument('--output_intermediate', type=int, default=0)
     parser.add_argument('--num_layers', type=int, default=2)
     parser.add_argument('--hidden_dimension', type=int, default=64)
     parser.add_argument('--edge_hidden', type=int, default=64)
     parser.add_argument('--dropout', type=float, default=0.5)
     parser.add_argument('--optim', type=str, default='adam')
     parser.add_argument('--warmup', default=10, type=int)
     parser.add_argument('--lr_gamma', default=0.998, type=float)
     parser.add_argument('--data_format', type=str, default='sbert')
     parser.add_argument('--early_stop_start', type=int, default=400)
     # parser.add_argument('--alpha', type=float, default=0.9)
     parser.add_argument('--low_label_test', type=int, default=0)
     parser.add_argument('--few_shot_test', type=int, default=0)
     parser.add_argument('--split', type=str, default='fixed')
     parser.add_argument("--sweep_round", type=int, default=50)
     parser.add_argument('--mode', type=str, default="main")
     parser.add_argument('--inductive', type=int, default = 0)
     parser.add_argument('--batchify', type=int, default = 0)
     parser.add_argument('--num_of_heads', type=int, default = 8)
     parser.add_argument('--num_of_out_heads', type=int, default = 1)
     parser.add_argument("--ensemble", nargs='+', type=str, default=[])
     parser.add_argument("--formats", nargs='+', type=str, default=[])
     parser.add_argument("--ensemble_string", type=str, default="")
     parser.add_argument("--pl_noise", type=float, default=0)
     parser.add_argument("--yaml_path",type=str,default="config.yaml")
     parser.add_argument("--no_val", type=int, default=0)
     parser.add_argument("--label_smoothing", type=float, default=0)
     parser.add_argument("--budget", type=int, default=20)
     parser.add_argument("--strategy", type=str, default="no")
     parser.add_argument("--filter_keep", type=int, default=0)
     parser.add_argument("--filter_strategy", type=str, default="none")
     parser.add_argument("--num_centers", type=int, default=1)
     parser.add_argument("--compensation", type=float, default=1)
     parser.add_argument("--save_logits", type=int, default=0)
     parser.add_argument("--save_data", type=int, default=0)
     parser.add_argument("--max_part", type=int, default=7)
     parser.add_argument("--debug", type=int, default=0)
     parser.add_argument("--train_vs_val", type=float, default = 3)
     parser.add_argument("--total_budget", type=int, default = -1)
     parser.add_argument("--initial_budget", type=float, default=0.5)
     parser.add_argument("--loss_type", type=str, default = 'ce')
     parser.add_argument("--second_filter", type=str, default = 'none')
     parser.add_argument("--debug_gt_label", type=int, default = 0)
     parser.add_argument("--train_stage", type=int, default = 1)
     parser.add_argument("--filter_all_wrong_labels", type=int, default = 0)
     parser.add_argument("--oracle", type=float, default = 1.0)
     parser.add_argument("--alpha", type=float, default = 0.1)
     parser.add_argument("--gcn2_alpha", type=float, default = 0.1)
     parser.add_argument("--appnp_alpha", type=float, default = 0.1)
     parser.add_argument("--shared_weights", type=bool, default = True)
     parser.add_argument("--beta", type=float, default = 0.1)
     parser.add_argument('--theta', type=float, default=.5,
                        help='theta for gcn2')
     parser.add_argument("--gamma", type=float, default = 0.1)
     parser.add_argument("--ratio", type=float, default = 0.3)
     parser.add_argument("--use_wandb", action='store_true')
     ### cluster arguments #######
     parser.add_argument("--stage", action='store_true')
     parser.add_argument("--clustering", type=str, default="DAEGC")
     parser.add_argument("-P", '--pretrain', dest="is_pretrain", default=False, action="store_true",
                    help="Whether to pretrain. Using '-P' to pretrain.")
     parser.add_argument("-TS", "--tsne", dest="plot_clustering_tsne", default=False, action="store_true",
                        help="Whether to draw the clustering tsne image. Using '-TS' to draw clustering TSNE.")
     parser.add_argument("-H", "--heatmap", dest="plot_embedding_heatmap", default=False, action="store_true",
                        help="Whether to draw the embedding heatmap. Using '-H' to draw embedding heatmap.")
     parser.add_argument("-N", "--cluster_norm", dest="adj_norm", default=False, action="store_true",
                        help="Whether to normalize the adj, default is False. Using '-N' to load adj with normalization.")
     parser.add_argument("-SLF", "--self_loop_false", dest="adj_loop", default=True, action="store_false",
                        help="Whether the adj has self-loop, default is True. Using '-SLF' to load adj without self-loop.")
     parser.add_argument("-SF", "--symmetric_false", dest="adj_symmetric", default=True, action="store_false",
                        help="Whether the normalization type is symmetric. Using '-SF' to load asymmetric adj.")
     parser.add_argument("-DS", "--desc", type=str, default="default",
                        help="The description of this experiment.")
     parser.add_argument("-M", "--model", type=str, dest="cluster_model_name", default="SDCN",
                        help="The model you want to run.")
     parser.add_argument("-D", '--cluster_dataset', type=str, dest="dataset_name", default="acm",
                        help="The dataset you want to use.")
     parser.add_argument("-R", "--root", type=str, default=None,
                        help="Input root path to switch relative path to absolute.")
     parser.add_argument("-K", "--k", type=int, default=None,
                        help="The k of KNN.")
     parser.add_argument("-T", "--t", type=int, default=None,
                        help="The order in GAT. 'None' denotes don't calculate the matrix M.")
     parser.add_argument("-LS", "--loops", type=int, default=1,
                        help="The Number of training rounds.")
     parser.add_argument("-F", "--feature", dest="feature_type", type=str, default="tensor", choices=["tensor", "npy"],
                        help="The datatype of feature. 'tenor' and 'npy' are available.")
     parser.add_argument("-L", "--label", dest="label_type", type=str, default="npy", choices=["tensor", "npy"],
                        help="The datatype of label. 'tenor' and 'npy' are available.")
     parser.add_argument("-A", "--adj", dest="adj_type", type=str, default="tensor", choices=["tensor", "npy"],
                        help="The datatype of adj. 'tenor' and 'npy' are available.")
     parser.add_argument("-S", "--seed", type=int, default=0,
                        help="The random seed. The default value is 0.")
    ########## arguments for our framework  ##########
     parser.add_argument("--confi_budget_size", type=int, default=100)
     parser.add_argument("--positive_sample_size", type=int, default=10)
     parser.add_argument("--negative_sample_size", type=int, default=10)
     parser.add_argument("--random_sample_size", type=int, default=10)
     parser.add_argument("--stage_num", type=int, default=1)
     parser.add_argument("--update_stage", type=int, default=1)
     parser.add_argument("--neighbor_size", type=int, default=10)
     parser.add_argument("--llm_mode", type=str, choices=["one_dim", "two_dim"], default="one_dim")
     parser.add_argument("--run_code", type=str, default="ours")
     parser.add_argument("--selection_ratio", type=float, default=0.1)
     parser.add_argument("--post_selection", type=int, default=1)
     parser.add_argument("--llm_confidence_score", type=int, default=80)
     parser.add_argument("--max_edge", type=float, default=5, help="max edge number times node number")
     parser.add_argument("--ensemble_alpha", type=float, default=0.5, help="ensemble alpha")
     # greedt_ET
     parser.add_argument("--greedy_k", type=int, default=1)
     parser.add_argument("--greedy_t", type=float, default=0.9999)
     parser.add_argument("--method", type=str, choices=['kmeans1', 'kmeans2', 'leiden'], default='kmeans2')
     parser.add_argument("--use_whitening", action='store_true')
     # NRGNN
     parser.add_argument("--n_p", type=int, default=50, 
                    help='number of positive pairs per node')
     parser.add_argument("--n_n", type=int, default=50, 
                        help='number of negitive pairs per node')
     parser.add_argument('--t_small',type=float, default=0.1, 
                    help='threshold of eliminating the edges')
     parser.add_argument('--p_u',type=float, default=0.8, 
                    help='threshold of adding pseudo labels')
     parser.add_argument('--label_correction_epoch',type=int, default=50)
     parser.add_argument('--edge_loss_alpha',type=float, default=1e-4)
     parser.add_argument('--edge_loss_beta',type=float, default=1e-6)
     parser.add_argument('--orig_edge_change_ratio',type=float, default=0.05)
     parser.add_argument('--candidate_edge_change_ratio',type=float, default=0.1)
     parser.add_argument('--edge_change_epoch',type=int, default=5)
     parser.add_argument('--update_structure',type=bool, default=False)
     parser.add_argument('--lc_lr',type=float, default=0.01)
     parser.add_argument('--given_feature',type=str, default='ensemble')
     # Subspace query 
     parser.add_argument("--subspace_alpha", type=float, default = 1.5)
     parser.add_argument("--svd_dim", type=str, default = 'k')
     parser.add_argument("--kmeans_dim", type=str, default = 'c')
     parser.add_argument("--subspace_t", type=int, default = 2)
     # ablation study
     parser.add_argument("--label_correction_ablation", type=int, default=0)
     parser.add_argument("--selection_ablation", type=int, default=0)
     args = parser.parse_args()
     return args