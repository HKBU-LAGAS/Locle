
#  # cora GCN
python src/main.py --dataset cora --model_name GCN --data_format sbert --main_seed_num 3 --split active --output_intermediate 0 --no_val 1 \
--strategy subspace --debug 1 --total_budget 350 --initial_budget 0.5 --filter_strategy consistency --loss_type ce --second_filter conf+entropy \
--epochs 20 --debug_gt_label 0 --early_stop_start 150 --filter_all_wrong_labels 0 --oracle 1 --ratio 0.3 --alpha 0.33 --beta 0.33 --positive_sample_size 80 \
--negative_sample 60 --stage_num 5 --run_code ours --svd_dim 128 --subspace_alpha 1 --device cuda:0 --post_selection 2 --n_p 3 --edge_loss_alpha 2e-2 \
--edge_loss_beta 1e-6 --edge_change_epoch 1 --label_correction_epoch 20 --orig_edge_change_ratio 0.1 --candidate_edge_change_ratio 0.05 --lc_lr 0.05 \
--llm_confidence 70 --label_correction_ablation 0 --selection_ablation 0 --max_edge 5 --device cuda:0 
 # cora GAT
CUDA_VISIBLE_DEVICES=1 python src/main.py --dataset cora --model_name GAT --data_format sbert --main_seed_num 3 --split active --output_intermediate 0 \
--no_val 1 --strategy subspace --debug 1 --total_budget 350 --initial_budget 0.4 --filter_strategy consistency --loss_type ce \
--second_filter conf+entropy --epochs 20 --debug_gt_label 0 --early_stop_start 150 --filter_all_wrong_labels 0 --oracle 1 --ratio 0.3 \
--alpha 0.33 --beta 0.33 --positive_sample_size 80 --negative_sample 60 --stage_num 5 --run_code ours --subspace_alpha 1 --device cuda:0 \
--post_selection 2 --n_p 3 --edge_loss_alpha 2e-2 --edge_loss_beta 1e-6 --edge_change_epoch 1 --svd_dim 128 --label_correction_epoch 20 \
--orig_edge_change_ratio 0.1 --candidate_edge_change_ratio 0.05 --lc_lr 0.05 --llm_confidence 70 --label_correction_ablation 0 --selection_ablation 0 \
--max_edge 5 --device cuda:0 
 # cora GCNII
CUDA_VISIBLE_DEVICES=2 python src/main.py --dataset cora --model_name GCNII --data_format sbert --main_seed_num 3 --split active \
--output_intermediate 0 --no_val 1 --strategy subspace --debug 1 --total_budget 350 --initial_budget 0.4 --filter_strategy consistency \
--loss_type ce --second_filter conf+entropy --epochs 100 --debug_gt_label 0 --early_stop_start 150 --filter_all_wrong_labels 0 --oracle 1 --ratio 0.3 \
--alpha 0.33 --beta 0.33 --positive_sample_size 80 --negative_sample 60 --stage_num 5 --run_code ours --subspace_alpha 1 --device cuda:0 --post_selection 2 \
--n_p 3 --edge_loss_alpha 2e-2 --edge_loss_beta 1e-6 --edge_change_epoch 1 --svd_dim 128 --label_correction_epoch 100 --orig_edge_change_ratio 0.0 \
--candidate_edge_change_ratio 0.01 --lc_lr 0.05 --llm_confidence 70 --label_correction_ablation 0 --selection_ablation 0 --max_edge 5 \
--device cuda:0 --lr 0.01 --num_layers 64 
 # citeseer GCN
CUDA_VISIBLE_DEVICES=3 python src/main.py --dataset citeseer2 --model_name GCN --data_format sbert --main_seed_num 3 --split active \
--output_intermediate 0 --no_val 1 --strategy subspace --debug 1 --total_budget 200 --initial_budget 0.5 --filter_strategy consistency \
--loss_type ce --second_filter conf+entropy --epochs 15 --debug_gt_label 0 --early_stop_start 150 --filter_all_wrong_labels 0 --oracle 1 --ratio 0.2 \
 --alpha 0.33 --beta 0.33 --alpha 0.33 --beta 0.33 --positive_sample_size 60 --negative_sample 20 --stage_num 5 --run_code ours --subspace_alpha 1 \
 --kmeans_dim 2 --svd_dim 128 --device cuda:0 --post_selection 1 --n_p 3 --edge_loss_alpha 1e-2 --edge_loss_beta 1e-6 --edge_change_epoch 1 \
 --label_correction_epoch 20 --orig_edge_change_ratio 0 --candidate_edge_change_ratio 0.02 --lc_lr 0.05 --llm_confidence 70 --label_correction_ablation 0 \
 --selection_ablation 0 --max_edge 4 
 # citeseer GCNII
python  src/main.py --dataset citeseer2 --model_name GCNII --lr 0.01 --data_format sbert --main_seed_num 3 --split active --output_intermediate 0 --no_val 1 --strategy subspace --debug 1 --total_budget 200 --initial_budget 0.3 --filter_strategy consistency --loss_type ce --second_filter conf+entropy --epochs 100 --debug_gt_label 0 --early_stop_start 150 --filter_all_wrong_labels 0 --oracle 1 --ratio 0 --alpha 0.33 --beta 0.33 --alpha 0.33 --beta 0.33 --positive_sample_size 60 --negative_sample 60 --stage_num 5 --run_code ours --subspace_alpha 1 --kmeans_dim 8 --svd_dim 256 --device cuda:0 --post_selection 2 --n_p 3 --edge_loss_alpha 1e-2 --edge_loss_beta 1e-6 --edge_change_epoch 1 --label_correction_epoch 100 --orig_edge_change_ratio 0. --candidate_edge_change_ratio 0.1 --lc_lr 0.05 --llm_confidence 70 --label_correction_ablation 0 --selection_ablation 0 --max_edge 4 --num_layers 32 
 # citeseer GAT
CUDA_VISIBLE_DEVICES=1 python src/main.py --dataset citeseer2 --model_name GAT --data_format sbert --main_seed_num 3 --split active \
--output_intermediate 0 --no_val 1 --strategy subspace --debug 1 --total_budget 200 --initial_budget 0.4 --filter_strategy consistency \
--loss_type ce --second_filter conf+entropy --epochs 15 --debug_gt_label 0 --early_stop_start 150 --filter_all_wrong_labels 0 --oracle 1 --ratio 0.3 \
--alpha 0.33 --beta 0.33 --alpha 0.33 --beta 0.33 --positive_sample_size 40 --negative_sample 20 --stage_num 5 --run_code ours --subspace_alpha 1 \
--kmeans_dim 2 --svd_dim 128 --device cuda:0 --post_selection 1 --n_p 3 --edge_loss_alpha 1e-2 --edge_loss_beta 1e-6 --edge_change_epoch 1 \
--label_correction_epoch 20 --orig_edge_change_ratio 0 --candidate_edge_change_ratio 0.01 --lc_lr 0.05 --llm_confidence 70 --label_correction_ablation 0 \
--selection_ablation 0 --max_edge 4 
# pubmed GCN final 
CUDA_VISIBLE_DEVICES=2 python  src/main.py --dataset pubmed --model_name GCN --data_format sbert --main_seed_num 3 --split active \
--output_intermediate 0 --no_val 1 --strategy subspace --debug 1 --total_budget 150 --initial_budget 0.5 --filter_strategy consistency \
--loss_type ce --second_filter conf+entropy --epochs 30 --debug_gt_label 0 --early_stop_start 150 --filter_all_wrong_labels 0 --oracle 1 \
--ratio 0. --alpha 0.33 --beta 0.33 --positive_sample_size 40 --negative_sample 20 --given_feature ensemble --stage_num 5 --run_code ours \
--subspace_alpha 1.2 --svd_dim 256 --edge_loss_beta 5e-7 --kmeans_dim 16 --n_p 5 --device cuda:0 --post_selection 1 --max_edge 6 \
--orig_edge_change_ratio 0 --candidate_edge_change_ratio 0 --edge_change_epoch 1 --llm_confidence_score 0 --label_correction_epoch 30 --edge_loss_alpha 1e-4 --lc_lr 0.1 --label_correction_ablation 0 --weight_decay 5e-4  &
 # pubmed GAT
python src/main.py --dataset pubmed --model_name GAT --data_format sbert --main_seed_num 3 --split active \
--output_intermediate 0 --no_val 1 --strategy subspace --debug 1 --total_budget 150 --initial_budget 0.5 \
--filter_strategy consistency --loss_type ce --second_filter conf+entropy --epochs 20 --debug_gt_label 0 --early_stop_start 150 \
--filter_all_wrong_labels 0 --oracle 1 --ratio 0 --alpha 0.33 --beta 0.33 --positive_sample_size 1 --negative_sample 20 \
--given_feature ensemble --stage_num 5 --run_code ours --subspace_alpha 1.2 --svd_dim 4 --edge_loss_beta 5e-7 --kmeans_dim 2 \
--n_p 5 --device cuda:0 --post_selection 1 --max_edge 5 --orig_edge_change_ratio 0. --candidate_edge_change_ratio 0.3 \
--edge_change_epoch 1 --llm_confidence_score 0 --label_correction_epoch 30 --edge_loss_alpha 1e-4 --lc_lr 0.1 --weight_decay 5e-4 \
--dropout 0.2 --hidden_dim 16 --label_correction_ablation 0 --num_of_out_heads 8 &
 # pubmed GCNII
CUDA_VISIBLE_DEVICES=0 python src/main.py --dataset pubmed --model_name GCNII --lr 0.01 --data_format sbert --main_seed_num 3 \
--split active --output_intermediate 0 --no_val 1 --strategy subspace --debug 1 --total_budget 150 --initial_budget 0.5 \
--filter_strategy consistency --loss_type ce --second_filter conf+entropy --epochs 100 --debug_gt_label 0 --early_stop_start 150 \
--filter_all_wrong_labels 0 --oracle 1 --ratio 0 --alpha 0.33 --beta 0.33 --positive_sample_size 60 --negative_sample 20 \
--given_feature ensemble --stage_num 4 --run_code ours --subspace_alpha 1.2 --svd_dim 4 --edge_loss_beta 5e-7 --kmeans_dim 16 \
--n_p 3 --device cuda:0 --num_layers 16 --label_correction_ablation 0 --lc_lr 0.01 --label_correction_epoch 100 &
# wikics GCN
python src/main.py --dataset wikics --model_name GCN --data_format sbert --main_seed_num 3 --split active --output_intermediate 0 --no_val 1 \
 --strategy subspace --debug 1 --total_budget 400 --filter_strategy consistency --loss_type ce --second_filter conf+entropy --epochs 30 \
 --initial_budget 0.5 --debug_gt_label 0 --early_stop_start 150 --filter_all_wrong_labels 0 --oracle 1 --ratio 0.2 --alpha 0.33 --beta 0.33 \
 --positive_sample_size 20 --negative_sample 20 --stage_num 5 --run_code ours --subspace_alpha 1 --kmeans_dim 8 --svd_dim 256 --device cuda:0 \
 --post_selection 1 --n_p 3 --edge_loss_alpha 1e-4 --edge_loss_beta 5e-7 --edge_change_epoch 1 --label_correction_epoch 30 --orig_edge_change_ratio 0.01 \
 --candidate_edge_change_ratio 0.3 --lc_lr 0.1 --llm_confidence 70 --label_correction_ablation 0 &
 # wikics GAT
CUDA_VISIBLE_DEVICES=2 python src/main.py --dataset wikics --model_name GAT --data_format sbert --main_seed_num 3 --split active --output_intermediate 0 \
--no_val 1 --strategy subspace --debug 1 --total_budget 400 --filter_strategy consistency --loss_type ce --second_filter conf+entropy --epochs 30 \
--initial_budget 0.3 --debug_gt_label 0 --early_stop_start 150 --filter_all_wrong_labels 0 --oracle 1 --ratio 0 --alpha 0.33 --beta 0.33 \
--positive_sample_size 80 --negative_sample 20 --stage_num 5 --run_code ours --subspace_alpha 1 --kmeans_dim 2 --svd_dim 128 --device cuda:0 --post_selection 2 \
--n_p 3 --edge_loss_alpha 1e-4 --edge_loss_beta 5e-7 --edge_change_epoch 1 --label_correction_epoch 30 --orig_edge_change_ratio 0. \
--candidate_edge_change_ratio 0.01 --lc_lr 0.05 --llm_confidence 0 --label_correction_ablation 0 &
 # wikics GCNII
python src/main.py --dataset wikics --model_name GCNII --data_format sbert --main_seed_num 3 --split active --output_intermediate 0 --no_val 1 --strategy subspace \
 --debug 1 --total_budget 400 --filter_strategy consistency --loss_type ce --second_filter conf+entropy --epochs 100 --initial_budget 0.4 --debug_gt_label 0 \
 --early_stop_start 150 --filter_all_wrong_labels 0 --oracle 1 --ratio 0.2 --alpha 0.33 --beta 0.33 --positive_sample_size 1 --negative_sample 20 --stage_num 5 \
 --run_code ours --subspace_alpha 1 --kmeans_dim 2 --svd_dim 128 --device cuda:0 --post_selection 2 --n_p 3 --edge_loss_alpha 1e-4 --edge_loss_beta 5e-7 \
 --edge_change_epoch 1 --label_correction_epoch 100 --orig_edge_change_ratio 0.1 --candidate_edge_change_ratio 0.3 --lc_lr 0.1 --llm_confidence 70 \
 --label_correction_ablation 0 --lr 0.01 --num_layers 8 &
 # dblp GCN final
CUDA_VISIBLE_DEVICES=3 python src/main.py --dataset dblp --model_name GCN --data_format sbert --main_seed_num 3 --split active --output_intermediate 0 \
--no_val 1 --lr 0.1 --strategy subspace --debug 1 --total_budget 320 --initial_budget 0.25 --filter_strategy consistency --loss_type ce \
--second_filter conf+entropy --epochs 20 --debug_gt_label 0 --early_stop_start 150 --filter_all_wrong_labels 0 --oracle 1 --ratio 0.1 --alpha 0.33 \
--beta 0.33 --positive_sample_size 80 --negative_sample 60 --stage_num 5 --run_code ours --subspace_alpha 1 --kmeans_dim k --svd_dim 64 --device cuda:0 \
--post_selection 2 --n_p 3 --edge_loss_alpha 1e-4 --edge_loss_beta 5e-7 --edge_change_epoch 1 --label_correction_epoch 20 --orig_edge_change_ratio 0 \
--candidate_edge_change_ratio 0.05 --lc_lr 0.1 --llm_confidence 0 --label_correction_ablation 0 --selection_ablation 0 --max_edge 70 &
# dblp GAT final
CUDA_VISIBLE_DEVICES=1 python src/main.py --dataset dblp --model_name GAT --data_format sbert --main_seed_num 3 --split active --output_intermediate 0 \
--no_val 1 --lr 0.01 --strategy subspace --debug 1 --total_budget 240 --initial_budget 0.4 --filter_strategy consistency --loss_type ce \
--second_filter conf+entropy --epochs 30 --debug_gt_label 0 --early_stop_start 150 --filter_all_wrong_labels 0 --oracle 1 --ratio 0.2 --alpha 0.33 \
--beta 0.33 --positive_sample_size 40 --negative_sample 60 --stage_num 5 --run_code ours --subspace_alpha 1 --kmeans_dim 8 --svd_dim 32 --device cuda:0 \
--post_selection 1 --n_p 3 --edge_loss_alpha 1e-4 --edge_loss_beta 5e-7 --edge_change_epoch 1 --label_correction_epoch 30 --orig_edge_change_ratio 0 \
--candidate_edge_change_ratio 0 --lc_lr 0.05 --llm_confidence 70 --label_correction_ablation 0 --selection_ablation 0 --max_edge 70 &
# dblp GCNII
CUDA_VISIBLE_DEVICES=2 python src/main.py --dataset dblp --model_name GCNII --data_format sbert --main_seed_num 3 --split active --output_intermediate 0 \
--no_val 1 --lr 0.05 --strategy subspace --debug 1 --total_budget 240 --initial_budget 0.3 --filter_strategy consistency --loss_type ce \
--second_filter conf+entropy --epochs 20 --debug_gt_label 0 --early_stop_start 150 --filter_all_wrong_labels 0 --oracle 1 --ratio 0.2 --alpha 0.33 \
--beta 0.33 --positive_sample_size 80 --negative_sample 60 --stage_num 5 --run_code ours --subspace_alpha 1 --kmeans_dim 1 --svd_dim 128 \
--device cuda:0 --post_selection 2 --n_p 3 --edge_loss_alpha 1e-4 --edge_loss_beta 5e-7 --edge_change_epoch 1 --label_correction_epoch 30 \
--orig_edge_change_ratio 0.01 --candidate_edge_change_ratio 0.01 --lc_lr 0.05 --llm_confidence 0 --label_correction_ablation 0 \
--selection_ablation 0 --max_edge 70 &
