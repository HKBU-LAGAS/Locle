import torch
import os
import numpy as np
from helper.utils import load_yaml, find_selected_node_attribute, paint_homo, find_similar_node_attribute
import json
from llm import pack_prompt, Experiment, calculate_cost_from_a_list_of_texts
from openail.utils import pair_wise_prediction, cluster_zero_shot_prompt, cluster_few_shot_prompt, cluster_similarity_prompt
from openail.utils import my_efficient_openai_text_api_label,my_efficient_openai_text_api,efficient_openai_text_api, set_endpoints, openai_text_api, openai_text_api_with_top_p, load_partial_openai_result, save_partial_openai_result, retrieve_dict, compute_ece, plot_calibration_curve, openai_text_api_with_backoff, num_tokens_from_string
import time
from helper.Rewire_GNN import Rewire_GNN
import copy
from helper.utils import measure_confidence, harmonicity, entrophy_confidence, rank_and_correct_v2
def one_dim(args, data, ensemble_ca_matrix, device, ensemble_pred, stage, ensemble_indices, ensemble_res,temperature=0.7, n=3, seed=42, query_num = 0):
    pred_right_num = 0
    pair_right_num = 0
    query_num = 0
    if args.dataset in ['pubmed', 'citeseer2', 'wikics', 'dblp']:
        raw_texts = data.raw_texts
    else:
        raw_texts = data.raw_text
    indices = torch.arange(len(ensemble_pred))
    filtered_pred = ensemble_pred[~data.train_mask]
    filtered_indices = indices[~data.train_mask.detach().cpu()]
    most_confident_indices, most_inconfident_indices = measure_confidence(ensemble_res, data.edge_index,confidence_num = int(args.selection_ratio*len(ensemble_pred)),inconfidence_num=int(args.selection_ratio*len(ensemble_pred)),\
        max_confidence_num= args.positive_sample_size,max_inconfidence_num= args.negative_sample_size, filtered_indices = filtered_indices)
    if args.selection_ablation==1:
        most_confident_nodes, most_confident_indices = torch.topk(filtered_pred.detach().cpu(), args.positive_sample_size)
    most_confident_indices = filtered_indices[most_confident_indices]
    if args.post_selection == 2 or args.selection_ablation==1:
        most_inconfident_nodes, most_inconfident_indices = torch.topk(filtered_pred.detach().cpu(), args.negative_sample_size, largest=False)
    most_inconfident_indices = filtered_indices[most_inconfident_indices]
    
    original_prediction_accuracy = torch.sum(ensemble_indices[most_inconfident_indices] == data.backup_y[most_inconfident_indices]) / len(most_inconfident_indices)
    negative_sample_num = args.negative_sample_size
    cur_confi_homo_list = []
    cur_inconfi_homo_list = []

    from openail.config import configs
    low_confi_label_num = 0
    low_confi_label_right_num = 0
    high_confi_label_num = 0
    high_confi_label_right_num = 0
    params_dict = load_yaml(args.yaml_path)
    key = params_dict['OPENAI_KEY']
    data_path = params_dict['DATA_PATH']
    exp = Experiment(data, key, data_path)
    exps = ['pair']
    print(f"Before add pseudo label, train node number: {data.train_mask.sum().item()}")
    dataname = args.dataset
    dataset_prompt = configs[dataname]['dataset-task']
    object_cat = configs[dataname]['zero-shot']['object-cat']
    question = configs[dataname]['zero-shot']['question']
    answer_format = configs[dataname]['zero-shot']['answer-format']
    examples = configs[dataname]['few-shot']['examples']
    initial_answers = configs[dataname]['initial answer']
    example_text = examples[0][0]
    example_labels = examples[0][1]
    few_shot_topk = configs[dataname]['few-shot-2']['examples']
    fst_example = few_shot_topk[0][0]
    fst_result = few_shot_topk[0][1]
    idxs = torch.arange(data.x.shape[0])
    select_flags = [False]*len(data.label_names)
    flag_texts = [""]*len(data.label_names)
    if stage > 0:
        for idx in most_confident_indices:
            data.y[idx] = ensemble_indices[idx]
            if select_flags[ensemble_indices[idx]] == False:
                select_flags[ensemble_indices[idx]] = True
                flag_texts[ensemble_indices[idx]] = raw_texts[ensemble_indices[idx]]
            data.train_mask[idx] = True   
            data.test_mask[idx] = False
            if data.backup_y[idx] == ensemble_indices[idx]:
                pred_right_num += 1
    for idx, flag in enumerate(select_flags):
        if flag == False:
            select_flags[idx] = True
            flag_texts[idx] = initial_answers[str(idx)][0]

    llm_pred_right_num = 0
    negetive_train_indices = []
    
    save_path = os.path.join("data/annotations/{}_temperature_{}_n_{}_output_seed_{}_save_zero.json".format(args.dataset,temperature, n, seed))
    # print(save_path)
    
    from filelock import FileLock

    lock = FileLock(f"{save_path}.lock")


    if os.path.exists(save_path):
        with lock:
            with open(save_path, "r") as f:
                llm_pred = json.load(f)
    # test original cache
    else :
        llm_pred = {}

    llm_confidence = []
    for node in most_inconfident_indices: 
        node = node.item()
        if str(node) in llm_pred:

            node = str(node)
            pred = llm_pred[node][0]
            llm_confidence.append(llm_pred[node][1])
            if llm_pred[node][1] > args.llm_confidence_score:
                high_confi_label_num += 1
                if pred == data.backup_y[int(node)]:
                    high_confi_label_right_num += 1
            elif llm_pred[node][1] <= args.llm_confidence_score:
                low_confi_label_num += 1
                if pred == data.backup_y[int(node)]:
                    low_confi_label_right_num += 1
                # continue
            node = int(node)
            negetive_train_indices.append(node)

            data.train_mask[node] = True
            data.test_mask[node] = False
            data.y[node] = pred
            if data.backup_y[node] == pred:
                llm_pred_right_num += 1
        else:

            input_text = []

     
            question = configs[dataname]['similar']['question']
            answer_format = configs[dataname]['zero-shot']['answer-format']
            reasoning = configs[dataname]['similar']['reasoning']
            if args.dataset in ['pubmed']:
                data.label_names[0] = data.label_names[0].replace(",", "")
            prompt = cluster_few_shot_prompt(dataset_prompt, raw_texts[node], flag_texts, data.label_names, need_tasks=True, object_cat="", question=question, answer_format=answer_format, reasoning=reasoning, dataname=dataname)
            input_text.append(prompt)
            input_filename = "data/cache/pair_wise_prompt_async_input_{}_temperature_{}_n_{}_input_seed_{}.json".format(args.dataset,temperature, n, seed)
            output_filename = "data/cache/pair_wise_prompt_async_input_{}_temperature_{}_n_{}_output_seed_{}.json".format(args.dataset,temperature, n, seed)
            openai_result, openai_confidence = my_efficient_openai_text_api(input_text, input_filename, output_filename, sp=60, ss=0, api_key="change to your key", request_url="change to your endpoint",
                                                         temperature=0.2, n = 1, rewrite = True, label_name=data.label_names, label_num=len(data.label_names))
            
            if query_num % 10 == 0:
                with lock:
                    with open(save_path, 'w') as f:
                        json.dump(llm_pred, f)
            llm_confidence.append(openai_confidence)
            query_num += 1
           
            if openai_result == -1:
                negative_sample_num -= 1
                continue
            else:
                negetive_train_indices.append(node)
                print(f"Node {node} is predicted as {openai_result}, the true label is {data.backup_y[node]}")
                data.train_mask[node] = True
                data.test_mask[node] = False
                data.y[node] = openai_result
                if data.backup_y[node] == openai_result:
                    llm_pred_right_num += 1
                llm_pred[node] = [int(openai_result), openai_confidence]
        
    if query_num > 0:
        with lock:
            with open(save_path, 'w') as f:
                json.dump(llm_pred, f)
    data.y = data.y.to(args.device)
    print(f"After add pseudo label, train node number: {data.train_mask.sum().item()}")
   
    '''
    add two parts: struture update and label correction
    '''
    if args.label_correction_ablation == 0:
        data.backup_edge_index = copy.deepcopy(data.edge_index)
        update_model = Rewire_GNN(args, args.device, most_confident_indices, most_inconfident_indices)
        # update_idx, update_pred = update_model.fit(data, ensemble_res)   
        if args.given_feature == 'orig':
            update_idx, update_pred, test_res, test_res1 = update_model.fit(data, data.x)
        elif args.given_feature == 'ensemble':
            update_idx, update_pred, test_res, test_res1 = update_model.fit(data, ensemble_res)
        if args.update_structure:
            data.edge_index = update_idx
     
        rank_and_correct_v2(llm_confidence, update_pred, most_inconfident_indices, data, args.llm_confidence_score)
     
        print(f"orig edge index: {data.edge_index.shape}, update edge index: {update_idx.shape}")
        

   
    post_process_right_num = 0
    
    for node in most_inconfident_indices:
       if data.backup_y[node] == data.y[node]:
           post_process_right_num += 1
    llm_inconfident_acc = 0
    post_inconfident_acc = 0
    if negative_sample_num > 0:
        llm_inconfident_acc = llm_pred_right_num/negative_sample_num*100

    post_inconfident_acc = post_process_right_num/negative_sample_num*100
    if args.debug_gt_label:
        data.y[most_inconfident_indices] = data.backup_y[most_inconfident_indices]
    if args.label_correction_ablation == 1:
        test_res = []
    else:
        test_res = [test_res , test_res1]
    return query_num, llm_inconfident_acc, post_inconfident_acc, original_prediction_accuracy, test_res
