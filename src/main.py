from helper.utils import load_yaml, pkl_and_write
from helper.args import get_command_line_args, replace_args_with_dict_values
from helper.noisy import NoiseAda
import torch
from helper.train_utils import train, test, get_optimizer, seed_everything, s_train, batch_train, batch_test
from models.nn import get_model
import numpy as np
import time
import logging
# print("OK")
import torch.nn.functional as F
from copy import deepcopy
from helper.data import get_dataset, Data, get_M
import os.path as osp
import optuna
from helper.hyper_search import hyper_search
import sys
from tqdm import tqdm
import copy
import os
from llm import pack_prompt, Experiment, calculate_cost_from_a_list_of_texts, my_llm_answer
from openail.utils import pair_wise_prediction
from helper.my_utils import one_dim
import json



def train_pipeline(seeds, args, epoch, data, need_train, need_save_logits, reliability_list, stage=False):
    device = args.device
    
    test_result_acc = []
    early_stop_accum = 0
    val_result_acc = []
    out_res = []
    best_val = 0
    debug_accs = []
    train_accs = []
    num_of_classes = data.y.max().item() + 1
    if args.model_name == 'S_model':
        noise_ada = NoiseAda(num_of_classes).to(device)
    else:
        noise_ada = None
    for i, seed in enumerate(seeds):
        cur_idx = i
        if len(reliability_list) > 0:
            reliability = reliability_list[0].to(device)
        seed_everything(seed)
        model = get_model(args).to(device)
        optimizer, scheduler = get_optimizer(args, model)
        if args.loss_type == 'ce':
            loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)            
        else:
            loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction='none')            
        if args.normalize:
            data.x = F.normalize(data.x, dim = -1)
        data = data.to(device)
        data.train_mask = data.train_masks[i]
        data.val_mask = data.val_masks[i]
        data.test_mask = data.test_masks[i]
        debug_acc = []
        this_train_acc = []
        if 'ft' in args.data_format and 'no_ft' not in args.data_format:
            data.x = data.xs[i]
            data.train_mask = data.train_masks[i]
            data.val_mask = data.val_masks[i]
            data.test_mask = data.test_masks[i]
        if 'pl' in args.split or 'active' in args.split:
            data.train_mask = data.train_masks[i]
            data.val_mask = data.val_masks[i]
            data.test_mask = data.test_masks[i]
            data.backup_y = data.y.clone()
            if not args.debug_gt_label:
                data.y = data.ys[i]
            else:
                print("Using ground truth label")
        if not stage:
            for i in tqdm(range(epoch)):
                train_mask = data.train_mask
                val_mask = data.val_mask
                if need_train:
                    if 'rim' in args.strategy or 'iterative' in args.strategy or args.split == 'active_train':
                        train_loss, val_loss, val_acc, train_acc = train(model, data, optimizer, loss_fn, train_mask, val_mask, args.no_val, reliability)
                    else:
                        if args.model_name == 'S_model':
                            train_loss, val_loss, val_acc, train_acc = s_train(model, data, optimizer, loss_fn, train_mask, val_mask, args.no_val, noise_ada)
                        else:
                            train_loss, val_loss, val_acc, train_acc = train(model, data, optimizer, loss_fn, train_mask, val_mask, args.no_val)
                    if scheduler:
                        scheduler.step()
                    if args.output_intermediate and not args.no_val:
                        print(f"Epoch {i}: Train loss: {train_loss}, Val loss: {val_loss}, Val acc: {val_acc[0]}")
                    if args.debug:
                        if args.filter_strategy == 'none':
                            test_acc, res = test(model, data, 1, data.test_mask)
                        else:
                            test_acc, res = test(model, data, 1, data.test_mask, data.backup_y)
                        # print(f"Epoch {i}: Test acc: {test_acc}")
                        debug_acc.append(test_acc)
                        this_train_acc.append(train_acc)
                    if not args.no_val:
                        if val_acc > best_val:
                            best_val = val_acc
                            best_model = deepcopy(model)
                            early_stop_accum = 0
                        else:
                            if i >= args.early_stop_start:
                                early_stop_accum += 1
                            if early_stop_accum > args.early_stopping and i >= args.early_stop_start:
                                print(f"Early stopping at epoch {i}")
                                break
                else:
                    best_model = model
            if 'pl' in args.split or 'active' in args.split:
                data.y = data.backup_y
            if args.no_val or best_model == None:
                best_model = model
            test_acc, res = test(best_model, data, args.return_embeds, data.test_mask)
            test_result_acc.append(test_acc)
            val_result_acc.append(best_val)
            out_res.append(res)
            best_val = 0
            best_model = None
            if args.debug:
                debug_accs.append(debug_acc)
                train_accs.append(this_train_acc)
            if need_save_logits:
                torch.save(out_res, f'../output/logits/{args.dataset}_{args.split}_{args.model_name}_{seed}_logits.pt')
        else:
            # import ipdb; ipdb.set_trace()
         #   each time store the original train_mask and label, after the clustering restore the original train_mask and label
            for i in tqdm(range(epoch)):
                # ipdb.set_trace()
                train_mask = data.train_mask
                val_mask = data.val_mask
                if need_train:
                    if 'rim' in args.strategy or 'iterative' in args.strategy or args.split == 'active_train':
                        train_loss, val_loss, val_acc, train_acc = train(model, data, optimizer, loss_fn, train_mask, val_mask, args.no_val, reliability)
                    else:
                        if args.model_name == 'S_model':
                            train_loss, val_loss, val_acc, train_acc = s_train(model, data, optimizer, loss_fn, train_mask, val_mask, args.no_val, noise_ada)
                        else:
                            train_loss, val_loss, val_acc, train_acc = train(model, data, optimizer, loss_fn, train_mask, val_mask, args.no_val)
                    if scheduler:
                        scheduler.step()
                    if args.output_intermediate and not args.no_val:
                        print(f"Epoch {i}: Train loss: {train_loss}, Val loss: {val_loss}, Val acc: {val_acc[0]}")
                    if args.debug:
                        if args.filter_strategy == 'none':
                            test_acc, res = test(model, data, 1, data.test_mask)
                        else:
                            test_acc, res = test(model, data, 1, data.test_mask, data.backup_y)
                        # print(f"Epoch {i}: Test acc: {test_acc}")
                        debug_acc.append(test_acc)
                        this_train_acc.append(train_acc)
                    if not args.no_val:
                        if val_acc > best_val:
                            best_val = val_acc
                            best_model = deepcopy(model)
                            early_stop_accum = 0
                        else:
                            if i >= args.early_stop_start:
                                early_stop_accum += 1
                            if early_stop_accum > args.early_stopping and i >= args.early_stop_start:
                                print(f"Early stopping at epoch {i}")
                                break
                    # if args.use_wandb:
                    #     wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc[0], 'train_acc': train_acc[0]})
                else:
                    best_model = model
            if 'pl' in args.split or 'active' in args.split:
                data.y = data.backup_y
            if args.no_val or best_model == None:
                best_model = model
            test_acc, res = test(best_model, data, args.return_embeds, data.test_mask)
            # test_result_acc.append(test_acc)
            # val_result_acc.append(best_val)
            out_res.append(res)
            best_val = 0
            best_model = None
            
            if args.debug:
                debug_accs.append(debug_acc)
                train_accs.append(this_train_acc)
            if need_save_logits:
                torch.save(out_res, f'../output/logits/{args.dataset}_{args.split}_{args.model_name}_{seed}_logits.pt')
           
            for i in tqdm(range(epoch)):
                # ipdb.set_trace()
                train_mask = data.train_mask
                val_mask = data.val_mask
                if need_train:
                    if 'rim' in args.strategy or 'iterative' in args.strategy or args.split == 'active_train':
                        train_loss, val_loss, val_acc, train_acc = train(model, data, optimizer, loss_fn, train_mask, val_mask, args.no_val, reliability)
                    else:
                        if args.model_name == 'S_model':
                            train_loss, val_loss, val_acc, train_acc = s_train(model, data, optimizer, loss_fn, train_mask, val_mask, args.no_val, noise_ada)
                        else:
                            train_loss, val_loss, val_acc, train_acc = train(model, data, optimizer, loss_fn, train_mask, val_mask, args.no_val)
                    if scheduler:
                        scheduler.step()
                    if args.output_intermediate and not args.no_val:
                        print(f"Epoch {i}: Train loss: {train_loss}, Val loss: {val_loss}, Val acc: {val_acc[0]}")
                    if args.debug:
                        if args.filter_strategy == 'none':
                            test_acc, res = test(model, data, 1, data.test_mask)
                        else:
                            test_acc, res = test(model, data, 1, data.test_mask, data.backup_y)
                        # print(f"Epoch {i}: Test acc: {test_acc}")
                        debug_acc.append(test_acc)
                        this_train_acc.append(train_acc)
                    if not args.no_val:
                        if val_acc > best_val:
                            best_val = val_acc
                            best_model = deepcopy(model)
                            early_stop_accum = 0
                        else:
                            if i >= args.early_stop_start:
                                early_stop_accum += 1
                            if early_stop_accum > args.early_stopping and i >= args.early_stop_start:
                                print(f"Early stopping at epoch {i}")
                                break
                else:
                    best_model = model
            if 'pl' in args.split or 'active' in args.split:
                data.y = data.backup_y
            if args.no_val or best_model == None:
                best_model = model
            test_acc, res = test(best_model, data, args.return_embeds, data.test_mask)
            test_result_acc.append(test_acc)
            val_result_acc.append(best_val)
            out_res.append(res)
            best_val = 0
            best_model = None
            if args.debug:
                debug_accs.append(debug_acc)
                train_accs.append(this_train_acc)
            if need_save_logits:
                torch.save(out_res, f'../output/logits/{args.dataset}_{args.split}_{args.model_name}_{seed}_logits.pt')
            # restore the original data
    if not args.debug:
        return test_result_acc, val_result_acc, out_res
    else:
        return test_result_acc, val_result_acc, out_res, debug_accs, train_accs


def my_training_pipeline(seeds, args, epoch, data, need_train, need_save_logits, reliability_list, stages=5, update_step = 1):
    '''iterative training, each update stage: 
    1.do training; 
    2.save best model; do the ensemble clustrering
    3.find prediction node parttern; calculate the prediction confidence and find reliable/unreliable nodes. put some reliable nodes into the confidence budget
    4.use LLM knowledge to correct the prediction of unreliable nodes
    5.use the label in reliable nodes to update the adjacency matrix
    6.update the data and do next stage training
'''
    # delete the duplicate edges ####
    from torch_geometric.utils import to_undirected
    tmp = data.edge_index.t()
    tmp = torch.unique(tmp, dim=0)
    data.edge_index = tmp.t()
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.y.shape[0])
    #################################
    device = args.device
    query_num = 0
    # data.edge_weight = torch.ones(data.edge_index.shape[1]).to(device)
    test_result_acc = []
    update_test_result_acc = []
    early_stop_accum = 0
    val_result_acc = []
    out_res = []
    best_val = 0
    debug_accs = []
    train_accs = []
    confi_homo_list = []
    inconfi_homo_list = []
    llm_inconfident_acc = []
    original_inconfident_acc = []
    correction_model_inconfident_acc = []
    num_of_classes = data.y.max().item() + 1
    if args.model_name == 'S_model':
        noise_ada = NoiseAda(num_of_classes).to(device)
    else:
        noise_ada = None
    for i, seed in enumerate(seeds):
        selection_result_path = os.path.join(data_path, f'subspace_selection/{args.dataset}_{int(args.total_budget*args.initial_budget)}_{args.kmeans_dim}_{args.svd_dim}_{args.ratio}_{args.epochs}_{seed}.json')
        if not os.path.exists(selection_result_path):
            with open(selection_result_path, 'w') as f:
                train_idx = torch.where(data.train_masks[seed]==True)[0].detach().cpu().tolist()
                json.dump(train_idx, f)
        llm_inconfident_acc.append([])
        correction_model_inconfident_acc.append([])
        original_inconfident_acc.append([])
        models = []
        cur_idx = i
        if len(reliability_list) > 0:
            reliability = reliability_list[0].to(device)
        seed_everything(seed)
        
        if args.loss_type == 'ce':
            loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)            
        else:
            loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction='none')            
        if args.normalize:
            data.x = F.normalize(data.x, dim = -1)
        data = data.to(device)
        data.train_mask = data.train_masks[i]
        data.val_mask = data.val_masks[i]
        data.test_mask = data.test_masks[i]

        debug_acc = []
        this_train_acc = []
        if 'ft' in args.data_format and 'no_ft' not in args.data_format:
            data.x = data.xs[i]
            data.train_mask = data.train_masks[i]
            data.val_mask = data.val_masks[i]
            data.test_mask = data.test_masks[i]
        if 'pl' in args.split or 'active' in args.split:
            data.train_mask = data.train_masks[i]
            data.val_mask = data.val_masks[i]
            data.test_mask = data.test_masks[i]
            data.backup_y = data.y.clone()
            if not args.debug_gt_label:
                data.y = data.ys[i]
            else:
                print("Using ground truth label")
        stage_test_acc = []
        update_stage_test_acc = []
        for stage in range(stages):
            data.test_mask = ~data.train_mask
            model = get_model(args).to(device)
            optimizer, scheduler = get_optimizer(args, model)
            for i in tqdm(range(epoch)):
                train_mask = data.train_mask
                val_mask = data.val_mask
                if need_train:
                    if 'rim' in args.strategy or 'iterative' in args.strategy or args.split == 'active_train':
                        train_loss, val_loss, val_acc, train_acc = train(model, data, optimizer, loss_fn, train_mask, val_mask, args.no_val, reliability)
                    else:
                        if args.model_name == 'S_model':
                            train_loss, val_loss, val_acc, train_acc = s_train(model, data, optimizer, loss_fn, train_mask, val_mask, args.no_val, noise_ada)
                        else:
                            train_loss, val_loss, val_acc, train_acc = train(model, data, optimizer, loss_fn, train_mask, val_mask, args.no_val)
                    if scheduler:
                        scheduler.step()
                    if args.output_intermediate and not args.no_val:
                        print(f"Epoch {i}: Train loss: {train_loss}, Val loss: {val_loss}, Val acc: {val_acc[0]}")
                    if args.debug:
                        if args.filter_strategy == 'none':
                            test_acc, res = test(model, data, 1, data.test_mask)
                        else:
                            test_acc, res = test(model, data, 1, data.test_mask, data.backup_y)
                        # print(f"Epoch {i}: Test acc: {test_acc}")
                        debug_acc.append(test_acc)
                        this_train_acc.append(train_acc)
                    if not args.no_val:
                        if val_acc > best_val:
                            best_val = val_acc
                            best_model = deepcopy(model)
                            early_stop_accum = 0
                        else:
                            if i >= args.early_stop_start:
                                early_stop_accum += 1
                            if early_stop_accum > args.early_stopping and i >= args.early_stop_start:
                                print(f"Early stopping at epoch {i}")
                                break

                else:
                    best_model = model
            if args.no_val or best_model == None:
                best_model = model
            # store current data.y
                data.store_y = copy.deepcopy(data.y)
            if 'pl' in args.split or 'active' in args.split:
                data.y = data.backup_y
            test_acc, res = test(best_model, data, args.return_embeds, data.test_mask)
            stage_test_acc.append(test_acc)
            val_result_acc.append(best_val)
            # softmax res
            # res = F.softmax(res, dim = -1)
            out_res.append(res)
            best_val = 0
            if need_save_logits:
                torch.save(out_res, f'../output/logits/{args.dataset}_{args.split}_{args.model_name}_{seed}_logits.pt')
            data.y = data.store_y
            if (stage+1) % update_step == 0:
                pred_right_num = 0
                pair_right_num = 0
                ensemble_res = torch.zeros_like(out_res[0])
                ensemble_ca_matrix = torch.zeros_like(out_res[0]@out_res[0].t())
                alpha = args.ensemble_alpha
                for i, res in enumerate(out_res):
                    w = (1-alpha)*(alpha**(stages-i))/(1-(alpha**stages))
                    ensemble_res += w*res
                    ensemble_ca_matrix += ensemble_res@ensemble_res.t()
                ensemble_res = torch.nn.functional.normalize(ensemble_res, dim = -1, p = 1)
                ensemble_ca_matrix = torch.nn.functional.normalize(ensemble_ca_matrix, dim = -1, p = 1)
                flat_res = ensemble_ca_matrix.flatten()
                ensemble_pred, ensemble_indices = torch.max(ensemble_res, dim = -1)
                indices = torch.arange(len(ensemble_pred))
                
                filtered_pred = ensemble_pred[~data.train_mask]
                filtered_indices = indices[~data.train_mask.detach().cpu()]
                
                most_confident_nodes, most_confident_indices,  = torch.topk(filtered_pred.detach().cpu(), args.positive_sample_size)
                most_confident_indices = filtered_indices[most_confident_indices]
                # use llm to predict the labels of the most inconfident labels
                
                print(f"stage num:{stage}")
                if stage < stages:
                    if args.llm_mode == 'one_dim':
                        query, llm_inconfi, post_inconfi, orig_inconfi, update_res = one_dim(args, data, ensemble_ca_matrix, device, ensemble_pred, stage, ensemble_indices,ensemble_res, query_num=query_num)
                        query_num += query
                        llm_inconfident_acc[-1].append(llm_inconfi)
                        correction_model_inconfident_acc[-1].append(post_inconfi)
                        original_inconfident_acc[-1].append(orig_inconfi)
                        update_stage_test_acc.append(update_res)
                
                


        if args.debug:
            debug_accs.append(debug_acc)
            train_accs.append(this_train_acc)
        if 'pl' in args.split or 'active' in args.split:
            data.y = data.backup_y
                
        test_result_acc.append(stage_test_acc)   
        update_test_result_acc.append(update_stage_test_acc)
    if not args.debug:
        return test_result_acc
    else:
        return test_result_acc, debug_accs, train_accs, update_test_result_acc
def main(data_path, args = None, custom_args = None, save_best = False):
    seeds = [i for i in range(args.main_seed_num)]
    device = args.device
    if custom_args != None:
        args = replace_args_with_dict_values(args, custom_args)
    params_dict = load_yaml(args.yaml_path)

    total_budget = args.total_budget
    args.total_budget = int(args.total_budget*args.initial_budget)
    if args.run_code == 'ours':
        args.negative_sample_size = int((1-args.initial_budget)*total_budget / (args.stage_num - 1))
    logit_path = params_dict['LOGIT_PATH']
    reliability_list = []
    data = get_dataset(seeds, args.dataset, args.split, args.data_format, data_path, logit_path, args.pl_noise, args.no_val, args.budget, args.strategy, args.num_centers, args.compensation, args.save_data, args.filter_strategy, args.max_part, args.oracle, reliability_list, args.total_budget, args.second_filter, True, False, args.filter_all_wrong_labels, args.alpha, args.beta, args.gamma, args.ratio, all_args=args).to(device)

    print(torch.where(data.train_masks[0])[0])
    epoch = args.epochs
    vars(args)['input_dim'] = data.x.shape[1]
    vars(args)['num_node'] = data.x.shape[0]
    vars(args)['num_classes'] = data.y.max().item() + 1
    if args.model_name == 'LP':
        need_train = False
    else:
        need_train = True

    if not args.batchify and args.ensemble_string == "":
        data.x = data.x.to(torch.float32)

        if not args.debug:
            test_result_acc, _, _ = train_pipeline(seeds, args, epoch, data, need_train, args.save_logits, reliability_list, args.stage)
        else:
            if args.run_code == 'ours':
                test_result_acc, debug_accs, train_accs, update_res = my_training_pipeline(seeds, args, epoch, data, need_train, args.save_logits, reliability_list, args.stage_num, args.update_stage)
            elif args.run_code == 'orig':
                test_result_acc, _, _, debug_accs, train_accs = train_pipeline(seeds, args, epoch, data, need_train, args.save_logits, reliability_list, args.stage)
        acc_list = []
        nmi_list = []
        ari_list = []
        f1_list = []
        update_acc_list = []
        update_nmi_list = []
        update_ari_list = []
        update_f1_list = []
        best_update_acc_list = []
        best_update_nmi_list = []
        best_update_ari_list = []
        best_update_f1_list = []
        best_update_res = []
        final_update_res = []
        if args.run_code == 'ours':
            for item in update_res:
                best_temp = []
                final_temp = []
                for item1 in item:
                    if len(item1) == 0:
                        best_temp.append({})
                        final_temp.append({})
                    else:
                        best_temp.append(item1[0])
                        final_temp.append(item1[1])
                best_update_res.append(best_temp)
                final_update_res.append(final_temp)
            test_result_acc = list(zip(*test_result_acc))
            update_res = list(zip(*update_res))
            best_update_res = list(zip(*best_update_res))
            final_update_res = list(zip(*final_update_res))
            update_final_res = []
            final_update_final_res = []
            final_result = []
            acc_avg = 0
            update_acc_avg = 0
            final_update_acc_avg = 0
            best_step = 0
            update_best_step = 0
            final_update_best_step = 0
            avg_step_acc = []
            for idx, item in enumerate(test_result_acc):
                acc_values = [d['acc'] for d in item if 'acc' in d]
                ari_values = [d['ari'] for d in item if 'ari' in d]
                nmi_values = [d['nmi'] for d in item if 'nmi' in d]
                f1_values = [d['f1'] for d in item if 'f1' in d]
                cur_acc_avg = sum(acc_values) / len(item)
                
                print(f"At stage {idx+1}, result is: Acc: {cur_acc_avg:.2f}, NMI: {sum(nmi_values) / len(item):.2f}, ARI: {sum(ari_values) / len(item):.2f}, F1: {sum(f1_values) / len(item):.2f}")
                if cur_acc_avg > acc_avg:
                    acc_avg = cur_acc_avg
                    best_step = idx+1
                    final_result = item
            for idx, item in enumerate(best_update_res):
                acc_values = [d['acc'] for d in item if 'acc' in d]
                cur_acc_avg = sum(acc_values) / len(item)
                if cur_acc_avg > update_acc_avg:
                    update_acc_avg = cur_acc_avg
                    update_best_step = idx+1
                    update_final_res = item
            for idx, item in enumerate(final_update_res):
                acc_values = [d['acc'] for d in item if 'acc' in d]
                cur_acc_avg = sum(acc_values) / len(item)
                if cur_acc_avg > final_update_acc_avg:
                    final_update_acc_avg = cur_acc_avg
                    final_update_best_step = idx+1
                    final_update_final_res = item
        elif args.run_code == 'orig':
            final_result = test_result_acc
        for acc in final_result:
            acc_list.append(acc['acc'])
            if acc['nmi'] != None:
                nmi_list.append(acc['nmi'])
                ari_list.append(acc['ari'])
                f1_list.append(acc['f1'])
        if args.run_code == 'ours':
            for acc in final_update_final_res:
                update_acc_list.append(acc['acc'])
                if acc['nmi'] != None:
                    update_nmi_list.append(acc['nmi'])
                    update_ari_list.append(acc['ari'])
                    update_f1_list.append(acc['f1'])
            for acc in update_final_res:
                best_update_acc_list.append(acc['acc'])
                if acc['nmi'] != None:
                    best_update_nmi_list.append(acc['nmi'])
                    best_update_ari_list.append(acc['ari'])
                    best_update_f1_list.append(acc['f1'])
        mean_test_acc = np.mean(acc_list)
        std_test_acc = np.std(acc_list)
        mean_nmi = np.mean(nmi_list)
        std_nmi = np.std(nmi_list)
        mean_ari = np.mean(ari_list)
        std_ari = np.std(ari_list)
        mean_f1 = np.mean(f1_list)
        std_f1 = np.std(f1_list)
        # update mean and std

        
        if args.debug:
            debug_acc_list = []
            debug_nmi_list = []
            debug_ari_list = []
            debug_f1_list = []
            for debug_acc in debug_accs:
                seed_acc_list = []
                seed_nmi_list = []
                seed_ari_list = []
                seed_f1_list = []
                for item in debug_acc:
                    seed_acc_list.append(item['acc'])
                    if item['nmi'] != None:
                        seed_nmi_list.append(item['nmi'])
                        seed_ari_list.append(item['ari'])
                        seed_f1_list.append(item['f1'])
                debug_acc_list.append(np.max(seed_acc_list))
                debug_nmi_list.append(np.max(seed_nmi_list))
                debug_ari_list.append(np.max(seed_ari_list))
                debug_f1_list.append(np.max(seed_f1_list))
            best_possible_test_acc = [np.max(res) for res in debug_acc_list]
            best_possible_test_nmi = [np.max(res) for res in debug_nmi_list]
            best_possible_test_ari = [np.max(res) for res in debug_ari_list]
            best_possible_test_f1 = [np.max(res) for res in debug_f1_list]
        res_train_accs = [x[-1]['acc'] for x in train_accs]
        res_train_nmi = [x[-1]['nmi'] for x in train_accs]
        res_train_ari = [x[-1]['ari'] for x in train_accs]
        res_train_f1 = [x[-1]['f1'] for x in train_accs]
        # if args.use_wandb:
        #     wandb.log({"Train Accuracy": np.mean(res_train_accs), "Train Accuracy std": np.std(res_train_accs), "Test Accuracy": mean_test_acc, "Test Accuracy std": std_test_acc, "Test NMI": mean_nmi, "Test NMI std": std_nmi, "Test ARI": mean_ari, "Test ARI std": std_ari, "Test F1": mean_f1, "Test F1 std": std_f1})

        command = ' '.join(sys.argv)
        print(f"Command: {command}")

        print(f"Train Accuracy: {np.mean(res_train_accs):.2f} ± {np.std(res_train_accs):.2f}")
        print(f"{args.dataset} Test Accuracy: {mean_test_acc:.2f} ± {std_test_acc:.2f}")
        print(f"{args.dataset} Test NMI: {mean_nmi:.2f} ± {std_nmi:.2f}")
        print(f"{args.dataset} Test ARI: {mean_ari:.2f} ± {std_ari:.2f}")
        print(f"{args.dataset} Test F1: {mean_f1:.2f} ± {std_f1:.2f}")
        if args.debug:
            print(f"Best possible accuracy: {np.mean(best_possible_test_acc) :.2f} ± {np.std(best_possible_test_acc) :.2f}")
            print(f"Best possible NMI: {np.mean(best_possible_test_nmi):.2f} ± {np.std(best_possible_test_nmi):.2f}")
            print(f"Best possible ARI: {np.mean(best_possible_test_ari):.2f} ± {np.std(best_possible_test_ari):.2f}")
            print(f"Best possible F1: {np.mean(best_possible_test_f1):.2f} ± {np.std(best_possible_test_f1):.2f}")
            # if args.use_wandb:
            #     wandb.log({"Best possible Accuracy": np.mean(best_possible_test_acc), "Best possible Accuracy std": np.std(best_possible_test_acc), "Best possible NMI": np.mean(best_possible_test_nmi), "Best possible NMI std": np.std(best_possible_test_nmi), "Best possible ARI": np.mean(best_possible_test_ari), "Best possible ARI std": np.std(best_possible_test_ari), "Best possible F1": np.mean(best_possible_test_f1), "Best possible F1 std": np.std(best_possible_test_f1)})
        print("Test acc: {}".format(test_result_acc))
    elif args.ensemble_string != "":
        pass
    else:
        pass
    if save_best:
        pkl_and_write(args, osp.join("./bestargs", f"{args.model_name}_{args.dataset}_{args.data_format}.pkl"))
    if args.debug:
        if args.debug_gt_label:
            pkl_and_write(debug_accs, osp.join("./debug", f"{args.model_name}_{args.split}_{args.dataset}_{args.data_format}_{args.pl_noise}_{args.budget}_{args.total_budget}_{args.strategy}_{args.filter_strategy}_{args.loss_type}_gt.pkl"))
            pkl_and_write(train_accs, osp.join("./debug_train", f"{args.model_name}_{args.split}_{args.dataset}_{args.data_format}_{args.pl_noise}_{args.budget}_{args.total_budget}_{args.strategy}_{args.filter_strategy}_{args.loss_type}_train_accs_gt.pkl"))
        elif args.filter_all_wrong_labels:
            pkl_and_write(debug_accs, osp.join("./debug", f"{args.model_name}_{args.split}_{args.dataset}_{args.data_format}_{args.pl_noise}_{args.budget}_{args.total_budget}_{args.strategy}_{args.filter_strategy}_{args.loss_type}_filtered.pkl"))
            pkl_and_write(train_accs, osp.join("./debug_train", f"{args.model_name}_{args.split}_{args.dataset}_{args.data_format}_{args.pl_noise}_{args.budget}_{args.total_budget}_{args.strategy}_{args.filter_strategy}_{args.loss_type}_train_accs_filtered.pkl"))
        else:
            pkl_and_write(debug_accs, osp.join("./debug", f"{args.model_name}_{args.split}_{args.dataset}_{args.data_format}_{args.pl_noise}_{args.budget}_{args.total_budget}_{args.strategy}_{args.filter_strategy}_{args.loss_type}.pkl"))
            pkl_and_write(train_accs, osp.join("./debug_train", f"{args.model_name}_{args.split}_{args.dataset}_{args.data_format}_{args.pl_noise}_{args.budget}_{args.total_budget}_{args.strategy}_{args.filter_strategy}_{args.loss_type}_train_accs.pkl"))

    # if args.use_wandb:
    #     wandb.finish()
                
def max_trial_callback(study, trial, max_try):
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE or t.state == optuna.trial.TrialState.RUNNING])
    n_total_complete = len([t for t in study.trials])
    if n_complete >= max_try or n_total_complete >= 2 * max_try:
        study.stop()
        torch.cuda.empty_cache()


def sweep(data_path, args = None):
    sweep_seeds = [i for i in range(args.sweep_seed_num)]

    device = args.device
    vars(args)['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = f"{args.dataset}_{args.model_name}_{args.data_format}_{args.split}"
    study = optuna.create_study(study_name=study_name, storage=None, direction='maximize', load_if_exists=True)
    param_f = hyper_search
    sweep_round = args.sweep_round
    study.optimize(lambda trial: sweep_run(trial, args, sweep_seeds, param_f, device, data_path), catch=(RuntimeError,), n_trials=sweep_round, callbacks=[lambda study, trial: max_trial_callback(study, trial, sweep_round)], show_progress_bar=True, gc_after_trial=True)
    main(args=args, custom_args = study.best_trial.params, save_best = True)
    print(study.best_trial.params)



def sweep_run(trial, args, sweep_seeds, param_f, device, data_path):
    params = param_f(trial, args.data_format, args.model_name, args.dataset)    
    args = replace_args_with_dict_values(args, params)
    params_dict = load_yaml(args.yaml_path)
    logit_path = params_dict['LOGIT_PATH']
    reliability_list = []
    data = get_dataset(sweep_seeds, args.dataset, args.split, args.data_format, data_path, logit_path, args.pl_noise, args.no_val, args.budget, args.strategy, args.num_centers, args.compensation, args.save_data, args.filter_strategy, args.max_part, args.oracle, reliability_list, args.total_budget, args.second_filter, True, False, args.filter_all_wrong_labels, args.alpha, args.beta, args.gamma, args.ratio).to(device)
    
    epoch = args.epochs
    vars(args)['input_dim'] = data.x.shape[1]
    vars(args)['num_node'] = data.x.shape[0]
    vars(args)['num_classes'] = data.y.max().item() + 1
    if args.model_name == 'LP':
        need_train = False
    else:
        need_train = True
    if not args.batchify and args.ensemble_string == "":
        data.x = data.x.to(torch.float32)
        test_result_acc, _, _ = train_pipeline(sweep_seeds, args, epoch, data, need_train, args.save_logits, reliability_list)
    elif args.ensemble_string != "":
        pass
    else:
        pass
    mean_test_acc = np.mean(test_result_acc)
    std_test_acc = np.std(test_result_acc)
    print(f"{args.dataset} Test Accuracy: {mean_test_acc} ± {std_test_acc}")
    return mean_test_acc






    


    
if __name__ == '__main__':
    current_time = int(time.time())
    # #logging.basicConfig(filename='../../logs/{}.log'.format(current_time),
    #                 filemode='a',
    #                 format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    #                 datefmt='%H:%M:%S',
    #                 level=logging.INFO)

    print("Start")
    args = get_command_line_args()    
    params_dict = load_yaml(args.yaml_path)
    data_path = params_dict['DATA_PATH']
    if args.mode == "main":
        main(data_path, args = args)
    else:
        sweep(data_path, args = args)



