import argparse
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from dataset import MetaPathDataset
from model import PathInstanceNetwork
from tpc_ds import QuerySample
import dill as cpickle
import pickle
import os
import numpy as np
from tpc_ds import TPCDSDataset
import multi_utils as mu
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
import tqdm
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import copy
from datetime import datetime
import time
import pandas as pd
from collections import Counter
import random
random.seed(42)
torch.manual_seed(42)
import ast
import glob
import json

def batch_loader(samples, batch_size):
    for i in range(0, len(samples), batch_size):
        yield samples[i:i + batch_size]

def collect_related_tab_desc_bytab(query_tab_name,index_file,threshold,query_tab_cols,tab_df_infos):
    col_joinability_idx = pickle.load(open(index_file,'rb'))
    compatible_tables, _ = mu.find_compatible_tables([query_tab_name], col_joinability_idx, threshold)
    desc_related_tab_col = {}
    for item in compatible_tables[query_tab_name]:
        tab2, col1, col2 = item
        if col1 in query_tab_cols:
            if col1 not in desc_related_tab_col:
                desc_related_tab_col[col1] = {}
            if tab2 not in desc_related_tab_col[col1] and col2 in tab_df_infos[tab2].columns:
                desc_related_tab_col[col1][tab2] = col2

        elif col2 in query_tab_cols:
            if col2 not in desc_related_tab_col:
                desc_related_tab_col[col2] = {}
            if tab2 not in desc_related_tab_col[col2] and col1 in tab_df_infos[tab2].columns:
                desc_related_tab_col[col2][tab2] = col1
    return desc_related_tab_col



def custom_collate_fn(batch):
    query_content_encoded, meta_path_instance_lists, meta_path_df_candidate_ent_lists, missing_vals, descriptor, meta_paths, meta_path_instance_filtered = zip(*batch)

    query_content_encoded = default_collate(query_content_encoded)
    missing_vals = default_collate(missing_vals)
    descriptor = default_collate(descriptor)

    # Define padding function
    def pad_tensor(tensor, target_shape):
        # Calculate the padding required for each dimension
        padding = []
        for dim in range(len(tensor.shape)):
            # Determine the padding needed at the end of this dimension
            pad_size = max(0, target_shape[dim] - tensor.shape[dim])
            # Padding is added as (before, after) for each dimension, but we only pad at the end
            padding = [0, pad_size] + padding

        # Apply the padding
        return F.pad(tensor, padding)

    # Find maximum counts
    max_meta_path_cnt = max(len(sample) for sample in meta_path_instance_lists)
    max_meta_path_instance_cnt = max(max(len(meta_path) for meta_path in sample) for sample in meta_path_instance_lists)
    max_node_cnt = max(max(max(instance.shape[0] for instance in meta_path) for meta_path in sample) for sample in meta_path_instance_lists)
    dim = meta_path_instance_lists[0][0][0].shape[-1]
 


    # Initialize padded arrays
    meta_path_instance_embeddings_padded = torch.zeros(len(batch), max_meta_path_cnt, max_meta_path_instance_cnt, max_node_cnt, dim)
    meta_path_instance_lengths = torch.zeros(len(batch), max_meta_path_cnt, max_meta_path_instance_cnt, dtype=torch.long)

    for batch_idx, sample in enumerate(meta_path_instance_lists):
        for meta_path_idx, meta_path in enumerate(sample):
            for instance_idx, instance in enumerate(meta_path):
                # instance is a instance_cnt*node_cnt*dim
                # print(instance.shape)
                padded_instance = pad_tensor(instance, (max_node_cnt, dim))
                # print(padded_instance.shape)
                meta_path_instance_embeddings_padded[batch_idx, meta_path_idx, instance_idx, :, :] = padded_instance
                meta_path_instance_lengths[batch_idx, meta_path_idx, instance_idx] = instance.shape[0]

    # Padding for meta_path_df_candidate_ent_lists
    max_candidate_ent_cnt = max(len(candidates) for candidates in meta_path_df_candidate_ent_lists)
    max_candidate_ent_length = max(max(ent.shape[0] for ent in candidates) for candidates in meta_path_df_candidate_ent_lists)
    candidate_dim = meta_path_df_candidate_ent_lists[0][0].shape[-1]

    meta_path_df_candidate_ent_embeddings_padded = torch.zeros(len(batch), max_candidate_ent_cnt, max_candidate_ent_length, candidate_dim)
    meta_path_df_candidate_ent_lengths = torch.zeros(len(batch), max_candidate_ent_cnt, dtype=torch.long)

    # for batch_idx, candidates in enumerate(meta_path_df_candidate_ent_lists):
    #     for candidate_idx, candidate in enumerate(candidates):
    #         padded_candidate = pad_tensor(candidate, (max_candidate_ent_length, candidate_dim))
    #         meta_path_df_candidate_ent_embeddings_padded[batch_idx, candidate_idx, :candidate.shape[0], :] = padded_candidate
    #         meta_path_df_candidate_ent_lengths[batch_idx, candidate_idx] = candidate.shape[0]
    for batch_idx, candidates in enumerate(meta_path_df_candidate_ent_lists):
        for candidate_idx, candidate in enumerate(candidates):
            padded_candidate = pad_tensor(candidate, (max_candidate_ent_length, candidate_dim))
            num_rows_to_assign = padded_candidate.shape[0]
            # Ensure that the slice of the padded array matches the shape of the padded tensor
            meta_path_df_candidate_ent_embeddings_padded[batch_idx, candidate_idx, :num_rows_to_assign, :] = padded_candidate[:num_rows_to_assign, :]
            meta_path_df_candidate_ent_lengths[batch_idx, candidate_idx] = candidate.shape[0]


    return query_content_encoded, meta_path_instance_embeddings_padded, meta_path_instance_lengths, meta_path_df_candidate_ent_embeddings_padded, meta_path_df_candidate_ent_lengths, missing_vals, descriptor, meta_paths, meta_path_instance_filtered



def compute_loss(weights, meta_path_instance_filtered, ground_truths, tab_df_infos,device):
    batch_loss = 0.0
    batch_size = len(weights)
    valid_examples = 0

    for i in range(batch_size):
        ground_truth = ground_truths[i]
        all_meta_paths = meta_path_instance_filtered[i]
        ground_truth_likelihoods = []

        # Flatten the list of lists (meta paths) to match the weights () (table_name, index)
        flattened_instances = [tab_df_infos[instance[-1][0]].loc[instance[-1][1]] for meta_path in all_meta_paths for instance in meta_path]

        # Check for ground truth in each instance and collect likelihood if present
        for j, instance in enumerate(flattened_instances):
            if ground_truth not in ['<NA>', 'nan', 'Unknown'] and ground_truth in [str(val_) for val_ in instance.values]:
                ground_truth_likelihoods.append(weights[i][j])

        # If there are valid instances with the ground truth, calculate the loss
        if ground_truth_likelihoods:
            valid_examples += 1
            # Sum the likelihoods of the ground truth over all valid instances
            ground_truth_likelihood_sum = torch.sum(torch.stack(ground_truth_likelihoods))
            # Avoid log(0) by adding a small epsilon
            batch_loss += -torch.log(ground_truth_likelihood_sum + 1e-10)

    # Normalize the batch loss by the number of valid examples
    if valid_examples > 0:
        return batch_loss / valid_examples
    else:
        return torch.tensor(0.0, device=device)  # Use the same device as weights

def evaluate_model(dataloader, model, tab_df_infos,  device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    num_valid_examples = 0

    
    with torch.no_grad():  # No need to track gradients for validation/testing
        for query_content_encoded, meta_path_instance_embeddings_padded, meta_path_instance_lengths, \
            meta_path_df_candidate_ent_embeddings_padded, meta_path_df_candidate_ent_lengths, \
            missing_vals, descriptor, meta_paths, meta_path_instance_filtered in dataloader:

            query_content_encoded = query_content_encoded.to(device)
            meta_path_instance_embeddings_padded = meta_path_instance_embeddings_padded.to(device)
            meta_path_instance_lengths = meta_path_instance_lengths.to(device)
            meta_path_df_candidate_ent_embeddings_padded = meta_path_df_candidate_ent_embeddings_padded.to(device)
            meta_path_df_candidate_ent_lengths = meta_path_df_candidate_ent_lengths.to(device)

            # Forward pass through the model to get weights for each table part
            weights = model(query_content_encoded, meta_path_instance_embeddings_padded,meta_path_instance_lengths,meta_path_df_candidate_ent_embeddings_padded,meta_path_df_candidate_ent_lengths)

            # Compute the loss for the batch
            loss = compute_loss(weights, meta_path_instance_filtered, missing_vals, tab_df_infos, device)

            if loss is not None:
                total_loss += loss.item() * len(weights)
                num_valid_examples += len(weights)
    
    # Calculate average loss
    avg_loss = total_loss / num_valid_examples if num_valid_examples > 0 else 0
    return avg_loss

def normalize_candidate(str1):
    try:
        # Try to convert both strings to floats
        num1 = float(str1)
        return num1
    except:
        return str(str1)

# def get_impute_result_forex(weights, query_samples):
def get_impute_result_forex(weights, query_samples, search=True):
    batch_size = len(weights)

    output_res = []
    for i in range(batch_size):
        sample_weight_series = weights[i].cpu().detach().numpy()
        cur_query_sample = query_samples[i]
        cur_query_desc = query_samples[i].descriptor
        cur_query_content = query_samples[i].query_content

        flattened_instances = [instance[-1].to_dict() for meta_path in cur_query_sample.all_df_series for instance in meta_path ]
        flattened_attrs = ast.literal_eval(query_samples[i].missing_val)
        assert len(flattened_attrs) == len(flattened_instances)

        candidate_scores = {}
        for candidate_ins, candidate_weight, candidate_attr_weight_info in zip(flattened_instances,sample_weight_series,\
                                        flattened_attrs):
            for elem_ in candidate_attr_weight_info:
                candidate_attr, attr_weight = elem_[0], elem_[1]
                candidate_ans = candidate_ins[candidate_attr]
                if candidate_ans not in candidate_scores:
                    candidate_scores[candidate_ans] = [0,0]
                candidate_scores[candidate_ans][0] += candidate_weight
                candidate_scores[candidate_ans][1] += attr_weight

        max_score_candidate = sorted(candidate_scores.items(), 
                             key=lambda item: (-item[1][0], -item[1][1]))[0][0]
        
        query_tab_iloc = cur_query_content['_original_index']
        if not search:
            output_res.append((query_tab_iloc, cur_query_desc, max_score_candidate))
        else:
            # if type(query_tab_iloc.iloc[0]) not in [str, int, float]:
            #     print(type(query_tab_iloc.iloc[0]))
            #     print(query_tab_iloc)
            output_res.append((query_tab_iloc, cur_query_desc, ['q__'+str(query_tab_iloc.iloc[0])+'__'+str(cur_query_desc),candidate_scores]))
    return output_res


def find_metapath_index(metapaths, node_sequence_index):
    cumulative_count = 0
    for i, metapath in enumerate(metapaths):
        # Add the number of node sequences in the current meta-path
        cumulative_count += len(metapath)
        # Check if the given index is less than the cumulative count
        if node_sequence_index < cumulative_count:
            # The index of the meta-path is i
            return i
    # If the node_sequence_index is out of bounds
    return None

def get_inductive_metapth(weights, query_samples,k=3):
    batch_size = len(weights)

    output_res, output_valid_pred = [], []
    for i in range(batch_size):
        sample_weight_series = weights[i].cpu().detach().numpy()
        index_of_maximum_weight = np.argmax(sample_weight_series)
        
        cur_meta_path_instances = query_samples[i].meta_path_instances
        cur_meta_path = query_samples[i].meta_paths
        cur_descriptor = query_samples[i].descriptor
        cur_all_df_series = query_samples[i].all_df_series
        cur_missing_val = query_samples[i].missing_val
        cur_query_content = query_samples[i].query_content

        # Now, if you want to get them in order, you can sort these indices based on the weights
        topk_indices = sample_weight_series.argsort()[-k:][::-1]


        # if cur_descriptor not in ['sr_returned_date_sk_d_day_name','sr_returned_date_sk_d_fy_year','sr_returned_date_sk_d_year','sr_hdemo_sk_hd_buy_potential','sr_cdemo_sk_cd_education_status']:continue
        # if cur_descriptor not in ['sr_returned_date_sk_d_day_name']:continue

        
        all_df_series = [item[-1] for meta_path in cur_all_df_series for item in meta_path]
        # max_df_series = [item[-1] for meta_path in cur_all_df_series for item in meta_path][index_of_maximum_weight].to_dict()

        missing_val_isin, non_missing_val, missing_val_isin, missing_val_isin_max = 0, 0, 0, 0
        if cur_missing_val not in ['<NA>', 'nan', 'Unknown']:
            non_missing_val = 1
            for i in range(len(all_df_series)):
                max_df_series = all_df_series[i]

                for key_, val_ in max_df_series.items():
                    # if not compare_strings_as_numbers(str(cur_missing_val), str(val_)):
                    #     print(cur_descriptor, str(cur_missing_val), str(val_))
                    if compare_strings_as_numbers(str(cur_missing_val), str(val_)):
                        if i in topk_indices:
                            missing_val_isin_max=1
                            max_meta_path = cur_meta_path[find_metapath_index(cur_meta_path_instances,i)]    
                            output_res.append((cur_descriptor,max_meta_path,key_))
                        missing_val_isin=1
            # if missing_val_isin==0:
            #     print(cur_descriptor)
            #     print(cur_missing_val)
            #     print(all_df_series)
        output_valid_pred.append((cur_descriptor, non_missing_val, missing_val_isin, missing_val_isin_max))

    return output_res, output_valid_pred

"""
    the output of this function should be a list of (primary_key_val, desc, top_val)
"""
def get_impute_res(batched_query_samples, dataloader, model, device):
    all_pred_res, all_pred_res_search = [], []
    # No need to track gradients for validation/testing
    for (query_content_encoded, meta_path_instance_embeddings_padded, meta_path_instance_lengths, \
        meta_path_df_candidate_ent_embeddings_padded, meta_path_df_candidate_ent_lengths, \
        missing_vals, descriptor, meta_paths, meta_path_instance_filtered), query_samples in tqdm(zip(dataloader,batched_query_samples),total = len(dataloader)):

        query_content_encoded = query_content_encoded.to(device)
        meta_path_instance_embeddings_padded = meta_path_instance_embeddings_padded.to(device)
        meta_path_instance_lengths = meta_path_instance_lengths.to(device)
        meta_path_df_candidate_ent_embeddings_padded = meta_path_df_candidate_ent_embeddings_padded.to(device)
        meta_path_df_candidate_ent_lengths = meta_path_df_candidate_ent_lengths.to(device)

        # Forward pass through the model to get weights for each table part
        weights = model(query_content_encoded, meta_path_instance_embeddings_padded,meta_path_instance_lengths,meta_path_df_candidate_ent_embeddings_padded,meta_path_df_candidate_ent_lengths)

        batch_res = get_impute_result_forex(weights, query_samples,False)
        all_pred_res_search.extend(get_impute_result_forex(weights, query_samples))
        all_pred_res.extend(batch_res)


    return all_pred_res,all_pred_res_search

"""
    the output of this function should be a list of (desc, top meta-path)
"""
def get_summary_metapath(batched_query_samples, dataloader, model, device):
    all_pred_res, all_valid_pred_cnt = [], []
    # No need to track gradients for validation/testing
    for (query_content_encoded, meta_path_instance_embeddings_padded, meta_path_instance_lengths, \
        meta_path_df_candidate_ent_embeddings_padded, meta_path_df_candidate_ent_lengths, \
        missing_vals, descriptor, meta_paths, meta_path_instance_filtered), query_samples in tqdm(zip(dataloader,batched_query_samples),total = len(dataloader)):

        query_content_encoded = query_content_encoded.to(device)
        meta_path_instance_embeddings_padded = meta_path_instance_embeddings_padded.to(device)
        meta_path_instance_lengths = meta_path_instance_lengths.to(device)
        meta_path_df_candidate_ent_embeddings_padded = meta_path_df_candidate_ent_embeddings_padded.to(device)
        meta_path_df_candidate_ent_lengths = meta_path_df_candidate_ent_lengths.to(device)

        # Forward pass through the model to get weights for each table part
        weights = model(query_content_encoded, meta_path_instance_embeddings_padded,meta_path_instance_lengths,meta_path_df_candidate_ent_embeddings_padded,meta_path_df_candidate_ent_lengths)

        batch_res, output_valid_pred = get_inductive_metapth(weights, query_samples)
        all_pred_res.extend(batch_res)
        all_valid_pred_cnt.extend(output_valid_pred)

    return all_pred_res, all_valid_pred_cnt


def compare_strings_as_numbers(str1, str2):
    try:
        # Try to convert both strings to floats
        num1 = float(str1)
        num2 = float(str2)
        # If both are numbers, compare as floats
        if num1 == num2:
            return True
        else:
            return False
    except ValueError:
        # If conversion fails, strings are not valid numbers
        # Compare as strings
        if str1 == str2:
            return True
        else:
            return False
        
def print_infos(query_attr_meta_path_dic,meta_path_infer_data, accu_=0.8):
    for query_attr_, meta_path_ in query_attr_meta_path_dic.items():
        print(query_attr_)
        print([item[0][0] for item in Counter(meta_path_).most_common(5)])   

    # Convert the list of lists into a DataFrame
    df = pd.DataFrame(meta_path_infer_data, columns=['attr', 'notnull_flag', 'valid_flag', 'max_flag'])
    grouped_df = df.groupby('attr').agg({'notnull_flag':'sum','valid_flag':'sum','max_flag':'sum'}).reset_index()
    print(grouped_df)

    valid_attrs = grouped_df[grouped_df['max_flag']/grouped_df['notnull_flag']>accu_]['attr'].values.tolist()
    print('the valid impute redundanct attribute is ',valid_attrs)
    return valid_attrs

# // "redun_attr": ["sr_addr_sk_ca_address_id", "sr_addr_sk_ca_city", "sr_addr_sk_ca_gmt_offset", "sr_addr_sk_ca_state", "sr_addr_sk_ca_street_name", "sr_addr_sk_ca_zip", "sr_cdemo_sk_cd_dep_college_count", "sr_cdemo_sk_cd_dep_count", "sr_cdemo_sk_cd_dep_employed_count", "sr_cdemo_sk_cd_education_status", "sr_cdemo_sk_cd_purchase_estimate", "sr_customer_sk_c_birth_country", "sr_customer_sk_c_birth_month", "sr_customer_sk_c_customer_id", "sr_customer_sk_c_email_address", "sr_customer_sk_c_first_name", "sr_customer_sk_c_last_name", "sr_customer_sk_c_salutation", "sr_item_sk_i_brand", "sr_item_sk_i_brand_id", "sr_item_sk_i_formulation", "sr_item_sk_i_item_desc", "sr_item_sk_i_item_id", "sr_item_sk_i_manager_id", "sr_item_sk_i_manufact", "sr_item_sk_i_product_name", "sr_return_time_sk_t_second", "sr_return_time_sk_t_time", "sr_return_time_sk_t_time_id", "sr_returned_date_sk_d_dow"]


# python tpc_ds3.py --config_file_path /research/local/feng/evaluation/dataconfig/tpc_ds.json --checkpoint_path /research/local/feng/evaluation/all_models/tpc_ds_202401092315.pt
# python tpc_ds2.py --config_file_path /research/local/feng/evaluation/dataconfig/eicu.json --checkpoint_path /research/local/feng/evaluation/all_models/eicu_202401091050.pt

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a Path Instance Network')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of the embeddings')
    parser.add_argument('--num_filters', type=int, default=128, help='Number of filters in the convolution layer')
    parser.add_argument('--kernel_size', type=int, default=1, help='Kernel size for the convolution layer')
    parser.add_argument('--num_heads', type=int, default=1, help='Number of heads in the multihead attention layer')
 
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--accuracy', type=float, default=0.8, help='Accuracy to filter the no valid attribute')
    parser.add_argument('--config_file_path', type=str, required=True, help='The parameter config for each database.')
    parser.add_argument('--checkpoint_path', type=str, required=True, \
                        help='If the model already saved, please use it for prediction...')
    parser.add_argument('--summary_path', type=str, required=True, \
                        help='the meta path filtered topk summary path...')
    # parser.add_argument('--method', type=str, required=True, \
    #                     help='method name for ablation study...')
    parser.add_argument('--method', type=str, default='dbimpute', \
                        help='method name for our study...')
    parser.add_argument('--mode', type=str, default="none", help='If u want to construct the inferrence samples, using it as sample.')
    parser.add_argument('--random_emb', action='store_true',  \
                        help='To evaluate the random generated embedding...')
    parser.add_argument('--input_para', type=int, default=5,  \
                        help='The parameters for our topk meta-path.')


    # Parse known arguments, ignoring unknown. This allows us to first get the path of the config file.
    args = parser.parse_args()
    # Load the config file
    config = mu.load_config(args.config_file_path)

    # Update the default values with the config file if they exist
    for key, value in config.items():
        setattr(args, key, value) 
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    db = TPCDSDataset(sample_size=args.sample_size,lsh_threshold=args.lsh_threshold,threshold=args.threshold,desc_meta_path_cnt=args.desc_meta_path_cnt,\
                database_name=args.database_name,mask_flag = np.nan,\
                query_table_name=args.query_tab_name,query_tab_path=args.query_tab_path,save_folder=args.save_folder,\
                csv_path=args.path,schema_path=args.schema_path)
    db.construct_all_graphs()
    

    # Initialize the model
    model = PathInstanceNetwork(args.embedding_dim, args.num_filters, args.kernel_size, args.num_heads)
    model.to(device) 
    # tab_df_infos, schema_infos = db.tab_df_infos, db.schema_infos
    if os.path.exists(args.checkpoint_path):
        print('predict using our model ', args.checkpoint_path)
        model.load_state_dict(torch.load(args.checkpoint_path))
        model.eval()


    # if not os.path.exists('test_'+args.database_name+'.pkl'):
    if not os.path.exists(args.summary_path):
        query_samples = cpickle.load(open(args.query_samples_path,'rb')) # Load query_samples from args.query_samples_path
        print('total sample size ', len(query_samples))

        # dataset = MetaPathDataset(query_samples, args.faiss_idx_dir, args.transtab_path, args.embedding_file_path, args.k)
        if args.random_emb:
            dataset = MetaPathDataset(query_samples, args.faiss_idx_dir, args.transtab_path, args.rand_embedding_file_path, args.k)
        else:
            dataset = MetaPathDataset(query_samples, args.faiss_idx_dir, args.transtab_path, args.embedding_file_path, args.k)
        
        # Determine the size of each set
        train_size = int(0.6 * len(dataset))  
        valid_size = int(0.2 * len(dataset))  
        test_size = len(dataset) - train_size - valid_size  # The rest for testing

        # Split the dataset
        train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])


        # meta-path inferrence referring
        # Assuming `test_dataset` is a Subset returned by random_split
        query_samples = [dataset.query_samples[i] for i in test_dataset.indices]
        print('induct from our test samples ', len(query_samples))
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
        # meta_path_res, meta_path_infer_data = get_summary_metapath(batch_loader(query_samples, args.batch_size), dataloader, model, device)
        meta_path_res, meta_path_infer_data = get_summary_metapath(batch_loader(query_samples, args.batch_size), test_dataloader, model, device)
        # print(len(meta_path_res))

        query_attr_meta_path_dic = {}
        for item in meta_path_res:
            query_attr_, meta_path_, attr_ = item
            if query_attr_ not in query_attr_meta_path_dic:
                query_attr_meta_path_dic[query_attr_] = []
            query_attr_meta_path_dic[query_attr_].append((meta_path_,attr_))
        # pickle.dump((query_attr_meta_path_dic, meta_path_res, meta_path_infer_data),open('test_'+args.database_name+'.pkl','wb'))
        pickle.dump((query_attr_meta_path_dic, meta_path_res, meta_path_infer_data),open(args.summary_path,'wb'))
    else:
        # query_attr_meta_path_dic, meta_path_res, meta_path_infer_data = pickle.load(open('test_'+args.database_name+'.pkl','rb'))
        query_attr_meta_path_dic, meta_path_res, meta_path_infer_data = pickle.load(open(args.summary_path,'rb'))

    valid_impute_attr_ = print_infos(query_attr_meta_path_dic,meta_path_infer_data, args.accuracy)
    # # construct_test_samples(self,query_attr_related_path_dict,desired_number_of_partitions=10)
    # if os.path.exists(args.checkpoint_path):
    #     model.load_state_dict(torch.load(args.checkpoint_path))
    #     model.eval()
    query_attr_related_path_dic = {}
    # for query_attr_, meta_path_ in query_attr_meta_path_dic.items():
    #     if query_attr_ not in valid_impute_attr_:continue
    #     #  saving for the top meta_path
    #     query_attr_related_path_dic[query_attr_] = [item[0] for item in Counter(meta_path_).most_common(5)]

    # set for filter the top5 candidates
    topk_metapaths = args.input_para
    for query_attr_, meta_path_ in query_attr_meta_path_dic.items():
        # Count occurrences of each element
        meta_path_counter = Counter(meta_path_)
        
        # Get total number of elements in meta_path_
        total_elements = sum(meta_path_counter.values())
        
        # Get the top 5 most common elements and their counts
        top5_common = meta_path_counter.most_common(topk_metapaths)
        
        # Compute the ratio for each of the top 5 elements
        top5_common_with_ratio = [(element, count / total_elements) for element, count in top5_common]
        
        cur_key_dict = {}
        for item in top5_common_with_ratio:
            if item[0][0] not in cur_key_dict:
                cur_key_dict[item[0][0]]=[]
            cur_key_dict[item[0][0]].append((item[0][1],item[1]))
        
        # Store the result in the dictionary
        query_attr_related_path_dic[query_attr_] = cur_key_dict
    print('we can only impute ',len(valid_impute_attr_),' attribute according to your set accuracy ', args.accuracy)

    
    # begin to test the impute the missing value
    start_time = time.time()


    if args.mode=='raw_sample':     
        inferrence_samples, query_error_df_keys = db.construct_eval_samples(query_attr_related_path_dic)
        infer_cnt = len(inferrence_samples)
        print('the test for inferring size is ', infer_cnt)
        cpickle.dump(inferrence_samples,open(os.path.join(args.infer_samples_path,args.database_name+'_'+str(infer_cnt)+'_raw.pkl'),'wb'))
    elif args.mode=='inductive_sample':
        inferrence_samples, query_error_df_keys = db.construct_test_samples(query_attr_related_path_dic)
        infer_cnt = len(inferrence_samples)
        print('the test for inferring size is ', infer_cnt)
        cpickle.dump(inferrence_samples,open(os.path.join(args.infer_samples_path,args.database_name+'_'+str(infer_cnt)+'_raw.pkl'),'wb'))
    else:
        data_sample_file = glob.glob(os.path.join(args.infer_samples_path,args.database_name+'_*.pkl'))[0]
        inferrence_samples = cpickle.load(open(data_sample_file,'rb'))
        sample_ratio = 0.01
        inferrence_samples = random.sample(inferrence_samples, int(len(inferrence_samples)*sample_ratio))
        # I would first do some sampling operation to test my test
        # inferrence_samples = inferrence_samples[:10]
        # if len(args.redun_attr)>0:
        #     inferrence_samples = [sample for sample in inferrence_samples if sample.descriptor in args.redun_attr]

    end_time = time.time()
    print('construct the sample time cost',end_time-start_time)
        

    query_raw_df = db.tab_df_infos[args.query_tab_name]
    # related_tab_desc_bytab = collect_related_tab_desc_bytab(args.query_tab_name,args.index_file,args.threshold,db.tab_df_infos[args.query_tab_name].columns,db.tab_df_infos)

    
    # this is for extracting the answers from our candidates based on the query desc
    query_tab_primary_keys = db.schema_infos[args.query_tab_name]['pkey_col']

    batched_query_samples = batch_loader(inferrence_samples, args.batch_size)
    # query_dataset = MetaPathDataset(inferrence_samples, args.faiss_idx_dir, args.transtab_path, args.embedding_file_path)
    if args.random_emb:
        query_dataset = MetaPathDataset(inferrence_samples, args.faiss_idx_dir, args.transtab_path, args.rand_embedding_file_path)
    else:
        query_dataset = MetaPathDataset(inferrence_samples, args.faiss_idx_dir, args.transtab_path, args.embedding_file_path)
    batched_query_dataloader = DataLoader(query_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    start_time = time.time()
    output_res, output_res_search = get_impute_res(batched_query_samples, batched_query_dataloader, model, device)
    end_time = time.time()
    print('construct the infer time cost',end_time-start_time)

    query_truth_df = pd.read_csv(args.input_file)

    # query_error_df_keys
    
    for item in output_res:
        row_idx, desc_attr, value = item

        try:
            column_data_type = query_raw_df[desc_attr].dtype
            # Convert the value to match the column data type
            if pd.api.types.is_integer_dtype(column_data_type):
                converted_value = pd.to_numeric(value, downcast='integer', errors='coerce')
            elif pd.api.types.is_float_dtype(column_data_type):
                converted_value = pd.to_numeric(value, errors='coerce')
            elif pd.api.types.is_string_dtype(column_data_type):
                converted_value = str(value)
        except:
            print(column_data_type)
            print(desc_attr)
            print(value, type(value), converted_value, type(converted_value))

        try:
            query_raw_df.loc[row_idx,desc_attr]=converted_value
        except:
            query_raw_df[desc_attr] = query_raw_df[desc_attr].astype(float)
            query_raw_df.loc[row_idx,desc_attr]=converted_value
    end_time = time.time()

    impute_time_cost = round(end_time-start_time)


    query_error_df = pd.read_csv(args.query_tab_path)
    query_truth_df = query_error_df[args.primary_columns].merge(query_truth_df, on=args.primary_columns, how='inner')
 
    query_truth_dict, query_pred_dict = {}, {}
    for item in output_res_search:
        row_idx, desc_attr, score_results = item
        # try:
        query_truth_dict['q__'+str(row_idx.iloc[0])+'__'+str(desc_attr)] = {normalize_candidate(query_truth_df.loc[row_idx,desc_attr].iloc[0]\
                                                                    if isinstance(query_truth_df.loc[row_idx,desc_attr],pd.Series) else query_truth_df.loc[row_idx,desc_attr]):1}
        # Continue as before with sorting and extracting the max score document
        sorted_docs = sorted(score_results[1].items(), key=lambda item: (item[1][0], item[1][1]))
        query_pred_dict[score_results[0]] = {normalize_candidate(doc_id[0]): rank for rank, doc_id in enumerate(sorted_docs, start=1)}
        # except:
        #     print(row_idx, ' meets some errors ')
    
    method = dbimpute
    run_json_f = os.path.join(args.trec_eval_dir,args.database_name+'_run_'+method+'.json')

    if args.method=='dbimpute':
        query_json_f = os.path.join(args.trec_eval_dir,args.database_name+'_qry.json')
        with open(query_json_f, 'w') as json_file:
            json_file.write(json.dumps(query_truth_dict, indent=4))
    with open(run_json_f, 'w') as json_file:
        json_file.write(json.dumps(query_pred_dict, indent=4))
    # with open(query_json_f, 'r') as json_file:
    #     query_truth_dict = json.load(json_file)
    # with open(run_json_f, 'r') as json_file:
    #     query_pred_dict = json.load(json_file)
    if type(db.schema_infos[db.query_tab_name]['pkey_col'])==str:
        impute_df = query_error_df_keys.merge(query_raw_df, how='left',on=args.primary_columns)
    else:
        impute_df = query_error_df_keys.merge(query_raw_df, how='left',on='__'.join(args.primary_columns))

    
    train_time_cost = -1
    impute_df.to_csv(os.path.join(args.impute_dir,args.database_name,\
                                        f'{args.query_tab_name}_{args.sample_size}_{method}_{train_time_cost}_{impute_time_cost}.csv'),index=False)