# train.py
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

def get_impute_result_forex(weights, query_samples, tab_df_infos,related_tab_desc_bytab, pkey_col):
    batch_size = len(weights)

    output_res = []
    for i in range(batch_size):
        sample_weight_series = weights[i].cpu().detach().numpy()
        cur_query_sample = query_samples[i]
        cur_query_desc = query_samples[i].descriptor
        cur_query_content = query_samples[i].query_content

        flattened_instances = []
        for meta_path in cur_query_sample.meta_path_instances:
            for instance in meta_path:
                tab_name, tab_cand_idx = instance[-1][0], instance[-1][1]
                if related_tab_desc_bytab[cur_query_desc][tab_name] not in tab_df_infos[tab_name].columns:
                    print(tab_name, related_tab_desc_bytab[cur_query_desc][tab_name])
                flattened_instances.append(tab_df_infos[tab_name].loc[tab_cand_idx,related_tab_desc_bytab[cur_query_desc][tab_name]])

        candidate_scores = {}
        for candidate_ans, candidate_weight in zip(flattened_instances,sample_weight_series):
            if candidate_ans not in candidate_scores:
                candidate_scores[candidate_ans] = 0
            else:
                candidate_scores[candidate_ans] += candidate_weight

        max_score_candidate = max(candidate_scores, key=candidate_scores.get)
        pkey_val = cur_query_content[[pkey_col]]
        output_res.append((pkey_val, cur_query_desc, max_score_candidate))
        
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

def get_inductive_metapth(weights, query_samples):
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


        max_meta_path = cur_meta_path[find_metapath_index(cur_meta_path_instances,index_of_maximum_weight)]    
        all_df_series = [item[-1] for meta_path in cur_all_df_series for item in meta_path]
        max_df_series = [item[-1] for meta_path in cur_all_df_series for item in meta_path][index_of_maximum_weight].to_dict()

        missing_val_isin, non_missing_val, missing_val_isin, missing_val_isin_max = 0, 0, 0, 0
        if cur_missing_val not in ['<NA>', 'nan', 'Unknown']:
            non_missing_val = 1
            for i in range(len(all_df_series)):
                max_df_series = all_df_series[i]

                for key_, val_ in max_df_series.items():
                    if str(cur_missing_val) == str(val_):
                        output_res.append((cur_descriptor,max_meta_path,key_))
                        if i==index_of_maximum_weight:missing_val_isin_max=1
                        missing_val_isin=1
        output_valid_pred.append((cur_descriptor, non_missing_val, missing_val_isin, missing_val_isin_max))

    return output_res, output_valid_pred

"""
    the output of this function should be a list of (primary_key_val, desc, top_val)
"""
def get_impute_res(batched_query_samples, dataloader, model, tab_df_infos, related_tab_desc_bytab, pkey_col, device):
    all_pred_res = []
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

        batch_res = get_impute_result_forex(weights, query_samples, tab_df_infos,related_tab_desc_bytab, pkey_col)
        all_pred_res.extend(batch_res)

    return all_pred_res

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


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a Path Instance Network')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of the embeddings')
    parser.add_argument('--num_filters', type=int, default=128, help='Number of filters in the convolution layer')
    parser.add_argument('--kernel_size', type=int, default=1, help='Kernel size for the convolution layer')
    parser.add_argument('--num_heads', type=int, default=1, help='Number of heads in the multihead attention layer')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--model_save_path', type=str, required=True, help='Path to save the best model')
    parser.add_argument('--config_file_path', type=str, required=True, help='The parameter config for each database.')
    parser.add_argument('--checkpoint_path', type=str, required=True, \
                        help='If the model already saved, please use it for prediction...')
    parser.add_argument('--random_emb', action='store_true',  \
                        help='To evaluate the random generated embedding...')


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
    

    # Initialize the model
    model = PathInstanceNetwork(args.embedding_dim, args.num_filters, args.kernel_size, args.num_heads)
    model.to(device) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    tab_df_infos, schema_infos = db.tab_df_infos, db.schema_infos

    test_dataset = None
    test_dataloader = None


    # load dataset for training
    if not os.path.exists(args.checkpoint_path):
        print('begin to train our model ...')
        # Load the dataset
        query_samples = cpickle.load(open(args.query_samples_path,'rb')) # Load query_samples from args.query_samples_path
        print('total sample size ', len(query_samples))

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

        # if query_samples is not 

        # Create data loaders for each dataset
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)


        best_val_loss = float('inf')
        early_stopping_counter = 0
        # Get current date and time
        current_time = datetime.now()

        # Format the date and time to desired format: 'YYYYMMDDHHMM'
        formatted_time = current_time.strftime('%Y%m%d%H%M')
        print(formatted_time)


        start_time = time.time()
        # Training loop
        for epoch in tqdm(range(args.num_epochs), total=args.num_epochs):
            model.train()
            total_loss = 0.0
            num_valid_examples = 0
            num_instances = 0

            # query_content_encoded, meta_path_instance_embeddings_padded, meta_path_instance_lengths, 
            # meta_path_df_candidate_ent_embeddings_padded, meta_path_df_candidate_ent_lengths, missing_vals, descriptor, meta_paths, meta_path_instance_filtered

            for batch_idx, (query_content_encoded, meta_path_instance_embeddings_padded, meta_path_instance_lengths, \
                meta_path_df_candidate_ent_embeddings_padded, meta_path_df_candidate_ent_lengths, \
                missing_vals, descriptor, meta_paths, meta_path_instance_filtered) in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):

                
                optimizer.zero_grad()

                query_content_encoded = query_content_encoded.to(device)
                meta_path_instance_embeddings_padded = meta_path_instance_embeddings_padded.to(device)
                meta_path_instance_lengths = meta_path_instance_lengths.to(device)
                meta_path_df_candidate_ent_embeddings_padded = meta_path_df_candidate_ent_embeddings_padded.to(device)
                meta_path_df_candidate_ent_lengths = meta_path_df_candidate_ent_lengths.to(device)

                # Forward pass through the model to get weights for each table part
                weights = model(query_content_encoded, meta_path_instance_embeddings_padded,meta_path_instance_lengths,meta_path_df_candidate_ent_embeddings_padded,meta_path_df_candidate_ent_lengths)
                
                # Compute the loss for the batch
                loss = compute_loss(weights, meta_path_instance_filtered, missing_vals, tab_df_infos, device)
                # test.tab_df_infos
                
                # If there was at least one valid example in the batch, perform backpropagation
                if loss is not None:
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(weights) # Multiply by batch size to scale up loss
                    num_valid_examples += len(weights)
                    # print(f"Train Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")
                else:
                    print(f"Train Epoch {epoch}, Batch {batch_idx}, no valid examples with ground truth.")


            print(f"Training for total {num_valid_examples} examples ")
            # Print average loss for the epoch (scaled back down)
            if num_valid_examples > 0:
                avg_loss = total_loss / num_valid_examples
                print(f"Epoch {epoch} average loss: {avg_loss}")
            else:
                print(f"Epoch {epoch} had no valid examples and was skipped.")

            # Validation phase
            val_loss = evaluate_model(valid_dataloader, model, tab_df_infos, device)
            print(f"Validation Epoch {epoch} loss: {val_loss}")

            # Check if the validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                # Save the best model
                best_model_wts = model.state_dict()
                print(f"Validation loss decreased ({best_val_loss:.6f} --> {val_loss:.6f}).  Saving model ...")
                torch.save(best_model_wts, os.path.join(args.model_save_path,args.database_name+'_'+formatted_time+'.pt'))
            else:
                early_stopping_counter += 1
                print(f"Validation loss did not improve from {best_val_loss:.6f}")
                if early_stopping_counter >= args.patience:
                    print("Early stopping")
                    break

            # Save the losses to file 
            with open(os.path.join(args.loss_save_path,args.database_name+'_'+formatted_time+'.txt'), 'a') as f:
                f.write(f"{epoch+1},{avg_loss},{val_loss}\n")

        end_time = time.time()
        print('training time is ', end_time-start_time)
        # after training, u need to induct the best meta-path for each attribute imputation
        model.load_state_dict(best_model_wts)
        model.eval()