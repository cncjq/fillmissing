from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import transtab
import pandas as pd
import multi_utils as mu
import faiss
import os
import pickle

"""
    I define this because I want to sample 
    the meta-path according to the path instance similarity
"""

class MetaPathSampler:
    def __init__(self, embedding_file_path, faiss_idx_dir, k=5, default_embedding_dim=64):
        self.all_embs = mu.load_embeddings(embedding_file_path)
        self.k = k
        self.default_embedding_dim = default_embedding_dim
        # You can also initialize a default embedding that will be used for all missing embeddings
        self.default_embedding = np.zeros(self.default_embedding_dim)

        self.faiss_idx_dir = faiss_idx_dir

    def search_index(self,query_embedding, table_name):
        # Load the specific table's FAISS index
        table_idx_file = os.path.join(self.faiss_idx_dir,f"{table_name}_faiss_index.index")
        if not os.path.exists(table_idx_file):
            embeddings = self.all_embs[table_name]
            embeddings = embeddings.astype(np.float32)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            faiss.write_index(index, table_idx_file)
        else:
            index = faiss.read_index(table_idx_file)

        # Perform the search
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = index.search(query_embedding, self.k)

        # Map index results back to row indices in the original table
        return [indices[0][i] for i in range(len(indices[0]))]
                
    

    def get_node_embedding(self, node_idx):
        table_name, table_idx = node_idx
        # Check if the table_name is in the dictionary and if the table_idx exists within the array
        if table_name in self.all_embs and table_idx < len(self.all_embs[table_name]):
            return self.all_embs[table_name][table_idx]
        else:
            # If not found, return the default embedding
            return self.default_embedding

    def calculate_similarity(self, node1_emb, node2_emb):
        similarity = np.dot(node1_emb, node2_emb) / (np.linalg.norm(node1_emb) * np.linalg.norm(node2_emb))
        return similarity

    def aggregate_path_similarity(self, path_instance):
        similarities = [self.calculate_similarity(
            self.get_node_embedding(path_instance[i]),
            self.get_node_embedding(path_instance[i + 1]))
            for i in range(len(path_instance) - 1)]
        return np.mean(similarities)

    def get_topk_path_instances_with_embeddings(self, meta_path_instances, all_df_series):
        path_similarities = [self.aggregate_path_similarity(instance) for instance in meta_path_instances]
        sorted_indices = sorted(range(len(path_similarities)), key=lambda idx: path_similarities[idx], reverse=True)[:self.k]
        selected_meta_path_instances = [meta_path_instances[idx] for idx in sorted_indices]
        # selected_all_df_series = [[item.to_list() for item in all_df_series[idx]] for idx in sorted_indices]
        # selected_all_df_series = [
        #     [[c for c in item.tolist() if not pd.isna(c)] for item in all_df_series[idx] ]
        #     for idx in sorted_indices
        # ]

        embedding_matrices = np.array([np.stack([self.get_node_embedding(node_idx) for node_idx in path_instance])
                                    for path_instance in selected_meta_path_instances])
        # print(selected_meta_path_instances)
        last_ent_matrices = np.array(np.stack([self.get_node_embedding(path_instance[-1]) for path_instance in selected_meta_path_instances])
                                    )

        return selected_meta_path_instances, embedding_matrices, last_ent_matrices
    
    def get_topk_path_instances_with_embeddings_tab(self, query_content_emb, tab_name):
        selected_meta_path_instances = [[(tab_name,related_tuple)] for related_tuple in self.search_index(self,query_content_emb, tab_name)]


        embedding_matrices = np.array([np.stack([self.get_node_embedding(node_idx) for node_idx in path_instance])
                                    for path_instance in selected_meta_path_instances])
        # print(selected_meta_path_instances)
        last_ent_matrices = np.array(np.stack([self.get_node_embedding(path_instance[-1]) for path_instance in selected_meta_path_instances])
                                    )

        return selected_meta_path_instances, embedding_matrices, last_ent_matrices




class MetaPathDataset(Dataset):
    def __init__(self, query_samples, faiss_idx_dir, transtab_path='', \
                 embedding_file_path='',k=5):
        self.query_samples = query_samples
        self.query_encoder = transtab.build_encoder(
                binary_columns=[],
                checkpoint = transtab_path
            )
        self.metapath_sampler = MetaPathSampler(embedding_file_path,faiss_idx_dir,k=k)

    
    def __len__(self):
        return len(self.query_samples)
    
    def __getitem__(self, idx):
        # Encode the query content, I will get a tensor tensor with 128 dim
        # meta_path_instance_list, meta_path_df_candidate_ent_embeddings
        query_content_encoded = self.query_encoder(self.query_samples[idx].re_query_content)
        missing_val = str(self.query_samples[idx].missing_val)
        descriptor = self.query_samples[idx].descriptor
        meta_paths = self.query_samples[idx].meta_paths
        meta_path_instances = self.query_samples[idx].meta_path_instances
        all_df_series = self.query_samples[idx].all_df_series


        meta_path_instance_filtered, embedding_ls, meta_path_df_candidate_ent_ls = [] , [],  []

        if len(meta_path_instances)>0:
            for meta_p, meta_df in zip(meta_path_instances, all_df_series):
                selected_meta_path_instances, embedding_matrices, last_ent_matrices = \
                    self.metapath_sampler.get_topk_path_instances_with_embeddings(meta_p, meta_df)
                meta_path_instance_filtered.append(selected_meta_path_instances)
                
                # Corrected tensor creation and appending
                embedding_tensor = torch.tensor(embedding_matrices, dtype=torch.float32)
                # print(embedding_tensor.shape)
                embedding_ls.append(embedding_tensor)

                last_ent_tensor = torch.tensor(last_ent_matrices, dtype=torch.float32)
                meta_path_df_candidate_ent_ls.append(last_ent_tensor)
            # print(len(embedding_ls))
        
        else:
            for item in meta_paths:
                table_ = item[1]

                selected_meta_path_instances, embedding_matrices, last_ent_matrices = \
                    self.metapath_sampler.get_topk_path_instances_with_embeddings_tab(query_content_encoded, table_)


                meta_path_instance_filtered.append(selected_meta_path_instances)
                
                # Corrected tensor creation and appending
                embedding_tensor = torch.tensor(embedding_matrices, dtype=torch.float32)
                embedding_ls.append(embedding_tensor)

                last_ent_tensor = torch.tensor(last_ent_matrices, dtype=torch.float32)
                meta_path_df_candidate_ent_ls.append(last_ent_tensor)

        # Convert lists of embeddings to tensors with the appropriate type
        meta_path_instance_embeddings = embedding_ls
        # meta_path_df_candidate_ent_embeddings = torch.tensor(np.stack(meta_path_df_candidate_ent_ls, axis=0), dtype=torch.float32)
        meta_path_df_candidate_ent_embeddings = meta_path_df_candidate_ent_ls


        return query_content_encoded, meta_path_instance_embeddings, meta_path_df_candidate_ent_embeddings, \
                missing_val, descriptor, meta_paths, meta_path_instance_filtered
        # return (query_content_encoded, meta_path_instance_embeddings, meta_path_df_candidate_ent_embeddings, 
        #         missing_val, descriptor, meta_paths, meta_path_instance_filtered)