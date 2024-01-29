import torch
import torch.nn as nn
import torch.nn.functional as F

class PathInstanceNetwork(nn.Module):
    def __init__(self, embedding_dim=128, num_filters=128, kernel_size=1, num_heads=1):
        super(PathInstanceNetwork, self).__init__()
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size)
        self.path_instance_self_attention = nn.MultiheadAttention(embed_dim=num_filters, num_heads=num_heads)
        self.meta_path_self_attention = nn.MultiheadAttention(embed_dim=num_filters, num_heads=num_heads)
        self.query_meta_path_interaction_attention = nn.MultiheadAttention(embed_dim=num_filters, num_heads=num_heads)
        self.scoring_layer = nn.Linear(embedding_dim + 2 * num_filters + embedding_dim, 1)
        self.num_filters = num_filters

    def forward(self, query_content_encoded, meta_path_instance_embeddings, meta_path_instance_lengths, meta_path_df_candidate_ent_embeddings, meta_path_df_candidate_ent_lengths):
        batch_size = query_content_encoded.size(0)
        all_weights = []

        for batch_idx in range(batch_size):
            batch_meta_path_repr_list = []
            for path_idx in range(meta_path_instance_embeddings[batch_idx].size(0)):
                path_instances = meta_path_instance_embeddings[batch_idx][path_idx]
                path_instance_lengths = meta_path_instance_lengths[batch_idx][path_idx]

                # Process each path instance
                path_instance_reprs = []
                for instance_idx in range(path_instances.size(0)):
                    instance = path_instances[instance_idx]
                    instance_length = path_instance_lengths[instance_idx]
                    # print(instance_length)
                    # print(instance)

                    # print(instance_length.shape)
                    # print(instance.shape)
                    if instance_length.item() > 0:
                        instance = instance[:instance_length.item()]  # Select valid (non-padded) part
                        instance = instance.unsqueeze(0).transpose(1, 2)  # Reshape for Conv1d
                        path_instance_repr = self.conv(instance).squeeze(0)
                        path_instance_repr = path_instance_repr.mean(dim=1)  # Mean pooling
                        path_instance_reprs.append(path_instance_repr)

                # Perform self-attention on path instances
                if path_instance_reprs:
                    path_instance_reprs = torch.stack(path_instance_reprs).unsqueeze(1)  # Reshape for self-attention torch.Size([3, 1, 128])
                    attn_output, _ = self.path_instance_self_attention(path_instance_reprs, path_instance_reprs, path_instance_reprs) #torch.Size([3, 1, 128])
                    meta_path_repr = attn_output.mean(dim=0)  # Aggregate representations
                    # print(meta_path_repr.shape) #orch.Size([1, 128])
                    batch_meta_path_repr_list.append(meta_path_repr)


            # Aggregate meta-path representations for the batch
            if batch_meta_path_repr_list:
                # print(len(batch_meta_path_repr_list))
                meta_path_reprs = torch.stack(batch_meta_path_repr_list)#cnt
                # print(batch_meta_path_repr_list[0].shape)
                # print(meta_path_reprs.shape) [3, 1, 128]
                v1, _ = self.meta_path_self_attention(meta_path_reprs, meta_path_reprs, meta_path_reprs)
                v1_aggregated = v1.mean(dim=0).squeeze(1)

            
            # Interaction attention between query_repr and each meta-path representation
            query_repr = query_content_encoded[batch_idx].unsqueeze(0)  # Shape: [1, embedding_dim]
            interaction_repr_list = []
            for meta_path_repr in batch_meta_path_repr_list:
                interaction_repr, _ = self.query_meta_path_interaction_attention(query_repr, meta_path_repr.unsqueeze(0), meta_path_repr.unsqueeze(0)) #torch.Size([1, 1, 128])
                interaction_repr_list.append(interaction_repr.squeeze(0))

            # Aggregate interaction representations
            v2_aggregated = torch.stack(interaction_repr_list).mean(dim=0)

            # Process each candidate entity and compute scores
            scores = []
            for idx, (candidate_ent_embedding, candidate_length) in enumerate(zip(meta_path_df_candidate_ent_embeddings[batch_idx], meta_path_df_candidate_ent_lengths[batch_idx])):
                if candidate_length.item()>0:
                    candidate = candidate_ent_embedding[:candidate_length]
                    seq_length = candidate.shape[0]
                    # Repeat (tile) the 2D tensors to match the sequence length
                    query_repr_tiled = query_repr.squeeze(0).repeat(seq_length, 1)  # [batch_size, seq_length, features]
                    v1_aggregated_tiled = v1_aggregated.repeat(seq_length, 1)  # [batch_size, seq_length, features]
                    v2_aggregated_tiled = v2_aggregated.repeat(seq_length, 1)
                    # print(candidate.shape, query_repr_tiled.shape, v1_aggregated_tiled.shape, v2_aggregated_tiled.shape)
                    # torch.Size([4, 128]) torch.Size([4, 128]) torch.Size([4, 128]) torch.Size([4, 128])

                    combined_repr = torch.cat((query_repr_tiled, v1_aggregated_tiled, v2_aggregated_tiled, candidate), dim=-1)
                    # torch.Size([1, 128]) torch.Size([1, 128]) torch.Size([1, 128]) torch.Size([4, 128])
                    score = self.scoring_layer(combined_repr)
                    scores.append(score) #4*1

            # Stack scores and apply softmax
            weights = F.softmax(torch.cat([tensor.view(-1) for tensor in scores]), dim=0)
            all_weights.append(weights)

        return all_weights