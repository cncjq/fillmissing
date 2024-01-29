import pandas as pd
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
from collections import deque
import tqdm
from tqdm import tqdm
import math
from load_table_representation import *
from queue import PriorityQueue
import json
import argparse

from datasketch import MinHash, MinHashLSHEnsemble

class ColumnLSHEnsembleIndex:
    def __init__(self, database, num_perm=128, threshold=0.8, num_part=32):
        self.database = database
        self.num_perm = num_perm
        self.threshold = threshold
        self.num_part = num_part
        self.lsh_ensemble = MinHashLSHEnsemble(threshold=self.threshold, num_perm=self.num_perm, num_part=self.num_part)
        self.minhashes = {}  # Change to dictionary to store MinHashes by column key

    def create_minhash(self, values):
        m = MinHash(num_perm=self.num_perm)
        for value in values:
            m.update(str(value).encode('utf-8'))
        return m

    def index(self):
        # Populate the minhashes dictionary with minhashes keyed by column
        print('begin to build the index for the database.')
        for table_name, df in tqdm(self.database.items(),total=len(self.database.items())):
            for column_name in df.columns:
                column_values = df[column_name].dropna().astype(str).values
                column_key = f"{table_name}.{column_name}"
                m = self.create_minhash(column_values)
                self.minhashes[column_key] = m  # Store the MinHash in the dictionary

        # Perform the indexing in a single call
        index_data = [(key, m, len(self.database[key.split('.')[0]][key.split('.')[1]].dropna())) for key, m in self.minhashes.items()]
        self.lsh_ensemble.index(index_data)
        print('lsh index built done...')

    def query(self, query_table_name, query_column_name):
        query_key = f"{query_table_name}.{query_column_name}"
        query_m = self.minhashes.get(query_key)
        if query_m is None:
            # MinHash for query column was not found, handle the error or rebuild it
            return []
        
        # Use the stored MinHash to query the LSH Ensemble
        result = self.lsh_ensemble.query(query_m, len(query_m))
        # ['apachepredvar.teachtype', 'apachepredvar.var03hspxlos', 'apachepredvar.saps3yesterday', 'apachepredvar.saps3today', 'apachepredvar.aids', 'apachepredvar.saps3day1']
        return [key for key in result if key != query_key]  # Exclude the query key itself

# Usage
# Assuming `database` is a dict with table names as keys and DataFrames as values
# and `database_pk_fks` is a dict with table names as keys and sets of primary key columns as values
def compute_joinability_index_lsh(database, database_pk_fks,threshold):
    lsh_index = ColumnLSHEnsembleIndex(database,threshold=threshold)
    lsh_index.index()
    
    print('begin to query for each column for the database tables...')
    joinability_index = {}
    for table_name, df in tqdm(database.items(),total = len(database.items())):
        for column_name in df.columns:
            if column_name not in database_pk_fks.get(table_name, []):
                similar_columns = lsh_index.query(table_name, column_name)
                for similar_column in similar_columns:
                    if similar_column != f"{table_name}.{column_name}":
                        related_tab_name, related_col_name = similar_column.split('.')
                        if related_tab_name!=table_name:
                            query_column_data = database[table_name][column_name].dropna()
                            target_column_data = database[related_tab_name][related_col_name].dropna()
                            # Calculate the equi-joinability score
                            score = calculate_equi_joinability(query_column_data, target_column_data)
                            # Store the score in the joinability index
                            joinability_index[(table_name, column_name, related_tab_name, related_col_name)] = score            
    return joinability_index


def convert_float_columns_to_int(df):
    for column in df.columns:
        if df[column].dtype == float:
            # Check if all non-NaN values in the column are integers
            if all((df[column].dropna() % 1) == 0):
                # Convert to pandas nullable integer type
                df[column] = df[column].astype('Int64')
    return df

def load_config(config_file):
    with open(config_file, 'r') as file:
        return json.load(file)

"""
    read all the database data
"""
def read_database_data(schema_path,path,query_tab_path,query_tab_name):
    r"""Process the raw files into a database."""
    old_schema_infos = json.load(open(schema_path,'r'))
    schema_infos = {}
    for key, val in old_schema_infos.items():
        if ',' in val['pkey_col']:
            schema_infos[key] = {'pkey_col':[m.strip() for m in val['pkey_col'].split(',')],'fkey_col_to_pkey_table':val['fkey_col_to_pkey_table']} 
        else:
            schema_infos[key] = {'pkey_col':val['pkey_col'].strip(),'fkey_col_to_pkey_table':val['fkey_col_to_pkey_table']}


    tab_df_infos, tab_pk_fk_infos = {}, {}
    for tab in tqdm(schema_infos.keys(), total = len(schema_infos)):
        if tab == query_tab_name: 
            df = pd.read_csv(query_tab_path)
        else:
            table_file_path = os.path.join(path, tab+'.csv')

            df = pd.read_csv(table_file_path)
        # df = self.convert_float_columns_to_int(df)
        # deal with the dataframe with multiple primary keys
        df.columns = [col.lower() for col in df.columns]
        primary_keys = schema_infos[tab]['pkey_col']
        df[primary_keys] = df[primary_keys].astype(str)

        foreign_keys = list(schema_infos[tab]['fkey_col_to_pkey_table'].keys())
        if type(primary_keys)!=str:
            df['__'.join(primary_keys)] = df[primary_keys].apply(lambda x: '__'.join(x), axis=1)
            df = df.drop(columns=[c for c in primary_keys if c not in foreign_keys])
            primary_keys = '__'.join(primary_keys)
        
        df = convert_float_columns_to_int(df)
        tab_df_infos[tab] = df

        if type(primary_keys)==str:
            tab_pk_fk_infos[tab] = list(set([primary_keys]) | set(foreign_keys))
        else:
            tab_pk_fk_infos[tab] = list(set(primary_keys) | set(foreign_keys))

    return tab_df_infos, schema_infos

def load_embeddings(file_path):
    try:
        with np.load(file_path, allow_pickle=True) as data:
            embs = {key: data[key] for key in data}
        return embs
    except FileNotFoundError:
        # If the embedding file is not found, create a default dictionary
        print(f"Embedding file {file_path} not found. Using default embeddings.")
        return {}

"""
    caulate the basic information for each path in the examples
"""
def calculate_metapath_complexity(tmp__df_ls1):
    meta_path_cnt, path_instance_cnt, path_instance_node_cnt = [], [], []   
    for example in tqdm(tmp__df_ls1,total=len(tmp__df_ls1)):
        meta_path_cnt.append(len(example.meta_paths))
        icnt = sum([len(item) for item in example.meta_path_instances])
        path_instance_cnt.append(icnt)
        path_instance_node_cnt.append(sum([sum([len(i) for i in item]) for item in example.meta_path_instances])/icnt)
    print('the total count for example ', len(tmp__df_ls1))
    print('the average meta-path count ' , np.percentile(np.array(meta_path_cnt),[25,50,75,90]))
    print('the average meta-path instance count ' , np.percentile(np.array(path_instance_cnt),[25,50,75,90]))
    print('the average node count for each meta-path instance ' , np.percentile(np.array(path_instance_node_cnt+1),[25,50,75,90]))

def trans_tab_transform(i_query_tab_df,pkey_col,fkey_col_to_pkey_table):
    query_tab_df = i_query_tab_df.copy()
    inferred_col_to_stype = infer_df_stype(query_tab_df)
    if pkey_col is not None:
        primary_keys = pkey_col
        if type(primary_keys)!=str:
            primary_keys = '__'.join(primary_keys)
        if primary_keys in inferred_col_to_stype:
            inferred_col_to_stype.pop(primary_keys)
    for fkey in fkey_col_to_pkey_table.keys():
        if fkey in inferred_col_to_stype:
            inferred_col_to_stype.pop(fkey)

    dtype_dict = {'num':[],'cat':[],'bin':[]}
    for col_, type_ in inferred_col_to_stype.items():
        dtype_dict[type_].append(col_)

    X = query_tab_df
    all_cols = [col.lower() for col in X.columns.tolist()]

    X.columns = all_cols

    bin_cols = [c.lower() for c in dtype_dict['bin']]
    cat_cols = [c.lower() for c in dtype_dict['cat']]
    num_cols = [c.lower() for c in dtype_dict['num']]

    # start processing features
    # process num
    if len(num_cols) > 0:
        for col in num_cols: 
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col].fillna(X[col].mode()[0], inplace=True)
        X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        X[cat_cols] = X[cat_cols].astype(str)

    if len(bin_cols) > 0:
        for col in bin_cols: X[col].fillna(X[col].mode()[0], inplace=True)
        X[bin_cols] = X[bin_cols].astype(str).applymap(lambda x: 1 if x.lower() in ['yes','true','1','t'] else 0).values        
        
        # if no dataset_config given, keep its original format
        # raise warning if there is not only 0/1 in the binary columns
        if (~X[bin_cols].isin([0,1])).any().any():
            raise ValueError(f'binary columns {bin_cols} contains values other than 0/1.')

    
    X = X[bin_cols + num_cols + cat_cols]

    return X

def nan_safe_equals(a, b):
    # Compares two values, treating NaN as equal to NaN
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or (a == b)
    else:
        return a == b

def tuples_are_equal(tup1, tup2):
    # Compares two tuples element-wise with nan_safe_equals
    if len(tup1) != len(tup2):
        return False
    for item1, item2 in zip(tup1, tup2):
        if not nan_safe_equals(item1[1], item2[1]):
            return False
    return True

def find_matching_key(dictionary, tup):
    # print('------',tup)
    # Searches for a tuple key in a dictionary that matches the given tuple
    for key in dictionary.keys():
        if tuples_are_equal(key, tup):
            return key
    return None



# def bfs_all_paths_with_edge_types(G, start, goal, max_depth=10, k=10):
#     queue = deque([(start, [start], 0)])  # Queue holds tuples of (current_node, path, path_length)
#     paths = PriorityQueue()  # Priority queue to store paths by length
#     visited = {start: 0}  # Dictionary to keep track of visited nodes and the shortest path length to them

#     while queue:
#         current_node, path, path_length = queue.popleft()

#         if current_node == goal:
#             # When goal is reached, add the path to the priority queue
#             paths.put((path_length, tuple(path)))  # Use tuple(path) to make it hashable
#             # If we've stored more than k paths, remove the longest one
#             if paths.qsize() > k:
#                 paths.get()
#             continue
        
#         # Check the length of the path
#         if path_length > max_depth:
#             continue

#         for next_node in G.neighbors(current_node):
#             # Check if the next node has been visited and if the path to it is not shorter
#             if next_node not in visited or path_length + 1 < visited[next_node]:
#                 visited[next_node] = path_length + 1
#                 edge_type = G.edges[current_node, next_node]['type']
#                 new_path = path + [(edge_type, next_node)]  # Include edge type in the path
#                 queue.append((next_node, new_path, path_length + 1))

#     # Extract the paths from the priority queue and return them
#     return [list(path) for _, path in paths.queue][:k]  # Convert tuples back to lists


# def bfs_all_paths_with_edge_types(G, start, goal, max_depth=10, k=10):
#     queue = deque([(start, [])])  # Initialize the queue with the start node and an empty path
#     paths = PriorityQueue()  # Priority queue to store paths by length

#     k=3
#     while queue and paths.qsize() < k:
#         current_node, path = queue.popleft()

#         # Check if the goal node is reached
#         if current_node == goal:
#             # Add the current path to the paths priority queue
#             paths.put((len(path) // 2, path))  # Length of path is half the number of elements in path list
#             continue

#         if len(path) // 2 >= max_depth:
#             continue  # Skip paths that are too long

#         for next_node in G.neighbors(current_node):
#             if next_node not in path:
#                 edge_type = G.edges[current_node, next_node]['type']
#                 # Append the edge type and next node to the path
#                 new_path = path + [edge_type, next_node]
#                 queue.append((next_node, new_path))

#     # Get up to k shortest paths, ensuring we don't have more paths than requested
#     shortest_paths = []
#     while not paths.empty() and len(shortest_paths) < k:
#         _, p = paths.get()
#         shortest_paths.append(p)

#     # Format the paths to the desired output
#     formatted_paths = []
#     for path in shortest_paths:
#         if len(path)==0:continue
#         formatted_path = [start]  # Start node
#         for i in range(0, len(path), 2):
#             formatted_path.append(path[i])  # Edge type
#             formatted_path.append(path[i + 1])  # Node
#         formatted_paths.append(formatted_path)

#     return formatted_paths

def bfs_all_paths_with_edge_types(G, start, goal, max_depth=10, k=10):
    queue = deque([(start, [])])  # Initialize the queue with the start node and an empty path
    shortest_paths = []  # List to store the shortest paths

    k=3
    while queue and len(shortest_paths) < k:
        current_node, path = queue.popleft()

        # Check if the goal node is reached
        if current_node == goal:
            shortest_paths.append(path)  # Add the current path to the shortest paths list
            continue

        if len(path) // 2 >= max_depth:
            continue  # Skip paths that are too long

        for next_node in G.neighbors(current_node):
            if next_node not in path:
                edge_type = G.edges[current_node, next_node]['type']
                # Append the edge type and next node to the path
                new_path = path + [edge_type, next_node]
                queue.append((next_node, new_path))

    # Format the paths to the desired output
    formatted_paths = []
    for path in shortest_paths:
        formatted_path = [start]  # Start node
        for i in range(0, len(path), 2):
            formatted_path.append(path[i])  # Edge type
            formatted_path.append(path[i + 1])  # Node
        formatted_paths.append(formatted_path)

    return formatted_paths


def find_rows_with_indexed_attributes(dataframe, attribute_dict):
    # Start with a mask that selects all rows
    mask = pd.Series([True] * len(dataframe))
    for key, indexed_values in attribute_dict.items():
        # We assume there's only one key-value pair in the dictionary and get the value
        actual_value = next(iter(indexed_values.values()))
        # Use the actual value for filtering
        if dataframe[key].dtypes==object:
            mask &= dataframe[key] == actual_value
        else:
            mask &= dataframe[key] == int(actual_value)
    
    return dataframe[mask].index

def set_str_index(df, column_name):
    if df[column_name].dtype != 'object' or (df[column_name].map(type) != str).any():
        df[column_name] = df[column_name].astype(str)
    return df


"""
    output:
    graph: each node is the corresponding table, each node is the attribute link pairs (pk-fk) (fk-pk) (desc-desc)
    tab_related_cols: for each table, collects the related table and attribute pairs for it
"""
def construct_table_semantic_graph(table_schema_info, table_join_info, threshold):
    G = nx.DiGraph()
    all_tab_names = table_schema_info.keys()
    # table1：[table2, column1, column2]
    tab_desc_related_cols, sim_scores = find_compatible_tables(all_tab_names, table_join_info, threshold)

    for tab1, related_key_infos in table_schema_info.items():
        primary_keys, fk_infos = related_key_infos['pkey_col'], related_key_infos['fkey_col_to_pkey_table']
        if type(primary_keys)!=str:
            primary_keys = '__'.join(primary_keys)
        for fkey, ftab in fk_infos.items():
            ftab_primary_keys = table_schema_info[ftab]['pkey_col']
            if type(ftab_primary_keys)!=str:
                G.add_edge(tab1, ftab, type=f'({fkey},{"__".join(ftab_primary_keys)})')
                G.add_edge(ftab, tab1, type=f'({"__".join(ftab_primary_keys),{fkey}})')
            else:
                G.add_edge(tab1, ftab, type=f'({fkey},{ftab_primary_keys})')
                G.add_edge(ftab, tab1, type=f'({ftab_primary_keys},{fkey})')
        # for item in tab_desc_related_cols.get(tab1,[]):
        #     table2, column1, column2 = item
        #     G.add_edge(tab1, table2, type=f'({column1},{column2})')
    return G, tab_desc_related_cols, sim_scores


"""
    build the index for all the description column pairs with their joinability
"""
# Function to calculate equi-joinability
def calculate_equi_joinability(query_column, target_column):
    query_column = query_column.dropna()
    target_column = target_column.dropna()
    
    query_set = set([str(c) for c in query_column])
    target_set = set([str(c) for c in target_column])
    intersection = query_set.intersection(target_set)
    return len(intersection) / len(query_set) if query_set else 0

# Function to process a pair of columns
def process_column_pair(pair):
    df1, table1, column1, df2, table2, column2 = pair
    joinability = calculate_equi_joinability(df1[column1], df2[column2])
    return joinability

def compute_joinability_index(database, database_pk_fks):
    # Pre-filter columns that are not primary keys
    non_pk_columns = {
        table: [col for col in database[table].columns if col not in database_pk_fks[table]]
        for table in database
    }

    # Use ThreadPoolExecutor to parallelize the computation
    joinability_index = {}
    with ThreadPoolExecutor() as executor:
        # Create a generator for all column pairs
        column_pairs_generator = (
            (database[table1], table1, col1, database[table2], table2, col2)
            for table1, table2 in combinations(database.keys(), 2)
            for col1 in non_pk_columns[table1]
            for col2 in non_pk_columns[table2]
        )

        # Submit all tasks to the executor and wrap the generator with tqdm for a progress bar
        future_to_pair = {executor.submit(process_column_pair, pair): (pair[1], pair[2], pair[4], pair[5]) for pair in column_pairs_generator}
        
        # Iterate over the as_completed iterator to preserve order and update the progress bar
        for future in tqdm(as_completed(future_to_pair), total=len(future_to_pair), desc="Computing Joinability Index"):
            pair = future_to_pair[future]
            joinability = future.result()  # This will also capture any exceptions thrown
            joinability_index[pair] = joinability

    return joinability_index



"""
    filter all the column pairs which meet the joinability is larger than the threshold
    output: table_name:{table_name', column1, column2}
"""
def find_compatible_tables(table_names, joinability_index, threshold):
    compatible_tables = {table1: set() for table1 in table_names}
    output_sim_scores = {}
    
    for key_, joinability in joinability_index.items():
        table1_name, column1, table2_name, column2 = key_
        if joinability > threshold:
            if table1_name in table_names:
                compatible_tables[table1_name].add(tuple([table2_name,column1,column2]))
                if '__'.join([table1_name,column1]) not in output_sim_scores:
                    output_sim_scores['__'.join([table1_name,column1])] = {}
                output_sim_scores['__'.join([table1_name,column1])][tuple([table2_name, column2])] = joinability
            if table2_name in table_names:
                compatible_tables[table2_name].add(tuple([table1_name,column2,column1]))
                if '__'.join([table2_name,column2]) not in output_sim_scores:
                    output_sim_scores['__'.join([table2_name,column2])] = {}
                output_sim_scores['__'.join([table2_name,column2])][tuple([table1_name, column1])] = joinability
    
    # Convert the sets to lists if necessary
    for table1 in compatible_tables:
        compatible_tables[table1] = list(compatible_tables[table1])
    
    return compatible_tables, output_sim_scores

def find_topk_joinable_pairs(output_sim_scores, table1, column1, k=3):
    # Construct the key for the given table1 and column1
    key = f"{table1}__{column1}"
    
    # Find the corresponding scores for the given table1 and column1
    if key in output_sim_scores:
        # Get all (table2, column2) pairs and their scores for the given table1 and column1
        table2_column2_scores = output_sim_scores[key]
        
        # Sort the (table2, column2) pairs based on their joinability score, in descending order
        sorted_pairs = sorted(table2_column2_scores.items(), key=lambda item: item[1], reverse=True)

        topk_table_ls = []
        for table_name, col_name in sorted_pairs:
            if table_name not in topk_table_ls:topk_table_ls.append(table_name)
            if len(topk_table_ls)==3:break
        
        # Get the top-k pairs
        # topk_pairs = sorted_pairs[:k]
        # Return the top-k pairs with their scores
        # return topk_pairs
        return topk_table_ls
    else:
        # If the table1 and column1 combination does not exist in the output_sim_scores, return an empty list
        return []
