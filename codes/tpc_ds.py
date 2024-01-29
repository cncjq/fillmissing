import os

import pandas as pd
import glob
import json
import numpy as np
from itertools import combinations
import pickle
import tqdm
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import multi_utils as mu
import pickle
import dill as dpickle
from collections import Counter
import argparse
import load_table_representation as ltr
import transtab
import warnings
warnings.filterwarnings('ignore')
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
import json
import dask.dataframe as dd

class Table:
    def __init__(self,df,pkey_col,fkey_col_to_pkey_table):
        self.df = df
        self.pkey_col = pkey_col
        self.fkey_col_to_pkey_table = fkey_col_to_pkey_table

#  some samples can be referred by the pk-fk relations, others can only be referred by the desc-desc relations
class QuerySample:
    def __init__(self, query_content, missing_val, descriptor, meta_paths, meta_path_instances,all_df_series,re_query_content,flag):
        self.query_content = query_content
        self.missing_val = missing_val
        self.descriptor = descriptor
        self.meta_paths = meta_paths
        self.meta_path_instances = meta_path_instances
        self.all_df_series = all_df_series
        self.re_query_content = re_query_content


# for database with large size, we tend to reduce its size for testing
class TPCDSDataset:
    def __init__(
        self,
        sample_size = 20,
        lsh_threshold = 0.3,
        threshold = 0.9,
        desc_meta_path_cnt = 10,
        database_name = "tpc_ds",
        mask_flag = np.nan,
        query_table_name = "store_returns",
        query_tab_path = "",
        save_folder = "",
        csv_path = "",
        schema_path = ''
    ):
       self.query_tab_name = query_table_name
       self.query_tab_path = query_tab_path
       self.path = csv_path
       self.lsh_threshold = lsh_threshold
       self.schema_path = schema_path
       self.save_folder = save_folder
       self.sample_size = sample_size
       self.database_name = database_name
       self.threshold = threshold
       self.mask_flag = mask_flag

#      the topk meta path number with less metapath  length 
       self.desc_meta_path_cnt = desc_meta_path_cnt

       if database_name=='tpc_ds':
           self.fact_tables = ['store_sales','catalog_sales','web_sales','store_returns','catalog_returns','web_returns']
    #    assert self.query_tab_name in self.fact_tables

       print('Begin to load raw data from the database')
    #    read all the tables in the database
       self.tab_df_infos, self.schema_infos = self.read_database_data()
       print('The database has been loaded...')



    def construct_all_graphs(self):
        #    construct the semantic related attributes
       print('begin to construct the column pair similarity index')
       self.col_joinability_idx = self.build_column_pair_index()
       print('The column pair similarity index build done...')
       print('begin to construct the semantic graph, nodes are tables and edges are relations')
    #    build the semantic graph according to the pk-fk and semantic relations
       self.semantic_graph, self.tab_desc_related_cols, self.sim_scores = mu.construct_table_semantic_graph(self.schema_infos, self.col_joinability_idx, self.threshold)
       print('The semantic graph construct done...')
       print('begin to construct the meta path for each attribute')
    #    construct the meta-paths for each attributes
       self.desc_meta_paths, self.desc_meta_paths_nopkfk = self.construct_meta_paths()
       print('The meta path construction done...')


    def construct_metapath_instances_pkfk_new_spark(self, query_tab_df, spark, meta_path,input_tab_df_infos):

        # Convert Pandas DataFrame to PySpark DataFrame
        query_tab_df_spark = spark.createDataFrame(query_tab_df)
        query_tab_df_spark = query_tab_df_spark.withColumn('_original_index', F.monotonically_increasing_id())

        # Pre-calculate join attributes per table for efficiency
        join_attributes_per_table = {}
        for i in range(1, len(meta_path), 2):
            join_attrs = meta_path[i][1:-1].split(',')
            for table_name, join_attr in zip(meta_path[i - 1:i + 2:2], join_attrs):
                if table_name not in join_attributes_per_table:
                    join_attributes_per_table[table_name] = set()
                join_attributes_per_table[table_name].add(join_attr)


        # Initialize partial_matches_spark with relevant columns from query table
        join_cols = list(join_attributes_per_table[self.query_tab_name]) + ['_original_index']
        partial_matches_spark = query_tab_df_spark.select(join_cols)


        for i in range(1, len(meta_path), 2):
            join_attrs = meta_path[i][1:-1].split(',')
            next_table_name = meta_path[i + 1]
            next_table_spark = input_tab_df_infos[next_table_name]

            # Perform the join using PySpark
            partial_matches_spark = partial_matches_spark.join(
                next_table_spark,
                partial_matches_spark[join_attrs[0]] == next_table_spark[join_attrs[1]],
                'inner'
            )

            # Check for excessive size after merge
            if partial_matches_spark.count() > query_tab_df_spark.count() * 20:
                print(f"Excessive size encountered: {partial_matches_spark.count()}")
                return None  # Return None to indicate early termination

            # Prepare for the next iteration if needed
            if i < len(meta_path) - 2:
                next_join_attrs = meta_path[i+2][1:-1].split(',')
                partial_matches_spark = partial_matches_spark.withColumnRenamed(next_join_attrs[0], 'next_' + next_join_attrs[0])

        # Extract the path instances
        meta_path_instances = {}
        next_table_names = [meta_path[i + 1] for i in range(1, len(meta_path), 2)]

        # # This part of the code needs careful adaptation as PySpark handles groupBy differently
        # for original_index, group in partial_matches_spark.groupBy('_original_index').agg(F.collect_list('_original_index_next_' + next_table_name for next_table_name in next_table_names)):
        #     meta_path_instance_data = []  # Build this list based on your specific requirements
        #     # ...
        #     # You will need to adapt this part to correctly extract and format the path instances
        #     meta_path_instances[original_index] = meta_path_instance_data

        # Assuming 'next_table_names' is a list of table names in the order they appear in the meta_path
        # Define a UDF to process the collected lists into the desired format
        def process_path_instance(groups):
            # Here you can implement the logic to transform the raw group data into your desired format
            # This is just a placeholder logic and should be replaced with your actual processing steps
            return json.dumps([{"table": table, "index": index} for table, index in zip(next_table_names, groups)])

        process_path_instance_udf = udf(process_path_instance, StringType())

        # Perform groupBy and aggregation
        grouped_data = partial_matches_spark.groupBy('_original_index').agg(*[F.collect_list('_original_index_next_' + table).alias(table) for table in next_table_names])

        # Apply the UDF to each grouped data
        meta_path_instances_df = grouped_data.withColumn("meta_path_instance", process_path_instance_udf(F.array(*next_table_names)))

        # Collecting the data back to the driver if the dataset is not too large
        meta_path_instances_collected = meta_path_instances_df.select('_original_index', 'meta_path_instance').collect()

        # Constructing the final meta_path_instances dictionary
        meta_path_instances = {row['_original_index']: json.loads(row['meta_path_instance']) for row in meta_path_instances_collected}


        return meta_path_instances


    #  if the metapaths can provide the pk-fk relationship, take them as priority to construct all the path instances
    def construct_metapath_instances_pkfk_new(self, query_tab_df, meta_path):
        meta_path_instances = {}
        partial_matches_size_threshold = len(query_tab_df) * 20
        # primary_keys = self.schema_infos[self.query_tab_name]['pkey_col']

        # if type(primary_keys)!=str:
        #     primary_keys = '__'.join(primary_keys)

        # Pre-calculate join attributes per table for efficiency
        join_attributes_per_table = {}
        for i in range(1, len(meta_path), 2):
            join_attrs = meta_path[i][1:-1].split(',')
            for table_name, join_attr in zip(meta_path[i-1:i+2:2], join_attrs):
                join_attributes_per_table.setdefault(table_name, set()).add(join_attr)

        # Process each table only once and efficiently
        for table_name, tab_df in self.tab_df_infos.items():
            if '_original_index' in tab_df.columns:
                tab_df.drop(columns=['_original_index'], inplace=True)
            if table_name == self.query_tab_name:
                continue
            tab_df['_original_index_next_' + table_name] = tab_df.index

        query_tab_df['_original_index'] = query_tab_df.index
        # Ensure the DataFrame index is set for efficient merging
        query_tab_df = query_tab_df.set_index(meta_path[1][1:-1].split(',')[0], drop=False)
        partial_matches = query_tab_df[list(join_attributes_per_table[self.query_tab_name])+['_original_index']].copy()


        for i in range(1, len(meta_path), 2):
            join_attrs = meta_path[i][1:-1].split(',')
            next_table_name = meta_path[i + 1]
            next_table = self.tab_df_infos[next_table_name].set_index(join_attrs[1], drop=False)
            
            if partial_matches.index.dtype != 'object':
                partial_matches.index = partial_matches.index.astype('str')

            # Do the same conversion for the 'next_table'
            if next_table.index.dtype != 'object':
                next_table.index = next_table.index.astype('str')
            # Perform the merge using indices
            partial_matches = partial_matches.merge(
                next_table[list(join_attributes_per_table[next_table_name]) + ['_original_index_next_' + next_table_name]],
                how='inner',
                left_index=True,
                right_index=True,
                suffixes=('', '_next')
            ).drop_duplicates(subset=[join_attrs[0], join_attrs[1]])
            
            

            # Check for excessive size after merge
            if len(partial_matches) > partial_matches_size_threshold:
                print(f"Excessive size encountered: {len(partial_matches)}")
                return None  # Return None to indicate early termination

            # Reset index if needed for the next iteration
            if i < len(meta_path) - 2:  # If not last iteration
                next_join_attrs = meta_path[i+2][1:-1].split(',') 
                partial_matches = partial_matches.reset_index(drop=True).set_index(next_join_attrs[0], drop=False)

        next_table_names = [meta_path[i + 1] for i in range(1, len(meta_path), 2)]


        # Create a dictionary that maps '_original_index' to a list of dictionaries, each representing a path instance
        for original_index, group in partial_matches.groupby('_original_index'):
            meta_path_instance_data = [
                [(table_, row['_original_index_next_' + table_]) for table_ in next_table_names]
                for _, row in group.iterrows()  # This is still using iterrows but on a much smaller group
            ]

            base_record = query_tab_df[query_tab_df['_original_index']==original_index].iloc[0].to_dict()
            base_record_key = tuple(base_record.items())

            meta_path_instances[base_record_key] = meta_path_instance_data

        return meta_path_instances
    
    def construct_metapath_instances_pkfk_new_dask(self, query_tab_df, meta_path, dd_dataframe, desired_number_of_partitions=10):
        query_tab_df['_original_index'] = query_tab_df.index
        query_tab_ddf = dd.from_pandas(query_tab_df, npartitions=desired_number_of_partitions)
        partial_matches_size_threshold = len(query_tab_ddf) * 20
        print(meta_path)

        # Set index for efficient merging in Dask
        # query_tab_ddf = query_tab_ddf.set_index(meta_path[1][1:-1].split(',')[0], drop=False)

        # Initialize empty dictionary for meta path instances
        meta_path_instances = {}

        # Pre-calculate join attributes per table for efficiency
        join_attributes_per_table = {}
        for i in range(1, len(meta_path), 2):
            join_attrs = meta_path[i][1:-1].split(',')
            for table_name, join_attr in zip(meta_path[i-1:i+2:2], join_attrs):
                join_attributes_per_table.setdefault(table_name, set()).add(join_attr)

        partial_matches = query_tab_ddf[['_original_index'] + list(join_attributes_per_table[self.query_tab_name])]

        # Perform the joins
        for i in range(1, len(meta_path), 2):
            join_attrs = meta_path[i][1:-1].split(',')
            next_table_name = meta_path[i + 1]
            next_table_ddf = dd_dataframe[next_table_name]
            # Ensure join attribute columns are of the same data type (string)
            join_attr_query = join_attrs[0]  # Join attribute in query_tab_ddf
            join_attr_next = join_attrs[1]  # Join attribute in next_table_ddf

            # Convert join attributes to string and set as index within each partition
            partial_matches = partial_matches.map_partitions(mu.set_str_index, join_attr_query)
            next_table_ddf = next_table_ddf.map_partitions(mu.set_str_index, join_attr_next)
            
            # print('before ', persist(partial_matches.head(1)[join_attr_query]))
            # print('before ', persist(next_table_ddf.head(1)[join_attr_next]))
            # Join with the next table in the meta path
            partial_matches = dd.merge(partial_matches,
                                    next_table_ddf[list(join_attributes_per_table[next_table_name]) +
                                    ['_original_index_next_' + next_table_name]],
                                    left_index=True,
                                    right_index=True,
                                    suffixes=('', '_next'))

            # Drop duplicates
            partial_matches = partial_matches.drop_duplicates(subset=[join_attrs[0], join_attrs[1]])

        # partial_matches = persist(partial_matches)
        # return partial_matches

        # Compute the result of joins
        partial_matches = partial_matches.compute()

        # Check for excessive size after merge
        if len(query_tab_ddf) > partial_matches_size_threshold:
            print(f"Excessive size encountered: {len(query_tab_ddf)}")
            return None  # Return None to indicate early termination

        # Efficiently build meta-path instances
        for original_index in partial_matches['_original_index'].unique():
            base_record = partial_matches.loc[original_index].to_dict()
            base_record_key = tuple(base_record.items())
            
            matching_extended_paths = partial_matches[partial_matches['_original_index'] == original_index]
            next_table_names = [meta_path[i + 1] for i in range(1, len(meta_path), 2)]

            meta_path_instance_data = []
            for _, row in matching_extended_paths.iterrows():
                meta_path_instance_data.append([(table_, row['_original_index_next_' + table_]) for table_ in next_table_names])

            meta_path_instances[base_record_key] = meta_path_instance_data
        return meta_path_instances

    
    #  if the metapaths can provide the pk-fk relationship, take them as priority to construct all the path instances
    def construct_metapath_instances_pkfk(self, query_tab_df, meta_path):
        meta_path_instances = {}

        partial_matches_size_threshold = len(query_tab_df)*20

        # Pre-calculate join attributes per table for efficiency
        join_attributes_per_table = {}
        for i in range(1, len(meta_path), 2):
            join_attrs = meta_path[i][1:-1].split(',')
            for table_name, join_attr in zip(meta_path[i-1:i+2:2], join_attrs):
                join_attributes_per_table.setdefault(table_name, set()).add(join_attr)

        # Process each table only once and efficiently
        for table_name, tab_df in self.tab_df_infos.items():
            if '_original_index' in tab_df.columns:
                tab_df.drop(columns=['_original_index'], inplace=True)
            if table_name == self.query_tab_name:
                continue
            tab_df['_original_index_next_' + table_name] = tab_df.index

        query_tab_df['_original_index'] = query_tab_df.index
        partial_matches = query_tab_df[['_original_index'] + list(join_attributes_per_table[self.query_tab_name])]

        for i in range(1, len(meta_path), 2):
            join_attrs = meta_path[i][1:-1].split(',')
            next_table_name = meta_path[i + 1]
            next_table = self.tab_df_infos[next_table_name]

            try:
                # Convert join attributes to string for consistency
                partial_matches[join_attrs[0]] = partial_matches[join_attrs[0]].astype(str)
                next_table[join_attrs[1]] = next_table[join_attrs[1]].astype(str)
            except:
                print(f"Terminating processing of meta-path {meta_path} at step {i} due to attribute {join_attrs[0],join_attrs[1]} not found.")
                return None

            # Efficiently select required columns
            next_table_columns = list(join_attributes_per_table[next_table_name]) + ['_original_index_next_' + next_table_name]
            partial_matches = partial_matches.merge(
                next_table[next_table_columns],
                how='inner',
                left_on=join_attrs[0],
                right_on=join_attrs[1],
                suffixes=('', '_next')
            ).drop_duplicates(subset=[join_attrs[0],join_attrs[1]])

            # Terminate if partial_matches becomes too large
            if len(partial_matches) > partial_matches_size_threshold:
                print(f"Terminating processing of meta-path {meta_path} at step {i} due to excessive size (size: {len(partial_matches)})")
                return None  # Return None to indicate early termination


        # Efficiently build meta-path instances
        for original_index in partial_matches['_original_index'].unique():
            base_record = query_tab_df.loc[original_index].to_dict()
            base_record_key = tuple(base_record.items())
            
            matching_extended_paths = partial_matches[partial_matches['_original_index'] == original_index]
            next_table_names = [meta_path[i + 1] for i in range(1, len(meta_path), 2)]

            meta_path_instance_data = []
            for _, row in matching_extended_paths.iterrows():
                meta_path_instance_data.append([(table_, row['_original_index_next_' + table_]) for table_ in next_table_names])

            meta_path_instances[base_record_key] = meta_path_instance_data

        return meta_path_instances


    """
        construct all the meta-path path instances
        I first treat the value I masked as nan
    """
    def construct_training_evaluation_samples(self):
        query_tab_df = self.tab_df_infos[self.query_tab_name]
        train_examples = []

        re_query_tab_df = mu.trans_tab_transform(query_tab_df,self.schema_infos[self.query_tab_name]['pkey_col'],\
                                                 self.schema_infos[self.query_tab_name]['fkey_col_to_pkey_table'])
        # trans_tab_transform(query_tab_df,pkey_col,fkey_col_to_pkey_table)

        print('begin to construct samples...')
        for desc, cur_meta_paths in tqdm(self.desc_meta_paths.items(),total=len(self.desc_meta_paths)):
            if len(cur_meta_paths)==0:continue
            if desc.lower() not in re_query_tab_df.columns:
                print("{desc} not belongs to numerical attributes or text attributes, so we skip it... ")
                continue
            # cur_meta_paths = sorted(cur_meta_paths, key=lambda x:len(x))[:self.desc_meta_path_cnt]

            # query_train_df query_tab_df the desc attribute is not nan,  also I want to sample the query_train_df based on sample_size  = 5000
            # query_evaluate_df query_tab_df the desc attribute is nan
            if self.sample_size>0:
                query_train_df = query_tab_df[query_tab_df[desc].notna()].sample(n=self.sample_size, random_state=42)
            else:
                query_train_df = query_tab_df[query_tab_df[desc].notna()]
            print(desc, ' not null sample size ', len(query_train_df))

            # Dictionary to store the content, meta-paths, and instances
            all_query_content_vals = {}
            all_query_content_meta_paths = {}
            all_query_content_instances = {}

            # Iterating through each meta-path
            for single_meta_path in tqdm(cur_meta_paths,total=len(cur_meta_paths)): 
                # construct_metapath_instances_pkfk
                all_meta_path_instances_key_val = self.construct_metapath_instances_pkfk(query_train_df, single_meta_path)

                if all_meta_path_instances_key_val is None:
                    print('1111')
                    continue
                
                # Iterating through each instance in the meta-path
                for query_content, instance_paths in all_meta_path_instances_key_val.items():
                    # Converting tuple query_content to a dictionary
                    query_content_dict = dict(query_content)
                    missing_val = query_content_dict[desc]
                    query_content_dict[desc] = self.mask_flag
                    query_tup = tuple(query_content_dict.items())

                    matching_key = mu.find_matching_key(all_query_content_vals, query_tup)

                    # If this is the first time we're seeing this query_tup
                    if matching_key is None:
                        all_query_content_vals[query_tup] = missing_val
                        all_query_content_meta_paths[query_tup] = [single_meta_path]
                        all_query_content_instances[query_tup] = [instance_paths]
                    else:
                        # Append the meta-path and corresponding instances
                        all_query_content_meta_paths[matching_key].append(single_meta_path)
                        all_query_content_instances[matching_key].append(instance_paths)
            # Construct training examples
            for query_tup, missing_val in all_query_content_vals.items():
                matching_key = mu.find_matching_key(all_query_content_meta_paths, query_tup)
                meta_paths = all_query_content_meta_paths[matching_key]
                matching_key = mu.find_matching_key(all_query_content_instances, query_tup)
                instance_paths_list = all_query_content_instances[matching_key]
                query_content_dict = dict(query_tup)
                query_content_index = query_content_dict.get('_original_index')

                # Retrieve the data series for each path instance
                all_df_series = [
                    [[self.tab_df_infos[path_item[0]].loc[path_item[1]] for path_item in path_ins] for path_ins in meta_path]
                    for meta_path in instance_paths_list
                ]

                re_query_row = re_query_tab_df.iloc[[query_content_index]].copy()
                re_query_row[desc.lower()] = self.mask_flag
                # Create the QuerySample object
                # query_content, missing_val, descriptor, meta_paths, meta_path_instances,all_df_series
                example = QuerySample(
                    query_content=pd.DataFrame([dict(query_tup)]), 
                    missing_val=missing_val, 
                    descriptor=desc, 
                    meta_paths=meta_paths, 
                    meta_path_instances=instance_paths_list,
                    all_df_series=all_df_series,  # Note that instance_paths is a list of list of dicts
                    re_query_content = re_query_row,
                    flag='pk-fk'
                )

                # Append the example to the train_examples list
                train_examples.append(example)

        #  if it cannot provide the pk-fk relationship, it just like matching two table.
        #  each meta_path only contains [('',''),table_name]
        print(f'{len(self.desc_meta_paths_nopkfk)} attributes do not have meta-paths, so we just loook for the related tables directly: ', self.desc_meta_paths_nopkfk.keys())
        for desc, cur_meta_paths in tqdm(self.desc_meta_paths_nopkfk.items(),total=len(self.desc_meta_paths_nopkfk)):
            if desc.lower() not in re_query_tab_df.columns:
                print("{desc} not belongs to numerical attributes or text attributes, so we skip it... ")
                continue

            # Count the frequency of each element
            counts = Counter(cur_meta_paths)
            # Sort elements by frequency in descending order
            cur_meta_paths = sorted(counts, key=lambda x: counts[x], reverse=True)[:self.desc_meta_path_cnt]
            # query_train_df query_tab_df the desc attribute is not nan,  also I want to sample the query_train_df based on sample_size  = 5000
            # query_evaluate_df query_tab_df the desc attribute is nan
            query_train_df = query_tab_df[query_tab_df[desc].notna()].sample(n=self.sample_size, random_state=42)
            print(desc, ' not null sample size ', len(query_train_df))

            for original_index in query_train_df.index:
                missing_val = query_train_df.iloc[original_index, desc]
                re_query_row = re_query_tab_df.loc[[original_index]].copy()
                re_query_row[desc]=self.mask_flag

                example = QuerySample(
                            query_content= query_tab_df.loc[[original_index]], 
                            missing_val=missing_val, 
                            descriptor=desc, 
                            meta_paths=cur_meta_paths, 
                            meta_path_instances= [],
                            all_df_series= [],  # Note that instance_paths is a list of list of dicts
                            re_query_content = re_query_row,
                            flag='desc-desc'
                        )
                train_examples.append(example)



        return train_examples
    
    """
        construct all the meta-path path instances
        I first treat the value I masked as nan
    """
    def construct_eval_samples(self,query_attr_related_path_dic,type_='default'):
        query_tab_df = self.tab_df_infos[self.query_tab_name]
        test_examples = []


        re_query_tab_df = mu.trans_tab_transform(query_tab_df,self.schema_infos[self.query_tab_name]['pkey_col'],\
                                                 self.schema_infos[self.query_tab_name]['fkey_col_to_pkey_table'])
        # trans_tab_transform(query_tab_df,pkey_col,fkey_col_to_pkey_table)

        if type_=='default':
            if len(query_tab_df)>50000:
                query_tab_df = query_tab_df.copy()
                query_tab_df = query_tab_df.sample(frac=0.01,random_state=10) #!!!!! to reduce our sample size to compare with the baselines grimp
                print('the table is so large, so we reduce its size to ',len(query_tab_df))

        print('begin to construct samples...')
        for desc, cur_meta_paths in tqdm(query_attr_related_path_dic.items(),total=len(query_attr_related_path_dic)):
            if len(cur_meta_paths)==0:continue
            if desc.lower() not in re_query_tab_df.columns:
                print("{desc} not belongs to numerical attributes or text attributes, so we skip it... ")
                continue

            query_test_df = query_tab_df[query_tab_df[desc].isna()]
            print(desc, ' to impute sample size ', len(query_test_df))

            # Dictionary to store the content, meta-paths, and instances
            all_query_content_meta_paths = {}
            all_query_content_instances = {}
            all_query_content_attrs = {}

            # Iterating through each meta-path
            for single_meta_path, mt_attr_infos in tqdm(cur_meta_paths.items(),total=len(cur_meta_paths)):

                all_meta_path_instances_key_val = self.construct_metapath_instances_pkfk(query_test_df, single_meta_path)


                if all_meta_path_instances_key_val is None:
                    print('1111')
                    continue
                
                # Iterating through each instance in the meta-path
                for query_content, instance_paths in all_meta_path_instances_key_val.items():
                    # Converting tuple query_content to a dictionary
                    query_content_dict = dict(query_content)
 
                    query_content_dict[desc] = self.mask_flag
                    query_tup = tuple(query_content_dict.items())

                    matching_key = mu.find_matching_key(all_query_content_meta_paths, query_tup)

                    # If this is the first time we're seeing this query_tup
                    if matching_key is None:
                        all_query_content_attrs[query_tup] = [mt_attr_infos]
                        all_query_content_meta_paths[query_tup] = [single_meta_path]
                        all_query_content_instances[query_tup] = [instance_paths]
                    else:
                        # Append the meta-path and corresponding instances
                        all_query_content_attrs[matching_key].append(mt_attr_infos)
                        all_query_content_meta_paths[matching_key].append(single_meta_path)
                        all_query_content_instances[matching_key].append(instance_paths)
            # Construct training examples
            for query_tup, meta_paths in all_query_content_meta_paths.items():
                matching_key = mu.find_matching_key(all_query_content_instances, query_tup)
                instance_paths_list = all_query_content_instances[matching_key]
                mt_attrs_list = repr(all_query_content_attrs[matching_key])
                query_content_dict = dict(query_tup)
                query_content_index = query_content_dict.get('_original_index')

                # Retrieve the data series for each path instance
                all_df_series = [
                    [[self.tab_df_infos[path_item[0]].loc[path_item[1]] for path_item in path_ins] for path_ins in meta_path]
                    for meta_path in instance_paths_list
                ]

                if type_=='default':re_query_row = re_query_tab_df.iloc[[query_content_index]].copy()
                else:re_query_row = re_query_tab_df.loc[[query_content_index]].copy()
                
                # Create the QuerySample object
                # query_content, missing_val, descriptor, meta_paths, meta_path_instances,all_df_series
                example = QuerySample(
                    query_content=pd.DataFrame([dict(query_tup)]), 
                    missing_val=mt_attrs_list, 
                    descriptor=desc, 
                    meta_paths=meta_paths, 
                    meta_path_instances=instance_paths_list,
                    all_df_series=all_df_series,  # Note that instance_paths is a list of list of dicts
                    re_query_content = re_query_row,
                    flag='pk-fk'
                )

                # Append the example to the train_examples list
                test_examples.append(example)



        if type(self.schema_infos[self.query_tab_name]['pkey_col'])==str:
            return test_examples, query_tab_df[[self.schema_infos[self.query_tab_name]['pkey_col']]]
        else:
            return test_examples, query_tab_df[['__'.join(self.schema_infos[self.query_tab_name]['pkey_col'])]]
     
    """
        construct the database test samples do that we can predict it
        dict: key val sample like desc_attr_:(metapath, attr_) 'sr_addr_sk_ca_street_number': (('store_returns',
            '(sr_item_sk,i_item_sk)',
            'item'),
            'ca_street_number')
    """
    def construct_test_samples(self,query_attr_related_path_dict=None,desired_number_of_partitions=10):
        if query_attr_related_path_dict is None:
            query_attr_related_path_dict = self.desc_meta_paths
            print('11')
        query_tab_df = self.tab_df_infos[self.query_tab_name]
        print(len(query_tab_df))
        
        test_examples = []

        re_query_tab_df = mu.trans_tab_transform(query_tab_df,self.schema_infos[self.query_tab_name]['pkey_col'],\
                                                    self.schema_infos[self.query_tab_name]['fkey_col_to_pkey_table'])
                    
        dd_dataframe = {}

        for table_name, tab_df in self.tab_df_infos.items():
            if table_name != self.query_tab_name:
                tab_df['_original_index_next_' + table_name] = tab_df.index
                dd_dataframe[table_name] = dd.from_pandas(tab_df, npartitions=desired_number_of_partitions)
            
        print('begin to convert the missing values to samples...')
        for desc, cur_path_attr_ in tqdm(query_attr_related_path_dict.items(),total=len(query_attr_related_path_dict)):
            if desc.lower() not in re_query_tab_df.columns:
                print("{desc} not belongs to numerical attributes or text attributes, so we skip it... ")
                continue
            # cur_meta_paths = sorted(cur_meta_paths, key=lambda x:len(x))[:self.desc_meta_path_cnt]
            query_test_df = query_tab_df[query_tab_df[desc].isna()]
            print(desc, ' to impute sample size ', len(query_test_df))

            # Dictionary to store the content, meta-paths, and instances
            all_query_content_attrs = {}
            all_query_content_meta_paths = {}
            all_query_content_instances = {}
            
            print(desc, cur_path_attr_, len(cur_path_attr_))
            # for single_meta_path in cur_path_attr_:
            for single_meta_path, mt_attr_infos in tqdm(cur_path_attr_.items(),total=len(cur_path_attr_)):
                print('test start ....')
                # Iterating through each meta-path
                all_meta_path_instances_key_val = self.construct_metapath_instances_pkfk_new_dask(query_test_df, single_meta_path, dd_dataframe)
                print('test end ....')

                if all_meta_path_instances_key_val is None:
                    print('1111')
                    continue
                
                # Iterating through each instance in the meta-path
                for query_content, instance_paths in all_meta_path_instances_key_val.items():
                    # Converting tuple query_content to a dictionary
                    query_content_dict = dict(query_content)
 
                    query_content_dict[desc] = self.mask_flag
                    query_tup = tuple(query_content_dict.items())

                    matching_key = mu.find_matching_key(all_query_content_meta_paths, query_tup)

                    # If this is the first time we're seeing this query_tup
                    if matching_key is None:
                        all_query_content_attrs[query_tup] = [mt_attr_infos]
                        all_query_content_meta_paths[query_tup] = [single_meta_path]
                        all_query_content_instances[query_tup] = [instance_paths]
                    else:
                        # Append the meta-path and corresponding instances
                        all_query_content_attrs[matching_key].append(mt_attr_infos)
                        all_query_content_meta_paths[matching_key].append(single_meta_path)
                        all_query_content_instances[matching_key].append(instance_paths)
            # Construct training examples
            for query_tup, meta_paths in all_query_content_meta_paths.items():
                matching_key = mu.find_matching_key(all_query_content_instances, query_tup)
                instance_paths_list = all_query_content_instances[matching_key]
                mt_attrs_list = repr(all_query_content_attrs[matching_key])
                query_content_dict = dict(query_tup)
                query_content_index = query_content_dict.get('_original_index')

                # Retrieve the data series for each path instance
                all_df_series = [
                    [[self.tab_df_infos[path_item[0]].loc[path_item[1]] for path_item in path_ins] for path_ins in meta_path]
                    for meta_path in instance_paths_list
                ]

                re_query_row = re_query_tab_df.loc[[query_content_index]].copy()
                
                # Create the QuerySample object
                # query_content, missing_val, descriptor, meta_paths, meta_path_instances,all_df_series
                example = QuerySample(
                    query_content=pd.DataFrame([dict(query_tup)]), 
                    missing_val=mt_attrs_list, 
                    descriptor=desc, 
                    meta_paths=meta_paths, 
                    meta_path_instances=instance_paths_list,
                    all_df_series=all_df_series,  # Note that instance_paths is a list of list of dicts
                    re_query_content = re_query_row,
                    flag='pk-fk'
                )

                # Append the example to the train_examples list
                test_examples.append(example)


        if type(self.schema_infos[self.query_tab_name]['pkey_col'])==str:
            return test_examples, query_tab_df[[self.schema_infos[self.query_tab_name]['pkey_col']]]
        else:
            return test_examples, query_tab_df[['__'.join(self.schema_infos[self.query_tab_name]['pkey_col'])]]


    """
        construct the database information with all the tables
    """
    def build_db(self):
        db = {}
        for table_name, table_content in self.tab_df_infos.items():
            if table_name==self.query_tab_name:continue
            db[table_name] = \
                Table(table_content,self.schema_infos[table_name]['pkey_col'],self.schema_infos[table_name]['fkey_col_to_pkey_table'])
        return db
    
    def test_build_db(self,frac=0.2):
        db = {}
        if self.database_name!='eicu':
            sample_df_infos = self.construct_database_sample_graph()
        else:
            sample_df_infos = self.tab_df_infos

        for table_name, table_content in sample_df_infos.items():
            if table_name==self.query_tab_name:continue
            db[table_name] = \
                Table(table_content,self.schema_infos[table_name]['pkey_col'],self.schema_infos[table_name]['fkey_col_to_pkey_table'])
        return db


    
    """
        for each desc attribute, collect all the meta-paths, which involves the meta-path from query table to different target table or 
        the same target table while with different number of meta-paths
    """
    def construct_meta_paths(self):
        meta_path_file = os.path.join(self.save_folder,'meta_paths', self.database_name+'.pkl' )
        if os.path.exists(meta_path_file):
            return pickle.load(open(meta_path_file,'rb'))
        # meta_path_file = os.path.join(self.save_folder,'meta_paths', self.database_name+'_old.pkl' )
        # if os.path.exists(meta_path_file):
        #     return pickle.load(open(meta_path_file,'rb'))
        desc_meta_paths = {}
        desc_target_tabs = {}
        desc_meta_paths_nopkfk = {}
        for item in self.tab_desc_related_cols[self.query_tab_name]:
            tab2, col1, col2 = item
            if col1 not in desc_target_tabs:
                desc_target_tabs[col1] = []
            desc_target_tabs[col1].append(tab2)
        for desc_attr in tqdm(self.tab_df_infos[self.query_tab_name].columns,total = len(self.tab_df_infos[self.query_tab_name].columns)):
            if desc_attr in self.tab_pk_fk_infos[self.query_tab_name]:continue
            cur_paths = set()
            for goal_tab in set(desc_target_tabs.get(desc_attr,[])):#(G, start, goal, max_depth=10, k=10):
                [cur_paths.add(tuple(p)) for p in mu.bfs_all_paths_with_edge_types(self.semantic_graph, self.query_tab_name, goal_tab, self.desc_meta_path_cnt)]
            
            if len(cur_paths)>0:
                # desc_meta_paths[desc_attr] = list(cur_paths)
                desc_meta_paths[desc_attr] = list(sorted(cur_paths,key=lambda x:len(x)))[:self.desc_meta_path_cnt]


                # output_cur_meta_paths = []
                # for i in range(len(desc_meta_paths[desc_attr])):
                #     cp = desc_meta_paths[desc_attr][i]
                #     ocp = [cp[0][0]]
                #     for j in range(1,len(cp)):
                #         ocp.append(cp[j][0])
                #         ocp.append(cp[j][1])
                #     output_cur_meta_paths.append(ocp)

                # desc_meta_paths[desc_attr] = output_cur_meta_paths
            else:
                # no pk-fk relationships we just match them with the related tables
                # each desc attribute have the correpsonding attributes, we sort them using the matched attribute count
                related_tabs = []
                for goal_tab in set(desc_target_tabs.get(desc_attr,[])):
                    related_tabs.append([('',''),goal_tab])
                if len(related_tabs)>0:
                    desc_meta_paths_nopkfk[desc_attr] = related_tabs  
                # for item in self.tab_desc_related_cols.get(self.query_tab_name,[]):
                #     table2, column1, column2 = item
                #     if column1==desc_attr:continue
                #     for topk in mu.find_topk_joinable_pairs(self.sim_scores, self.query_tab_name, column1, k=3):
                #         table2, column2 = topk[0]
                #         cur_paths.add(tuple([(self.query_tab_name,None),(f'({column1},{column2})',table2)]))
            

            
        pickle.dump((desc_meta_paths, desc_meta_paths_nopkfk),open(meta_path_file,'wb'))
        return desc_meta_paths,desc_meta_paths_nopkfk
            
    """
        construct the joinability for the desc attribute pairs in the database
    """
    def build_column_pair_index(self):
        
        index_dir = os.path.join(self.save_folder, 'col_index')
        index_file = os.path.join(index_dir, self.database_name+'.pkl')
        if not os.path.exists(index_dir):
            os.mkdir(index_dir)
        if not os.path.exists(index_file):
            if args.sample_size>0:
                sample_df_infos = self.construct_database_sample_graph()
            else:
                sample_df_infos = self.tab_df_infos
            # col_joinability_idx = mu.compute_joinability_index(sample_df_infos,self.tab_pk_fk_infos,self.threshold)
            col_joinability_idx = mu.compute_joinability_index_lsh(sample_df_infos,self.tab_pk_fk_infos,self.lsh_threshold)
                
            pickle.dump(col_joinability_idx, open(index_file,'wb'))
        else:
            col_joinability_idx = pickle.load(open(index_file,'rb'))
            
        return col_joinability_idx
    
    
    """
        for each table, construct the self mutual graph, the entities should be (table, cell value, column, row index), 
        the relations should be row
    """
    def construct_row_graph(self, tab_name, df, pk_cols, fk_cols):
        triples = []
        entities = set()

        desc_cols = [col for col in df.columns if col not in pk_cols and col not in fk_cols]

        assert isinstance(pk_cols, str)

        df_length = len(df)
    
        pk_values = df[pk_cols].values
        fk_values = {f_col: df[f_col].values for f_col in fk_cols}

        # for each primary key, link the primary key and desc attrs
        for d_col in desc_cols:
            d_values = df[d_col].values
            for idx in range(df_length):
                if pd.notna(pk_values[idx]) and pd.notna(d_values[idx]):
                    val1_ent = (tab_name, pk_cols, str(pk_values[idx]))
                    val2_ent = (tab_name, d_col, str(d_values[idx]))

                    entities.update([val1_ent, val2_ent])

                    triples.append((val1_ent, 'pk_desc', val2_ent))
                    triples.append((val2_ent, 'desc_pk', val1_ent))

        # for each foreign key, link other foreign key and desc attrs
        for f_col in fk_cols:
            f_values = fk_values[f_col]
            for d_col in desc_cols:
                d_values = df[d_col].values
                for idx in range(df_length):
                    if pd.notna(f_values[idx]) and pd.notna(d_values[idx]):
                        val1_ent = (tab_name, f_col, str(f_values[idx]))
                        val2_ent = (tab_name, d_col, str(d_values[idx]))

                        entities.update([val1_ent, val2_ent])

                        triples.append((val1_ent, 'fk_desc', val2_ent))
                        triples.append((val2_ent, 'desc_fk', val1_ent))

        return triples, list(entities)

    def convert_float_columns_to_int(self,df):
        for column in df.columns:
            if df[column].dtype == float:
                # Check if all non-NaN values in the column are integers
                if all((df[column].dropna() % 1) == 0):
                    # Convert to pandas nullable integer type
                    df[column] = df[column].astype('Int64')
        return df


    """
        read all the database data
    """
    def read_database_data(self):
        r"""Process the raw files into a database."""
        old_schema_infos = json.load(open(self.schema_path,'r'))
        schema_infos = {}
        for key, val in old_schema_infos.items():
            if ',' in val['pkey_col']:
                schema_infos[key] = {'pkey_col':[m.strip() for m in val['pkey_col'].split(',')],'fkey_col_to_pkey_table':val['fkey_col_to_pkey_table']} 
            else:
                schema_infos[key] = {'pkey_col':val['pkey_col'].strip(),'fkey_col_to_pkey_table':val['fkey_col_to_pkey_table']}


        tab_df_infos, tab_pk_fk_infos = {}, {}
        for tab in tqdm(schema_infos.keys(), total = len(schema_infos)):
            # if tab != self.query_tab_name:continue
            # if tab == 'inventory':continue
            # if tab != 'date_dim':continue
            if tab == self.query_tab_name: 
                df = pd.read_csv(self.query_tab_path)
            else:
                table_file_path = os.path.join(self.path, tab+'.csv')
                # if not os.path.isfile(table_file_path):
                #     table_file_path = os.path.join(self.raw_csv_path, tab+'.csv')

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
            
            df = self.convert_float_columns_to_int(df)
            tab_df_infos[tab] = df

            if type(primary_keys)==str:
                tab_pk_fk_infos[tab] = list(set([primary_keys]) | set(foreign_keys))
            else:
                tab_pk_fk_infos[tab] = list(set(primary_keys) | set(foreign_keys))

            # break

        self.schema_infos = schema_infos
        self.tab_pk_fk_infos = tab_pk_fk_infos
        # self.dimension_relations = {table: {val_:key_ for key_,val_ in info.items()} for table, info in schema_infos.items() if tab not in self.fact_tables and tab != 'inventory'}
        return tab_df_infos, schema_infos

    """
        since the database size is so large, 
        so I tend to sample some rows from query table and build its graph
    """
    def construct_database_sample_graph_low(self):
        print('begin to load raw data in database')
        self.tab_df_infos = self.read_database_data()
        print('the database has been loading done...')

        basic_query_df = self.tab_df_infos[self.query_tab_name].sample(n=self.sample_size,random_state=10)
        basic_query_pks = self.schema_infos[self.query_tab_name]['pkey_col'] 
        if type(basic_query_pks)!=str:basic_query_pks = '__'.join(basic_query_pks)
        basic_query_ents =  {(self.query_tab_name, basic_query_pks, str(val), 'pk') for val in basic_query_df[basic_query_pks].tolist()}

        filtered_database_df = {self.query_tab_name:basic_query_df}

        # build the pk-fk relations for construct such sample graph
        all_entity_rel_triples = []
        # for table_name, table_meta_info in tqdm(self.schema_infos.items(), total = len(self.schema_infos)):
        for table_name in tqdm(self.tab_df_infos.keys(), total = len(self.tab_df_infos)):
            table_meta_info = self.schema_infos[table_name]
            df = self.tab_df_infos[table_name]

            query_pk_cols = table_meta_info['pkey_col']
            if type(query_pk_cols)!=str:query_pk_cols = '__'.join(query_pk_cols)
            primary_entities = {(table_name, query_pk_cols, str(val), 'pk') for val in df[query_pk_cols].tolist()}

            for fk_key, fk_tab in table_meta_info['fkey_col_to_pkey_table'].items():
                fk_values = df[fk_key].to_list()   
                # Create entity pairs
                left_entities = {(table_name, fk_key, str(val), 'fk') for val in fk_values}
                
                for left_ent, right_ent in zip(primary_entities, left_entities):
                    all_entity_rel_triples.append((left_ent, right_ent))
                    all_entity_rel_triples.append((right_ent, left_ent))

                related_tab_col = self.schema_infos[fk_tab]['pkey_col']
                if type(related_tab_col)!=str:
                    related_tab_col = '__'.join(related_tab_col)

                right_entities = {(fk_tab, related_tab_col, str(val), 'pk') for val in fk_values}

                # Create and append the relationship triples
                for left_entity, right_entity in zip(left_entities, right_entities):
                    all_entity_rel_triples.append((left_entity, right_entity))
                    all_entity_rel_triples.append((right_entity, left_entity))
        print('begin to construct our graph with edges undirected ',len(all_entity_rel_triples))
        G = nx.DiGraph()
        G.add_edges_from(all_entity_rel_triples)
        print('graph has been constructed done...')

        print('begin to filter the valid primary key for each table...')
        table_pk_vals = defaultdict(lambda: defaultdict(list))

        for node in tqdm(basic_query_ents, total=len(basic_query_ents)):
            connected_nodes_dfs = set(nx.dfs_preorder_nodes(G, node))
            for item in connected_nodes_dfs:
                tab_name, pk_name, val, type_ = item
                if type_ == 'pk' and val not in table_pk_vals[tab_name][pk_name]:
                    table_pk_vals[tab_name][pk_name].append(val)


        
        for table, pk_vals in table_pk_vals.items():
            p_key, vals = pk_vals
            cur_df = self.tab_df_infos[table]
            filtered_database_df[table] = cur_df[cur_df[p_key].isin(vals)]
        
        return filtered_database_df

    """
        For this kind of sample, I will find all the pks for dimension tables based on the fact tables.
    """
    def construct_database_sample_graph_mine(self):
        print('begin to load raw data in database')
        self.tab_df_infos = self.read_database_data()
        print('the database has been loading done...')

        filtered_database_df = {}
        tab_filtered_keys = {}
        for fact_table in tqdm(self.fact_tables, total = len(self.fact_tables)):
            sample_fact_table_df = self.tab_df_infos[fact_table].sample(n=self.sample_size,random_state=10)
            filtered_database_df[fact_table] = sample_fact_table_df

            for f_key, f_tab in self.schema_infos[fact_table]['fkey_col_to_pkey_table'].items():
                if f_tab not in tab_filtered_keys:
                    tab_filtered_keys[f_tab] = set()
                try:
                    unique_vals = sample_fact_table_df[f_key].astype(int).unique()
                except:
                    unique_vals = sample_fact_table_df[f_key].unique()
                    print(f_key, unique_vals[0])
                tab_filtered_keys[f_tab].update(unique_vals)

        # after getting the fk values for these dimension tables, 
        print('begin to filter the dimension tables....')
        single_table = []
        for dimension_table in self.tab_df_infos.keys():
            if dimension_table not in self.fact_tables and dimension_table in tab_filtered_keys:
                tab_keys = tab_filtered_keys[dimension_table]
                tab_pk = self.schema_infos[dimension_table]['pkey_col']
                assert type(tab_pk)==str
                tab_df = self.tab_df_infos[dimension_table]
                try:
                    tab_df[tab_pk] = tab_df[tab_pk].astype(int)
                except:
                    print(dimension_table, ' primary key ', tab_pk, ' not int type')
                dimension_df = tab_df[tab_df[tab_pk].isin(tab_keys)]
                print(dimension_table, len(dimension_df))

                filtered_database_df[dimension_table] = dimension_df


            elif dimension_table not in self.fact_tables:
                single_table.append(dimension_table)
        
        print('begin to filter the other tables....')
        for each_tab in single_table:
            print(each_tab,' not reflected with fact tables')
            each_tab_ls = []
            for tab, item in self.schema_infos.items():
                for fk_key,fk_table in item['fkey_col_to_pkey_table'].items():
                    if fk_table==each_tab and tab in filtered_database_df:
                        print(tab, fk_key, fk_table)
                        tab_df = self.tab_df_infos[each_tab]
                        tab_pkeys = self.schema_infos[each_tab]['pkey_col']

                        try:
                            tab_df[tab_pkeys] = tab_df[tab_pkeys].astype(int)
                        except:
                            print(each_tab, ' primary key ', tab_pk, ' not int type')

                        try:
                            filtered_database_df[tab][fk_key] = filtered_database_df[tab][fk_key].astype(int)
                            unique_vals = filtered_database_df[tab][fk_key].unique()
                        except:
                            unique_vals = filtered_database_df[tab][fk_key].unique()

                        each_tab_ls.append(tab_df[tab_df[tab_pkeys].isin(unique_vals)])

            filtered_database_df[each_tab] = pd.concat(each_tab_ls, axis=0).drop_duplicates()
            print(each_tab, '.....', len(filtered_database_df[each_tab]))

        return filtered_database_df

    def process_table(self, table_name, tab_filtered_keys, filtered_database_df, processed_tables, pending_tables):
        tab_df = self.tab_df_infos[table_name]
        primary_keys = self.schema_infos[table_name]['pkey_col']
        if isinstance(primary_keys, list):
            primary_keys = '__'.join(primary_keys)

        if table_name in tab_filtered_keys and (table_name not in processed_tables or tab_filtered_keys[table_name] != processed_tables[table_name]):
            keys = tab_filtered_keys[table_name]
            new_filtered_df = tab_df[tab_df[primary_keys].isin([str(k) for k in keys])] if tab_df[primary_keys].dtype == object else tab_df[tab_df[primary_keys].isin(keys)]
        
            if table_name in filtered_database_df:
                existing_df = filtered_database_df[table_name]
                combined_df = pd.concat([existing_df, new_filtered_df]).drop_duplicates()
                filtered_database_df[table_name] = combined_df
            else:
                filtered_database_df[table_name] = new_filtered_df
        
            processed_tables[table_name] = tab_filtered_keys[table_name].copy()

            for fkey, related_table in self.schema_infos[table_name].get('fkey_col_to_pkey_table', {}).items():
                related_keys = set(filtered_database_df[table_name][fkey].dropna().unique())
                if related_table in tab_filtered_keys:
                    if not related_keys.issubset(tab_filtered_keys[related_table]):
                        tab_filtered_keys[related_table].update(related_keys)
                        if related_table in processed_tables:
                            pending_tables.add(related_table)
                else:
                    tab_filtered_keys[related_table] = related_keys
                    pending_tables.add(related_table)

    def construct_database_sample_graph(self,frac=0.2):
        filtered_database_df = {}
        tab_filtered_keys = {}
        processed_tables = {}

        for fact_table in tqdm(self.fact_tables, total=len(self.fact_tables)):
            fact_df = self.tab_df_infos[fact_table]
            sampled_fact_df = fact_df.sample(frac = frac, random_state=10)
            filtered_database_df[fact_table] = sampled_fact_df

            for f_key, d_table in self.schema_infos[fact_table]['fkey_col_to_pkey_table'].items():
                new_keys = set(sampled_fact_df[f_key].dropna().unique())
                if d_table in tab_filtered_keys:
                    tab_filtered_keys[d_table].update(new_keys)
                else:
                    tab_filtered_keys[d_table] = new_keys

        changes_made = True
        while changes_made:
            changes_made = False
            pending_tables = set(self.tab_df_infos.keys()) - set(self.fact_tables)

            while pending_tables:
                next_table = pending_tables.pop()
                keys_before = processed_tables.get(next_table, set())
                self.process_table(next_table, tab_filtered_keys, filtered_database_df, processed_tables, pending_tables)

                if processed_tables.get(next_table, set()) != keys_before:
                    changes_made = True

        print('Database sample construction complete.')
        return filtered_database_df
    
    def filter_valid_cols(self, tab_col_unique_vals, query_tab_name, query_desc_cols):
        valid_cols = []
        for d_col in tqdm(query_desc_cols, total = len(query_desc_cols)):
            set_d_col_values = tab_col_unique_vals[query_tab_name][d_col]
            match_found = False  # Flag to track if a match is found
            match_tab_col = None

            for tab, col_vals in tab_col_unique_vals.items():
                if tab == query_tab_name:
                    continue
                for col_, val_ in col_vals.items():
                    # if set_d_col_values.issubset(val_):
                    if self.jaccard_similarity(set_d_col_values, val_)>0.8:
                        match_found = True
                        match_tab_col = (tab, col_)
                        break  # Break out of the inner loop as match is found

                if match_found:
                    valid_cols.append(d_col)
                    break  # Break out of the outer loop as match is found

        return valid_cols
    
    """
     build the index for finding the entities
    """
    def jaccard_similarity(self,set1, set2):
        # Calculate the intersection and union of the sets
        intersection = set1.intersection(set2)
        union = set1.union(set2)

        # Calculate the Jaccard similarity
        jaccard_sim = len(intersection) / len(union) if union else 1  # Handle case when both sets are empty
        return jaccard_sim


    
    # given the imcomplete table as query, make the graph
    def construct_entity_graph(self):
        r"""Process the raw files into a database."""

        tab_df_infos = self.construct_database_sample_graph()
        
        entity_rel_triples, entities = [], []
        tab_col_unique_vals = {}
        # for tab in tqdm(schema_infos.keys(), total = len(schema_infos)):
        for tab in tqdm(tab_df_infos.keys(), total = len(tab_df_infos)):
            primary_keys = self.schema_infos[tab]['pkey_col']
            if type(primary_keys)!=str:
                primary_keys = '__'.join(primary_keys)

            foreign_keys = list(self.schema_infos[tab]['fkey_col_to_pkey_table'].keys())
            df = tab_df_infos[tab]
            # first construct the structure information for the cell value in each row
            row_facts, row_entities = self.construct_row_graph(tab, df, primary_keys, foreign_keys)
            tab_col_unique_vals[tab] = {col:set(df[col].unique()) for col in df.columns if col not in primary_keys and col not in foreign_keys}

            entity_rel_triples.extend(row_facts)
            entities.extend(row_entities)

            if tab == self.query_tab_name:
                query_desc_cols = [col for col in df.columns if col not in primary_keys and col not in foreign_keys]
                query_entities = row_entities

        self.entities = entities

        # for each desc cols in the query table, find if it exists any column values for it in other tables
        query_valid_query_cols = self.filter_valid_cols(tab_col_unique_vals, self.query_tab_name, query_desc_cols)
        print('the valid columns for the query is ', query_valid_query_cols)
        self.valid_queries = [ent_ for ent_ in query_entities if ent_[1] in query_valid_query_cols]

        print('table reading done ...', len(tab_df_infos))
        print('the number of the entities ',len(entities))
        print(len(query_entities))

        for tab, tab_df in tab_df_infos.items():
            print(tab, len(tab_df))


        # print(tab_df_infos['household_demographics'])
        # print([ent for ent in entities if ent[0]=='household_demographics'])

        for table_name in tqdm(tab_df_infos.keys(), total=len(tab_df_infos)):
            table_meta_info = self.schema_infos[table_name]
            df = tab_df_infos[table_name]

            for fk_key, fk_tab in table_meta_info['fkey_col_to_pkey_table'].items():
                # Use pandas operations to handle data efficiently
                fk_values = df[fk_key].dropna().astype(int)

                related_tab_col = self.schema_infos[fk_tab]['pkey_col']
                if not isinstance(related_tab_col, str):
                    related_tab_col = '__'.join(related_tab_col)

                # Vectorized operations to create paired_entities
                left_entities = pd.DataFrame({'table': [table_name]*len(fk_values), 'key': [fk_key]*len(fk_values), 'value': fk_values.astype(str)})
                right_entities = pd.DataFrame({'table': [fk_tab]*len(fk_values), 'key': [related_tab_col]*len(fk_values), 'value': fk_values.astype(str)})
                paired_entities = pd.concat([left_entities, right_entities], axis=1).to_numpy()

                # Efficient filtering
                entities_set = set(self.entities)
                valid_pairs = [(tuple(pair[:3]),tuple(pair[3:])) for pair in paired_entities if tuple(pair[:3]) in entities_set and tuple(pair[3:]) in entities_set]

                # Append relationship triples
                for left_entity, right_entity in valid_pairs:
                    entity_rel_triples.append((left_entity, 'fk_pk', right_entity))
                    entity_rel_triples.append((right_entity, 'pk_fk', left_entity))

                print(f"{table_name}, {fk_key}, {fk_tab}, done...")

        self.facts = entity_rel_triples

    def preprocessing(self):
        self.construct_entity_graph()

        assert len(self.entities)>0
        assert len(self.facts)>0

        ent=set(self.entities)
        rel=set(['fk_pk','pk_fk','pk_desc','desc_pk','fk_desc','desc_fk'])
        ent2type = {}
        for e in ent:
            ent2type[e] = (e[0],e[1])
        facts = self.facts

        assert len(ent)==len(ent2type)
        ty=set([])
        for typelist in ent2type.values():
            for t in typelist:
                ty.add(t)
        savefolder = self.save_folder
        if not os.path.exists(savefolder):
            os.mkdir(savefolder)

        with open(savefolder+'/entity2id.pkl','wb') as fin:
            entity2id={v:k for k,v in enumerate(ent)}
            pickle.dump(entity2id,fin)
        with open(savefolder+'/relation2id.pkl','wb') as fin:
            relation2id={v:k for k,v in enumerate(rel)}
            pickle.dump(relation2id,fin)
        with open(savefolder+'/type2id.pkl','wb') as fin:
            type2id={v:k for k,v in enumerate(ty)}
            pickle.dump(type2id,fin)
        with open(savefolder+'/ent2type.pkl','wb') as fin:
            pickle.dump(ent2type,fin)
        with open(savefolder+'/graph.pkl','wb') as fin:
            facts=[[i[0],i[2],i[1]] for i in facts]
            pickle.dump(facts,fin)
        with open(savefolder+'/queries.pkl','wb') as fin:
            pickle.dump(self.valid_queries,fin)

    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Prepare for the data....')
    # parser.add_argument('--sample_size', type=int, default=800, help='If u want to sample the database, use this parameter')
    # parser.add_argument('--lsh_threshold', type=float, default=0.3, help='Used for filtering the related attribute pairs')
    # parser.add_argument('--threshold', type=float, default=0.9, help='Used for filtering the related attribute pairs')
    # parser.add_argument('--desc_meta_path_cnt', type=int, default=10, help='The maxmimum size for the meta path count')
    # parser.add_argument('--query_tab_name', type=str, required=True, help='Table name for the incomplete table')
    # parser.add_argument('--database_name', type=str, required=True, help='Database name')
    # parser.add_argument('--query_tab_path', type=str, required=True, help='The query table file path which inject with missing values')
    # parser.add_argument('--save_folder', type=str, required=True, help='The directory to save our processed data')
    # parser.add_argument('--path', type=str, required=True, help='The path which stored all the tables for the database')
    # parser.add_argument('--schema_path', type=str, required=True, help='The schema path for the database')

    # parser.add_argument('--transtab_path', type=str, required=True, help='The path to save for checkpoint of pretrained embeddings')
    # parser.add_argument('--embedding_file_path', type=str, required=True, help='The path to save for the embeddings')
    parser.add_argument('--config_file_path', type=str, required=True, help='The parameter config for each database.')
    args = parser.parse_args()

    config = mu.load_config(args.config_file_path)

    # Update the default values with the config file if they exist
    for key, value in config.items():
        setattr(args, key, value) 

    test = TPCDSDataset(sample_size=args.sample_size,lsh_threshold=args.lsh_threshold,threshold=args.threshold,desc_meta_path_cnt=args.desc_meta_path_cnt,\
                    database_name=args.database_name,mask_flag = np.nan,\
                    query_table_name=args.query_tab_name,query_tab_path=args.query_tab_path,save_folder=args.save_folder,\
                    csv_path=args.path,schema_path=args.schema_path)


    training_sample_filepath = args.save_folder+'/train_samples/'+args.database_name+'_'+str(args.sample_size)+'.pkl'
    if not os.path.exists(training_sample_filepath):
        print('begin to prepare everything for the graph construction ...')
        test.construct_all_graphs()
        print('begin to construct the training examples ...')
        train_examples = test.construct_training_evaluation_samples()
        print(f'{len(train_examples)} example construction finished ...')
        dpickle.dump(train_examples,open(training_sample_filepath,'wb'))

    if not os.path.exists(args.transtab_path):
        print('begin to training the model for the database')
        ltr.train_embedding_model(test, args.transtab_path)
    if not os.path.exists(args.embedding_file_path):
        print('begin to save the embedding for the database except the query table')
        ltr.load_save_embedding_bycheckpoint_path(test, args.transtab_path, args.embedding_file_path+'')