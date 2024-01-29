import pandas as pd
import glob
import json
import numpy as np
from itertools import combinations
import pickle
import tqdm
from tqdm import tqdm
from collections import defaultdict
import glob
import os
import argparse

# for database with large size, we tend to reduce its size for testing
class TPCDSRedundant:
    def __init__(
        self,
        basic_data_path = "",
        save_data_path = "",
        schema_path = '',
        related_tab_nums =5,
        unique_val_cnt = 5,
        notnull_ratio = 0.8
    ):
        self.basic_data_path = basic_data_path
        self.schema_path = schema_path
        self.schema_infos = json.load(open(self.schema_path,'r'))
        self.related_tab_nums = related_tab_nums
        self.unique_val_cnt = unique_val_cnt
        self.not_null_ratio = notnull_ratio
        self.save_data_path = save_data_path
        if not os.path.exists(save_data_path):
            os.mkdir(save_data_path)

        self.init_database()
        
    def init_database(self):
        self.database_df = {}
        for csv_f in glob.glob(self.basic_data_path+'/*.csv'):
            csv_name = csv_f.split('/')[-1][:-4]
            self.database_df[csv_name] = pd.read_csv(csv_f)

        tab_parent_lvls, tab_parent_edges = {}, []
        tab_desc_dict = {}
        for tab, tab_meta_infos in self.schema_infos.items():
            if ',' in tab_meta_infos['pkey_col']:
                tab_desc_dict[tab] = {'p_col':[m.strip() for m in tab_meta_infos['pkey_col'].split(',')],'f_col':tab_meta_infos['fkey_col_to_pkey_table'].keys()} 
            else:
                tab_desc_dict[tab] = {'p_col':[tab_meta_infos['pkey_col'].strip()],'f_col':tab_meta_infos['fkey_col_to_pkey_table'].keys()} 

            for f_key, f_tab in tab_meta_infos['fkey_col_to_pkey_table'].items():
                if f_tab not in tab_parent_lvls:
                    tab_parent_lvls[f_tab] = []
                if tab not in tab_parent_lvls[f_tab]: tab_parent_lvls[f_tab].append(tab)
                if (tab, f_tab) not in tab_parent_edges:tab_parent_edges.append((tab, f_tab))
        
        overlap_tabs = [tab for tab, parent_tabs in tab_parent_lvls.items() if len(parent_tabs)>self.related_tab_nums]

        over_tab_dict = {}
        
        for f in overlap_tabs:
            df1 = self.database_df[f]
            cur_primary_keys = self.schema_infos[f]['pkey_col']
            if type(cur_primary_keys)==str:
                cur_primary_keys = [cur_primary_keys]
            full_cols = []
            cur_foreign_keys = list(self.schema_infos[f]['fkey_col_to_pkey_table'])
            full_cols.extend(cur_primary_keys)
            for col in df1.columns:
                if col not in cur_primary_keys and col not in cur_foreign_keys and len(df1[col].unique())>self.unique_val_cnt and  df1[col].dtype in [str,int,float,object] and df1[col].count()/len(df1)>self.not_null_ratio:
                    full_cols.append(col)

            over_tab_dict[f] = df1.dropna(subset=full_cols)[full_cols]
        self.reduant_data = over_tab_dict
        self.tab_desc_dict = tab_desc_dict

    """
        for each table has raw data stored in self.database_df
        for each reduant table stored in self.reduant_data
        then for each table, I will find the fk linked reduant table for it and then merge them 
    """
    def update_tables_bypkfk(self):
        updated_dict = {} 
        for tab, tab_meta_infos in tqdm(self.schema_infos.items(), total = len(self.schema_infos)):
            cur_df = self.database_df[tab]
            raw_cnt = len(cur_df)
            for f_key, f_tab in tab_meta_infos['fkey_col_to_pkey_table'].items():
                if f_tab not in self.reduant_data:continue
                cur_df1 = self.reduant_data[f_tab]
                cur_df1 = cur_df1.rename(columns={k:f_key+'_'+k for k in cur_df1.columns if k not in self.tab_desc_dict[f_tab]['p_col'] and k not in self.tab_desc_dict[f_tab]['f_col']})
                cur_df = pd.merge(cur_df, cur_df1, left_on=f_key, right_on=self.schema_infos[f_tab]['pkey_col'], how='left')
                cur_df = cur_df.drop(columns=[self.schema_infos[f_tab]['pkey_col']])
            updated_dict[tab] = cur_df
            assert len(cur_df)==raw_cnt

            cur_df.to_csv(os.path.join(self.save_data_path,tab+'.csv'),index=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Prepare for the data....')
    parser.add_argument('--basic_data_path', type=str, required=True, help='The path for raw dataset without redundant data')
    parser.add_argument('--save_data_path', type=str, required=True, help='The path for our processed injected dataset with redundant data')
    parser.add_argument('--schema_path', type=str, required=True, help='The path for saving the database schema')
    
    args = parser.parse_args()

    tr = TPCDSRedundant(args.basic_data_path, args.save_data_path, args.schema_path)
    tr.update_tables_bypkfk()
        
            

