# train.py
import argparse
import dill as cpickle
import pickle
import os
import numpy as np
from tpc_ds import TPCDSDataset
import multi_utils as mu
from scipy.spatial.distance import jensenshannon
import tqdm
from tqdm import tqdm
from datetime import datetime
import math
from math import sqrt
import pandas as pd
import numpy as np
import scipy.stats
from sklearn import preprocessing
from load_table_representation import *
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error,mean_squared_error,mean_absolute_error
import glob
import pytrec_eval
import collections
from collections import Counter

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


def list_to_prob_dist(lst):
    series = pd.Series(lst)
    return series.value_counts(normalize=True)

def calculate_q_error(true_vals, pred_vals, epsilon=1e-1):
    all_errors = []
    for estimated, actual in zip(pred_vals, true_vals):
        if estimated*actual<0:continue
        # Handling cases with zeros
        if estimated == 0:
            estimated = epsilon if actual>0 else -epsilon
        if actual == 0:
            actual =  epsilon if estimated>0 else -epsilon
            
        cur_error = max(estimated / actual, actual / estimated)
        all_errors.append(cur_error)
    if len(all_errors)>0:
        return np.quantile([round(r,4) for r in all_errors],[0.25,0.5,0.75,0.9]), len(all_errors)
    else:
        return [np.nan,np.nan,np.nan,np.nan],0

def evaluate_accuracy(true_vals, pred_vals):
    accuracy = np.mean(np.array([1 if compare_strings_as_numbers(t,p) else 0 for t,p in zip(true_vals,pred_vals)]))
    try:
        mae = mean_absolute_error(true_vals, pred_vals)
        rmse = sqrt(mean_squared_error(true_vals, pred_vals))

        mape_true_vals,mape_pred_vals = [],[]
        for a_, b_ in zip(mape_true_vals, mape_pred_vals):
            if a_==0:continue
            mape_true_vals.append(a_)
            mape_pred_vals.append(b_)
        mape = np.mean(np.abs((np.array(true_vals) - np.array(pred_vals)) / (np.array(true_vals) ))) * 100


        dist_a, bins = np.histogram(true_vals, bins=20, density=True)
        dist_b, _ = np.histogram(pred_vals, bins=bins, density=True)

        # Ensure histograms are normalized (sum to 1)
        dist_a /= dist_a.sum()
        dist_b /= dist_b.sum()

        # Calculate JSD
        jsd = jensenshannon(dist_a, dist_b)

        qerror = calculate_q_error(true_vals, pred_vals)


        return  round(rmse,4), round(mae,4), round(mape,4), qerror, round(accuracy,4)
    except:
        return np.nan, np.nan, np.nan, [(np.nan, np.nan, np.nan, np.nan),0], round(accuracy,4)
    

def evaluate_topk_res(attr_dict, query_truth_dict1,query_pred_dict1, initialized_desc):
    eval_metrics = {'recip_rank', 'ndcg_cut_1', 'ndcg_cut_3', 'ndcg_cut_5', 'success_1'}


    common_keys = set(query_truth_dict1.keys()) & set(query_pred_dict1.keys())
    query_truth_dict1 = {k: query_truth_dict1[k] for k in common_keys}
    query_pred_dict1 = {k: query_pred_dict1[k] for k in common_keys}

    query_truth_dict, query_pred_dict = {}, {} 
    for key,val in query_truth_dict1.items():
        attr_ = key.split('__')[-1]
        if attr_ not in initialized_desc:continue
        query_truth_dict[key] = val
        # assert query_pred_dict1[key] is not None
        query_pred_dict[key] = query_pred_dict1[key]


    evaluator = pytrec_eval.RelevanceEvaluator(query_truth_dict,eval_metrics)
    results = evaluator.evaluate(query_pred_dict)


    # print(json.dumps(results, indent=1))
    query_groups = {'num':[],'cat':[],'all':[]}
    for key in query_truth_dict.keys():
        attr_ = key.split('__')[-1]
        if attr_ not in initialized_desc:continue
        if attr_ not in query_groups:
            query_groups[attr_] = []
        query_groups[attr_].append(key)
        if attr_dict[attr_]=='num':query_groups['num'].append(key)
        else:
            query_groups['cat'].append(key)
        query_groups['all'].append(key)



    # Initialize a dictionary to store the metric sums for each group
    group_metrics_sum = {group: {metric: 0 for metric in eval_metrics} for group in query_groups}
    group_counts = {group: 0 for group in query_groups}

    # Initialize a dictionary to store the mean metrics for each group
    group_mean_metrics = {group: {} for group in query_groups}

    # Iterate over each group
    for group_name, group_queries in query_groups.items():
        # Filter results for the current group
        group_results = {q: results[q] for q in group_queries if q in results}
        group_counts[group_name] = len(group_results)
        
        # Sum the metrics for the current group
        for query_measures in group_results.values():
            for metric in eval_metrics:
                group_metrics_sum[group_name][metric] += query_measures[metric]

        if group_counts[group_name] > 0:
            group_mean_metrics[group_name] = {
                metric: (group_metrics_sum[group_name][metric] / group_counts[group_name])
                for metric in eval_metrics
            }
        
        # Calculate the mean for each metric by dividing by the number of queries in the group
        num_queries = len(group_results)
        if num_queries>0:
            group_mean_metrics[group_name] = {metric: (group_metrics_sum[group_name][metric] / num_queries)
                                        for metric in eval_metrics}


    data_for_df = []
    # Add data for each group
    for group_name, mean_metrics in group_mean_metrics.items():
        row = {'group_name': group_name,'group_count': group_counts[group_name], }
        row.update(mean_metrics)
        data_for_df.append(row)

    return pd.DataFrame(data_for_df)

def normalize_candidate(str1):
    try:
        # Try to convert both strings to floats
        num1 = float(str1)
        return num1
    except:
        return str(str1)
    
def normalize_num_values(values, min_value, max_value, mean, std):
    normalized = [(val - min_value) / (max_value - min_value) for val in values]
    return normalized
    # normalized = [(val - mean) / std for val in values]
    # return normalized

def evaluate_on_attributes(args, base_df, candidate_methods, attr_ls, attr_nan_dict, attr_type='num'):
    if len(attr_ls)==0:return None  

    output_res_df = pd.DataFrame({'method':[],'impute_attr':[], 'impute_attr_type':[],'missing_cnt':[],'rmse':[],\
                         'mae':[],'is_redundant':[],'redundancy':[],'qerror_25':[],'qerror_50':[],'qerror_75':[],\
                            'qerror_90':[],'accu':[],'training_time':[],'test_time':[],'max_value':[],'min_value':[]})
    
    for f in glob.glob(os.path.join(args.impute_dir, args.database_name, '*.csv')):
        arrs = f.split('/')[-1].strip('.csv').split('_')
        method, train_time, test_time = arrs[-3], arrs[-2], arrs[-1]
        # if method in ['dbrandmodel','dbrandemb','hyperimpute']:continue
        if method in ['dbrandemb','hyperimpute']:continue
        if len(candidate_methods)>0 and method not in candidate_methods:continue


        impute_df = pd.read_csv(f)

        
        if len(args.primary_columns)>1 and ('dbimpute' in method  or 'dbrandmodel'==method):
            # Split the column into multiple columns
            split_df = impute_df['__'.join(args.primary_columns)].str.split('__', expand=True)
            # Convert the split columns to integers
            for col in split_df.columns:
                split_df[col] = split_df[col].astype(int)
            # Assign the split integer columns back to the original DataFrame
            impute_df[args.primary_columns] = split_df

        all_gt_vals_num, all_impute_vals_num, all_cnt_num, all_not_added_cnt, all_valid_cnt, all_gt_vals_num1,all_impute_vals_num1  = [], [], 0, 0, 0, [], []
        for key_, re_item_info in attr_nan_dict.items():
            pk_fk_attr_, max_, min_,mean_,std_, median_, unique_cnt = re_item_info
            if key_ not in attr_ls:continue

            filtered_base_df = pk_fk_attr_.merge(base_df, on=args.primary_columns, how='inner')
            filtered_impute_df = pk_fk_attr_.merge(impute_df, on=args.primary_columns, how='inner')
            try:
                gt_vals, impute_vals = filtered_base_df[key_].values, filtered_impute_df[key_].values
            except:
                try:
                    if key_+'_imputed' not in filtered_impute_df.columns:
                        print(method, key_, f, 'error')
                    gt_vals, impute_vals = filtered_base_df[key_].values, filtered_impute_df[key_+'_imputed'].values
                except:
                    continue
            
            count_not_added = 0
            if method in ['dbimpute','dbrandmodel','acl','grimp','gpt', 'gpt1','dbrandemb'] or 'dbimpute' in method:
                # impute_val_cnt = len(impute_vals)
                # impute_val_valid_cnt = len([c for c in impute_vals if not pd.isna(c)])
                # if key_ not in initialized_desc:continue
                filtered_gt_vals, filtered_impute_vals = [], []
                for a_, b_ in zip(gt_vals, impute_vals):
                    if pd.isna(b_):
                        count_not_added += 1
                        b_ = median_

                    # Normalize the values and check types
                    normalized_a = normalize_candidate(a_)
                    normalized_b = normalize_candidate(b_)
                    if attr_type == 'num' and isinstance(normalized_b, float):
                        filtered_gt_vals.append(normalized_a)
                        filtered_impute_vals.append(normalized_b)
                    elif attr_type == 'cat' and isinstance(normalized_b, str):
                        filtered_gt_vals.append(normalized_a)
                        filtered_impute_vals.append(normalized_b)  
                    else:
                        count_not_added += 1
                if len(gt_vals) > 0:
                    ratio_not_added = count_not_added / len(gt_vals)
                    all_not_added_cnt += count_not_added
                    all_valid_cnt += len(gt_vals)
                _, _, mape, qerrors, _ = evaluate_accuracy(filtered_gt_vals, filtered_impute_vals)
                all_gt_vals_num1.extend(filtered_gt_vals)
                all_impute_vals_num1.extend(filtered_impute_vals)

                if min_ is not None and max_ is not None:
                    filtered_gt_vals = normalize_num_values(filtered_gt_vals, min_, max_,mean_,std_,)
                    filtered_impute_vals = normalize_num_values(filtered_impute_vals, min_, max_,mean_,std_,)

                rmse, mae, _, _, accuracy = evaluate_accuracy(filtered_gt_vals, filtered_impute_vals)
                all_gt_vals_num.extend(filtered_gt_vals)
                all_impute_vals_num.extend(filtered_impute_vals)
                all_cnt_num+=len(filtered_impute_vals)

            else:
                _, _, mape, qerrors, _ = evaluate_accuracy(gt_vals, impute_vals)
                all_gt_vals_num1.extend(gt_vals)
                all_impute_vals_num1.extend(impute_vals)

                if min_ is not None and max_ is not None:
                    gt_vals = normalize_num_values(gt_vals, min_, max_,mean_,std_,)
                    impute_vals = normalize_num_values(impute_vals, min_, max_,mean_,std_,)

                rmse, mae, _, _,  accuracy = evaluate_accuracy(gt_vals, impute_vals)
                all_gt_vals_num.extend(gt_vals)
                all_impute_vals_num.extend(impute_vals)
                all_cnt_num+=len(impute_vals)

            res_dict = {'method':[method], 'impute_attr':[key_], 'impute_attr_type':[dtype_dic[key_]], 'missing_cnt':[len(filtered_impute_df)],
                        'rmse':[rmse], 'mae':[mae], 'is_redundant':[key_ in attr_ls], 'redundancy':[converrate_dict.get(key_,-1)],'qerror_25':[qerrors[0][0]], 'qerror_50':[qerrors[0][1]], 
                        'qerror_75':[qerrors[0][2]], 'qerror_90':[qerrors[0][3]], 'accu':[accuracy], 
                        'training_time':[train_time], 'test_time':[test_time],'max_value':[max_],'min_value':[min_], 'unique_cnt':[unique_cnt]}
            
            res_dict.update({
                'count_not_added': count_not_added,
                'ratio_not_added': ratio_not_added if count_not_added!=0 else 0,
                'mape': mape,
                'qerror_neg': qerrors[1]
            })

            # Use pd.concat instead of append
            output_res_df = pd.concat([output_res_df, pd.DataFrame(res_dict)], ignore_index=True)
        
        # if len(all_gt_vals_num)==0 or len(all_impute_vals_num)==0:
        #     print('-----',f)
        _, _, _,qerrors, _ = evaluate_accuracy(all_gt_vals_num1, all_impute_vals_num1)
        rmse, mae,mape, _, accuracy = evaluate_accuracy(all_gt_vals_num, all_impute_vals_num)
        print(all_gt_vals_num)
        print(all_impute_vals_num)

        res_dict = {'method':[method], 'impute_attr':['all'], 'impute_attr_type':[attr_type], 'missing_cnt':[all_cnt_num],
                    'rmse':[rmse], 'mae':[mae], 'is_redundant':['all'],'redundancy':['all'],\
                    # 'mape':[mape], 'jsd':[jsd], 
                    'qerror_25':[qerrors[0][0]], 'qerror_50':[qerrors[0][1]], 
                    'qerror_75':[qerrors[0][2]], 'qerror_90':[qerrors[0][3]], 'accu':[accuracy], 
                    'training_time':[-1], 'test_time':[-1],'max_value':[np.nan],'min_value':[np.nan],'unique_cnt':[np.nan]}
        res_dict.update({
                'count_not_added': all_not_added_cnt,
                'ratio_not_added': all_not_added_cnt/all_valid_cnt if all_valid_cnt>0 else 0,
                'mape': mape,
                'qerror_neg': qerrors[1]
            })
        
        # Use pd.concat instead of append
        output_res_df = pd.concat([output_res_df, pd.DataFrame(res_dict)], ignore_index=True)
    return output_res_df

"""
    here the base_df represents the error_df, 
    if the impute position has value while error_df not, 
    then we take it as our query
"""
def report_time(args, base_df):

    output_res_df = pd.DataFrame({'method':[],'total_impute':[], 'test_time':[]})
    
    for f in glob.glob(os.path.join(args.impute_dir, args.database_name, '*.csv')):
        arrs = f.split('/')[-1].strip('.csv').split('_')
        method, train_time, test_time = arrs[-3], arrs[-2], arrs[-1]

        impute_df = pd.read_csv(f)

        if method!='datawig':
            method_impute_cols = [col for col in impute_df.columns if col in base_df.columns and (col not in args.primary_columns and col not in args.foreign_columns)]
        else:
            method_impute_cols = [col.replace('_imputed','') for col in impute_df.columns if col.replace('_imputed','') in base_df.columns and (col not in args.primary_columns and col not in args.foreign_columns)]

        assert len(method_impute_cols)>0
        # if method!='hyperimpute':
        #     continue
        if len(args.primary_columns)>1 and method in ['dbimpute','dbrandmodel','dbrandemb']:
            # Split the column into multiple columns
            split_df = impute_df['__'.join(args.primary_columns)].str.split('__', expand=True)
            # Convert the split columns to integers
            for col in split_df.columns:
                split_df[col] = split_df[col].astype(int)
            # Assign the split integer columns back to the original DataFrame
            impute_df[args.primary_columns] = split_df
        
        base_compare_df = impute_df[args.primary_columns].merge(base_df, how='inner', on=args.primary_columns)

        valid_cnt = 0
        for col in method_impute_cols:
            base_col_vals = base_compare_df[col].values.tolist()
            if method=='datawig':
                impute_col_vals = impute_df[col+'_imputed'].values.tolist()
            else:
                impute_col_vals = impute_df[col].values.tolist()
            for error_val, impute_val in zip(base_col_vals,impute_col_vals):
                if pd.isna(error_val) and not pd.isna(impute_val):
                    valid_cnt+=1

        res_dict = {'method':[method],'total_impute':[valid_cnt], 'test_time':[test_time]}
            
        # Use pd.concat instead of append
        output_res_df = pd.concat([output_res_df, pd.DataFrame(res_dict)], ignore_index=True)

    return output_res_df



if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate all the methods')
    parser.add_argument('--config_file_path', type=str, required=True, help='The parameter config for each database.')
    args = parser.parse_args()
    # Load the config file
    config = mu.load_config(args.config_file_path)

    # Update the default values with the config file if they exist
    for key, value in config.items():
        setattr(args, key, value) 

    base_df = pd.read_csv(args.input_file)
    error_df = pd.read_csv(args.query_tab_path)

    dtype_dic = {}
    for col in error_df.columns:
        if error_df[col].dtype=='object':
            dtype_dic[col]='cat'
        else:
            if args.database_name=='tpc_ds' and (col.endswith('_id') or col.endswith('_zip') or col.endswith('street_number')):
                dtype_dic[col]='cat'
            else:
                dtype_dic[col]='num'

    
    for f in glob.glob(os.path.join(args.impute_dir, args.database_name, '*.csv')):
        arrs = f.split('/')[-1].strip('.csv').split('_')
        method, train_time, test_time = arrs[-3], arrs[-2], arrs[-1]
        if method!='dbimpute':continue
        dbimpute_df = pd.read_csv(f)

    attr_nan_dict = {}
    for attr_, type_ in dtype_dic.items():
        # if type_!='num':continue
        attr_nan_pks = []
        if attr_ not in args.primary_columns and attr_ not in args.foreign_columns:   

            error_df_null = error_df[error_df[attr_].isna()]
            base_df_not_null = base_df[~base_df[attr_].isna()]
            attr_nan_pks = pd.merge(error_df_null[args.primary_columns], base_df_not_null[args.primary_columns], on=args.primary_columns)

            if len(args.primary_columns)>1:
                split_df = dbimpute_df['__'.join(args.primary_columns)].str.split('__', expand=True)
                for col in split_df.columns:
                    split_df[col] = split_df[col].astype(int)
                # Assign the split integer columns back to the original DataFrame
                dbimpute_df[args.primary_columns] = split_df
            
            attr_nan_pks = pd.merge(attr_nan_pks,dbimpute_df[args.primary_columns], on=args.primary_columns)
            if type_=='num':
                max_, min_ = base_df[attr_].max(), base_df[attr_].min()
                mean_, std_ = base_df[attr_].mean(), base_df[attr_].std()
                median_ = base_df[attr_].median()
            else:
                max_, min_, mean_, std_, median_  = None, None, None, None, None
            unique_cnt = len(base_df[attr_].unique())

            if len(attr_nan_pks)>0: attr_nan_dict[attr_] = [attr_nan_pks,max_,min_, mean_, std_, median_, unique_cnt]

    print('total impute numeric attributes ', len(attr_nan_dict), ' in database ', args.database_name)     
    output_res_df = pd.DataFrame({'method':[],'impute_attr':[], 'missing_cnt':[],'rmse':[],\
                        'is_redundant':[],'redundancy':[],'qerror_25':[],'qerror_50':[],'qerror_75':[],\
                            'qerror_90':[],'accu':[],'training_time':[],'test_time':[]})
    
    initialized_desc = []
    query_attr_meta_path_dic, meta_path_res, meta_path_infer_data = pickle.load(open(args.partial_res_path,'rb'))

    
    query_attr_related_path_dic = {}
    for query_attr_, meta_path_ in query_attr_meta_path_dic.items():
        query_attr_related_path_dic[query_attr_] = Counter(meta_path_).most_common()[0][0]
    print(query_attr_related_path_dic)

    df = pd.DataFrame(meta_path_infer_data, columns=['attr', 'notnull_flag', 'valid_flag', 'max_flag'])
    grouped_df = df.groupby('attr').agg({'notnull_flag':'sum','valid_flag':'sum','max_flag':'sum'}).reset_index()
    valid_attr_ = grouped_df[grouped_df['valid_flag']/grouped_df['notnull_flag']>0.95]['attr'].values.tolist() 
    # print('redundant attribute is ',len(valid_attr_),', the list is ',valid_attr_)

    grouped_df['coverrate'] = grouped_df["valid_flag"]/grouped_df['notnull_flag']
    converrate_dict = pd.Series(grouped_df['coverrate'].values, index=grouped_df['attr']).to_dict()

    
    initialized_desc = args.initialized_desc
    # print(initialized_desc)
    # print([c for c in initialized_desc if dtype_dic[c]=='cat'])
    # print([c for c in initialized_desc if dtype_dic[c]=='num'])

    print('we evaluate on our provided desc attribute', len(initialized_desc), initialized_desc)
    # ,'dbimpute','grimp','gpt4','retriever'
    print('we evaluate on our provided desc attribute with categorical attributes', len([c for c in initialized_desc if dtype_dic[c]=='cat']), [c for c in initialized_desc if dtype_dic[c]=='cat'])
    print('we evaluate on our provided desc attribute with numerical attributes', len([c for c in initialized_desc if dtype_dic[c]=='num']), [c for c in initialized_desc if dtype_dic[c]=='num'])


    output_res_df1 = evaluate_on_attributes(args, base_df, ['dbimpute'], [c for c in initialized_desc if dtype_dic[c]=='cat'], attr_nan_dict, \
                                            'cat')
    output_res_df2 = evaluate_on_attributes(args, base_df, [], [c for c in initialized_desc if dtype_dic[c]=='num'], attr_nan_dict, 'num')
    
    if output_res_df1 is not None and output_res_df2 is not None:
        output_res_df = pd.concat([output_res_df1, output_res_df2], ignore_index=True)
    elif output_res_df1 is not None:
        output_res_df = output_res_df1.copy()
    elif output_res_df2 is not None:
        output_res_df = output_res_df2.copy()
    else:
        output_res_df = None

    if len(output_res_df)>0:
        output_res_df.to_csv(f'results_ana/{args.database_name}_result_alls.csv',index=False)
    else:
        print('we cannot get the results... check it!!!')


    query_json_f = os.path.join(args.trec_eval_dir,args.database_name+'_qry.json')
    run_json_f_path = os.path.join(args.trec_eval_dir,args.database_name+'_run_*.json')
    with open(query_json_f, 'r') as json_file:
        query_truth_dict = json.load(json_file)

    
    all_res_dfs = []
    for run_json_f in glob.glob(run_json_f_path):
        method = run_json_f.split('/')[-1].strip('.json').split('_')[-1]

        with open(run_json_f, 'r') as json_file:
            query_pred_dict = json.load(json_file)

        if 'dbimpute' in method:
            query_pred_dict = {key:val for key, val in query_pred_dict.items() if val is not None} 

            query_pred_dict = {key:query_pred_dict.get(key,{str(normalize_candidate(attr_nan_dict.get(key.split('__')[-1])[5]))\
                                                        :1}) for key in query_truth_dict.keys()}
        else:
            query_pred_dict = {key:val for key, val in query_pred_dict.items() if val is not None}
            
        # if method!='acl':continue
        topk_res_df = evaluate_topk_res(dtype_dic, query_truth_dict,query_pred_dict, initialized_desc)
        
        topk_res_df['method'] = method
        all_res_dfs.append(topk_res_df)

    pd.concat(all_res_dfs,axis=0).to_csv(f'results_ana/{args.database_name}_topk_result_alls.csv',index=False)
    

    