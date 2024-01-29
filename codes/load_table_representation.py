import os
import pdb

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Any, Dict, List, Optional
import pandas.api.types as ptypes
import transtab
# set random seed
transtab.random_seed(42)
from tpc_ds import *

from torch_frame.typing import DataFrame, Series


def _lst_is_all_type(
    lst,types
) -> bool:
    assert isinstance(lst, list)
    return all(isinstance(x, types) for x in lst)


def _lst_is_free_of_nan_and_inf(lst):
    assert isinstance(lst, list)
    return all(not math.isnan(x) and not math.isinf(x) for x in lst)


def _min_count(ser: Series) -> int:
    return ser.value_counts().min()


def infer_series_stype(ser: Series):
    """Infer :obj:`stype` given :class:`Series` object. The inference may not
    be always correct/best for your data. We recommend you double-checking the
    correctness yourself before actually using it.

    Args:
        ser (Series): Input series.

    Returns:
        Optional[stype]: Inferred :obj:`stype`. Returns :obj:`None` if
            inference failed.
    """
    has_nan = ser.isna().any()
    if has_nan:
        ser = ser.dropna()

    if len(ser) == 0:
        return None

    # Categorical minimum counting threshold. If the count of the most minor
    # categories is larger than this value, we treat the column as categorical.
    cat_min_count_thresh = 4

    if isinstance(ser.iloc[0], list):
        # Candidates: embedding, sequence_numerical, multicategorical

        # True if all elements in all lists are numerical
        is_all_numerical = True
        # True if all elements in all lists are string
        is_all_string = True
        # True if all lists are of the same length and all elements are float
        # and free of nans.
        is_embedding = True

        length = len(ser.iloc[0])
        for lst in ser:
            if not isinstance(lst, list):
                return None
            if _lst_is_all_type(lst, (int, float)):
                if not (length == len(lst) and _lst_is_all_type(lst, float)
                        and _lst_is_free_of_nan_and_inf(lst)):
                    is_embedding = False
            else:
                is_all_numerical = False
            if not _lst_is_all_type(lst, str):
                is_all_string = False

        if is_all_numerical:
            if is_embedding:
                return stype.embedding
            else:
                return stype.sequence_numerical
        elif is_all_string:
            return stype.multicategorical
        else:
            return None
    else:
        # Candidates: numerical, categorical, multicategorical, and
        # text_(embedded/tokenized)

        if ptypes.is_numeric_dtype(ser):
            # Candidates: numerical, categorical
            if ptypes.is_float_dtype(ser) and not (has_nan and
                                                   (ser % 1 == 0).all()):
                return 'num'
            else:
                if _min_count(ser) > cat_min_count_thresh:
                    return 'cat'
                else:
                    return 'num'
        else:
            return 'cat'


def infer_df_stype(df: DataFrame):
    """Infer :obj:`col_to_stype` given :class:`DataFrame` object.

    Args:
        df (DataFrame): Input data frame.

    Returns:
        col_to_stype: Inferred :obj:`col_to_stype`, mapping a column name to
            its inferred :obj:`stype`.
    """
    col_to_stype = {}
    for col in df.columns:
        stype = infer_series_stype(df[col])
        if stype is not None:
            col_to_stype[col] = stype
    return col_to_stype

def get_stype_proposal(db):
    r"""Propose stype for columns of a set of tables in the given database.

    Args:
        db (Database): : The database object containing a set of tables.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping table name into
            :obj:`col_to_stype` (mapping column names into inferred stypes).
    """

    inferred_col_to_stype_dict = {}
    for table_name, table in db.items():
        # Take the first 10,000 rows for quick stype inference.
        inferred_col_to_stype = infer_df_stype(table.df)

        # Remove pkey, fkey columns since they will not be used as input
        # feature.
        if table.pkey_col is not None:
            primary_keys = table.pkey_col
            if type(primary_keys)!=str:
                primary_keys = '__'.join(primary_keys)
            if primary_keys in inferred_col_to_stype:
                inferred_col_to_stype.pop(primary_keys)
        for fkey in table.fkey_col_to_pkey_table.keys():
            if fkey in inferred_col_to_stype:
                inferred_col_to_stype.pop(fkey)

        dtype_dict = {'num':[],'cat':[],'bin':[]}
        for col_, type_ in inferred_col_to_stype.items():
            dtype_dict[type_].append(col_)

        inferred_col_to_stype_dict[table_name] = dtype_dict

    return inferred_col_to_stype_dict


def load_data(db, dataset_config=None, pred_load_set = None, encode_cat=False, data_cut=None, seed=123, mode='train',to_get_datanames=None):
    '''Load datasets from the local device or from openml.datasets.

    Parameters
    ----------
    dataname: str or int
        the dataset name/index intended to be loaded from openml. or the directory to the local dataset.
    
    dataset_config: dict
        the dataset configuration to specify for loading. Please note that this variable will
        override the configuration loaded from the local files or from the openml.dataset.
    
    encode_cat: bool
        whether encoder the categorical/binary columns to be discrete indices, keep False for TransTab models.
    
    data_cut: int
        how many to split the raw tables into partitions equally; set None will not execute partition.

    seed: int
        the random seed set to ensure the fixed train/val/test split.

    Returns
    -------
    all_list: list or tuple
        the complete dataset, be (x,y) or [(x1,y1),(x2,y2),...].

    train_list: list or tuple
        the train dataset, be (x,y) or [(x1,y1),(x2,y2),...].

    val_list: list or tuple
        the validation dataset, be (x,y) or [(x1,y1),(x2,y2),...].

    test_list: list
        the test dataset, be (x,y) or [(x1,y1),(x2,y2),...].

    cat_col_list: list
        the list of categorical column names.

    num_col_list: list
        the list of numerical column names.

    bin_col_list: list
        the list of binary column names.

    '''
    
    # load a list of datasets, combine together and outputs
    num_col_list, cat_col_list, bin_col_list = [], [], []
    all_list = []
    tab_names = []
    train_list, val_list, test_list = [], [], []

    for dataname_, data_info in db.items():
        if to_get_datanames is not None:
            if dataname_ not in to_get_datanames:continue
        if mode=='train':
            if len(data_info.df)<=100:continue
            if len(data_info.df)>10000:continue
            allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = \
                load_single_data(data_info.df, dataset_config=dataset_config[dataname_], pre_load_sets = None, encode_cat=encode_cat, data_cut=data_cut, seed=seed)
            
            num_col_list.extend(num_cols)
            cat_col_list.extend(cat_cols)
            bin_col_list.extend(bin_cols)
            all_list.append(allset)
            train_list.append(trainset)
            val_list.append(valset)
            test_list.append(testset)
            tab_names.append(dataname_)
        else:
            allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = \
                load_single_data(data_info.df, dataset_config=dataset_config[dataname_], pre_load_sets = pred_load_set, encode_cat=encode_cat, data_cut=data_cut, seed=seed)

            num_col_list.extend(num_cols)
            cat_col_list.extend(cat_cols)
            bin_col_list.extend(bin_cols)
            all_list.append(allset)
            train_list.append(trainset)
            val_list.append(valset)
            test_list.append(testset)
            tab_names.append(dataname_)
    # for item in val_list:
    #     print(item[0])
    return tab_names, all_list, train_list, val_list, test_list, cat_col_list, num_col_list, bin_col_list


def load_single_data(df, dataset_config=None, pre_load_sets=None, encode_cat=False, data_cut=None, seed=123):
    '''Load tabular dataset from local or from openml public database.
    args:
        dataname: Can either be the data directory on `./data/{dataname}` or the dataname which can be found from the openml database.
        dataset_config: 
            A dict like {'dataname':{'bin': [col1,col2,...]}} to indicate the binary columns for the data obtained from openml.
            Also can be used to {'dataname':{'cols':[col1,col2,..]}} to assign a new set of column names to the data
        encode_cat:  Set `False` if we are using transtab, otherwise we set it True to encode categorical values into indexes.
        data_cut: The number of cuts of the training set. Cut is performed on both rows and columns.
    outputs:
        allset: (X,y) that contains all samples of this dataset
        trainset, valset, testset: the train/val/test split
        num_cols, cat_cols, bin_cols: the list of numerical/categorical/binary column names
    '''
    print('####'*10)

    X = df
    all_cols = [col.lower() for col in X.columns.tolist()]


    X.columns = all_cols

    if dataset_config is not None:
        if 'bin' in dataset_config:
            bin_cols = dataset_config['bin']
        
        if 'cat' in dataset_config:
            cat_cols = dataset_config['cat']

        if 'num' in dataset_config:
            num_cols = dataset_config['num']

    # update cols by loading dataset_config
    if pre_load_sets is not None:
        bin_cols = [c for c in all_cols if c in pre_load_sets['bin_cols']]
        cat_cols = [c for c in all_cols if c in pre_load_sets['cat_cols']]
        num_cols = [c for c in all_cols if c in pre_load_sets['num_cols']]
        if len(bin_cols)+len(cat_cols)+len(num_cols)==0:
            assert dataset_config is not None
            if 'bin' in dataset_config:
                bin_cols = dataset_config['bin']
            
            if 'cat' in dataset_config:
                cat_cols = dataset_config['cat']

            if 'num' in dataset_config:
                num_cols = dataset_config['num']

    # start processing features
    # process num
    if len(num_cols) > 0:
        for col in num_cols: 
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col].fillna(X[col].mode()[0], inplace=True)
        X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        for col in cat_cols: X[col].fillna(X[col].mode()[0], inplace=True)
        # process cate
        if encode_cat:
            X[cat_cols] = OrdinalEncoder().fit_transform(X[cat_cols])
        else:
            X[cat_cols] = X[cat_cols].astype(str)

    if len(bin_cols) > 0:
        for col in bin_cols: X[col].fillna(X[col].mode()[0], inplace=True)
        if 'binary_indicator' in dataset_config:
            X[bin_cols] = X[bin_cols].astype(str).applymap(lambda x: 1 if x.lower() in dataset_config['binary_indicator'] else 0).values
        else:
            X[bin_cols] = X[bin_cols].astype(str).applymap(lambda x: 1 if x.lower() in ['yes','true','1','t'] else 0).values        
        
        # if no dataset_config given, keep its original format
        # raise warning if there is not only 0/1 in the binary columns
        if (~X[bin_cols].isin([0,1])).any().any():
            raise ValueError(f'binary columns {bin_cols} contains values other than 0/1.')

    
    X = X[bin_cols + num_cols + cat_cols]


    # split train/val/test

    train_dataset, test_dataset = train_test_split(X,  test_size=0.2, random_state=seed, shuffle=True)
    val_size = int(len(X)*0.1)
    val_dataset = train_dataset.iloc[-val_size:]
    train_dataset = train_dataset.iloc[:-val_size]

    print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}'.format(len(X), len(all_cols), len(cat_cols), len(bin_cols), len(num_cols) ))
    return (X,pd.DataFrame({'target_col':[1 for _ in range(len(X))]})), (train_dataset,pd.DataFrame({'target_col':[1 for _ in range(len(train_dataset))]})),\
          (val_dataset,pd.DataFrame({'target_col':[1 for _ in range(len(val_dataset))]})), \
        (test_dataset, pd.DataFrame({'target_col':[1 for _ in range(len(test_dataset))]})), cat_cols, num_cols, bin_cols

def calculate_similarity_mat(user_embeddings, item_embeddings):
    # Move tensors to CPU if they are on GPU
    if user_embeddings.is_cuda:
        user_embeddings = user_embeddings.cpu()
    if item_embeddings.is_cuda:
        item_embeddings = item_embeddings.cpu()
    
    # Convert tensors to NumPy arrays
    user_embeddings_np = user_embeddings.detach().numpy()
    item_embeddings_np = item_embeddings.detach().numpy()

    # Calculate cosine similarity using NumPy operations
    user_norms = np.linalg.norm(user_embeddings_np, axis=1, keepdims=True)
    item_norms = np.linalg.norm(item_embeddings_np, axis=1, keepdims=True)
    
    # Use dot product for cosine similarity since the vectors are normalized
    cosine_similarity_matrix = np.dot(user_embeddings_np, item_embeddings_np.T) / (user_norms * item_norms.T)

    # Find the top N items for each user
    N = 5  # For example, find top 5 similar items
    top_N_items_indices = np.argsort(-cosine_similarity_matrix, axis=1)[:, :N]

    return user_embeddings, item_embeddings, top_N_items_indices

def get_all_embeddings(enc,embeddings,small_tab_names,small_list,batch_size=128):
    for i in tqdm(range(len(small_tab_names)),total=len(small_tab_names)):
        # try:
        df_full = small_list[i][0]
        total_rows = len(df_full)
        embeddings[small_tab_names[i]] = []  # Initialize with an empty list

        # Process the dataframe in batches
        for start_idx in range(0, total_rows, batch_size):
            end_idx = start_idx + batch_size
            df_batch = df_full[start_idx:end_idx]
            
            # Process the batch and move the result to CPU memory immediately
            output_ = enc(df_batch).cpu().detach().numpy()
            
            # Append the result to the embeddings list
            embeddings[small_tab_names[i]].extend(output_)
        embeddings[small_tab_names[i]] = np.array(embeddings[small_tab_names[i]])
        # except:
        #     print(f"Error processing : {small_tab_names[i]}")
        # Convert the list of embeddings to a numpy array
    return embeddings

def train_embedding_model(tpc_dataset, save_checkpoint_path): 
    # for this, we just sample some data and pretraining the model
    tpc_db = tpc_dataset.test_build_db() 
    stype_dict = get_stype_proposal(tpc_db)
    print(stype_dict)

    tab_names, allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = load_data(tpc_db, stype_dict, None, encode_cat=False, data_cut=None, seed=123,\
                                                                                                                        mode='train')

  
    # make a fast pre-train of TransTab contrastive learning model
    # build contrastive learner, set supervised=True for supervised VPCL
    model, collate_fn = transtab.build_contrastive_learner(
        cat_cols, num_cols, bin_cols, 
        supervised=True, # if take supervised CL
        num_partition=2, # num of column partitions for pos/neg sampling
        overlap_ratio=0.5, # specify the overlap ratio of column partitions during the CL
    )

    # start contrastive pretraining training
    training_arguments = {
        'num_epoch':30,
        'batch_size':128,
        'lr':1e-4,
        'eval_metric':'val_loss',
        'eval_less_is_better':True,
        'output_dir':save_checkpoint_path,
        "ignore_duplicate_cols":True
        }

    transtab.train(model, trainset, valset, collate_fn=collate_fn, **training_arguments)

def load_save_embedding_bycheckpoint_path(tpc_dataset, save_checkpoint_path, save_emb_path):
    # we would emb all the data tables except the query table
    tpc_db = tpc_dataset.build_db()
    stype_dict = get_stype_proposal(tpc_db)
    print(stype_dict)

    file_path = save_emb_path
    if not os.path.exists(file_path):

        enc = transtab.build_encoder(
            binary_columns=[],
            checkpoint = save_checkpoint_path
        )
        pred_load_set = {'num_cols':enc.numerical_columns,'cat_cols':enc.categorical_columns,'bin_cols':enc.binary_columns}

        tab_names, allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = load_data(tpc_db, stype_dict, pred_load_set, encode_cat=False, data_cut=None, seed=123,mode='predict',to_get_datanames=None)

        embeddings = get_all_embeddings(enc,{},tab_names,allset,batch_size=128)

        np.savez_compressed(save_emb_path, **embeddings)
