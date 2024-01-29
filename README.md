
# Using Search to Solve The Missing Data Problem in Large Relational Databases

## Overview
This repository contains datasets and code associated with our paper titled "Using Search to Solve The Missing Data Problem in Large Relational Databases." This work represents the forefront of data imputation efforts aimed at resolving the missing data issue in large relational databases.

Our codes can be referred in directory **codes**. 

## Requirements
```bash
pip install -r requirements.txt
```

## Experimental Config Overview
See the configs in dataconfig/xx.json
- `transtab_path`: The model for training [trans_tab](https://github.com/RyanWangZf/transtab/tree/main) model to emb the data for each tuple .
- `input_file`: The path for the raw data of the query table.
- `embedding_file_path`: The path for each tuple of the whole database tuples.
- `query_sample_path`: The path for the training samples.
- `infer_samples_path`: The path for the inferring samples.
- `schema_path`: The path for the schema file of the database. [schema_path]
- `path`: The path for the raw dataset. [basic\_data\_path]
- `baseline_tab_dir`: The path for experiments for baselines.
- `query_tab_path`: The path for injected error query table. (See the output of error_injection.py)
- `query_tab_name`: The name of the query table.
- `faiss_idx_dir`: The path for the saving faiss index.
- `loss_save_path`: The path for saving our loss history.
- `index_file`: The path for the index of column joinability.
- `impute_dir`: The path to save the imputed data.
- `database_name`: The name for the database
- `sample_size`: The sample size for training model (-1 means utilizing all the completed values for each attribute).
- `partial_res_path`: The path for saving the inferring meta-paths of each attribute.
- `trec_eval_dir`: The path for saving top-k ranking missing values.
- `primary_columns`: Primary key column
- `foreign_columns`: Foreign key column
- `initialized_desc`: The redundant attributes in the query table.

## Dataset Overview

### Raw Datasets
We have employed two datasets in our research:

- **eICU**: Can be downloaded directly from [eICU - PhysioNet](https://physionet.org/content/eicu-crd/2.0/).
- **TPC-DS**: The TPC-DS dataset can be generated using the code found at [TPC-DS Kit - GitHub](https://github.com/gregrahn/tpcds-kit/tree/5a3a81796992b725c2a8b216767e142609966752). For our experiments, we have set the SCALE parameter to 1

### Creating Redundant Data
To create redundant data from your dataset, execute the following script:
```bash
python codes/inject_redundancy.py --basic_data_path [path_to_raw_dataset] --save_data_path [path_to_save_processed_dataset] --schema_path [path_to_database_schema]
```
- `basic_data_path`: The path where the raw dataset is stored.
- `save_data_path`: The path where the processed dataset is to be saved.
- `schema_path`: The path to the database schema. Example schemas are provided in `datasets/eicu/database_schema.json` and `datasets/tpc_ds/database_schema.json`.

### Injecting Errors in Query Tables
To inject errors into a query table, use the following script:
```bash
python codes/error_injection.py -i [path_to_processed_dataset]/[query_table csv file] --primary_columns [primary_key_columns] --foreign_columns [foreign_key_columns] --target_all_columns --drop
```
- The `-i` option indicates the path of the query table to the processed dataset with redundant data. 
- Specify the primary and foreign key columns as per your dataset's schema.

### Processed Dataset Access
Our processed TPC-DS dataset is available [here](https://zenodo.org/uploads/10579801). Place it in `datasets/tpc_ds`.

## Methodology

### Semantic Modeling
Execute the following for semantic modeling:
```bash
python codes/tpc_ds.py -config [path_to_config_file]
```

### Meta Path Learning
#### Model Training
```bash
python codes/train_my_method.py --config_file_path [path_to_config_file] --checkpoint_path '' --model_save_path [path_to_save_model_path]
```

#### Missing Value Imputation
```bash
python codes/impute_eicu.py --config_file_path [path_to_config_file] --checkpoint_path [path_to_save_model_path]  --summary_path [path_to_save_inferred_path]  --method dbimpute --mode raw_sample
```

#### Evaluation
```
python codes/evaluate.py --config_file_path [path_to_config_file]
```
