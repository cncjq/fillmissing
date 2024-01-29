"""
This script is used for injecting random missing values in a given dataframe according to some simple rules.
It is possible to target either a subset, or all columns in the dataframe.
Columns can be converted to string values so they are treated as categorical values.
"""
import argparse
import os
import os.path as path
import random

import numpy as np
import pandas as pd
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', action='store', required=True,
                        help='Input file to be modfied by adding noise.')
    parser.add_argument('--error_fraction', default=.2, action='store', type=float,
                        help='Fraction of errors to be added to the columns.')
    parser.add_argument('--primary_columns', '-p', action='store', nargs='*', required=True,
                        help='The primary key columns for the dataset. ')
    parser.add_argument('--foreign_columns', '-f', action='store', nargs='*', required=True,
                        help='The foreign key columns for the dataset. ')
    parser.add_argument('--seed', action='store', type=int, default=None,
                        help='Set a specific random seed for repeatability. ')
    parser.add_argument('--missing_value_flag', action='store', default=None,
                        help='Optional argument to pass to the dataset parser as a null value flag. ')
    parser.add_argument('--target_all_columns', action='store_true',
                        help='Inject "error_fraction/n_columns" errors in each column. ')
    parser.add_argument('--save_folder', action='store', default=None,
                        help='Force saving the dirty datasets in the given folder.')
    parser.add_argument('--map_columns', '-m', action='store', nargs='*',
                        help='The columns which need to be mapped and transformed in the raw file. ')
    parser.add_argument('--json_string', type=str, help='A JSON string to be parsed.')
    parser.add_argument('--drop', action='store_true', help='Set a switch to true.')
    


    args = parser.parse_args()
    return args

def parse_json_input(json_string):
    """
    Parses the provided JSON string and returns the corresponding JSON object.

    :param json_string: A string containing JSON data.
    :return: A JSON object.
    """
    try:
        return {k: (v if len(v) > 0 else np.NaN) for k,v in json.loads(json_string).items()}
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON input: {e}")

def column_error_injection(column: pd.Series, mask, na_value=np.nan):
    column_dirty = column.copy(deep=True)
    column_dirty.loc[mask] = na_value
    count_injected_values = sum((column_dirty.isna()))
    return column_dirty, count_injected_values


def get_mask(df, error_fraction):
    return df.sample(frac=error_fraction).index

def write_hivae_version(df_dirty, df_base, args):
    # HI-VAE requires a specific error format to work take datasets as input. This function prepares all files required
    # by HI-VAE in the proper format.

    print('Writing additional files for HI-VAE.')
    input_file_name = args.input_file

    clean_name, ext = path.splitext(input_file_name)
    basename = path.basename(clean_name)
    error_frac = str(int(args.error_fraction*100))
    output_dataset_path = f'data/dirty_datasets/{basename}_hivae_{error_frac}'
    input_dataset_path = f'data/{basename}_hivae_{error_frac}{ext}'

    output_dataset_path += f'{ext}'

    for df, fpath in zip([df_base, df_dirty], [input_dataset_path, output_dataset_path]):
        n_uniques = convert_to_number(df, args.primary_columns, args.foreign_columns)
        convert_to_hivae_format(df_base, df, fpath, args.primary_columns, args.foreign_columns)

    try:
        prepare_dtype_file(df_base, n_uniques)
    except:
        print('nothing...')

    return output_dataset_path


def simple_error_injection(args, df):
    df2 = df.copy(deep=True)
    error_fraction = args.error_fraction
    if args.target_all_columns and args.primary_columns and args.foreign_columns:
        target_columns = [col for col in df.columns if col not in args.primary_columns and col not in args.foreign_columns]
    error_count = {k: 0 for k in target_columns}

    for attr in target_columns:
        mask = get_mask(df2, error_fraction)
        df2[attr], injected = column_error_injection(df2[attr], mask)
        error_count[attr] = injected
        # This encodes nan/none as -1
        # df2.loc[df2.sample(frac=args.error_fraction).index, attr] = -1
        print(f'Column {attr}: {injected} errors in {len(df2[attr])} values.')
    tot_errors = df2.isna().sum().sum()
    print(
        f'Injected a total of {tot_errors} errors. {tot_errors}/{len(df2.values.ravel())}={tot_errors / len(df2.values.ravel()):.2f}')
    return df2


def write_dirty_df(df_dirty, args):
    error_frac = str(int(args.error_fraction*100))
    input_file_name = args.input_file
    clean_name, ext = path.splitext(input_file_name)
    basename = path.basename(clean_name)
    output_name = f'{basename}_all_columns_{error_frac}{ext}'

    if args.save_folder is not None:
        output_path = f'{args.save_folder}'
    else:
        output_path = f'data/{basename}/'
    if not path.exists(output_path):
        print(f'Creating new folder: {output_path}')
        os.makedirs(output_path)

    output_name = os.path.join(output_path, output_name)

    print(f'Output file: {output_name}')
    # df_dirty.to_csv(output_name, index=False)

    return output_name


def convert_to_number(df, p_keys, f_keys):
    df_copy = df.copy()
    n_uniques = {}
    for idx, col in enumerate(df.columns):
        if df[col].dtype == 'O' and col not in p_keys and col not in f_keys:
            uniques = df_copy[col].unique().tolist()
            dict_uniques = {uniques[_i]: _i for _i in range(len(uniques))}
            df_copy[col] = df_copy[col].apply(lambda x: dict_uniques[x])
            n_uniques[idx] = len(uniques)
    return n_uniques


def convert_to_hivae_format(df_base, df_tgt, path, p_keys, f_keys):
    df_copy = df_tgt.copy()
    dict_uniques = dict()
    for col in df_base.columns:
        if df_base[col].dtype == 'O' and col not in p_keys and col not in f_keys:
            uniques = df_base[col].unique().tolist()
            dict_uniques[col] = {uniques[_i]: _i for _i in range(len(uniques))}
            dict_uniques[col][np.nan] = ''
            # df_copy[col] = df_copy[col].apply(lambda x: dict_uniques[col][x])
            df_copy[col] = df_copy[col].apply(lambda x:'' if pd.isna(x) else dict_uniques[col].get(x,dict_uniques[col].get(str(x))))
    df_copy.to_csv(path, index=False)
    return dict_uniques


def prepare_dtype_file(df, n_uniques):
    # This function creates a new dataframe that contains a row for each column in the dataframe.
    # Rows are either "categorical" or "real" depending on whether the dtype is "object", or numerical.
    dd = {_: df.columns[_] for _ in range(len(df.columns))}
    for idx, dt in enumerate(df.dtypes):
        # NOTE: this is not robust to weird datatypes: this check assumes that dtypes are either object, or real values.
        if dt == 'object' or dt == 'str':
            dd[idx] = ['categorical', n_uniques[idx], n_uniques[idx]]
        else:
            dd[idx] = ['real', 1, '']
    try:
        dtypes = pd.DataFrame(columns=['type', 'dim', 'nclass']).from_dict(dd)
    except:
        dtypes = pd.DataFrame.from_dict(dd, orient='index', columns=['type', 'dim', 'nclass'])
    return dtypes

def convert_pk_fk_columns(args, df_base):
    for col in args.primary_columns:
        df_base[col] = df_base[col].apply(lambda x: str(x))
    for col in args.foreign_columns:
        df_base[col] = df_base[col].apply(lambda x: str(x))
    return df_base

def convert_map_columns(args, df_base, map_str):
    for col in args.map_columns:
        df_base[col] = df_base[col].astype(str)
        for val in df_base[col].unique():
            if not pd.isna(val):
                assert val in map_str
        df_base[col] = df_base[col].apply(lambda x: map_str.get(x,''))
    return df_base


if __name__ == '__main__':
    # This script will run BART on a given dataset and will add errors according to the provided values
    args = parse_args()
    random.seed(args.seed)

    df_base = pd.read_csv(args.input_file, na_values=args.missing_value_flag)
    #  for the raw query table with less missing values
    if args.drop:
        df_base = df_base.dropna()

    convert_pk_fk_columns(args, df_base)

    if args.map_columns is not None and len(args.map_columns) > 0:
        map_str = parse_json_input(args.json_string)
        convert_map_columns(args, df_base, map_str)

    # df_dirty = simple_error_injection(args, df_base)
    # output_name = write_dirty_df(df_dirty, args)
    # write_hivae_version(df_dirty, df_base, args)

    # for rewriting the pattern
    output_name = write_dirty_df(None, args)
    write_hivae_version(pd.read_csv(output_name), df_base, args)