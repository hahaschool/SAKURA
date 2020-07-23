import pickle

import numpy as np
import pandas as pd

from utils.data_splitter import DataSplitter

# Read phenotype table for reference
pheno_csv_path = '../../dataset/10X_PBMC_integrated_purified_and_cite/pheno.csv'
pheno_df = pd.read_csv(pheno_csv_path, index_col=0)

splits = dict()
# Generate dummy "all" mask
splits['all'] = np.ones(pheno_df.shape[0])

# Generate dataset_wise split mask
splits['protein_v3'] = np.array(pheno_df.data_key == 'V3_with_protein_barcode', dtype=np.int)
splits['pure_v1'] = np.array(pheno_df.data_key != 'V3_with_protein_barcode', dtype=np.int)

# Data splitters
data_splitter = DataSplitter()

# V1 Purified: train-test
v1_overall_train_dec = 8
v1_overall_train_seed = 764764

# Make v1 train/test split
v1_dec_bin = data_splitter.auto_random_k_bin_labelling(base=splits['pure_v1'],
                                                       k=10,
                                                       seed=v1_overall_train_seed)
overall_train_test_split = data_splitter.get_incremental_train_test_split(base=v1_dec_bin,
                                                                          k=v1_overall_train_dec)
splits['pure_v1_overall_train'] = overall_train_test_split['train'].astype(np.bool)
splits['pure_v1_overall_test'] = overall_train_test_split['test'].astype(np.bool)

# V3 Protein: train-test
v3_overall_train_dec = 8
v3_overall_train_seed = 810764

# Make v3 train/test split
v3_dec_bin = data_splitter.auto_random_k_bin_labelling(base=splits['protein_v3'],
                                                       k=10,
                                                       seed=v3_overall_train_seed)
overall_train_test_split = data_splitter.get_incremental_train_test_split(base=v3_dec_bin,
                                                                          k=v3_overall_train_dec)
splits['protein_v3_overall_train'] = overall_train_test_split['train'].astype(np.bool)
splits['protein_v3_overall_test'] = overall_train_test_split['test'].astype(np.bool)

# Make merged overall train/test split
splits['merged_overall_train'] = splits['pure_v1_overall_train'] | splits['protein_v3_overall_train']
splits['merged_overall_test'] = splits['pure_v1_overall_test'] | splits['protein_v3_overall_test']

# Export splits.pkl
with open('splits.pkl', 'wb') as f:
    pickle.dump(splits, f)

print(splits.keys())
