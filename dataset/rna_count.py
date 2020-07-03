import json

import numpy as np
import pandas as pd
import sklearn.preprocessing as skprep
import torch
from torch.utils.data import Dataset


class ToTensor(object):
    """
    Convert object to PyTorch Tensors
    """

    def __call__(self, sample, input_type='gene'):
        if input_type == 'gene':
            if type(sample) is pd.core.frame.DataFrame:
                return torch.from_numpy(sample.astype(np.float).values).transpose(0, 1).float()
            elif type(sample) is pd.core.series.Series:
                return torch.from_numpy(sample.astype(np.float).values).unsqueeze(0).float()
            else:
                raise NotImplementedError
        elif input_type == 'pheno':
            if type(sample) is pd.core.frame.DataFrame:
                return torch.from_numpy(sample.astype(np.float).values)
            elif type(sample) is np.ndarray:
                return torch.from_numpy(sample)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


class ToOnehot(object):
    """
    Expected to be used on Phenotype only
    Convert categorical labels to one-hot encodings
    Useful when the loss is not compatible directly with class labels
    """

    def __call__(self, sample, order='auto'):
        # Adaptations
        if order is not 'auto':
            if type(order) is list:
                order = [order]

        ohtrs = skprep.OneHotEncoder(categories=order, sparse=False).fit(sample)
        return ohtrs.transform(sample)


class ToOrdinal(object):
    """
    Expected to be used on Phenotype only
    Convert categorical labels to Ordinals (1,2,3...)
    Useful for losses like torch.nn.CrossEntropyLoss
    """

    def __call__(self, sample, order='auto'):
        # Adaptations
        if order is not 'auto':
            if type(order) is list:
                order = [order]

        ortrs = skprep.OrdinalEncoder(categories=order, dtype=np.int).fit(sample)
        return ortrs.transform(sample)


class ToKBins(object):
    """
    Discretize continuous data
    By default, binarize
    """

    def __call__(self, sample, n_bins=2, encode='ordinal', strategy='quantile'):
        kbintrs = skprep.KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
        return kbintrs.fit_transform(sample)


class LogNormalize(object):
    """
    Log scale, center, and normalize data
    (pending feature)
    """

    def __call__(self, center=True, scale=True, normalize=True):
        raise NotImplementedError


class SCRNASeqCountData(Dataset):
    """
    General scRNA-Seq Count Dataset

    Input:
    genotype_csv:
        * Assuming rows are genes, colmuns are samples(/cells)
        * rownames are gene identifiers (gene name, or ensembl ID)
        * colnames are sample identifiers (cell name)
    genotype_meta_csv:
        * pre_procedure: transformations that will perform when *load* the dataset
        * post_procedure: transformations that will perform when *export* requested samples
    phenotype_csv:
        * Assuming rows are samples, columns are metadata contents
        * rownames are sample identifiers ()
    phenotype_meta_csv:
        * A json file to define Type, Range, and Order for phenotype columns
        * Storage entity is a 'dict'
        * Type: 'categorical', 'numeric', 'ordinal' (tbd)
        * For 'categorical':
            * Range: array of possible values, *ordered*
        * pre_procedure
        * post_procedure


    Options:
        * Mode

    Modes:
        * all
        * sample_id
        * expr
        * pheno



    Transformations:
        * ToTensor:
        * ToOneHot: transform categorical data to one-hot encoding, an order of classes should be specified, otherwise
                    will use sorted labels, assuming the range of labels are from input
        * ToOrdinal:
        * ToKBins:
        * LogNormalize:
    """

    """
    Sample:
    
    
    """

    def __init__(self, gene_csv_path, pheno_csv_path,
                 gene_meta_json_path=None, pheno_meta_json_path=None,
                 gene_meta=None, pheno_meta=None,
                 mode='all', verbose=False):

        # Verbose console logging
        self.verbose=verbose

        # Persist argument list
        self.gene_csv_path = gene_csv_path
        self.gene_meta_json_path = gene_meta_json_path
        self.pheno_csv_path = pheno_csv_path
        self.pheno_meta_json_path = pheno_meta_json_path
        self.mode = mode

        # Register transformers
        self.to_tensor = ToTensor()
        self.to_onehot = ToOnehot()
        self.to_ordinal = ToOrdinal()
        self.to_kbins = ToKBins()

        # Read gene expression matrix
        self._gene_expr_mat_orig = pd.read_csv(self.gene_csv_path, index_col=0, header=0)
        self.gene_expr_mat = self._gene_expr_mat_orig.copy()

        if self.verbose:
            print('==========================')
            print('rna_count dataset:')
            print("Imported gene expression matrix CSV from:", self.gene_csv_path)
            print(self.gene_expr_mat.shape)
            print(self.gene_expr_mat.head(3))

        # Read gene expression matrix metadata
        self.gene_meta = gene_meta
        if self.gene_meta is None:
            self.gene_meta = {
                "all": {
                    "gene_list": "*",
                    "pre_procedure": [],
                    'post_procedure': [
                        {
                            "type": "ToTensor"
                        }
                    ]
                }
            }
            if self.verbose:
                print('No external gene expression set provided, using dummy.')
        if self.gene_meta_json_path is not None:
            with open(self.gene_meta_json_path, 'r') as f:
                self.gene_meta = json.load(f)
                if self.verbose:
                    print("Gene expression set metadata imported from:", self.gene_meta_json_path)
        if self.verbose:
            print("Gene expression set metadata:")
            print(self.gene_meta)

        # Read phenotype data frame
        self._pheno_df_orig = pd.read_csv(self.pheno_csv_path, index_col=0, header=0)
        self.pheno_df = self._pheno_df_orig.copy()

        if self.verbose:
            print("Phenotype data from CSV from:", self.pheno_csv_path)
            print(self.pheno_df.shape)
            print(self.pheno_df)

        # Read phenotype colmun metadata
        self.pheno_meta = pheno_meta
        if pheno_meta_json_path is not None:
            if self.verbose:
                print("Reading phenotype metadata json from:", self.pheno_meta_json_path)
            with open(self.pheno_meta_json_path, 'r') as f:
                self.pheno_meta = json.load(f)

        # Cell list
        self._cell_list_orig = self._gene_expr_mat_orig.columns.values
        self.cell_list = self._cell_list_orig.copy()

        # Sanity check
        # Cell should be consistent between expr matrix and phenotype table
        if (self._gene_expr_mat_orig.columns.values != self._pheno_df_orig.index.values).any():
            raise ValueError

    def subset_reset(self):
        self.gene_expr_mat = self._gene_expr_mat_orig.copy()
        self.pheno_df = self._pheno_df_orig.copy()
        self.cell_list = self._cell_list_orig.copy()
        return self

    def subset_gene(self, gene_list, exclude_mode=False):
        # Subset the dataset with selected genes
        # e.g. subset HVG
        if exclude_mode:
            self.gene_expr_mat.drop(gene_list, axis=0, inplace=True)
            if self.train_test_split_ratio is not None:
                self.gene_expr_mat_test.drop(gene_list, axis=0, inplace=True)
                self.gene_expr_mat_all.drop(gene_list, axis=0, inplace=True)
        else:
            self.gene_expr_mat = self.gene_expr_mat.loc[gene_list, :]
            if self.train_test_split_ratio is not None:
                self.gene_expr_mat_test = self.gene_expr_mat_test.loc[gene_list, :]
                self.gene_expr_mat_all = self.gene_expr_mat_all.loc[gene_list, :]
        return self

    def subset_pheno(self, pheno_list, exclude_mode=False):
        # Return a pheno subsetted object (original data backuped)
        if exclude_mode:
            self.pheno_df.drop(pheno_list, axis=1, inplace=True)
            if self.train_test_split_ratio is not None:
                self.pheno_df_test.drop(pheno_list, axis=1, inplace=True)
                self.pheno_df_all.drop(pheno_list, axis=1, inplace=True)
        else:
            self.pheno_df = self.pheno_df.loc[:, pheno_list]
            if self.train_test_split_ratio is not None:
                self.pheno_df_test = self.pheno_df_test.loc[:, pheno_list]
                self.pheno_df_all = self.pheno_df_all.loc[:, pheno_list]
        return self

    def subset_cell(self, cell_list, exclude_mode=False):
        # Return a cell subsetted object (original data backuped)
        # Adapt type
        if type(cell_list) is list:
            cell_list = np.array(cell_list)
        if exclude_mode:
            self.gene_expr_mat.drop(cell_list, axis=1, inplace=True)
            self.pheno_df.drop(cell_list, axis=0, inplace=True)
            if self.train_test_split_ratio is not None:
                self.gene_expr_mat_test.drop(cell_list, axis=1, inplace=True)
                self.pheno_df_test.drop(cell_list, axis=0, inplace=True)
                self.gene_expr_mat_all.drop(cell_list, axis=1, inplace=True)
                self.pheno_df_all.drop(cell_list, axis=0, inplace=True)
        else:
            self.gene_expr_mat = self.gene_expr_mat.loc[:, cell_list]
            self.pheno_df = self.pheno_df.loc[cell_list, :]
            if self.train_test_split_ratio is not None:
                self.gene_expr_mat_test = self.gene_expr_mat_test.loc[:, cell_list]
                self.pheno_df_test = self.pheno_df_test.loc[cell_list, :]
                self.gene_expr_mat_all = self.gene_expr_mat_all.loc[:, cell_list]
                self.pheno_df_all = self.pheno_df_all.loc[cell_list, :]
        self.cell_list = cell_list
        return self

    def export_data(self, item,
                    include_raw=True,
                    include_proc=True,
                    include_cell_key=True):
        """
        Export a batch of data given 'item' as index.
        :param item: index
        :param include_raw: should the unprocessed, subsetted expression matrix and phenotype data frame be exported
        :param include_proc: should the processed data (following procedures specified in the configs) be exported
        :param include_cell_key: should names/keys of the cells be exported
        :return:
        """

        # Type adaptation: when 'item' is a single index, convert it to list
        if type(item) is int:
            item = [item]

        # Extract required cells and prepare required structure
        ret = dict()

        if include_cell_key is True:
            ret['cell_key'] = self.gene_expr_mat.columns.values[item]

        # Prepare raw gene output (if needed)
        if include_raw is True:
            ret['expr_mat'] = self.gene_expr_mat.iloc[:, item].copy()
        if include_proc is True:
            ret['expr'] = dict()
            for cur_expr_key in self.gene_meta.keys():
                cur_expr_meta = self.gene_meta[cur_expr_key]
                cur_expr_mat = None
                # Gene Selection
                if type(cur_expr_meta['gene_list']) is list or type(cur_expr_meta['gene_list']) is np.ndarray:
                    # Select given genes
                    cur_expr_mat = self.gene_expr_mat.loc[cur_expr_meta['gene_list'], :].iloc[:, item].copy()
                elif cur_expr_meta['gene_list'] == '*':
                    # Select all genes
                    cur_expr_mat = self.gene_expr_mat.iloc[:, item].copy()
                elif cur_expr_meta['gene_list'] == '-':
                    # Deselect given genes
                    cur_expr_mat = self.gene_expr_mat.drop(cur_expr_meta['exclude_list'], axis=0).iloc[:, item].copy()
                ret['expr'][cur_expr_key] = cur_expr_mat

                # Post Transformation
                for cur_procedure in cur_expr_meta['post_procedure']:
                    if cur_procedure['type'] == 'ToTensor':
                        ret['expr'][cur_expr_key] = self.to_tensor(cur_expr_mat, input_type='gene')
                    else:
                        print("Unsupported post-transformation")
                        raise NotImplementedError

        # Prepare phenotype output
        if include_raw is True:
            ret['pheno_df'] = self.pheno_df.iloc[item, :].copy()
        if include_proc is True:
            ret['pheno'] = dict()
            for pheno_output_key in self.pheno_meta.keys():
                cur_pheno_meta = self.pheno_meta[pheno_output_key]
                pheno_df_key = cur_pheno_meta['pheno_df_key']
                ret['pheno'][pheno_output_key] = self.pheno_df.loc[:, [pheno_df_key]].iloc[item, :].copy()
                # Process phenotype label as required
                for cur_procedure in cur_pheno_meta['post_procedure']:
                    if cur_procedure['type'] == 'ToTensor':
                        ret['pheno'][pheno_output_key] = self.to_tensor(sample=ret['pheno'][pheno_output_key],
                                                                        input_type='pheno')
                    elif cur_procedure['type'] == 'ToOnehot':
                        ret['pheno'][pheno_output_key] = self.to_onehot(sample=ret['pheno'][pheno_output_key],
                                                                        order=cur_pheno_meta['order'])
                    elif cur_procedure['type'] == 'ToOrdinal':
                        ret['pheno'][pheno_output_key] = self.to_ordinal(sample=ret['pheno'][pheno_output_key],
                                                                         order=cur_pheno_meta['order'])
                    elif cur_procedure['type'] == 'ToKBins':
                        ret['pheno'][pheno_output_key] = self.to_kbins(sample=ret['pheno'][pheno_output_key],
                                                                       n_bins=cur_procedure['n_bins'],
                                                                       encode=cur_procedure['encode'],
                                                                       strategy=cur_procedure['strategy'])
                    else:
                        raise NotImplementedError
        return ret

    def __len__(self):
        # Length of the dataset is considered as the number of cells
        return self.gene_expr_mat.shape[1]

    def __getitem__(self, item):
        if self.mode is 'all':
            return self.export_data(item,
                                    include_raw=True,
                                    include_proc=True,
                                    include_cell_key=True)
        elif self.mode is 'key':
            return self.export_data(item,
                                    include_raw=False,
                                    include_proc=False,
                                    include_cell_key=True)
        else:
            return self.export_data(item,
                                    include_raw=False,
                                    include_proc=True,
                                    include_cell_key=False)

