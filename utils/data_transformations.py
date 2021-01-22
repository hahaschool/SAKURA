import numpy as np
import pandas as pd
import sklearn.preprocessing as skprep
import torch


class ToTensor(object):
    """
    Convert object to PyTorch Tensors
    """

    def __call__(self, sample, input_type='gene', force_tensor_type=None):
        ret = None
        if input_type == 'gene':
            if type(sample) is pd.core.frame.DataFrame:
                ret = torch.from_numpy(sample.astype(np.float).values).transpose(0, 1).float()
            elif type(sample) is pd.core.series.Series:
                ret = torch.from_numpy(sample.astype(np.float).values).unsqueeze(0).float()
            else:
                raise NotImplementedError
        elif input_type == 'pheno':
            if type(sample) is pd.core.frame.DataFrame:
                ret = torch.from_numpy(sample.astype(np.float).values)
            elif type(sample) is np.ndarray:
                ret = torch.from_numpy(sample)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        if force_tensor_type is not None:
            if force_tensor_type == 'float':
                ret = ret.float()
            elif force_tensor_type == 'int':
                ret = ret.int()
            elif force_tensor_type == 'double':
                ret = ret.double()
            else:
                raise NotImplementedError('Expected tensor type not supported yet')
        return ret

class ToOnehot(object):
    """
    Expected to be used on Phenotype only
    Convert categorical labels to one-hot encodings
    Useful when the loss is not compatible directly with class labels
    """

    def __call__(self, sample, order='auto'):
        # Adaptations
        if order != 'auto':
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

    def __call__(self, sample, order='auto', handle_unknown='use_encoded_value', unknown_value=None):
        # Adaptations
        if order != 'auto':
            if type(order) is list:
                order = [order]

        ortrs = skprep.OrdinalEncoder(categories=order, handle_unknown=handle_unknown, unknown_value=unknown_value, dtype=np.int).fit(sample)
        return ortrs.transform(sample)


class ToKBins(object):
    """
    Discretize continuous data
    By default, binarize
    """

    def __call__(self, sample, n_bins=2, encode='ordinal', strategy='quantile'):
        kbintrs = skprep.KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
        return kbintrs.fit_transform(sample)


class StandardNormalizer(object):
    """
    Allows log-transformation (like in Seurat, first multiply with a size factor, then plus a pseudocount, then log),
     standardization (scaling and centering, to obtain z-score)
    (pending feature)
    """

    def __call__(self, center=True, scale=True, normalize=True):
        raise NotImplementedError
