import numpy as np

class DataSplitter(object):

    def auto_random_k_bin_labelling(self, base: np.ndarray, k: int, seed=None) -> np.ndarray:
        """
        Obtain a label vector (containing 1~k for included points, 0 for not included points)
        :param base:
        :param k:
        :param seed: a temporary seed
        :return: a numpy.integer array, consisting 0 and 1~k
        """
        # Sanity check

        if type(base) is not np.ndarray:
            raise ValueError
        if np.issubdtype(base.dtype, np.integer) is False:
            raise ValueError
        if k <= 0:
            raise ValueError
        if len(base.shape) != 1:
            raise ValueError

        # Set random seed when required, otherwise inherit
        random_state_backup = np.random.get_state()
        if seed is not None:
            np.random.seed(seed)

        base_nonzero_idx = np.arange(0, len(base))[base != 0]
        base_nonzero = base[base != 0]
        rep = len(base_nonzero) // k
        rmd = len(base_nonzero) % k
        bin = np.concatenate([np.repeat(np.arange(1, k+1), rep), np.arange(1, rmd+1)])
        np.random.shuffle(bin)

        ret = base.copy()
        ret[base_nonzero_idx] = bin

        # Restore random state if seed is set (nullify the effect of setting seed)
        if seed is not None:
            np.random.set_state(random_state_backup)

        return ret

    def auto_random_stratified_k_bin_labelling(self, base, label, k, seed=None):
        """
        :param base:
        :param label:
        :param k:
        :param seed:
        :return:
        """
        raise NotImplementedError

    def get_incremental_select_unselect_split(self, base: np.ndarray, k: int) -> np.ndarray:
        """
        Obtain 1 split code from label vector, points labelled from 1~k are considered as selected (1),
        otherwise not selected(0)
        Useful when determining overall train/test split directly from predefined labels

        :param base:
        :param k:
        :return: numpy.integer array, consisting 0 and 1
        """
        # Sanity check
        if type(base) is not np.ndarray:
            raise ValueError
        if np.issubdtype(base.dtype, np.integer) is False:
            raise ValueError
        if len(base.shape) != 1:
            raise ValueError

        base_leq_k_idx = np.arange(0, len(base))[(base != 0) & (base <= k)]
        ret = np.zeros(len(base), dtype=np.integer)
        ret[base_leq_k_idx] = 1

        return ret


    def get_incremental_train_test_split(self, base: np.ndarray, k: int) -> dict:
        """
        Obtain 2 split codes from label vector, points labelled from 1~k are considered as train (1 in first vector),
        rest of selected (non-zero) cells are test(1 in second vector),
        unselected points remain unchanged(0 in all vectors)
        Useful when planning to increase points in supervision incrementally (e.g. select 30% cells with known certain known labels)
        :param base: which label vector should be processed
        :param k:
        :return: a dictionary, "train" and "test" split codes
        """

        # Sanity check is omitted as performed in get_incremental_select_unselect_split()

        train_vec = self.get_incremental_select_unselect_split(base, k)
        test_vec = np.ones(len(base), dtype=np.integer)
        test_vec[(train_vec != 0) | (base == 0)] = 0

        return {
            'train' : train_vec,
            'test' : test_vec
        }

    def get_k_fold_cv_split(self, base: np.ndarray) -> dict:
        """
        Obtain cross validation foldings directly from 1~k labels, 0 considered to be not selected
        :param base:
        :return: a dictionary, key is labels, mapped to disctionarys of test and train
        """

        # Sanity check
        if type(base) is not np.ndarray:
            raise ValueError
        if np.issubdtype(base.dtype, np.integer) is False:
            raise ValueError
        if len(base.shape) != 1:
            raise ValueError

        selected_vec = np.zeros(len(base), dtype = np.integer)
        selected_vec[base != 0] = 1
        ret = dict()
        for i in np.unique(base[base != 0]):
            ret[str(i)] = dict()
            cur_test_vec = np.zeros(len(base), dtype = np.integer)
            cur_test_vec[base == i] = 1
            cur_train_vec = selected_vec ^ cur_test_vec
            ret[str(i)] = {
                'train': cur_train_vec,
                'test': cur_test_vec
            }
        return ret


    def auto_random_k_fold_cv_split(self, base: np.ndarray, k: int, seed=None):
        """
        Obtain k*2 split codes based on random K-Fold split, where train or test are labelled as 1 (corresp.)
        Useful when planning for a K-Fold cross validation.

        :param base: which split code/mask should be built on, 0: not included, 1: included
        :param k: number of bins
        :param seed: random seed
        :return: A dict() of
        """

        k_bin_label = self.auto_random_k_bin_labelling(base, k, seed)
        ret = self.get_k_fold_cv_split(k_bin_label)
        return ret
