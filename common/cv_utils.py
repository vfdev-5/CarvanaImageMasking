
import numpy as np
from sklearn.model_selection import KFold

from .data_utils import UNIQUE_TRAIN_CAR_MAKERS, get_trainval_data_ids


def generate_k_folds_trainval_data_ids(n_folds):
    """
    """
    kf = KFold(n_splits=n_folds)
    np_unique_train_car_makers = np.array(UNIQUE_TRAIN_CAR_MAKERS)

    for train_car_makers_indices, val_car_makers_indices in kf.split(UNIQUE_TRAIN_CAR_MAKERS):
        train_car_makers = np_unique_train_car_makers[train_car_makers_indices]
        val_car_makers = np_unique_train_car_makers[val_car_makers_indices]

        train_data_ids = get_trainval_data_ids(train_car_makers)
        val_data_ids = get_trainval_data_ids(val_car_makers)

        yield train_data_ids, val_data_ids

