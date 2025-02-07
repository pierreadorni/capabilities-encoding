import pandas as pd
from typing import List, Tuple
import numpy as np

def load_data(path='data.csv'):
    # the header is the first two rows. first row is dataset, second is modality. we merge them
    data = pd.read_csv(path, header=[0,1], index_col=0)
    new_cols = []
    last_dataset_name = ''
    for col in data.columns.values:
        dataset, modality = col
        if "Unnamed" in dataset:
            dataset = last_dataset_name
        else:
            last_dataset_name = dataset
        new_cols.append(dataset if "Unnamed" in modality else dataset + ' ' + modality)
    data.columns = new_cols
    # drop columns and rows with too many missing values
    last_shape = (0,0)
    while last_shape != data.shape:
        last_shape = data.shape
        data = data.dropna(axis=1, thresh=5)
        data = data.dropna(axis=0, thresh=5)
        
    # numbers are stored as strings with a comma in place of the dot, so we convert them to floats
    data = data.apply(lambda x: x.str.replace(',', '.').astype(float) if x.dtype == 'object' else x)
    # replace raw accuracies with distance to best accuracy for each col
    data = data.apply(lambda x: x.max() - x)
    return data

def to_relative(data):
    """
    Convert the data to relative distances by dividing each distance by the maximum distance in its column.
    """
    data = np.nan_to_num(data, nan=-1)
    relative_distances = data / data.max(axis=0, keepdims=True)
    relative_distances[relative_distances < 0] = np.nan
    return relative_distances


def mask_gt(gt, n_masked=10) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Mask some of the ground truth values to simulate missing data.

    :returns: the masked ground truth and a list of masked indexes
    """
    gt_masked = gt.copy()
    masked_indexes = np.zeros_like(gt, dtype=bool)
    non_inf_indexes = np.argwhere(~np.isnan(gt))
    # select n_masked indexes to mask
    selected_indexes = np.random.choice(non_inf_indexes.shape[0], n_masked, replace=False)

    for idx in selected_indexes:
        masked_indexes[tuple(non_inf_indexes[idx])] = True
        gt_masked[tuple(non_inf_indexes[idx])] = np.nan
    return gt_masked, masked_indexes
