from typing import Literal

import numpy as np

nodata = 0

label_lr_class = {
    'ESA_GLC10_label': [0, 10, 20, 30, 40, 50, 60, 80, 90, 95, 100],
    'Esri_GLC10_label': [0, 1, 2, 4, 5, 7, 8, 10, 11],
    'FORM_GLC10_label': [0, 1, 2, 3, 5, 6, 8, 9],
    'GLC_FCS30_label': [0, 10, 11, 12, 60, 61, 70, 90, 130, 150, 180, 190, 210],
    'HR_ground_truth': [0, 1, 2, 3, 4, 5, 6, 7, 8],

    'ESA_GLC10_label_train': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Esri_GLC10_label_train': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'FORM_GLC10_label_train': [0, 1, 2, 3, 4, 5, 6, 7],
    'GLC_FCS30_label_train': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
}


def get_lr_and_train_map(
        type: Literal[
            'ESA_GLC10_label', 'Esri_GLC10_label', 'FORM_GLC10_label', 'GLC_FCS30_label',
            'HR_ground_truth',
        ] = '', ):
    value = 0  # 序列对应的值
    LR_Label_Class = label_lr_class[type]

    lr_to_train_map = []
    train_to_lt_map = np.full(shape=len(LR_Label_Class), fill_value=nodata)

    for label_idx in range(LR_Label_Class[-1] + 1):
        if label_idx in LR_Label_Class:
            lr_to_train_map.append(value)
            train_to_lt_map[value] = label_idx
            value += 1
        else:
            lr_to_train_map.append(nodata)

    return np.array(lr_to_train_map), np.array(train_to_lt_map)


mappings = {

    1: {
        'label': 'Tree canopy',
        'color': (0, 83, 39),
        'HR_ground_truth': [5],
        'FORM_GLC10_label': [2],
        'Esri_GLC10_label': [2],
        'ESA_GLC10_label': [10, 95],
        'GLC_FCS30_label': [60, 61,62, 70,72,120, 90, 180],
    },
    2: {
        'label': 'Low vegetation',
        'color': (226, 221, 193),
        'HR_ground_truth': [2, 7],
        'FORM_GLC10_label': [1, 3, 4],
        'Esri_GLC10_label': [3, 5, 6, 11],
        'ESA_GLC10_label': [20, 30, 40, 100],
        'GLC_FCS30_label': [10, 11, 12, 130, 150],
    },
    3: {
        'label': 'Water',
        'color': (58, 101, 157),
        'HR_ground_truth': [6],
        'FORM_GLC10_label': [5, 6],
        'Esri_GLC10_label': [1, 4],
        'ESA_GLC10_label': [80, 90],
        'GLC_FCS30_label': [210],
    },
    4: {
        'label': 'Built-up',
        'color': (221, 0, 0),
        'HR_ground_truth': [1, 3, 4, 8],
        'FORM_GLC10_label': [8, 9],
        'Esri_GLC10_label': [7, 8],
        'ESA_GLC10_label': [50, 60],
        'GLC_FCS30_label': [190],

    },
}


def get_lr_to_4_cls_label_map(in_type=''):
    label_config = label_lr_class[in_type]
    num_class = label_config[-1]
    in_to_out_map = np.full(num_class + 1, fill_value=nodata)

    for category in mappings.keys():
        in_type_idx = mappings.get(category)[in_type]
        in_to_out_map[in_type_idx] = category
    return in_to_out_map


def get_train_to_4_cls_label_map(in_type):
    in_type = in_type.split('_train')[0]
    lr_to_train = get_lr_and_train_map(in_type)[0]
    lr_to_4_cls = get_lr_to_4_cls_label_map(in_type)

    num_class = max(lr_to_train) + 1
    train_to_hr = np.zeros((num_class, num_class), dtype=int)

    for lr_idx in range(len(lr_to_train)):
        train_idx = lr_to_train[lr_idx]
        hr_idx = lr_to_4_cls[lr_idx]
        train_to_hr[train_idx, hr_idx] = 1

    return np.argmax(train_to_hr, axis=1)
