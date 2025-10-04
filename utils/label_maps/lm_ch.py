import numpy as np

IMAGE_MEANS = np.array([117.67, 130.39, 121.52, 162.92])  # The setting here is for Chesapeake dataset
IMAGE_STDS = np.array([39.25, 37.82, 24.24, 60.03])

# LR中有哪些类是用得上的
LR_Label_Class = [0, 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]  # 这里0代表nodata
HR_Label_Class = [1, 2, 3, 4, 5, 6]

nodata = 0  # label里面都没0，可以用

# LR、HR、17类、4类的关系
mappings = {

    1: {
        'NLCD': [41, 42, 43, 90],
        'CCLC': [2],
        'cvpr': [8, 9, 10, 15],
        'label': 'Tree canopy',
        'color': (0, 83, 39)
    },
    2: {
        'NLCD': [31, 52, 71, 81, 82, 95],
        'CCLC': [3],
        'cvpr': [7, 11, 12, 13, 14, 16],
        'label': 'Low vegetation',
        'color': (226, 221, 193)
    },
    3: {
        'NLCD': [11],
        'CCLC': [1],
        'cvpr': [1],
        'label': 'Water',
        'color': (58, 101, 157)
    },
    4: {
        'NLCD': [21, 22, 23, 24],
        'CCLC': [4, 5, 6],
        'cvpr': [3, 4, 5, 6],
        'label': 'Impervious',
        'color': (221, 0, 0)
    },
}

def get_per_coarse_train_to_fine_map():
    """
    构建每个 coarse 类别下，train_id → cvpr_fine_id 的映射
    返回一个 dict：{ coarse_id: array, ... }
    """
    result = {}

    for coarse_id, info in mappings.items():
        fine_ids = sorted(info['cvpr'])  # 当前 coarse 的 fine 类别列表（cvpr ID）
        train_to_fine_map = np.array(fine_ids, dtype=np.int64)  # 顺序就是 train_id

        result[coarse_id] = train_to_fine_map

    return result


def get_per_coarse_fine_to_train_map():
    """
    构建每个 coarse 类别下，fine_id → 模型内部ID（0~N） 映射
    返回一个 dict：{ coarse_id: array, ... }
    """
    result = {}

    for coarse_id, info in mappings.items():
        fine_ids = sorted(info['cvpr'])  # 当前 coarse 的 fine 类别列表
        max_fine_id = 16
        fine_to_train_map = np.full(max_fine_id + 1, fill_value=0, dtype=np.int64)

        for local_id, fine_id in enumerate(fine_ids):
            fine_to_train_map[fine_id] = local_id+1

        result[coarse_id] = fine_to_train_map

    return result

def get_lr_to_17_cls():
    lr_to_17_cls_map = []

    value = 0  # 序列对应的值

    for label_idx in range(LR_Label_Class[-1] + 1):
        if label_idx in LR_Label_Class:
            lr_to_17_cls_map.append(value)
            value += 1
        else:
            lr_to_17_cls_map.append(nodata)

    lr_to_17_cls_map = np.array(lr_to_17_cls_map).astype(np.int64)
    return lr_to_17_cls_map


def get_lr_to_4_cls():
    lr_to_4_cls_map = np.full(LR_Label_Class[-1] + 1, fill_value=nodata)

    for category in mappings:
        nlcd = mappings.get(category)['NLCD']

        lr_to_4_cls_map[nlcd] = category

    return lr_to_4_cls_map
    pass


def get_hr_to_4_cls():
    hr_to_4_cls_map = np.full(HR_Label_Class[-1] + 1, fill_value=nodata)

    for category in mappings:
        cclc = mappings.get(category)['CCLC']

        hr_to_4_cls_map[cclc] = category

    return hr_to_4_cls_map
    pass


def get_17_cls_to_4_cls():
    _17_cls_to_4_cls = np.full(17, fill_value=nodata)

    for category in mappings:
        cvpr = mappings.get(category)['cvpr']

        _17_cls_to_4_cls[cvpr] = category

    return _17_cls_to_4_cls
    pass


def get_17_cls_to_lr():
    # 模型输出17分类后，需要转换回去然后才能上色
    _17_cls_to_lr_map = np.full(17, fill_value=nodata)

    for category in mappings:
        nlcd = mappings.get(category)['NLCD']
        cvpr = mappings.get(category)['cvpr']
        # print(nlcd,cvpr)

        _17_cls_to_lr_map[cvpr] = nlcd

    return _17_cls_to_lr_map
