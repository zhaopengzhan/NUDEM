import numpy as np

fine_label_classes = list(range(25))
ignored_fine_class_ids = [8, 16, 17]  # 公园 雪地 裸地
# 从 fine_label_classes 中移除 ignored_fine_class_ids 中的所有元素
fine_label_classes = [cls for cls in fine_label_classes if cls not in ignored_fine_class_ids]

coarse_label_classes = list(range(6))
nodata = 0

mappings = {
    1: {
        'fbp': [1, 21, 12, 22, 18, 23, 19, 24, 20],  # 9个类别
        'label': 'impervious',
        'color': (221, 0, 0)
    },
    2: {
        'fbp': [2, 3, 4],  # paddy field, irrigated field, dry cropland
        'label': 'crop',
        'color': (150, 200, 150)
    },
    3: {
        'fbp': [5, 6, 7],  # arbor forest, shrub forest
        'label': 'forest',
        'color': (0, 83, 39)
    },
    4: {
        'fbp': [9, 10],  # natural meadow, artificial meadow
        'label': 'grassland',
        'color': (226, 221, 193)
    },
    5: {
        'fbp': [11, 13, 14, 15],  # river, lake, pond, fish pond
        'label': 'water',
        'color': (58, 101, 157)
    },

}


def get_fine_to_train():
    max_idx = max(fine_label_classes)
    fine_to_train_map = np.full(max_idx + 1, fill_value=nodata, dtype=np.int64)

    for train_id, fine_id in enumerate(sorted(fine_label_classes)):
        fine_to_train_map[fine_id] = train_id
    return fine_to_train_map

def get_train_to_fine():
    len_train = len(fine_label_classes)
    train_to_fine_map  = np.full(len_train, fill_value=nodata, dtype=np.int64)

    for train_id, fine_id in enumerate(sorted(fine_label_classes)):
        train_to_fine_map [train_id] = fine_id
    return train_to_fine_map

def get_fine_to_coarse():
    max_idx = max(fine_label_classes)
    fine_to_coarse_cls_map = np.full(max_idx + 1, fill_value=nodata)

    for coarse_id, mapping in mappings.items():
        for fine_id in mapping['fbp']:
            fine_to_coarse_cls_map[fine_id] = coarse_id

    return fine_to_coarse_cls_map
    pass


if __name__ == '__main__':
    #
    # print(get_fine_to_train())
    # mm = np.random.randint(0, 25, 100)
    # rr = get_fine_to_train()[mm]
    # print(max(rr))

    #
    print(get_train_to_fine())
    mm = np.random.randint(0, 22, 1000)
    rr = get_train_to_fine()[mm]
    print(max(rr))
