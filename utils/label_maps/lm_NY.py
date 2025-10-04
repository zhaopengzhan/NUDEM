import numpy as np

IMAGE_MEANS = np.array([102.42, 101.74, 86.01])  # The setting here is for Chesapeake dataset
IMAGE_STDS = np.array([54.01, 49.89, 58.33])

FCS10_Label_Class = [
    0, 11, 12, 20, 51, 52, 61, 62, 71, 72, 81, 82, 91, 92, 121, 122, 130, 140, 181, 182, 183, 184, 185, 186, 187, 191,
    192, 200, 210, 220,0
]
UrbanWatch_Label_Class = list(range(9))
nodata = 0

train_mappings = {
    1: {
        'UrbanWatch': [4],
        'FCS10': [51, 52, 61, 62, 71, 72, 81, 82, 91, 92],
        'label': 'Tree Canopy',
        'color': (34, 139, 34)
    },
    2: {
        'UrbanWatch': [5],
        'FCS10': [121, 122, 130, 150, 140],
        'label': 'Grass/Shrub',
        'color': (128, 236, 104)
    },
    3: {
        'UrbanWatch': [6],
        'FCS10': [11, 12, 20],
        'label': 'Agriculture',
        'color': (255, 193, 37)
    },
    4: {
        'UrbanWatch': [7],
        'FCS10': [210],
        'label': 'Water',
        'color': (0, 0, 255)
    },
}
# urban -> GLC_FCS10 mapping
HR_to_LR_mappings = {
    1: {

        'label': 'Building',
        'color': (255, 0, 0)
    },
    2: {

        'label': 'Road',
        'color': (133, 133, 133)
    },
    3: {
        'label': 'Parking Lot',
        'color': (255, 0, 192)
    },
    4: {
        'FCS10': [51, 52, 61, 62, 71, 72, 81, 82, 91, 92],
        'label': 'Tree Canopy',
        'color': (34, 139, 34)
    },
    5: {
        'FCS10': [121, 122, 130, 150, 140],
        'label': 'Grass/Shrub',
        'color': (128, 236, 104)
    },
    6: {
        'FCS10': [11, 12, 20],
        'label': 'Agriculture',
        'color': (255, 193, 37)
    },
    7: {
        'FCS10': [210],
        'label': 'Water',
        'color': (0, 0, 255)
    },
    8: {
        'label': 'Barren',
        'color': (128, 0, 0)
    },

}

def get_UW_to_train():
    max_idx = max(UrbanWatch_Label_Class)
    fine_to_train_map = np.full(max_idx + 1, fill_value=nodata, dtype=np.int64)

    for train_id, fine_id in enumerate(sorted(UrbanWatch_Label_Class)):
        fine_to_train_map[fine_id] = train_id
    return fine_to_train_map

def get_FCS10_to_train():
    max_idx = max(FCS10_Label_Class)
    fine_to_train_map = np.full(max_idx + 1, fill_value=nodata, dtype=np.int64)

    for train_id, fine_id in enumerate(sorted(FCS10_Label_Class)):
        fine_to_train_map[fine_id] = train_id
    return fine_to_train_map

def get_train_to_FCS10():
    len_train = len(FCS10_Label_Class)
    _map = np.full(len_train, fill_value=nodata, dtype=np.int64)

    for train_id, fine_id in enumerate(sorted(FCS10_Label_Class)):
        _map[train_id] = fine_id
    return _map

def get_UW_to_4cls():
    max_idx = max(UrbanWatch_Label_Class)
    _map = np.full(max_idx + 1, fill_value=nodata)

    for coarse_id, mapping in train_mappings.items():
        for fine_id in mapping['UrbanWatch']:
            _map[fine_id] = coarse_id

    return _map

if __name__ == '__main__':
    print(get_train_to_FCS10())