from .lm_NY import get_UW_to_4cls, get_FCS10_to_train, get_train_to_FCS10
from .lm_ch import get_lr_to_4_cls, get_hr_to_4_cls, get_17_cls_to_4_cls, get_lr_to_17_cls, get_17_cls_to_lr
from .lm_pl import get_lr_and_train_map, get_train_to_4_cls_label_map, get_lr_to_4_cls_label_map
from .lm_FPB import get_train_to_fine,get_fine_to_train,get_fine_to_coarse
__all__ = ['get_xx_label_map']


def get_xx_label_map(in_type='', out_type=''):
    '''
        Literal[
        'nlcd_label', 'lc_label',
        'Target_4_cls',
        'nlcd_label_train',
        ]
    '''

    if out_type == 'Target_4_cls':
        if in_type == 'nlcd_label':  # LR label
            return get_lr_to_4_cls()

        if in_type == 'lc_label':  # HR label
            return get_hr_to_4_cls()

        if in_type == 'nlcd_label_train':  # LR Train label
            return get_17_cls_to_4_cls()

    if in_type == 'nlcd_label_train' and out_type == 'nlcd_label':
        return get_17_cls_to_lr()

    if in_type == 'nlcd_label' and out_type == 'nlcd_label_train':
        return get_lr_to_17_cls()

    '''
        Literal[
        'ESA_GLC10_label', 'Esri_GLC10_label', 'FORM_GLC10_label', 'GLC_FCS30_label',
         'HR_ground_truth', 'Target_4_cls',
        'ESA_GLC10_label_train', 'Esri_GLC10_label_train', 'FORM_GLC10_label_train', 'GLC_FCS30_label_train'
        ]
    '''
    if '_train' in out_type and in_type in ['ESA_GLC10_label', 'Esri_GLC10_label', 'FORM_GLC10_label',
                                            'GLC_FCS30_label', 'HR_ground_truth']:
        return get_lr_and_train_map(in_type)[0]

    if '_train' in in_type and out_type in ['ESA_GLC10_label', 'Esri_GLC10_label', 'FORM_GLC10_label',
                                            'GLC_FCS30_label', 'HR_ground_truth']:
        return get_lr_and_train_map(out_type)[-1]

    if out_type == 'Target_4_cls':
        if '_train' in in_type:
            return get_train_to_4_cls_label_map(in_type)
        else:
            return get_lr_to_4_cls_label_map(in_type)


    '''
        Literal[
        'fpb_fine', 'fpb_coarse',
        'fpb_fine_train',
        ]
    '''
    if in_type == 'fpb_fine':
        if out_type == 'fpb_fine_train':
            return get_fine_to_train()
        if out_type == 'fpb_fine_coarse':
            return get_fine_to_coarse()

    if in_type == 'fpb_fine_train' and out_type == 'fpb_fine':
        return get_train_to_fine()

    '''
        Literal[
        'UW_label', 'FCS10_label',
        'UW_label_train', 'FCS10_label_train',
        'UW_4cls_label'
        ]
    '''
    if in_type == 'UW_label' and out_type == 'UW_4cls_label':
        return get_UW_to_4cls()

    if in_type == 'FCS10_label' and out_type == 'FCS10_label_train':
        return get_FCS10_to_train()

    if in_type == 'FCS10_label_train' and out_type == 'FCS10_label':
        return get_train_to_FCS10()

    raise ValueError("function 'get_xx_label_map' get an invalid input parameter", in_type, out_type)
