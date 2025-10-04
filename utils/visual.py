import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# from osgeo import gdal

plt.rc('font', family='Times New Roman')


def mask_to_png(mask_array: np.ndarray, color_map: dict) -> np.ndarray:
    """
    Converts a category mask array to a color image array.

    Parameters:
    mask_array (numpy.ndarray) [H, W]: 2D array containing the mask image, where each value represents a category.
    color_map (dict): Dictionary where the keys are category values and the values are dictionaries containing color and label.
                      For example:
                      {
                          0: {'color': [0, 0, 0], 'label': 'background'},
                          1: {'color': [255, 0, 0], 'label': 'category1'},
                          ...
                      }

    Returns:
    numpy.ndarray: 3D array representing the color image, where each pixel's color corresponds to the category value in the input mask.
    """
    # 获取图像尺寸 [H, W]
    height, width = mask_array.shape

    # 创建一个新的彩色图像数组
    color_array = np.zeros((height, width, 3), dtype=np.uint8)

    # 使用矢量化操作，将每个类别值映射到对应的颜色
    for category in color_map:
        color = color_map.get(category)['color']
        label = color_map.get(category)['label']

        color_array[mask_array == category] = color

    return color_array


def save_mask(color_array, save_path=None):
    # 将彩色数组转换为Pillow图像
    color_image = Image.fromarray(color_array)

    if save_path:
        # 保存彩色图像
        color_image.save(save_path)

    # 显示彩色图像
    # color_image.show()


def plot_mask(mask_array: np.ndarray, color_map: dict, isShow=True, isLegend=True, save_path=None, show_percent=True):
    color_array = mask_to_png(mask_array, color_map)
    uni_list, count_list = np.unique(mask_array, return_counts=True)
    # print(uni_list, count_list)
    height, width = mask_array.shape
    #
    # fig, ax = plt.subplots(figsize=(height // 10, width // 10), dpi=10)
    fig, ax = plt.subplots()
    fig_width, fig_height = fig.get_size_inches()
    font_size = round(fig_height * 2)
    plt.rcParams.update({'font.size': font_size})

    # 隐藏坐标轴
    ax.axis('off')

    plt.imshow(color_array)

    if isLegend:
        # 创建图例项
        patches = []
        for category, color_info in color_map.items():
            if category in uni_list:
                color = np.array(color_info['color']) / 255.0  # 将颜色转换为matplotlib可用的格式
                label = color_info['label']

                if show_percent:
                    label = label + f' {count_list[np.where(uni_list == category)][0] / count_list.sum():.2%}'

                patches.append(mpatches.Patch(color=color, label=label))

        # 添加图例
        # plt.legend(handles=patches, loc='best', bbox_to_anchor=(0, 0))
        plt.legend(handles=patches, loc='best')

    if save_path:
        # 保存图例
        plt.savefig(save_path)

    if isShow:
        # 显示图例
        plt.show()

    pass
