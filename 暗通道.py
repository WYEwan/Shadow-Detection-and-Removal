import numpy as np
import matplotlib.pyplot as plt
import rasterio
from skimage.morphology import remove_small_objects, binary_dilation, disk
from skimage import measure


def dark_channel_prior_shadow_extraction(tif_path, window_size=20, threshold=15, min_shadow_size=1000):
    """
    基于暗通道先验的阴影提取算法。

    :param tif_path: str, TIF 文件路径
    :param window_size: int, 窗口大小，用于计算暗通道。
    :param threshold: float, 阴影检测的暗通道值阈值。
    :param min_shadow_size: int, 阴影区域的最小面积（像素）。
    :return: tuple, (img, dark_channel, dark_channel_min, shadow_mask, contours)
    """
    with rasterio.open(tif_path) as src:
        img = src.read([1, 2, 3])  # 读取红、绿、蓝通道
        img = np.moveaxis(img, 0, -1)  # 转换成(H, W, C)的形状

    # 计算暗通道图像
    dark_channel = np.min(img, axis=-1)  # 对RGB通道取最小值
    pad_size = window_size // 2
    padded_dark_channel = np.pad(dark_channel, pad_size, mode='edge')
    dark_channel_min = np.zeros_like(dark_channel)

    for i in range(dark_channel.shape[0]):
        for j in range(dark_channel.shape[1]):
            dark_channel_min[i, j] = np.min(
                padded_dark_channel[i:i + window_size, j:j + window_size]
            )

    # 阈值处理，标记潜在阴影区域
    shadow_mask = dark_channel_min < threshold

    # 形态学操作（膨胀）
    shadow_mask = binary_dilation(shadow_mask, footprint=disk(3))

    # 移除小面积区域
    shadow_mask = remove_small_objects(shadow_mask, min_size=min_shadow_size)

    # 提取阴影轮廓
    contours = measure.find_contours(shadow_mask, 0.5)

    return img, dark_channel, dark_channel_min, shadow_mask, contours


def visualize_process(img, dark_channel, dark_channel_min, shadow_mask, contours):
    """
    可视化处理过程和最终阴影轮廓。

    :param img: ndarray, 原始图像数据
    :param dark_channel: ndarray, 原始暗通道图像
    :param dark_channel_min: ndarray, 最小滤波后的暗通道图像
    :param shadow_mask: ndarray, 阴影二值掩码
    :param contours: list, 阴影轮廓坐标列表
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 显示原始图像
    axes[0, 0].imshow(img / 255.0)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 显示原始暗通道图像
    axes[0, 1].imshow(dark_channel, cmap='gray')
    axes[0, 1].set_title('Original Dark Channel')
    axes[0, 1].axis('off')

    # 显示最小滤波后的暗通道图像
    axes[1, 0].imshow(dark_channel_min, cmap='gray')
    axes[1, 0].set_title('Filtered Dark Channel (Min)')
    axes[1, 0].axis('off')

    # 显示阴影二值掩码图像
    axes[1, 1].imshow(shadow_mask, cmap='gray')
    for contour in contours:
        axes[1, 1].plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)
    axes[1, 1].set_title('Shadow Binary Mask with Contours')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


# 示例调用
if __name__ == "__main__":
    tif_path = r"D:\\通用文件夹\\遥感原理与数字图像处理\\期末作业\\data\\tif\\4.tif"
    img, dark_channel, dark_channel_min, shadow_mask, contours = dark_channel_prior_shadow_extraction(tif_path)
    visualize_process(img, dark_channel, dark_channel_min, shadow_mask, contours)
