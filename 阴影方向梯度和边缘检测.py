import numpy as np
import matplotlib.pyplot as plt
import rasterio
from skimage.filters import sobel
from skimage.morphology import remove_small_objects, binary_dilation, disk
from skimage import measure
from scipy.ndimage import laplace


def shadow_extraction(tif_path, gradient_threshold=1.5, min_shadow_size=1000):
    """
    基于梯度方向和边缘检测的阴影提取算法。

    :param tif_path: str, TIF 文件路径
    :param gradient_threshold: float, 梯度幅值阈值，用于检测低梯度区域。
    :param min_shadow_size: int, 阴影区域的最小面积（像素）。
    :return: tuple, (img, gray_img, gradient_magnitude, enhanced_edges, shadow_mask, contours)
    """
    with rasterio.open(tif_path) as src:
        img = src.read([1, 2, 3])  # 读取红、绿、蓝通道
        img = np.moveaxis(img, 0, -1)  # 转换成(H, W, C)的形状

    gray_img = np.mean(img, axis=-1)  # 转为灰度图像

    # 使用 Sobel 算子计算梯度幅值
    gradient_magnitude = sobel(gray_img)

    # 阈值处理，标记潜在阴影区域
    shadow_mask = gradient_magnitude < gradient_threshold

    # 使用拉普拉斯算子增强边缘
    enhanced_edges = laplace(gray_img)
    shadow_mask = shadow_mask & (enhanced_edges < 0)  # 结合低梯度与负边缘特性

    # 形态学操作（膨胀）
    shadow_mask = binary_dilation(shadow_mask, footprint=disk(3))

    # 移除小面积区域
    shadow_mask = remove_small_objects(shadow_mask, min_size=min_shadow_size)

    # 提取阴影轮廓
    contours = measure.find_contours(shadow_mask, 0.5)

    return img, gray_img, gradient_magnitude, enhanced_edges, shadow_mask, contours


def visualize_process(img, gray_img, gradient_magnitude, enhanced_edges, shadow_mask, contours):
    """
    可视化处理过程和最终阴影轮廓。

    :param img: ndarray, 原始图像数据
    :param gray_img: ndarray, 灰度图像
    :param gradient_magnitude: ndarray, 梯度幅值图像
    :param enhanced_edges: ndarray, 增强边缘图像
    :param shadow_mask: ndarray, 阴影二值掩码
    :param contours: list, 阴影轮廓坐标列表
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))

    # 显示原始图像
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 显示灰度图像
    axes[0, 1].imshow(gray_img, cmap='gray')
    axes[0, 1].set_title('Gray Image')
    axes[0, 1].axis('off')

    # 显示梯度幅值图像
    axes[1, 0].imshow(gradient_magnitude, cmap='viridis')
    axes[1, 0].set_title('Gradient Magnitude')
    axes[1, 0].axis('off')

    # 显示增强边缘图像
    axes[1, 1].imshow(enhanced_edges, cmap='coolwarm')
    axes[1, 1].set_title('Enhanced Edges')
    axes[1, 1].axis('off')

    # 显示阴影二值掩码
    axes[2, 0].imshow(shadow_mask, cmap='gray')
    axes[2, 0].set_title('Shadow Binary Mask')
    axes[2, 0].axis('off')

    # 显示原始图像和阴影轮廓
    axes[2, 1].imshow(img)
    for contour in contours:
        axes[2, 1].plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)
    axes[2, 1].set_title('Shadow Contours')
    axes[2, 1].axis('off')

    plt.tight_layout()
    plt.show()


# 示例调用
if __name__ == "__main__":
    tif_path = r"D:\\通用文件夹\\遥感原理与数字图像处理\\期末作业\\data\\tif\\4.tif"
    img, gray_img, gradient_magnitude, enhanced_edges, shadow_mask, contours = shadow_extraction(tif_path)
    visualize_process(img, gray_img, gradient_magnitude, enhanced_edges, shadow_mask, contours)
