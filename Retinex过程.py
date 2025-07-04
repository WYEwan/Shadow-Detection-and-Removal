import numpy as np
import matplotlib.pyplot as plt
import rasterio
from skimage import filters, measure
from skimage.morphology import remove_small_objects
from scipy.ndimage import gaussian_filter

def shadow_extraction(tif_path, scales=[15, 80, 250], threshold_factor=0.75, min_shadow_size=1000):
    """
    阴影提取算法，输入影像文件路径，输出阴影区域的二值掩码。

    :param tif_path: str, TIF文件路径
    :param scales: list, 多尺度Retinex算法的模糊尺度
    :param threshold_factor: float, 阈值调整因子
    :param min_shadow_size: int, 阴影区域的最小面积（像素）
    :return: tuple, (img, gray_img, msr_img, shadow_mask, contours)
    """
    with rasterio.open(tif_path) as src:
        img = src.read([1, 2, 3])  # 读取红、绿、蓝通道
        img = np.moveaxis(img, 0, -1)  # 转换成(H, W, C)的形状

    def msr(image, scales):
        result = np.zeros_like(image, dtype=np.float32)
        for scale in scales:
            blurred = gaussian_filter(image, sigma=scale)
            result += np.log1p(image / (blurred + 1e-6))
        return result

    gray_img = np.mean(img, axis=-1)  # 转为灰度图像
    msr_img = msr(gray_img, scales)  # 应用MSR算法

    shadow_threshold = filters.threshold_otsu(msr_img) * threshold_factor
    shadow_mask = msr_img < shadow_threshold  # 阴影二值掩码
    shadow_mask = remove_small_objects(shadow_mask, min_size=min_shadow_size)  # 移除小面积区域

    contours = measure.find_contours(shadow_mask, 0.5)  # 提取阴影轮廓

    return img, gray_img, msr_img, shadow_mask, contours

def visualize_processing_steps(img, gray_img, msr_img, shadow_mask, contours):
    """
    可视化阴影提取的中间过程和最终结果。

    :param img: ndarray, 原始图像数据
    :param gray_img: ndarray, 灰度图像
    :param msr_img: ndarray, MSR图像
    :param shadow_mask: ndarray, 阴影二值掩码
    :param contours: list, 阴影轮廓坐标列表
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 原始图像
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # 灰度图像
    axes[0, 1].imshow(gray_img, cmap="gray")
    axes[0, 1].set_title("Grayscale Image")
    axes[0, 1].axis("off")

    # MSR图像
    axes[0, 2].imshow(msr_img, cmap="gray")
    axes[0, 2].set_title("MSR Image")
    axes[0, 2].axis("off")

    # 阴影二值掩码
    axes[1, 0].imshow(shadow_mask, cmap="gray")
    axes[1, 0].set_title("Shadow Mask")
    axes[1, 0].axis("off")

    # 最终结果叠加轮廓
    axes[1, 1].imshow(img)
    for contour in contours:
        axes[1, 1].plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)
    axes[1, 1].set_title("Shadow Contours on Image")
    axes[1, 1].axis("off")

    # 隐藏多余的子图
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()

# 示例调用
if __name__ == "__main__":
    tif_path = r"D:\\通用文件夹\\遥感原理与数字图像处理\\期末作业\\data\\tif\\4.tif"
    img, gray_img, msr_img, shadow_mask, contours = shadow_extraction(tif_path)
    visualize_processing_steps(img, gray_img, msr_img, shadow_mask, contours)
