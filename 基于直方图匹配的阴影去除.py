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
    :return: tuple, (img, shadow_mask, contours)
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

    return img, shadow_mask, contours

def histogram_matching(source, reference):
    """
    将source图像的直方图匹配到reference图像的直方图。

    :param source: ndarray, 要调整的图像（阴影区域）
    :param reference: ndarray, 目标图像（非阴影区域）
    :return: ndarray, 匹配后的图像
    """
    old_shape = source.shape
    source = source.ravel()
    reference = reference.ravel()

    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    r_values, r_counts = np.unique(reference, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64) / source.size
    r_quantiles = np.cumsum(r_counts).astype(np.float64) / reference.size

    interp_r_values = np.interp(s_quantiles, r_quantiles, r_values)
    return interp_r_values[bin_idx].reshape(old_shape)

def shadow_removal(img, shadow_mask):
    """
    基于直方图匹配的阴影去除算法。

    :param img: ndarray, 原始图像数据
    :param shadow_mask: ndarray, 阴影区域的二值掩码
    :return: ndarray, 阴影去除后的图像
    """
    shadow_area = img[shadow_mask]  # 阴影区域像素值
    non_shadow_area = img[~shadow_mask]  # 非阴影区域像素值

    # 初始化结果图像
    corrected_img = img.copy()
    
    for channel in range(img.shape[-1]):  # 对每个颜色通道独立操作
        corrected_img[shadow_mask, channel] = histogram_matching(
            shadow_area[:, channel], non_shadow_area[:, channel]
        )

    return corrected_img

def visualize_shadow_removal(original_img, corrected_img, shadow_mask):
    """
    可视化阴影去除前后的对比。

    :param original_img: ndarray, 原始图像数据
    :param corrected_img: ndarray, 阴影去除后的图像
    :param shadow_mask: ndarray, 阴影区域的二值掩码
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(shadow_mask, cmap="gray")
    axes[1].set_title("Shadow Mask")
    axes[1].axis("off")

    axes[2].imshow(corrected_img)
    axes[2].set_title("Corrected Image")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

# 示例调用
if __name__ == "__main__":
    tif_path = r"D:\\通用文件夹\\遥感原理与数字图像处理\\期末作业\\data\\tif\\4.tif"
    img, shadow_mask, contours = shadow_extraction(tif_path)
    corrected_img = shadow_removal(img, shadow_mask)
    visualize_shadow_removal(img, corrected_img, shadow_mask)
