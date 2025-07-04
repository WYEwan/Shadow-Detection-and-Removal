import numpy as np
import matplotlib.pyplot as plt
import rasterio
from skimage import filters, measure
from skimage.morphology import remove_small_objects
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2hsv, rgb2lab

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

def estimate_alpha(img, shadow_mask):
    """
    估计阴影区域的光照衰减因子。

    :param img: ndarray, 原始图像数据
    :param shadow_mask: ndarray, 阴影区域的二值掩码
    :return: ndarray, 每个像素的光照衰减因子（范围 0~1）
    """
    gray_img = np.mean(img, axis=-1)  # 转为灰度图像
    non_shadow_intensity = np.median(gray_img[~shadow_mask])  # 非阴影区域的中位亮度
    shadow_intensity = gray_img * shadow_mask  # 阴影区域的亮度

    alpha = np.ones_like(gray_img)
    alpha[shadow_mask] = shadow_intensity[shadow_mask] / non_shadow_intensity
    alpha[alpha > 1] = 1  # 确保最大值不超过1
    alpha[alpha < 0.1] = 0.1  # 避免过小值导致不稳定

    return alpha

def shadow_removal(img, shadow_mask):
    """
    基于区域特征匹配的阴影去除算法。

    :param img: ndarray, 原始图像数据
    :param shadow_mask: ndarray, 阴影区域的二值掩码
    :return: ndarray, 阴影去除后的图像
    """
    corrected_img = img.copy()
    
    # 获取非阴影区域的平均值和标准差
    non_shadow_mask = ~shadow_mask
    mean_values = np.mean(img[non_shadow_mask], axis=0)
    std_values = np.std(img[non_shadow_mask], axis=0)

    # 获取阴影区域的平均值和标准差
    shadow_mean = np.mean(img[shadow_mask], axis=0)
    shadow_std = np.std(img[shadow_mask], axis=0)

    # 调整阴影区域的像素值
    for channel in range(img.shape[-1]):
        corrected_img[..., channel][shadow_mask] = (
            (img[..., channel][shadow_mask] - shadow_mean[channel]) / shadow_std[channel]
        ) * std_values[channel] + mean_values[channel]

    corrected_img[corrected_img > 255] = 255  # 防止溢出
    corrected_img[corrected_img < 0] = 0      # 防止负值
    corrected_img = corrected_img.astype(np.uint8)

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

def calculate_and_plot_histograms(original_img, corrected_img, shadow_mask):
    """
    计算并绘制不同场景下的灰度值直方图。

    :param original_img: ndarray, 原始图像数据
    :param corrected_img: ndarray, 阴影去除后的图像
    :param shadow_mask: ndarray, 阴影区域的二值掩码
    """
    # 转换为灰度图像
    gray_original = np.mean(original_img, axis=-1).astype(np.uint8)
    gray_corrected = np.mean(corrected_img, axis=-1).astype(np.uint8)

    # 计算直方图
    hist_original = np.histogram(gray_original, bins=256, range=(0, 255))[0]
    hist_shadow_area = np.histogram(gray_original[shadow_mask], bins=256, range=(0, 255))[0]
    hist_corrected_non_shadow = np.histogram(gray_corrected[~shadow_mask], bins=256, range=(0, 255))[0]
    hist_corrected_shadow_area = np.histogram(gray_corrected[shadow_mask], bins=256, range=(0, 255))[0]

    # 绘制直方图
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    axes[0, 0].plot(hist_original, color='blue')
    axes[0, 0].set_title("Original Image Histogram")
    
    axes[0, 1].plot(hist_shadow_area, color='red')
    axes[0, 1].set_title("Original Shadow Area Histogram")
    
    axes[1, 0].plot(hist_corrected_non_shadow, color='green')
    axes[1, 0].set_title("Corrected Non-Shadow Area Histogram")
    
    axes[1, 1].plot(hist_corrected_shadow_area, color='purple')
    axes[1, 1].set_title("Corrected Shadow Area Histogram")
    
    for ax in axes.flatten():
        ax.set_xlim(0, 255)
        ax.set_xlabel("Gray Level")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

from skimage.color import rgb2hsv, rgb2lab

def calculate_color_spaces(img, mask=None):
    """
    计算HSV和LAB色彩空间的分量。

    :param img: ndarray, 输入RGB图像
    :param mask: ndarray, 可选，二值掩码，若提供则仅计算掩码区域的分量
    :return: dict, 包含HSV和LAB分量的字典
    """
    hsv_img = rgb2hsv(img)
    lab_img = rgb2lab(img)

    if mask is not None:
        mask = mask.astype(bool)
        hsv_values = {channel: hsv_img[..., channel][mask] for channel in range(3)}
        lab_values = {channel: lab_img[..., channel][mask] for channel in range(3)}
    else:
        hsv_values = {channel: hsv_img[..., channel].flatten() for channel in range(3)}
        lab_values = {channel: lab_img[..., channel].flatten() for channel in range(3)}

    return {
        "HSV": hsv_values,
        "LAB": lab_values
    }

def visualize_color_spaces(color_spaces, title_prefix=""):
    """
    可视化HSV和LAB色彩空间的分量。

    :param color_spaces: dict, 包含HSV和LAB分量的字典
    :param title_prefix: str, 图像标题前缀
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    space_labels = ["H", "S", "V", "L", "A", "B"]

    for i, (space_name, values) in enumerate(color_spaces.items()):
        for j, (channel, data) in enumerate(values.items()):
            ax = axes[i, j]
            ax.hist(data, bins=256, color='gray', alpha=0.7)
            ax.set_title(f"{title_prefix} {space_name} - {space_labels[i * 3 + j]}")
            ax.set_xlim(0, 1 if space_name == "HSV" else 100)

    plt.tight_layout()
    plt.show()

# 示例调用新的功能
if __name__ == "__main__":
    tif_path = r"D:\\通用文件夹\\遥感原理与数字图像处理\\期末作业\\data\\tif\\1.tif"
    img, shadow_mask, contours = shadow_extraction(tif_path)
    corrected_img = shadow_removal(img, shadow_mask)

    # 处理前的HSV和LAB色彩空间分量
    original_color_spaces = calculate_color_spaces(img)
    visualize_color_spaces(original_color_spaces, title_prefix="Original")

    # 处理前阴影区的HSV和LAB色彩空间分量
    shadow_color_spaces = calculate_color_spaces(img, shadow_mask)
    visualize_color_spaces(shadow_color_spaces, title_prefix="Shadow Area")

    # 处理后的非阴影区的HSV和LAB色彩空间分量
    corrected_color_spaces = calculate_color_spaces(corrected_img, ~shadow_mask)
    visualize_color_spaces(corrected_color_spaces, title_prefix="Corrected Non-Shadow Area")

    # 处理后的阴影区的HSV和LAB色彩空间分量
    corrected_shadow_color_spaces = calculate_color_spaces(corrected_img, shadow_mask)
    visualize_color_spaces(corrected_shadow_color_spaces, title_prefix="Corrected Shadow Area")

