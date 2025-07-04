import numpy as np
import matplotlib.pyplot as plt
import rasterio
from skimage import filters, measure
from skimage.morphology import remove_small_objects
from scipy.ndimage import gaussian_filter
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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

def calculate_spectral_curves(img, shadow_mask, corrected_img):
    """
    计算并展示不同区域的光谱曲线，以及对应频段的均值和标准差。

    :param img: ndarray, 原始图像数据
    :param shadow_mask: ndarray, 阴影区域的二值掩码
    :param corrected_img: ndarray, 阴影去除后的图像
    """
    def calculate_stats(region_data, label):
        """
        计算某区域数据的均值、标准差，并打印结果。

        :param region_data: ndarray, 区域数据
        :param label: str, 区域标签
        """
        mean_values = np.mean(region_data, axis=(0, 1))
        std_values = np.std(region_data, axis=(0, 1))
        print(f"{label} - Mean: {mean_values}, Std: {std_values}")
        return mean_values, std_values

    # 计算总体图像的光谱曲线
    total_mean, total_std = calculate_stats(img, "Overall Image")

    # 计算处理前阴影区的光谱曲线
    shadow_mean, shadow_std = calculate_stats(img * shadow_mask[..., np.newaxis], "Shadow Area (Before Correction)")

    # 计算处理后非阴影区的光谱曲线
    non_shadow_mean, non_shadow_std = calculate_stats(
        corrected_img * (~shadow_mask[..., np.newaxis]),
        "Non-Shadow Area (After Correction)"
    )

    # 计算处理后阴影区的光谱曲线
    corrected_shadow_mean, corrected_shadow_std = calculate_stats(
        corrected_img * shadow_mask[..., np.newaxis],
        "Shadow Area (After Correction)"
    )

    # 绘制光谱曲线
    bands = np.arange(1, img.shape[-1] + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(bands, total_mean, label="Overall Image", marker='o')
    plt.plot(bands, shadow_mean, label="Shadow Area (Before Correction)", marker='o')
    plt.plot(bands, non_shadow_mean, label="Non-Shadow Area (After Correction)", marker='o')
    plt.plot(bands, corrected_shadow_mean, label="Shadow Area (After Correction)", marker='o')
    plt.fill_between(bands, total_mean - total_std, total_mean + total_std, alpha=0.2)
    plt.fill_between(bands, shadow_mean - shadow_std, shadow_mean + shadow_std, alpha=0.2)
    plt.fill_between(bands, non_shadow_mean - non_shadow_std, non_shadow_mean + non_shadow_std, alpha=0.2)
    plt.fill_between(bands, corrected_shadow_mean - corrected_shadow_std, corrected_shadow_mean + corrected_shadow_std, alpha=0.2)
    plt.xlabel("Spectral Band")
    plt.ylabel("Reflectance")
    plt.title("Spectral Curves with Mean and Std")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))

    # 绘制光谱曲线
    plt.plot(bands, total_mean, label="Overall Image", marker='o', color='blue')
    plt.plot(bands, shadow_mean, label="Shadow Area (Before Correction)", marker='o', color='orange')
    plt.plot(bands, non_shadow_mean, label="Non-Shadow Area (After Correction)", marker='o', color='green')
    plt.plot(bands, corrected_shadow_mean, label="Shadow Area (After Correction)", marker='o', color='red')

    # 填充标准差区域并添加图例
    plt.fill_between(bands, total_mean - total_std, total_mean + total_std, color='blue', alpha=0.2, label="Overall Image Std")
    plt.fill_between(bands, shadow_mean - shadow_std, shadow_mean + shadow_std, color='orange', alpha=0.2, label="Shadow Area (Before Correction) Std")
    plt.fill_between(bands, non_shadow_mean - non_shadow_std, non_shadow_mean + non_shadow_std, color='green', alpha=0.2, label="Non-Shadow Area (After Correction) Std")
    plt.fill_between(bands, corrected_shadow_mean - corrected_shadow_std, corrected_shadow_mean + corrected_shadow_std, color='red', alpha=0.2, label="Shadow Area (After Correction) Std")

    plt.xlabel("Spectral Band")
    plt.ylabel("Reflectance")
    plt.title("Spectral Curves with Mean and Std")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))

    # 绘制光谱曲线
    plt.plot(bands, total_mean, label="Overall Image", marker='o', color='blue')
    plt.plot(bands, shadow_mean, label="Shadow Area (Before Correction)", marker='o', color='orange')
    plt.plot(bands, non_shadow_mean, label="Non-Shadow Area (After Correction)", marker='o', color='green')
    plt.plot(bands, corrected_shadow_mean, label="Shadow Area (After Correction)", marker='o', color='red')

    # 使用虚线表示标准差
    plt.plot(bands, total_mean - total_std, linestyle='--', color='blue', alpha=0.6)
    plt.plot(bands, total_mean + total_std, linestyle='--', color='blue', alpha=0.6)
    plt.plot(bands, shadow_mean - shadow_std, linestyle='--', color='orange', alpha=0.6)
    plt.plot(bands, shadow_mean + shadow_std, linestyle='--', color='orange', alpha=0.6)
    plt.plot(bands, non_shadow_mean - non_shadow_std, linestyle='--', color='green', alpha=0.6)
    plt.plot(bands, non_shadow_mean + non_shadow_std, linestyle='--', color='green', alpha=0.6)
    plt.plot(bands, corrected_shadow_mean - corrected_shadow_std, linestyle='--', color='red', alpha=0.6)
    plt.plot(bands, corrected_shadow_mean + corrected_shadow_std, linestyle='--', color='red', alpha=0.6)

    plt.xlabel("Spectral Band")
    plt.ylabel("Reflectance")
    plt.title("Spectral Curves with Mean and Std")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))

    # 绘制光谱曲线
    plt.plot(bands, total_mean, label="整体图像均值", marker='o', color='blue')
    plt.plot(bands, shadow_mean, label="阴影区域（校正前）均值", marker='o', color='orange')
    plt.plot(bands, non_shadow_mean, label="非阴影区域（校正后）均值", marker='o', color='green')
    plt.plot(bands, corrected_shadow_mean, label="阴影区域（校正后）均值", marker='o', color='red')

    # 填充标准差区域
    plt.fill_between(bands, total_mean - total_std, total_mean + total_std, color='blue', alpha=0.1, label="整体图像标准差")
    plt.fill_between(bands, shadow_mean - shadow_std, shadow_mean + shadow_std, color='orange', alpha=0.1, label="阴影区域（校正前）标准差")
    plt.fill_between(bands, non_shadow_mean - non_shadow_std, non_shadow_mean + non_shadow_std, color='green', alpha=0.1, label="非阴影区域（校正后）标准差")
    plt.fill_between(bands, corrected_shadow_mean - corrected_shadow_std, corrected_shadow_mean + corrected_shadow_std, color='red', alpha=0.1, label="阴影区域（校正后）标准差")

    plt.xlabel("光谱波段")
    plt.ylabel("反射率")
    plt.title("带有均值和标准差的光谱曲线")
    plt.legend()
    plt.grid()
    plt.show()




# 调用示例
if __name__ == "__main__":
    tif_path = r"D:\\通用文件夹\\遥感原理与数字图像处理\\期末作业\\data\\tif\\4.tif"
    img, shadow_mask, contours = shadow_extraction(tif_path)
    corrected_img = shadow_removal(img, shadow_mask)
    calculate_spectral_curves(img, shadow_mask, corrected_img)

