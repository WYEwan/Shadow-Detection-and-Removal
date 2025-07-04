import cv2
import rasterio
import numpy as np
import tkinter as tk
from osgeo import gdal
from skimage import measure
from scipy.ndimage import label
import matplotlib.pyplot as plt
from skimage.filters import gabor
from skimage.filters import sobel
from scipy.ndimage import laplace
from sklearn.cluster import KMeans
from skimage.filters import laplace
from skimage import filters, measure
from skimage.filters import gaussian
from matplotlib.widgets import Line2D
from scipy.ndimage import gaussian_filter
from tkinter import filedialog, messagebox
from skimage.exposure import equalize_hist
from skimage.color import rgb2hsv, rgb2lab
from skimage.exposure import match_histograms
from skimage.morphology import remove_small_objects
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.morphology import remove_small_objects, binary_dilation, disk

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
current_function = None

def shadow_extraction_msr_filter(tif_path, scales=[15, 80, 250], threshold_factor=0.75, min_shadow_size=1000):
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

def visualize_shadow_1(img, contours):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)  # 显示原始图像
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)  # 绘制轮廓
    ax.set_title('Shadow Extraction with Contours')
    ax.axis('off')
    plt.show()

def shadow_extraction_msr_extraction_process(tif_path, scales=[15, 80, 250], threshold_factor=0.75, min_shadow_size=1000):
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

def visualize_processing_steps_1(img, gray_img, msr_img, shadow_mask, contours):
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

def shadow_extraction_gabor(tif_path, gabor_frequencies=[0.1, 0.2, 0.3], threshold_factor=0.50, min_shadow_size=1000):
    with rasterio.open(tif_path) as src:
        img = src.read([1, 2, 3])  # 读取红、绿、蓝通道
        img = np.moveaxis(img, 0, -1)  # 转换成(H, W, C)的形状

    gray_img = np.mean(img, axis=-1)  # 转为灰度图像

    # 应用 Gabor 滤波器提取纹理特征
    gabor_responses = np.zeros_like(gray_img, dtype=np.float32)
    for freq in gabor_frequencies:
        response, _ = gabor(gray_img, frequency=freq)
        gabor_responses += np.abs(response)  # 累加不同频率下的响应

    # 基于 Otsu 方法计算全局阈值，并调整以检测阴影区域
    shadow_threshold = np.mean(gabor_responses) * threshold_factor
    shadow_mask = gabor_responses < shadow_threshold  # 阴影二值掩码

    # 移除小面积区域
    shadow_mask = remove_small_objects(shadow_mask, min_size=min_shadow_size)

    # 提取阴影轮廓
    contours = measure.find_contours(shadow_mask, 0.5)

    return img, shadow_mask, contours

def visualize_shadow_gabor(img, contours):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)  # 显示原始图像
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)  # 绘制轮廓
    ax.set_title('Shadow Extraction with Contours')
    ax.axis('off')
    plt.show()

def shadow_extraction_gabor_process(tif_path, gabor_frequencies=[0.1, 0.2, 0.3], threshold_factor=0.50, min_shadow_size=1000):
    with rasterio.open(tif_path) as src:
        img = src.read([1, 2, 3])  # 读取红、绿、蓝通道
        img = np.moveaxis(img, 0, -1)  # 转换成(H, W, C)的形状

    gray_img = np.mean(img, axis=-1)  # 转为灰度图像

    # 应用 Gabor 滤波器提取纹理特征
    gabor_responses = np.zeros_like(gray_img, dtype=np.float32)
    for freq in gabor_frequencies:
        response, _ = gabor(gray_img, frequency=freq)
        gabor_responses += np.abs(response)  # 累加不同频率下的响应

    # 基于 Otsu 方法计算全局阈值，并调整以检测阴影区域
    shadow_threshold = np.mean(gabor_responses) * threshold_factor
    shadow_mask = gabor_responses < shadow_threshold  # 阴影二值掩码

    # 移除小面积区域
    shadow_mask = remove_small_objects(shadow_mask, min_size=min_shadow_size)

    # 提取阴影轮廓
    contours = measure.find_contours(shadow_mask, 0.5)

    return img, gray_img, gabor_responses, shadow_mask, contours

def visualize_process_gabor_process(img, gray_img, gabor_responses, shadow_mask, contours):
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # 显示灰度图像
    axes[0, 0].imshow(gray_img, cmap='gray')
    axes[0, 0].set_title('Gray Image')
    axes[0, 0].axis('off')

    # 显示 Gabor 响应累加图像
    axes[0, 1].imshow(gabor_responses, cmap='viridis')
    axes[0, 1].set_title('Gabor Responses')
    axes[0, 1].axis('off')

    # 显示阴影二值掩码
    axes[1, 0].imshow(shadow_mask, cmap='gray')
    axes[1, 0].set_title('Shadow Binary Mask')
    axes[1, 0].axis('off')

    # 显示原始图像和阴影轮廓
    axes[1, 1].imshow(img)
    for contour in contours:
        axes[1, 1].plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)
    axes[1, 1].set_title('Shadow Contours')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

def shadow_extraction_direction(tif_path, gradient_threshold=1.5, min_shadow_size=1000):
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

    return img, shadow_mask, contours

def visualize_shadow_direction(img, contours):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)  # 显示原始图像
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)  # 绘制轮廓
    ax.set_title('Shadow Extraction with Contours')
    ax.axis('off')
    plt.show()

def shadow_extraction_direction_process(tif_path, gradient_threshold=1.5, min_shadow_size=1000):
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

def visualize_process_direction_process(img, gray_img, gradient_magnitude, enhanced_edges, shadow_mask, contours):
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

def dark_channel_prior_shadow_extraction_dcp(tif_path, window_size=20, threshold=15, min_shadow_size=1000):
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

    return img, shadow_mask, contours

def visualize_shadow_dcp(img, contours):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img / 255.0)  # 显示原始图像，归一化以便于显示
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)  # 绘制轮廓
    ax.set_title('Shadow Extraction with Dark Channel Prior')
    ax.axis('off')
    plt.show()

def dark_channel_prior_shadow_extraction_dcp_process(tif_path, window_size=20, threshold=15, min_shadow_size=1000):
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

def visualize_process_dcp_process(img, dark_channel, dark_channel_min, shadow_mask, contours):
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

#function9:histogram
def shadow_extraction_histogram(tif_path, num_clusters=3, min_shadow_size=1000):
    with rasterio.open(tif_path) as src:
        img = src.read([1, 2, 3])  # 读取红、绿、蓝通道
        img = np.moveaxis(img, 0, -1)  # 转换成(H, W, C)的形状

    gray_img = np.mean(img, axis=-1)  # 转为灰度图像

    # 直方图均衡化增强对比度
    enhanced_img = equalize_hist(gray_img)

    # 高通滤波提取高频信息
    high_pass_img = np.abs(laplace(enhanced_img))  # 使用绝对值保证高频信息为正值

    # 将高频信息与均衡化图像结合，增强阴影对比
    combined_img = enhanced_img - 0.5 * high_pass_img
    combined_img = np.clip(combined_img, 0, 1)  # 限制在[0, 1]范围内

    # 聚类分割（K-means）
    pixels = combined_img.reshape(-1, 1)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(pixels)
    clustered_img = kmeans.labels_.reshape(gray_img.shape)

    # 假定最暗簇为阴影区域
    shadow_cluster = np.argmin(kmeans.cluster_centers_)
    shadow_mask = clustered_img == shadow_cluster

    # 移除小区域并膨胀
    shadow_mask = remove_small_objects(shadow_mask, min_size=min_shadow_size)
    shadow_mask = binary_dilation(shadow_mask, footprint=disk(3))

    # 提取阴影轮廓
    labeled_mask, _ = label(shadow_mask)
    contours = measure.find_contours(labeled_mask, 0.5)

    return img, shadow_mask, contours

def visualize_shadow_histogram(img, contours):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img / 255)  # 显示原始图像，归一化到[0, 1]
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)  # 绘制轮廓
    ax.set_title('Shadow Extraction with Contours')
    ax.axis('off')
    plt.show()

#function10:histogram process
def shadow_extraction_histogram_process(tif_path, num_clusters=3, min_shadow_size=1000):
    with rasterio.open(tif_path) as src:
        img = src.read([1, 2, 3])  # 读取红、绿、蓝通道
        img = np.moveaxis(img, 0, -1)  # 转换成(H, W, C)的形状

    gray_img = np.mean(img, axis=-1)  # 转为灰度图像

    # 直方图均衡化增强对比度
    enhanced_img = equalize_hist(gray_img)

    # 高通滤波提取高频信息
    high_pass_img = np.abs(laplace(enhanced_img))  # 使用绝对值保证高频信息为正值

    # 将高频信息与均衡化图像结合，增强阴影对比
    combined_img = enhanced_img - 0.5 * high_pass_img
    combined_img = np.clip(combined_img, 0, 1)  # 限制在[0, 1]范围内

    # 聚类分割（K-means）
    pixels = combined_img.reshape(-1, 1)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(pixels)
    clustered_img = kmeans.labels_.reshape(gray_img.shape)

    # 假定最暗簇为阴影区域
    shadow_cluster = np.argmin(kmeans.cluster_centers_)
    shadow_mask = clustered_img == shadow_cluster

    # 移除小区域并膨胀
    shadow_mask = remove_small_objects(shadow_mask, min_size=min_shadow_size)
    shadow_mask = binary_dilation(shadow_mask, footprint=disk(3))

    # 提取阴影轮廓
    labeled_mask, _ = label(shadow_mask)
    contours = measure.find_contours(labeled_mask, 0.5)

    return img, enhanced_img, high_pass_img, combined_img, shadow_mask, contours

def visualize_process_histogram_process(img, enhanced_img, high_pass_img, combined_img, shadow_mask, contours):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 原始图像
    axes[0, 0].imshow(img / 255.0)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # 直方图均衡化后的图像
    axes[0, 1].imshow(enhanced_img, cmap="gray")
    axes[0, 1].set_title("Histogram Equalized")
    axes[0, 1].axis("off")

    # 高通滤波后的图像
    axes[0, 2].imshow(high_pass_img, cmap="gray")
    axes[0, 2].set_title("High-pass Filtered")
    axes[0, 2].axis("off")

    # 结合后的图像
    axes[1, 0].imshow(combined_img, cmap="gray")
    axes[1, 0].set_title("Combined Image")
    axes[1, 0].axis("off")

    # 阴影二值掩码
    axes[1, 1].imshow(shadow_mask, cmap="gray")
    for contour in contours:
        axes[1, 1].plot(contour[:, 1], contour[:, 0], color="red", linewidth=2)
    axes[1, 1].set_title("Shadow Mask with Contours")
    axes[1, 1].axis("off")

    # 占位（留空或补充其他内容）
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()

#function11:histogramcomparison
def shadow_extraction_histogramcomparison(tif_path, scales=[15, 80, 250], threshold_factor=0.75, min_shadow_size=1000):
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

def histogram_matching_histogramcomparison(source, reference):
    old_shape = source.shape
    source = source.ravel()
    reference = reference.ravel()

    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    r_values, r_counts = np.unique(reference, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64) / source.size
    r_quantiles = np.cumsum(r_counts).astype(np.float64) / reference.size

    interp_r_values = np.interp(s_quantiles, r_quantiles, r_values)
    return interp_r_values[bin_idx].reshape(old_shape)

def shadow_removal_histogramcomparison(img, shadow_mask):
    shadow_area = img[shadow_mask]  # 阴影区域像素值
    non_shadow_area = img[~shadow_mask]  # 非阴影区域像素值

    # 初始化结果图像
    corrected_img = img.copy()
    
    for channel in range(img.shape[-1]):  # 对每个颜色通道独立操作
        corrected_img[shadow_mask, channel] = histogram_matching_histogramcomparison(
            shadow_area[:, channel], non_shadow_area[:, channel]
        )

    return corrected_img

def visualize_shadow_removal_histogramcomparison(original_img, corrected_img, shadow_mask):
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

#function12:statistics
def shadow_extraction_statistics(tif_path, scales=[15, 80, 250], threshold_factor=0.75, min_shadow_size=1000):
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

def visualize_shadow_statistics(img, contours):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)  # 显示原始图像
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)  # 绘制轮廓
    ax.set_title('Shadow Extraction with Contours')
    ax.axis('off')
    plt.show()

def shadow_removal_with_matching_statistics(img, shadow_mask, blending_radius=1):
    img_float = img.astype(np.float32) / 255.0  # 转为浮点型图像
    output_img = img_float.copy()
    
    # 获取非阴影区域的掩码
    non_shadow_mask = ~shadow_mask

    for channel in range(img_float.shape[-1]):
        # 获取阴影和非阴影区域的像素值
        shadow_pixels = img_float[..., channel][shadow_mask]
        non_shadow_pixels = img_float[..., channel][non_shadow_mask]
        
        # 计算非阴影区域的均值和标准差
        mean_non_shadow = np.mean(non_shadow_pixels)
        std_non_shadow = np.std(non_shadow_pixels)
        
        # 计算阴影区域的均值和标准差
        mean_shadow = np.mean(shadow_pixels)
        std_shadow = np.std(shadow_pixels)
        
        # 线性调整阴影区域的亮度和对比度
        if std_shadow > 0:  # 避免除零错误
            adjusted_shadow = (shadow_pixels - mean_shadow) / std_shadow * std_non_shadow + mean_non_shadow
        else:
            adjusted_shadow = shadow_pixels
        
        # 更新图像的阴影区域
        output_img[..., channel][shadow_mask] = adjusted_shadow

    # 在阴影边界进行平滑过渡
    from skimage.filters import gaussian
    shadow_mask_blurred = gaussian(shadow_mask.astype(float), sigma=blending_radius)
    for channel in range(output_img.shape[-1]):
        output_img[..., channel] = (
            output_img[..., channel] * shadow_mask_blurred +
            img_float[..., channel] * (1 - shadow_mask_blurred)
        )

    # 转换回[0, 255]的整数型图像
    output_img = (np.clip(output_img, 0, 1) * 255).astype(np.uint8)
    return output_img

def visualize_removal_statistics(original_img, processed_img, shadow_mask):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(shadow_mask, cmap='gray')
    axes[1].set_title('Shadow Mask')
    axes[1].axis('off')

    axes[2].imshow(processed_img)
    axes[2].set_title('Shadow Removed')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

#function13:featureapplication
def shadow_extraction_featureapplication(tif_path, scales=[15, 80, 250], threshold_factor=0.75, min_shadow_size=1000):
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

def estimate_alpha_featureapplication(img, shadow_mask):
    gray_img = np.mean(img, axis=-1)  # 转为灰度图像
    non_shadow_intensity = np.median(gray_img[~shadow_mask])  # 非阴影区域的中位亮度
    shadow_intensity = gray_img * shadow_mask  # 阴影区域的亮度

    alpha = np.ones_like(gray_img)
    alpha[shadow_mask] = shadow_intensity[shadow_mask] / non_shadow_intensity
    alpha[alpha > 1] = 1  # 确保最大值不超过1
    alpha[alpha < 0.1] = 0.1  # 避免过小值导致不稳定

    return alpha

def shadow_removal_featureapplication(img, shadow_mask):
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

def visualize_shadow_removal_featureapplication(original_img, corrected_img, shadow_mask):
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

def calculate_and_plot_histograms_featureapplication(original_img, corrected_img, shadow_mask):
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

#function14:spectral
def shadow_extraction_spectral(tif_path, scales=[15, 80, 250], threshold_factor=0.75, min_shadow_size=1000):
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

def estimate_alpha_spectral(img, shadow_mask):
    gray_img = np.mean(img, axis=-1)  # 转为灰度图像
    non_shadow_intensity = np.median(gray_img[~shadow_mask])  # 非阴影区域的中位亮度
    shadow_intensity = gray_img * shadow_mask  # 阴影区域的亮度

    alpha = np.ones_like(gray_img)
    alpha[shadow_mask] = shadow_intensity[shadow_mask] / non_shadow_intensity
    alpha[alpha > 1] = 1  # 确保最大值不超过1
    alpha[alpha < 0.1] = 0.1  # 避免过小值导致不稳定

    return alpha

def shadow_removal_spectral(img, shadow_mask):
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

def visualize_shadow_removal_spectral(original_img, corrected_img, shadow_mask):
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

def calculate_spectral_curves_spectral(img, shadow_mask, corrected_img):
    def calculate_stats(region_data, label):
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

#function15:gray
def shadow_extraction_gray(tif_path, scales=[15, 80, 250], threshold_factor=0.75, min_shadow_size=1000):
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

def estimate_alpha_gray(img, shadow_mask):
    gray_img = np.mean(img, axis=-1)  # 转为灰度图像
    non_shadow_intensity = np.median(gray_img[~shadow_mask])  # 非阴影区域的中位亮度
    shadow_intensity = gray_img * shadow_mask  # 阴影区域的亮度

    alpha = np.ones_like(gray_img)
    alpha[shadow_mask] = shadow_intensity[shadow_mask] / non_shadow_intensity
    alpha[alpha > 1] = 1  # 确保最大值不超过1
    alpha[alpha < 0.1] = 0.1  # 避免过小值导致不稳定

    return alpha

def shadow_removal_gray(img, shadow_mask):
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

def visualize_shadow_removal_gray(original_img, corrected_img, shadow_mask):
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

def calculate_and_plot_histograms_gray(original_img, corrected_img, shadow_mask):
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
    axes[0, 0].set_title("原始图像直方图")
    
    axes[0, 1].plot(hist_shadow_area, color='red')
    axes[0, 1].set_title("原始阴影区域直方图")
    
    axes[1, 0].plot(hist_corrected_non_shadow, color='green')
    axes[1, 0].set_title("校正后非阴影区域直方图")
    
    axes[1, 1].plot(hist_corrected_shadow_area, color='purple')
    axes[1, 1].set_title("校正后阴影区域直方图")
    
    for ax in axes.flatten():
        ax.set_xlim(0, 255)
        ax.set_xlabel("灰度级")
        ax.set_ylabel("频率")

    plt.tight_layout()
    plt.show()

#function16:color
def shadow_extraction_color(tif_path, scales=[15, 80, 250], threshold_factor=0.75, min_shadow_size=1000):
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

def estimate_alpha_color(img, shadow_mask):
    gray_img = np.mean(img, axis=-1)  # 转为灰度图像
    non_shadow_intensity = np.median(gray_img[~shadow_mask])  # 非阴影区域的中位亮度
    shadow_intensity = gray_img * shadow_mask  # 阴影区域的亮度

    alpha = np.ones_like(gray_img)
    alpha[shadow_mask] = shadow_intensity[shadow_mask] / non_shadow_intensity
    alpha[alpha > 1] = 1  # 确保最大值不超过1
    alpha[alpha < 0.1] = 0.1  # 避免过小值导致不稳定

    return alpha

def shadow_removal_color(img, shadow_mask):
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

def visualize_shadow_removal_color(original_img, corrected_img, shadow_mask):
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

def calculate_and_plot_histograms_color(original_img, corrected_img, shadow_mask):
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

def plot_line_profile_color(img, line_coords):
    # 获取线段上的像素值
    x1, y1 = line_coords[0]
    x2, y2 = line_coords[1]
    num_points = max(abs(x2 - x1), abs(y2 - y1))
    x, y = np.linspace(x1, x2, num_points).astype(int), np.linspace(y1, y2, num_points).astype(int)
    
    # 提取像素值
    rgb_values = img[y, x, :]
    
    # 转换为HSV和LAB
    hsv_values = rgb2hsv(rgb_values / 255.0)
    lab_values = rgb2lab(rgb_values / 255.0)
    
    # 绘制RGB变化
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(rgb_values)
    plt.title("RGB曲线")
    plt.xlabel("像素索引")
    plt.ylabel("值")
    plt.legend(['红色', '绿色', '蓝色'])
    
    # 绘制HSV变化
    plt.subplot(1, 3, 2)
    plt.plot(hsv_values)
    plt.title("HSV曲线")
    plt.xlabel("像素索引")
    plt.ylabel("值")
    plt.legend(['色调', '饱和度', '亮度'])
    
    # 绘制LAB变化
    plt.subplot(1, 3, 3)
    plt.plot(lab_values)
    plt.title("LAB曲线")
    plt.xlabel("像素索引")
    plt.ylabel("值")
    plt.legend(['L', 'A', 'B'])
    
    plt.tight_layout()
    plt.show()

def draw_line_on_image_color(img):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title('Draw a line to analyze color profile')
    line, = ax.plot([], [], 'r-', linewidth=2)  # 预设空线条
    coords = []

    def on_click(event):
        if event.inaxes != ax:
            return
        coords.append((int(event.xdata), int(event.ydata)))
        line.set_data([p[0] for p in coords], [p[1] for p in coords])
        fig.canvas.draw()

        # 如果已经绘制了两个点，开始分析
        if len(coords) == 2:
            plot_line_profile_color(img, coords)
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

def function1(file_path):#msr filter 
    global current_function
    current_function = "Function1"
    print(f"Function1 executed with {file_path}")
    img, shadow_mask, contours = shadow_extraction_msr_filter(file_path)
    visualize_shadow_1(img, contours)

def function2(file_path):#msr filter process
    global current_function
    current_function = "Function2"
    print(f"Function2 executed with {file_path}")
    img, gray_img, msr_img, shadow_mask, contours = shadow_extraction_msr_extraction_process(file_path)
    visualize_processing_steps_1(img, gray_img, msr_img, shadow_mask, contours)

def function3(file_path):#gabor
    global current_function
    current_function = "Function3"
    print(f"Function3 executed with {file_path}")
    img, shadow_mask, contours = shadow_extraction_gabor(file_path)
    visualize_shadow_gabor(img, contours)

def function4(file_path):#gabor process
    global current_function
    current_function = "Function4"
    print(f"Function4 executed with {file_path}")
    img, gray_img, gabor_responses, shadow_mask, contours = shadow_extraction_gabor_process(file_path)
    visualize_process_gabor_process(img, gray_img, gabor_responses, shadow_mask, contours)

def function5(file_path):#direction
    global current_function
    current_function = "Function5"
    print(f"Function5 executed with {file_path}")
    img, shadow_mask, contours = shadow_extraction_direction(file_path)
    visualize_shadow_direction(img, contours)

def function6(file_path):#direction process
    global current_function
    current_function = "Function6"
    print(f"Function6 executed with {file_path}")
    img, gray_img, gradient_magnitude, enhanced_edges, shadow_mask, contours = shadow_extraction_direction_process(file_path)
    visualize_process_direction_process(img, gray_img, gradient_magnitude, enhanced_edges, shadow_mask, contours)

def function7(file_path):#dcp
    global current_function
    current_function = "Function7"
    print(f"Function7 executed with {file_path}")
    img, shadow_mask, contours = dark_channel_prior_shadow_extraction_dcp(file_path)
    visualize_shadow_dcp(img, contours)

def function8(file_path):#dcp process
    global current_function
    current_function = "Function8"
    print(f"Function8 executed with {file_path}")
    img, dark_channel, dark_channel_min, shadow_mask, contours = dark_channel_prior_shadow_extraction_dcp_process(file_path)
    visualize_process_dcp_process(img, dark_channel, dark_channel_min, shadow_mask, contours)

def function9(file_path):#histogram
    global current_function
    current_function = "Function9"
    print(f"Function9 executed with {file_path}")
    img, shadow_mask, contours = shadow_extraction_histogram(file_path)
    visualize_shadow_histogram(img, contours)

def function10(file_path):#dhistogram process
    global current_function
    current_function = "Function10"
    print(f"Function10 executed with {file_path}")
    img, enhanced_img, high_pass_img, combined_img, shadow_mask, contours = shadow_extraction_histogram_process(file_path)
    visualize_process_histogram_process(img, enhanced_img, high_pass_img, combined_img, shadow_mask, contours)

def function11(file_path):#histogram comparison
    global current_function
    current_function = "Function11"
    print(f"Function11 executed with {file_path}")
    img, shadow_mask, contours = shadow_extraction_histogramcomparison(file_path)
    corrected_img = shadow_removal_histogramcomparison(img, shadow_mask)
    visualize_shadow_removal_histogramcomparison(img, corrected_img, shadow_mask)

def function12(file_path):#statistics
    global current_function
    current_function = "Function12"
    print(f"Function12 executed with {file_path}")
    img, shadow_mask, contours = shadow_extraction_statistics(file_path)
    visualize_shadow_statistics(img, contours)
    shadow_removed_img = shadow_removal_with_matching_statistics(img, shadow_mask)
    visualize_removal_statistics(img, shadow_removed_img, shadow_mask)

def function13(file_path):#featureapplication
    global current_function
    current_function = "Function13"
    print(f"Function13 executed with {file_path}")
    img, shadow_mask, contours = shadow_extraction_featureapplication(file_path)
    corrected_img = shadow_removal_featureapplication(img, shadow_mask)
    visualize_shadow_removal_featureapplication(img, corrected_img, shadow_mask)
    calculate_and_plot_histograms_featureapplication(img, corrected_img, shadow_mask)

def function14(file_path):#spectral
    global current_function
    current_function = "Function14"
    print(f"Function14 executed with {file_path}")
    img, shadow_mask, contours = shadow_extraction_spectral(file_path)
    corrected_img = shadow_removal_spectral(img, shadow_mask)
    calculate_spectral_curves_spectral(img, shadow_mask, corrected_img)

def function15(file_path):#gray
    global current_function
    current_function = "Function15"
    print(f"Function15 executed with {file_path}")
    img, shadow_mask, contours = shadow_extraction_gray(file_path)
    corrected_img = shadow_removal_gray(img, shadow_mask)
    visualize_shadow_removal_gray(img, corrected_img, shadow_mask)
    calculate_and_plot_histograms_gray(img, corrected_img, shadow_mask)

def function16(file_path):#color
    global current_function
    current_function = "Function16"
    print(f"Function16 executed with {file_path}")
    img, shadow_mask, contours = shadow_extraction_color(file_path)
    corrected_img = shadow_removal_color(img, shadow_mask)

    # 在阴影去除后的图像上绘制线并分析
    draw_line_on_image_color(corrected_img)

def import_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        if file_path.lower().endswith('.tif'):
            # 可视化TIF文件
            visualize_tif(file_path)
            print(f"File {file_path} imported successfully.")
            # 更新状态以传递给功能函数
            main_window.file_path = file_path
        else:
            messagebox.showerror("文件错误", "文件必须为tif格式！")

def visualize_tif(file_path):
    # 打开TIF文件
    dataset = gdal.Open(file_path)
    if dataset is None:
        messagebox.showerror("错误", "无法打开TIF文件！")
        return

    # 获取波段数量
    num_bands = dataset.RasterCount

    if num_bands >= 3:
        # 读取RGB三个波段
        r_band = dataset.GetRasterBand(1).ReadAsArray()
        g_band = dataset.GetRasterBand(2).ReadAsArray()
        b_band = dataset.GetRasterBand(3).ReadAsArray()

        # 将波段合成为RGB图像
        rgb_image = np.dstack((r_band, g_band, b_band))
    elif num_bands == 1:
        # 读取单波段（灰度图）
        r_band = dataset.GetRasterBand(1).ReadAsArray()
        rgb_image = r_band
    else:
        messagebox.showerror("错误", "TIF文件的波段数不支持！")
        return

    # 创建matplotlib图形
    fig, ax = plt.subplots()
    if num_bands >= 3:
        ax.imshow(rgb_image)
    else:
        ax.imshow(rgb_image, cmap='gray')
    ax.set_title('TIF Visualization')
    ax.axis('off')

    # 嵌入图形到Tkinter窗口
    canvas = FigureCanvasTkAgg(fig, master=main_window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


def on_start_button_click():
    # 关闭第一个主页面
    start_window.destroy()
    # 打开第二个主页面
    main_window.update()
    main_window.deiconify()

def stop_current_function():
    global current_function
    if current_function:
        print(f"Stopping {current_function}")
        current_function = None
    else:
        messagebox.showinfo("提示", "当前没有正在运行的功能。")

def close_app():
    app.quit()  # 停止主循环
    app.destroy()  # 销毁所有窗口

# 在主窗口上绑定关闭事件
def on_window_close():
    app.quit()  # 停止主循环
    app.destroy()  # 销毁所有窗口

# 创建主应用程序窗口
app = tk.Tk()
app.withdraw()  # 隐藏主应用程序窗口

# 绑定关闭事件到 on_window_close 函数
app.protocol("WM_DELETE_WINDOW", on_window_close)

# 创建第一个主页面
start_window = tk.Toplevel(app)
start_window.title("开始页面")
start_window.geometry("400x200")

start_label1 = tk.Label(start_window, text="欢迎使用由王逸 2022104023开发的应用程序", font=("Arial", 14))
start_label1.pack(pady=10)

start_label2 = tk.Label(start_window, text="遥感原理和图像处理期末大作业应用程序", font=("Arial", 12))
start_label2.pack(pady=10)

start_button = tk.Button(start_window, text="点击此处开始运行", command=on_start_button_click)
start_button.pack(pady=20)

# 创建第二个主页面（隐藏）
main_window = tk.Toplevel(app)
main_window.title("主页面")
main_window.geometry("800x600")
main_window.withdraw()

# 添加菜单栏
menu_bar = tk.Menu(main_window)
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="导入新的文件", command=import_file)
menu_bar.add_cascade(label="文件", menu=file_menu)

# 添加功能菜单
function_menu = tk.Menu(menu_bar, tearoff=0)
function_menu.add_command(label="阴影提取-MSR和滤波", command=lambda: function1(main_window.file_path))
function_menu.add_command(label="阴影提取过程图-MSR和滤波", command=lambda: function2(main_window.file_path))
function_menu.add_command(label="阴影提取-Gabor与纹理", command=lambda: function3(main_window.file_path))
function_menu.add_command(label="阴影提取过程图-Gabor与纹理", command=lambda: function4(main_window.file_path))
function_menu.add_command(label="阴影提取-方向梯度和边缘检测", command=lambda: function5(main_window.file_path))
function_menu.add_command(label="阴影提取过程图-方向梯度和边缘检测", command=lambda: function6(main_window.file_path))
function_menu.add_command(label="阴影提取-暗通道先验", command=lambda: function7(main_window.file_path))
function_menu.add_command(label="阴影提取过程图-暗通道先验", command=lambda: function8(main_window.file_path))
function_menu.add_command(label="阴影提取-直方图对比和高通滤波", command=lambda: function9(main_window.file_path))
function_menu.add_command(label="阴影提取过程图-直方图对比和高通滤波", command=lambda: function10(main_window.file_path))
function_menu.add_command(label="阴影去除-直方图匹配", command=lambda: function11(main_window.file_path))
function_menu.add_command(label="阴影去除-统计学匹配和平滑过渡", command=lambda: function12(main_window.file_path))
function_menu.add_command(label="阴影去除-区域特征匹配", command=lambda: function13(main_window.file_path))
function_menu.add_command(label="检验-光谱曲线绘制", command=lambda: function14(main_window.file_path))
function_menu.add_command(label="检验-灰度直方图绘制", command=lambda: function15(main_window.file_path))
function_menu.add_command(label="检验-点击两点画出横跨线并进行色彩空间变化图绘制", command=lambda: function16(main_window.file_path))

menu_bar.add_cascade(label="功能", menu=function_menu)

# 添加关闭功能和关闭程序按钮
control_menu = tk.Menu(menu_bar, tearoff=0)
control_menu.add_command(label="停止当前功能", command=stop_current_function)
control_menu.add_command(label="关闭程序", command=close_app)

menu_bar.add_cascade(label="控制", menu=control_menu)

main_window.config(menu=menu_bar)

# 设置默认文件路径为空
main_window.file_path = None

app.mainloop()
