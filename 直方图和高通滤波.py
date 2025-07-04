import numpy as np
import matplotlib.pyplot as plt
import rasterio
from skimage.exposure import equalize_hist
from skimage.filters import laplace
from sklearn.cluster import KMeans
from skimage.morphology import remove_small_objects, binary_dilation, disk
from skimage import measure
from scipy.ndimage import label


def shadow_extraction(tif_path, num_clusters=3, min_shadow_size=1000):
    """
    基于直方图对比和高通滤波的阴影提取方法。

    :param tif_path: str, TIF 文件路径
    :param num_clusters: int, 聚类的簇数。
    :param min_shadow_size: int, 阴影区域的最小面积（像素）。
    :return: tuple, (img, enhanced_img, high_pass_img, combined_img, shadow_mask, contours)
    """
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


def visualize_process(img, enhanced_img, high_pass_img, combined_img, shadow_mask, contours):
    """
    可视化处理过程和最终阴影轮廓。

    :param img: ndarray, 原始图像数据
    :param enhanced_img: ndarray, 直方图均衡化后的图像
    :param high_pass_img: ndarray, 高通滤波后的图像
    :param combined_img: ndarray, 结合后的图像
    :param shadow_mask: ndarray, 阴影二值掩码
    :param contours: list, 阴影轮廓坐标列表
    """
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


# 示例调用
if __name__ == "__main__":
    tif_path = r"D:\\通用文件夹\\遥感原理与数字图像处理\\期末作业\\data\\tif\\4.tif"
    img, enhanced_img, high_pass_img, combined_img, shadow_mask, contours = shadow_extraction(tif_path)
    visualize_process(img, enhanced_img, high_pass_img, combined_img, shadow_mask, contours)
