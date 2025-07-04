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
    :return: tuple, (img, shadow_mask, contours)
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

    return img, shadow_mask, contours


def visualize_shadow(img, contours):
    """
    可视化阴影轮廓。

    :param img: ndarray, 原始图像数据
    :param contours: list, 阴影轮廓坐标列表
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img / 255)  # 显示原始图像，归一化到[0, 1]
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)  # 绘制轮廓
    ax.set_title('Shadow Extraction with Contours')
    ax.axis('off')
    plt.show()


# 示例调用
if __name__ == "__main__":
    tif_path = r"D:\\通用文件夹\\遥感原理与数字图像处理\\期末作业\\data\\tif\\1.tif"
    img, shadow_mask, contours = shadow_extraction(tif_path)
    visualize_shadow(img, contours)
