import numpy as np
import matplotlib.pyplot as plt
import rasterio
from skimage.filters import gabor
from skimage.morphology import remove_small_objects
from skimage import measure

def shadow_extraction(tif_path, gabor_frequencies=[0.1, 0.2, 0.3], threshold_factor=0.50, min_shadow_size=1000):
    """
    基于 Gabor 滤波和纹理分析的阴影提取算法。

    :param tif_path: str, TIF 文件路径
    :param gabor_frequencies: list, Gabor 滤波器的频率参数列表，用于提取不同尺度的纹理信息。
    :param threshold_factor: float, 阈值调整因子，控制阴影检测的灵敏度。
    :param min_shadow_size: int, 阴影区域的最小面积（像素）。
    :return: tuple, (img, shadow_mask, contours)
    """
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

def visualize_shadow(img, contours):
    """
    可视化阴影轮廓。

    :param img: ndarray, 原始图像数据
    :param contours: list, 阴影轮廓坐标列表
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)  # 显示原始图像
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)  # 绘制轮廓
    ax.set_title('Shadow Extraction with Contours')
    ax.axis('off')
    plt.show()

# 示例调用
if __name__ == "__main__":
    tif_path = r"D:\\通用文件夹\\遥感原理与数字图像处理\\期末作业\\data\\tif\\4.tif"
    img, shadow_mask, contours = shadow_extraction(tif_path)
    visualize_shadow(img, contours)
