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
    控制 MSR 算法中高斯模糊的尺度。尺度越大，模拟的光照变化范围越广。
    小尺度：突出细节，强调局部对比度，但可能引入噪声。大尺度：更平滑地模拟光照分布，有助于去除大范围光照不均，但可能掩盖细节。
    选择多个尺度的组合，兼顾局部细节和全局光照变化。

    :param threshold_factor: float, 阈值调整因子
    控制阴影检测的灵敏度，是基于 Otsu 阈值的乘数调整。范围：通常为 0.5 到 1.0 默认值 0.75。
    值较低（如0.5）：检测到更多阴影区域，包括一些亮度较低但非阴影的区域。适合处理高对比度或低光照图像。
    值较高（如0.9）：更严格地选择阴影区域，可能忽略部分真实的阴影。适合处理光照均匀、阴影对比明显的图像。
    根据图像亮度范围动态调整。例如，高亮度图像使用较低因子，低亮度图像使用较高因子。

    :param min_shadow_size: int, 阴影区域的最小面积（像素）
    用于移除小连通区域的阈值，单位为像素面积。默认值 1000 表示删除小于 1000 个像素的阴影区域。
    值较小（如100）：更宽松，保留小阴影区域，但可能引入噪声。适合高分辨率图像或目标区域较小的应用。
    值较大（如5000）：更严格，仅保留大面积阴影，可能丢失小的真实阴影。适合低分辨率图像或关注主要阴影的应用。
    根据图像分辨率和目标场景调整。例如，分辨率为 1m/像素时，1000 像素面积相当于 1000 平方米。

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
