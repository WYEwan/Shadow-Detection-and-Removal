import numpy as np
import matplotlib.pyplot as plt
import rasterio
from skimage import filters, measure
from skimage.morphology import remove_small_objects
from scipy.ndimage import gaussian_filter
from skimage.exposure import match_histograms
from skimage.filters import gaussian

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

def shadow_removal_with_matching(img, shadow_mask, blending_radius=1):
    """
    改进的阴影去除算法，通过与周边非阴影区域的匹配进行调整。

    :param img: ndarray, 原始图像数据，形状为(H, W, C)
    :param shadow_mask: ndarray, 阴影区域的二值掩码，形状为(H, W)
    :param blending_radius: int, 阴影边界平滑过渡的半径
    :return: ndarray, 去除阴影后的图像
    """
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

def visualize_removal(original_img, processed_img, shadow_mask):
    """
    可视化阴影去除结果。

    :param original_img: ndarray, 原始图像数据
    :param processed_img: ndarray, 去除阴影后的图像数据
    :param shadow_mask: ndarray, 阴影区域的二值掩码
    """
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

# 示例调用
if __name__ == "__main__":
    tif_path = r"D:\\通用文件夹\\遥感原理与数字图像处理\\期末作业\\data\\tif\\4.tif"
    img, shadow_mask, contours = shadow_extraction(tif_path)
    visualize_shadow(img, contours)
    shadow_removed_img = shadow_removal_with_matching(img, shadow_mask)
    visualize_removal(img, shadow_removed_img, shadow_mask)