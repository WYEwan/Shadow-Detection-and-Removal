import numpy as np
import matplotlib.pyplot as plt
import rasterio
from skimage import filters, measure
from skimage.morphology import remove_small_objects
from scipy.ndimage import gaussian_filter
from skimage.filters import gaussian
from skimage.exposure import match_histograms

def shadow_extraction(tif_path, scales=[15, 80, 250], threshold_factor=0.75, min_shadow_size=1000):
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

def shadow_removal_method1(img, shadow_mask):
    corrected_img = img.copy()
    # ... implement method 1 specific code ...
    return corrected_img

def shadow_removal_method2(img, shadow_mask):
    corrected_img = img.copy()
    # ... implement method 2 specific code ...
    return corrected_img

def shadow_removal_method3(img, shadow_mask):
    corrected_img = img.copy()
    # ... implement method 3 specific code ...
    return corrected_img

def plot_spectral_curves(original_img, corrected_imgs, shadow_mask, method_names):
    plt.figure(figsize=(15, 5))

    # Plot original image spectrum
    plt.subplot(1, len(corrected_imgs) + 1, 1)
    plt.title("Original Image Spectrum")
    plt.plot(np.mean(original_img[shadow_mask], axis=0), label='Original')
    plt.legend()

    for i, (corrected_img, method_name) in enumerate(zip(corrected_imgs, method_names), start=2):
        plt.subplot(1, len(corrected_imgs) + 1, i)
        plt.title(f"{method_name} Spectrum")
        plt.plot(np.mean(corrected_img[shadow_mask], axis=0), label=method_name)
        plt.legend()

    # Overall comparison plot
    plt.figure(figsize=(10, 5))
    plt.title("All Methods Comparison")
    plt.plot(np.mean(original_img[shadow_mask], axis=0), label='Original', linestyle='--')
    for corrected_img, method_name in zip(corrected_imgs, method_names):
        plt.plot(np.mean(corrected_img[shadow_mask], axis=0), label=method_name)
    plt.legend()
    plt.show()

# 示例调用
if __name__ == "__main__":
    tif_path = "D:\\通用文件夹\\遥感原理与数字图像处理\\期末作业\\data\\tif\\1.tif"
    img, shadow_mask, contours = shadow_extraction(tif_path)

    corrected_img1 = shadow_removal_method1(img, shadow_mask)
    corrected_img2 = shadow_removal_method2(img, shadow_mask)
    corrected_img3 = shadow_removal_method3(img, shadow_mask)

    plot_spectral_curves(
        img,
        [corrected_img1, corrected_img2, corrected_img3],
        shadow_mask,
        ["Method 1", "Method 2", "Method 3"]
    )
