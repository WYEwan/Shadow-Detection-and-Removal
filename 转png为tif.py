from PIL import Image
import os

# 输入PNG文件的路径
input_directory = r"D:\通用文件夹\遥感原理与数字图像处理\期末作业\data"

# 遍历1.png到15.png
for i in range(1, 16):
    # 构建PNG文件的完整路径
    input_path = os.path.join(input_directory, f"{i}.png")
    
    # 读取PNG文件
    try:
        with Image.open(input_path) as img:
            # 构建输出TIF文件的完整路径
            output_path = os.path.join(input_directory, f"{i}.tif")
            
            # 保存为TIF格式
            img.save(output_path, "TIFF")
            print(f"成功转换: {input_path} -> {output_path}")
    except Exception as e:
        print(f"转换失败: {input_path}，错误信息: {e}")
