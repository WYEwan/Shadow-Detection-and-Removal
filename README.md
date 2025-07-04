# Shadow-Detection-and-Removal

This is a remote sensing image processing program I submitted for a course, aimed at detecting and removing shadows from TIFF files of remote sensing images in densely built-up areas.
The most important file is "main.py," which can fully run all functions in a compilation environment. After running, it presents a complete visualization interface that allows importing any TIFF file for shadow detection and removal. It offers several algorithmic solutions for processing.
Upon running, an introductory interface appears; click the button to proceed.
You then enter the main interface's blank page. Click "文件" in the top-left corner to add a new TIFF file; other file types are not supported.
<img width="341" alt="主界面空界面" src="https://github.com/user-attachments/assets/a99d151f-3560-4927-8170-83d49a67a09a" />
<img width="343" alt="成功导入TIFF文件" src="https://github.com/user-attachments/assets/7eaf34b9-de3a-420b-a65b-0ea20fb5ecb8" />
Click "功能" in the top-left corner to run various shadow extraction and removal algorithms and display some visual results.
<img width="341" alt="可以点击相应功能" src="https://github.com/user-attachments/assets/8cac6d29-78e2-4c88-81f2-f5d74e5dbf89" />
Functions prefixed with "检验" in "功能" can only be run after an algorithm has been executed. One interesting feature is the last one, which allows you to click on any two points to draw a line segment and analyze the color space changes along that line.
It is crucial to click "控制" in the top-left corner to stop the current function before running a new one.
<img width="344" alt="注意在每次运行完功能之后，点击这里取消当前功能" src="https://github.com/user-attachments/assets/1e295229-0c14-45de-9674-31c25d82e29a" />
There is also a corresponding paper titled "阴影提取和去除方法.pdf." This paper documents the theories used in the script. Since the paper has not been reviewed and was written by a beginner who is just starting to explore this topic, it may contain many theoretical errors. Please read it with caution. If you find any mistakes, feel free to point them out to me.
Other scripts are mostly sub-scripts of "main.py" or initial versions of these sub-scripts. If you are interested in the specific implementation of certain functions, you can check them out, but I do not recommend running them. The actual execution should be done using "main.py," as it is more complete and has been tested for execution.
I am quite lazy and will not explain each sub-script individually, except for "main.py." From the file names, you should be able to roughly guess their functions, which are based on the theories mentioned in the aforementioned paper.
