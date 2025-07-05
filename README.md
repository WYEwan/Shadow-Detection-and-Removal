# Shadow-Detection-and-Removal阴影识别与去除

# For more detailed information, please refer to the attached PDF file.（Note that both the attached file and the code comments are in Chinese.）
# 具体的论文参见PDF附件（注意，本附件及代码注释均为中文）

This is a remote sensing image processing program I submitted for a course, aimed at detecting and removing shadows from TIFF files of remote sensing images in densely built-up areas.
这是我在一个课程中提交的遥感图像处理程序，旨在用于对密集建筑区遥感图像的TIFF文件进行阴影识别和去除

The most important file is "main.py," which can fully run all functions in a compilation environment. After running, it presents a complete visualization interface that allows importing any TIFF file for shadow detection and removal. It offers several algorithmic solutions for processing.
其中最重要的是“main.py”，这个脚本在编译环境下可以完整运行所有功能。运行之后，会呈现一个完善的可视化界面，可以导入任何TIFF格式文件并进行阴影识别和去除处理，提供了若干个算法方案来进行处理。

1.Upon running, an introductory interface appears; click the button to proceed.
1.运行之后会有一个引入界面，点击按钮继续运行。

2.You then enter the main interface's blank page. Click "文件" in the top-left corner to add a new TIFF file; other file types are not supported.
2.之后进入主界面的空页面，点击左上角的“文件”以加入新的TIFF文件，其它文件类型均不支持

<img width="341" alt="主界面空界面" src="https://github.com/user-attachments/assets/a99d151f-3560-4927-8170-83d49a67a09a" />

<img width="343" alt="成功导入TIFF文件" src="https://github.com/user-attachments/assets/7eaf34b9-de3a-420b-a65b-0ea20fb5ecb8" />

3.Click "功能" in the top-left corner to run various shadow extraction and removal algorithms and display some visual results.
3.点击左上角的“功能”，可以运行各种类型的阴影提取和去除算法，并且显示一些可视化运行结果

<img width="341" alt="可以点击相应功能" src="https://github.com/user-attachments/assets/8cac6d29-78e2-4c88-81f2-f5d74e5dbf89" />

4.Functions prefixed with "检验" in "功能" can only be run after an algorithm has been executed. One interesting feature is the last one, which allows you to click on any two points to draw a line segment and analyze the color space changes along that line.
4.“功能”中前缀带有“检验”的功能，注意要是运行完算法之后才可以运行的，其中一个比较有趣的是最后一个功能，其可以点击任意两个点取一条线段，分析这条线段上的不同色彩空间变化

5.It is crucial to click "控制" in the top-left corner to stop the current function before running a new one.
5.一定要注意每次运行完一个功能之后要点击左上角的控制，停止此项功能的运行，然后才能运行新的。

<img width="344" alt="注意在每次运行完功能之后，点击这里取消当前功能" src="https://github.com/user-attachments/assets/1e295229-0c14-45de-9674-31c25d82e29a" />

There is also a corresponding paper titled "阴影提取和去除方法.pdf." This paper documents the theories used in the script. Since the paper has not been reviewed and was written by a beginner who is just starting to explore this topic, it may contain many theoretical errors. Please read it with caution. If you find any mistakes, feel free to point them out to me.
其中还有一篇对应的论文“建筑密集区阴影识别和去除.pdf”，这篇论文撰写了这个脚本中所运用的理论，由于这篇论文未经审查，只是一个刚刚开始接触这个内容的新手所写出的较为稚嫩的文章，可能会有很多理论错误等，谨慎阅读，如果发现有错误，欢迎向我指出。

Other scripts are mostly sub-scripts of "main.py" or initial versions of these sub-scripts. If you are interested in the specific implementation of certain functions, you can check them out, but I do not recommend running them. The actual execution should be done using "main.py," as it is more complete and has been tested for execution.
其它的脚本基本都是main.py脚本中的子脚本或者这些子脚本的初始版本，如果对于其中具体功能的实现感兴趣，可以点击查看，但是不建议运行，实际运行还是采用main.py，因为它更加完善，且经测试可执行。

I am quite lazy and will not explain each sub-script individually, except for "main.py." From the file names, you should be able to roughly guess their functions, which are based on the theories mentioned in the aforementioned paper.
我比较懒，就不一一对除了main.py之外的子脚本进行说明了，大家看文件名应该也能大概猜出是干什么的，其具体依据的理论就是上述提到论文中的理论。

