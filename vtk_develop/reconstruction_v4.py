'''
v4:分支以不同颜色突出显示
v3:模型导出功能stl、vtk，优化代码结构（颜色、材质宏定义）
v2:jpg
v1：png
3D断层建模
输入：插值后的图片序列
'''
import os
import time
from vtk import (
    vtkJPEGReader,
    vtkPolyDataReader,
    vtkSmoothPolyDataFilter,
    vtkImageGaussianSmooth,
    vtkWindowedSincPolyDataFilter,
    vtkFlyingEdges3D,
    vtkSTLWriter,
    vtkPolyDataWriter,
    vtkUnsignedCharArray
)
from vtkmodules.vtkRenderingCore import (
    vtkPolyDataMapper,
    vtkProperty,
    vtkActor,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor
)
from vtkmodules.vtkCommonColor import vtkNamedColors    # 颜色
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera

# 读取JPG序列
def jpgReader(filepath:str, width, height, spacing = [4, 4, 7]):
    jpg_files = [file for file in os.listdir(filepath) if file.lower().endswith('.jpg')]   # 读取jpg文件列表
    jpg_count = len(jpg_files)  # 统计jpg的数量
    JPG_Reader = vtkJPEGReader()
    JPG_Reader.SetNumberOfScalarComponents(1)  # 设置图像中每个像素的标量分量数量，灰度图1，彩色图3
    JPG_Reader.SetFileDimensionality(2)  # 维度
    JPG_Reader.SetFilePattern(os.path.join(filepath, "%06d.jpg"))  # 这是设置文件名的格式"xxx/%06d.jpg"
    JPG_Reader.SetFileNameSliceOffset(0)  # 设置从哪个数字开始命名
    JPG_Reader.SetDataExtent(0, width-1, 0, height-1, 0, jpg_count-1)  # 读取PNG数据的空间范围，z范围代表图片序列数,用来定义要读取的图像序列的范围
    JPG_Reader.Update()
    JPG_Reader.GetOutput().SetSpacing(spacing)  # 仅作用于数据输出对象，在读取完数据后可以动态调整间距。用户根据实际撤回速度，给定数值。
    print('JPG reading completed')
    return JPG_Reader.GetOutput()

# 读取已有的vtk模型
def vtkReader(vtkfilepath:str, vtkfilename:str):
    vtk_reader = vtkPolyDataReader()
    vtk_reader.SetFileName(os.path.join(vtkfilepath,vtkfilename))  # 指定要读取的 .vtk 文件路径
    vtk_reader.Update() # 执行读取操作
    print("模型已成功从.vtk文件读取！")
    return vtk_reader.GetOutput()   # 获取 PolyData 对象

# 没有就建立文件夹
def folder_preparation(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print('输出文件夹已建立')
    else: print('输出文件夹寻址成功')

# 输出stl模型
def save_model_as_stl(vtkpolydata, filepath:str, filename:str):
    folder_preparation(filepath)  # 保证模型输出文件夹可用
    stl_writer = vtkSTLWriter()
    stl_writer.SetFileName(os.path.join(filepath, filename + ".stl"))
    stl_writer.SetInputData(vtkpolydata)  # 将PolyData传递给STL Writer
    stl_writer.Write()  # 执行导出
    print("模型已成功导出为STL文件！")

# 输出vtk模型
def save_model_as_vtk(vtkpolydata, foldername:str, filename:str):
    folder_preparation(foldername)  # 保证模型输出文件夹可用
    vtk_writer = vtkPolyDataWriter()
    vtk_writer.SetFileName(os.path.join(foldername,filename+".vtk"))
    vtk_writer.SetInputData(vtkpolydata)    # 将PolyData传递给vtk Writer
    vtk_writer.Write()  # 执行导出
    print("模型已成功导出为vtk文件！")

colors = vtkNamedColors()  # 创建颜色对象
color_vessel= map(lambda x: x / 255.0, [204, 0, 0, 255])  # 将传入的函数依次作用到序列的每个元素。lambda表示x的匿名函数x/255
colors.SetColor("Vessel", *color_vessel)  # 自定义血管颜色
color_branch= map(lambda x: x / 255.0, [8, 79, 184, 255])  # 将传入的函数依次作用到序列的每个元素。lambda表示x的匿名函数x/255
colors.SetColor("Branch", *color_branch)  # 自定义分支颜色
color_probe= map(lambda x: x / 255.0, [192, 192, 192, 255])  # 将传入的函数依次作用到序列的每个元素。lambda表示x的匿名函数x/255
colors.SetColor("Probe", *color_probe)  # 自定义探针颜色
colors.SetColor("Background",colors.GetColor3d('CornflowerBlue'))   # 自定义背景颜色

colors_array = vtkUnsignedCharArray()
colors_array.SetNumberOfComponents(3)  # 3 分量表示 RGB
colors_array.SetName("whole")  # 设置数组名称

property_vessel = vtkProperty()
# property_vessel.SetColor(colors.GetColor3d("Vessel"))
# property_vessel.SetDiffuse(0.7)  # 设置漫反射
# property_vessel.SetSpecular(0.4)  # 设置镜面反射
# property_vessel.SetSpecularPower(20)  # 设置镜面反射强度

start_3d = time.time()

# 读取jpg序列
jpgmodel = jpgReader("interpolation_out_3d_in", 512, 512,[4,4,7])

# 高斯平滑
gauss = vtkImageGaussianSmooth()
gauss.SetInputData(jpgmodel)
gauss.SetStandardDeviations(1, 1, 1) #标准差
gauss.SetRadiusFactors(1, 1, 1)  #半径
gauss.Update()
spacing = [4, 4, 7]
gauss.GetOutput().SetSpacing(spacing)
print('gauss completed')

# 创建 vtkFlyingEdges3D 对象
flying_edges = vtkFlyingEdges3D()
flying_edges.SetInputConnection(gauss.GetOutputPort())
flying_edges.SetValue(0, 128)  # 设置边缘提取的阈值
flying_edges.Update()  # 更新过滤器
print('flying_edges completed')

'''
# 拉普拉斯网络平滑算法
smoothFilter = vtkSmoothPolyDataFilter()
smoothFilter.SetInputConnection(contour.GetOutputPort())
smoothFilter.SetNumberOfIterations(100)    # 设置迭代次数
smoothFilter.SetRelaxationFactor(0.5)     # 设置松弛因子，影响点移动的幅度 (0~1)
smoothFilter.FeatureEdgeSmoothingOn()    # 开启特征边缘平滑
smoothFilter.BoundarySmoothingOn()        # 打开边界平滑
smoothFilter.Update()
'''

smoothFilter = vtkWindowedSincPolyDataFilter()
smoothFilter.SetInputConnection(flying_edges.GetOutputPort())
smoothFilter.SetNumberOfIterations(150)
smoothFilter.SetPassBand(0.1)
smoothFilter.BoundarySmoothingOn()
smoothFilter.FeatureEdgeSmoothingOff()
smoothFilter.NonManifoldSmoothingOn()
smoothFilter.NormalizeCoordinatesOn()
smoothFilter.Update()

# 分段设置颜色
# 获取所有的点
points = smoothFilter.GetOutput().GetPoints()
# 遍历所有的点，根据 z 坐标进行颜色设置
for i in range(points.GetNumberOfPoints()):
    # 获取第 i 个点的坐标 (x, y, z)
    x, y, z = points.GetPoint(i)

    # 如果 z 坐标 >= 0，则设置为红色 (255, 0, 0)，否则设置为蓝色 (0, 0, 255)
    if 1582 <= z <= 1694 or 2177 <= z <= 2345 or 5208 <= z <= 5348 or 7021 <= z <= 7252:
        colors_array.InsertNextTuple3(*[int(c * 255) for c in colors.GetColor3d("Branch")])  # 蓝色

    else:
        colors_array.InsertNextTuple3(*[int(c * 255) for c in colors.GetColor3d("Vessel")])  # 红色

# 将颜色数组关联到 polyData 的顶点数据
smoothFilter.GetOutput().GetPointData().SetScalars(colors_array)

mapper = vtkPolyDataMapper()
mapper.SetInputConnection(smoothFilter.GetOutputPort())
mapper.ScalarVisibilityOn()    # 必须开启这个

actor = vtkActor()
actor.SetMapper(mapper)
actor.SetProperty(property_vessel)
actor.RotateY(-90)
print("演员创建完成")

# 创建渲染器，负责显示演员，并设置渲染器的背景颜色
ren = vtkRenderer()
ren.AddActor(actor)  # 将演员添加到渲染器中
ren.SetBackground(colors.GetColor3d('Background'))  # 设置背景颜色为午夜蓝

# 创建渲染窗口，将渲染器添加到窗口中，设置窗口大小和标题
window = vtkRenderWindow()
window.AddRenderer(ren)  # 将渲染器放入渲染窗口
window.SetSize(1024, 576)  # 设置窗口大小
window.SetWindowName("IVOCT三维重建")  # 设置窗口标题

interactor = vtkRenderWindowInteractor()  # 创建交互器
interactor.SetRenderWindow(window)  # 将交互器与渲染窗口关联
style = vtkInteractorStyleTrackballCamera()  # 设置交互样式为"轨迹球摄像机"
interactor.SetInteractorStyle(style)

# 开始显示
if __name__ == '__main__':
    # 先渲染窗口
    window.Render()
    '''
    # 输出3D模型
    save_model_as_stl(smoothFilter.GetOutput(), "3dmodel", "3dstl")
    save_model_as_vtk(smoothFilter.GetOutput(),"3dmodel","3dvtk")
    '''
    end_3d = time.time()
    print(f'进程总耗时：{end_3d - start_3d:.4f} 秒')
    interactor.Initialize()  # 再初始化交互
    interactor.Start()  # 最后启动交互
