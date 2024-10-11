'''
v1：png
3D断层建模
输入：插值后的图片序列

'''

import time
from vtk import vtkPNGReader
from vtk import vtkSmoothPolyDataFilter
from vtk import vtkImageReslice
from vtk import vtkImageGaussianSmooth
from vtk import vtkWindowedSincPolyDataFilter
from vtk import vtkMarchingCubes
from vtk import vtkFlyingEdges3D
from vtk import vtkStripper
from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLRenderer
from vtkmodules.vtkCommonColor import vtkNamedColors    # 颜色
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import (
    vtkPolyDataMapper,
    vtkProperty,
    vtkActor,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor
)

start_3d = time.time()
colors = vtkNamedColors()  # 创建颜色对象
color_vessel= map(lambda x: x / 255.0, [204, 0, 0, 255])  # 将传入的函数依次作用到序列的每个元素。lambda表示x的匿名函数x/255
colors.SetColor("Vessel", *color_vessel)  # 自定义血管颜色

# 定义个图片读取接口
PNG_Reader = vtkPNGReader()
PNG_Reader.SetNumberOfScalarComponents(1)    # 设置图像中每个像素的标量分量数量，灰度图1，彩色图3
PNG_Reader.SetFileDimensionality(2)  # 维度
# 读取PNG序列
PNG_Reader.SetFilePattern("interpolation_out_3d_in/%06d.png")     # 这是设置文件名的格式
PNG_Reader.SetFileNameSliceOffset(0)    # 设置从哪个数字开始命名
PNG_Reader.SetDataExtent(0, 2047, 0, 2047, 0, 1076)# 读取PNG数据的空间范围，z范围代表图片序列数,用来定义要读取的图像序列的范围
PNG_Reader.Update()
# JPG_Reader.SetDataByteOrderToLittleEndian() 手动设置字节序，不必要
spacing = [4, 4, 7]  #xyz方向上，vtkImageData的每个数据点的间隔单位数
PNG_Reader.GetOutput().SetSpacing(spacing)  # 仅作用于数据输出对象，在读取完数据后可以动态调整间距。用户根据实际撤回速度，给定数值。
print('PNG reading completed')


# 高斯平滑
gauss = vtkImageGaussianSmooth()
gauss.SetInputConnection(PNG_Reader.GetOutputPort())
gauss.SetStandardDeviations(1, 1, 1) #标准差
gauss.SetRadiusFactors(1, 1, 1)  #半径
gauss.Update()
spacing = [4, 4, 7]
gauss.GetOutput().SetSpacing(spacing)
print('gauss completed')


'''
# 计算轮廓的方法
contour = vtkMarchingCubes()    # 提取三维等值面
contour.SetInputConnection(JPG_Reader.GetOutputPort())
contour.ComputeNormalsOn()
contour.SetValue(0,100)     # 可以提取黑白边缘
'''

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


mapper = vtkPolyDataMapper()
mapper.SetInputConnection(smoothFilter.GetOutputPort())
mapper.ScalarVisibilityOff()    # 关闭标量可视化后，所有几何体的颜色将统一，而不会根据标量数据来变化。

proper = vtkProperty()
proper.SetColor(colors.GetColor3d("Vessel"))
# property_vessel.SetDiffuse(0.7)  # 设置漫反射
# property_vessel.SetSpecular(0.4)  # 设置镜面反射
# property_vessel.SetSpecularPower(20)  # 设置镜面反射强度

actor = vtkActor()
actor.SetMapper(mapper)
actor.SetProperty(proper)
actor.RotateY(-90)

# 创建渲染器，负责显示演员，并设置渲染器的背景颜色
ren = vtkRenderer()
ren.AddActor(actor)  # 将演员添加到渲染器中
ren.SetBackground(colors.GetColor3d('CornflowerBlue'))  # 设置背景颜色为午夜蓝

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
    window.Render()     # 先渲染窗口
    end_3d = time.time()
    print(f'渲染耗时：{end_3d - start_3d:.4f} 秒')
    interactor.Initialize()     # 再初始化交互
    interactor.Start()  # 最后启动交互
