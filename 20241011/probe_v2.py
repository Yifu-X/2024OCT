'''
v2:只捕捉鼠标在模型上的坐标
v1:实现显示切片图像
'''
import time
from vtk import vtkCylinderSource
from vtk import vtkJPEGReader
from vtk import vtkCellPicker
from vtk import vtkImageGaussianSmooth
from vtk import vtkFlyingEdges3D
from vtk import vtkWindowedSincPolyDataFilter
from vtkmodules.vtkCommonColor import vtkNamedColors    # 颜色
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import (
    vtkPolyDataMapper,
    vtkProperty,
    vtkImageActor,
    vtkActor,
    vtkRenderer,  
    vtkRenderWindow,
    vtkRenderWindowInteractor
)

pic_num = 1076

# 颜色
start_3d = time.time()
colors = vtkNamedColors()  # 创建颜色对象
color_vessel= map(lambda x: x / 255.0, [204, 0, 0, 255])  # 将传入的函数依次作用到序列的每个元素。lambda表示x的匿名函数x/255
colors.SetColor("Vessel", *color_vessel)  # 自定义血管颜色
color_probe= map(lambda x: x / 255.0, [192, 192, 192, 255])  # 将传入的函数依次作用到序列的每个元素。lambda表示x的匿名函数x/255
colors.SetColor("Probe", *color_probe)  # 自定义探针颜色
colors.SetColor("Background",colors.GetColor3d('CornflowerBlue'))

# actor属性
property_vessel = vtkProperty()
property_vessel.SetColor(colors.GetColor3d("Vessel"))
# property_vessel.SetDiffuse(0.7)  # 设置漫反射
# property_vessel.SetSpecular(0.4)  # 设置镜面反射
# property_vessel.SetSpecularPower(20)  # 设置镜面反射强度

property_probe = vtkProperty()
property_probe.SetColor(colors.GetColor3d("Probe"))
property_probe.SetOpacity(0.5)    # 透明度

# 创建渲染器
ren = vtkRenderer()
# 创建渲染窗口，将渲染器添加到窗口中，设置窗口大小和标题
window = vtkRenderWindow()
window.AddRenderer(ren)  # 将渲染器放入渲染窗口
window.SetSize(1024, 576)  # 设置窗口大小
window.SetWindowName("IVOCT三维重建")  # 设置窗口标题
# 设置窗口交互方式
interactor = vtkRenderWindowInteractor()  # 创建交互器
interactor.SetRenderWindow(window)  # 将交互器与渲染窗口关联
style = vtkInteractorStyleTrackballCamera()  # 设置交互样式为"轨迹球摄像机"
interactor.SetInteractorStyle(style)

# 读取PNG序列
JPG_Reader = vtkJPEGReader()
JPG_Reader.SetNumberOfScalarComponents(1)    # 设置图像中每个像素的标量分量数量，灰度图1，彩色图3
JPG_Reader.SetFileDimensionality(2)  # 维度
JPG_Reader.SetFilePattern("interpolation_out_3d_in/%06d.jpg")     # 这是设置文件名的路径和格式
JPG_Reader.SetFileNameSliceOffset(0)    # 设置从哪个数字开始命名
JPG_Reader.SetDataExtent(0, 512, 0, 512, 0, pic_num)# 读取PNG数据的空间范围，z范围代表图片序列数,用来定义要读取的图像序列的范围
JPG_Reader.Update()
spacing = [4, 4, 7]  #xyz方向上，vtkImageData的每个数据点的间隔单位数.512*512
JPG_Reader.GetOutput().SetSpacing(spacing)  # 仅作用于数据输出对象，在读取完数据后可以动态调整间距。用户根据实际撤回速度，给定数值。
print('JPG reading completed')

# 高斯平滑
gauss = vtkImageGaussianSmooth()
gauss.SetInputConnection(JPG_Reader.GetOutputPort())
gauss.SetStandardDeviations(1, 1, 2) #标准差
gauss.SetRadiusFactors(1, 1, 2)  #半径
gauss.Update()
gauss.GetOutput().SetSpacing(spacing)
print('gauss completed')

# 创建 vtkFlyingEdges3D 对象
flying_edges = vtkFlyingEdges3D()
flying_edges.SetInputConnection(gauss.GetOutputPort())
flying_edges.SetValue(0, 128)  # 设置边缘提取的阈值
flying_edges.Update()  # 更新过滤器
print('flying_edges completed')

smoothFilter = vtkWindowedSincPolyDataFilter()
smoothFilter.SetInputConnection(flying_edges.GetOutputPort())
smoothFilter.SetNumberOfIterations(150)
smoothFilter.SetPassBand(0.1)
smoothFilter.BoundarySmoothingOn()
smoothFilter.FeatureEdgeSmoothingOff()
smoothFilter.NonManifoldSmoothingOn()
smoothFilter.NormalizeCoordinatesOn()
smoothFilter.Update()
print('smooth completed')

mapper = vtkPolyDataMapper()
mapper.SetInputConnection(smoothFilter.GetOutputPort())
mapper.ScalarVisibilityOff()    # 关闭标量可视化后，所有几何体的颜色将统一，而不会根据标量数据来变化。

actor = vtkActor()
actor.SetMapper(mapper)
actor.SetProperty(property_vessel)
actor.SetPosition([-512*4/2,-512*4/2,0])

# 将演员放入渲染器，并设置背景颜色
ren.AddActor(actor)  # 将血管添加到渲染器中
ren.SetBackground(colors.GetColor3d('Background'))  # 设置背景颜色为午夜蓝

# 创建指示探针位置的薄片
box = vtkCylinderSource()
box.SetHeight(7)       # 设置圆柱的高度
box.SetRadius(512*2)       # 设置圆柱的半径
box.SetResolution(50)    # 设置圆柱的分辨率（越高越圆滑）
boxMapper = vtkPolyDataMapper()
boxMapper.SetInputConnection(box.GetOutputPort())
boxActor = vtkActor()
boxActor.SetMapper(boxMapper)
boxActor.SetProperty(property_probe)
boxActor.RotateX(90)
ren.AddActor(boxActor)
print("指针创建成功")

slice_reader = vtkJPEGReader()
slice_actor = vtkImageActor()

# 创建探针
picker = vtkCellPicker()
picker.SetTolerance(0.001)
picker.AddPickList(actor)   # 将血管模型加入拾取列表中，只捕捉鼠标在模型上的坐标
picker.PickFromListOn()  # 启用拾取列表


def OnMouseMove(obj, event):
    # 清除上一次的切片图像，避免堆积
    ren.RemoveActor(slice_actor)
    # 获取鼠标位置
    x, y = obj.GetEventPosition()
    if picker.Pick(x, y, 0, ren):   # 将窗口坐标转换为世界坐标
        point = picker.GetPickPosition()
        image_index = int(point[2] / spacing[2])  # 根据鼠标位置获取对应的图片索引,假设 z 方向上的间距
        # 加载切片图片
        file_name = "interpolation_out_3d_in/%06d.jpg"% image_index
        slice_reader.SetFileName(file_name)  # 这是设置文件名的路径和格式
        slice_reader.Update()
        slice_reader.GetOutput().SetSpacing(spacing[0], spacing[1], 1)
        slice_actor.SetInputData(slice_reader.GetOutput())
        slice_actor.SetPosition(512*2, -512*2, point[2])
        ren.AddActor(slice_actor)
        # 更新指针的位置
        boxActor.SetPosition(0, 0, point[2])
        ren.AddActor(boxActor)
        # print(f"切片加载成功")
    else:
        ren.RemoveActor(boxActor)
        # print("超出切片范围")

    # 更新渲染结果
    window.Render()

interactor.AddObserver("MouseMoveEvent", OnMouseMove)

# 开始显示
if __name__ == '__main__':
    window.Render()     # 先渲染窗口
    end_3d = time.time()
    print(f'渲染耗时：{end_3d - start_3d:.4f} 秒')
    interactor.Initialize()     # 再初始化交互
    interactor.Start()  # 最后启动交互
