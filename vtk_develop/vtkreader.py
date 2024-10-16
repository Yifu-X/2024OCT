'''
读取现有的vtk模型
'''
from vtk import vtkPolyDataReader
import os
import time
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
# 读取已经渲染好的vtk模型，可以直接导入渲染
def vtkReader(vtkfilepath:str, vtkfilename:str):
    vtk_reader = vtkPolyDataReader()
    vtk_reader.SetFileName(os.path.join(vtkfilepath,vtkfilename))  # 指定要读取的 .vtk 文件路径
    vtk_reader.Update() # 执行读取操作
    print("模型已成功从.vtk文件读取！")
    return vtk_reader.GetOutput()   # 获取 PolyData 对象

start_3d = time.time()
colors = vtkNamedColors()  # 创建颜色对象
color_vessel= map(lambda x: x / 255.0, [204, 0, 0, 255])  # 将传入的函数依次作用到序列的每个元素。lambda表示x的匿名函数x/255
colors.SetColor("Vessel", *color_vessel)  # 自定义血管颜色
color_probe= map(lambda x: x / 255.0, [192, 192, 192, 255])  # 将传入的函数依次作用到序列的每个元素。lambda表示x的匿名函数x/255
colors.SetColor("Probe", *color_probe)  # 自定义探针颜色
colors.SetColor("Background",colors.GetColor3d('CornflowerBlue'))

vtkmodel = vtkReader("123456","123456.vtk")

mapper = vtkPolyDataMapper()
mapper.SetInputData(vtkmodel)
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
    window.Render()     # 先渲染窗口
    end_3d = time.time()
    print(f'进程总耗时：{end_3d - start_3d:.4f} 秒')
    interactor.Initialize()  # 再初始化交互
    interactor.Start()  # 最后启动交互