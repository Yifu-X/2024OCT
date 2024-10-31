'''
***该版本被GUI直接调用，谨慎修改***
无后缀：git同步版本
v2: 该版本可以被GUI直接调用，传回render；同时加入指针功能
v1:分离了建模与渲染过程，更适配GUI操作；
'''
import os
import time
from vtk import (
    vtkJPEGReader,
    vtkPolyDataReader,
    vtkCylinderSource,
    vtkImageGaussianSmooth,
    vtkWindowedSincPolyDataFilter,
    vtkFlyingEdges3D,
    vtkSTLWriter,
    vtkPolyDataWriter,
    vtkUnsignedCharArray,
    vtkCellPicker
)
from vtkmodules.vtkRenderingCore import (
    vtkPolyDataMapper,
    vtkProperty,
    vtkActor,
    vtkRenderer
)
from vtkmodules.vtkCommonColor import vtkNamedColors    # 颜色

class Modeling:
    def __init__(self, project_folder: str, width: int, height: int, spacing=[4, 4, 7]):
        self.project_folder   = project_folder    # jpg文件夹地址
        self.jpg_folder = os.path.join(self.project_folder, os.path.basename(self.project_folder) + "_interpolation")
        self.vtkname = os.path.basename(self.project_folder) + ".vtk"
        self.width = width   # 横向分辨率
        self.height = height    # 纵向分辨率
        self.spacing = spacing  # 空间间隔
        self.vtk_polydata = None  # 存储处理后的vtkPolyData对象
        self.model_probe = None # 储存探针模型

    # 自定义颜色
    def setup_colors(self):
        color_vessel = list(map(lambda x: x / 255.0, [204, 0, 0, 255]))
        self.colors.SetColor("Vessel", *color_vessel)  # 血管红色
        color_branch = list(map(lambda x: x / 255.0, [8, 79, 184, 255]))
        self.colors.SetColor("Branch", *color_branch)  # 分支蓝色
        color_probe = list(map(lambda x: x / 255.0, [192, 192, 192, 255]))
        self.colors.SetColor("Probe", *color_probe)  # 探针银色
        self.colors.SetColor("Background", self.colors.GetColor3d('CornflowerBlue'))  # 背景蓝色

    # 读取已有模型
    def vtkReader(self, vtkfilepath: str, vtkfilename: str):
        vtk_reader = vtkPolyDataReader()
        vtk_reader.SetFileName(os.path.join(vtkfilepath, vtkfilename))  # 指定要读取的 .vtk 文件路径
        vtk_reader.Update()  # 执行读取操作
        print(f"模型已成功从{vtkfilename}文件读取！")
        return vtk_reader.GetOutput()  # 获取 PolyData 对象

    # 读取JPG序列
    def jpgReader(self):
        jpg_files = [file for file in os.listdir(self.jpg_folder) if file.lower().endswith('.jpg')] # 去读取插值文件夹
        jpg_count = len(jpg_files)
        jpg_reader = vtkJPEGReader()
        jpg_reader.SetNumberOfScalarComponents(1)
        jpg_reader.SetFileDimensionality(2)
        jpg_reader.SetFilePattern(os.path.join(self.jpg_folder, "%06d.jpg"))
        jpg_reader.SetFileNameSliceOffset(0)
        jpg_reader.SetDataExtent(0, self.width - 1, 0, self.height - 1, 0, jpg_count - 1)
        jpg_reader.Update()
        jpg_reader.GetOutput().SetSpacing(self.spacing)
        print('JPG reading completed')
        return jpg_reader.GetOutput()

    # 保存stl文件
    def save_model_as_stl(self):
        if self.vtk_polydata is None:
            print("错误: 需要先运行process方法生成vtkpolydata!")
            return
        stl_writer = vtkSTLWriter()
        stl_writer.SetFileName(os.path.join(self.project_folder, self.project_folder + ".stl"))
        stl_writer.SetInputData(self.vtk_polydata)
        stl_writer.Write()
        print(f"stl模型已成功保存为'{self.project_folder}.stl'，保存路径：'{self.project_folder}'")

    # 保存vtk文件到图片序列的原文件夹
    def save_model_as_vtk(self, vtkpolydata, vtkfolder:str, vtkname:str):
        if vtkpolydata is None:
            print("错误: 需要先运行process方法生成vtkpolydata!")
            return
        vtk_writer = vtkPolyDataWriter()
        vtk_writer.SetFileName(os.path.join(vtkfolder, vtkname))
        vtk_writer.SetInputData(vtkpolydata)
        vtk_writer.Write()
        print(f"vtk模型已成功保存为'{vtkname}'，保存路径：'{vtkfolder}'")

    # 建模的主方法
    def process(self):
        start_3d = time.time()
        if self.vtkname in os.listdir(self.project_folder):
            # 读取已有模型
            print("检测到已有模型，读取中......")
            self.vtk_polydata = self.vtkReader(self.project_folder, self.vtkname)
            end_3d = time.time()
            print(f'读取完成！总耗时：{end_3d - start_3d:.4f} 秒')
        else:
            # 无已有模型，建模
            print("未检测到已有模型，开始建模")

            if not os.path.exists(self.jpg_folder):
                print("请先对二值图像进行插值，再进行建模")
            else:
                # 读取jpg序列
                jpgmodel = self.jpgReader()

                # 高斯平滑
                gauss = vtkImageGaussianSmooth()
                gauss.SetInputData(jpgmodel)
                gauss.SetStandardDeviations(1, 1, 1)    # 标准差
                gauss.SetRadiusFactors(1, 1, 1)     # 半径
                gauss.Update()
                gauss.GetOutput().SetSpacing(self.spacing)
                print('gauss completed')

                # 创建 vtkFlyingEdges3D 对象
                flying_edges = vtkFlyingEdges3D()
                flying_edges.SetInputConnection(gauss.GetOutputPort())
                flying_edges.SetValue(0, 128)   # 设置边缘提取的阈值
                flying_edges.Update()
                print('flying_edges completed')


                # 平滑
                smoothFilter = vtkWindowedSincPolyDataFilter()
                smoothFilter.SetInputConnection(flying_edges.GetOutputPort())
                smoothFilter.SetNumberOfIterations(150)
                smoothFilter.SetPassBand(0.1)
                smoothFilter.BoundarySmoothingOn()
                smoothFilter.FeatureEdgeSmoothingOff()
                smoothFilter.NonManifoldSmoothingOn()
                smoothFilter.NormalizeCoordinatesOn()
                smoothFilter.Update()
                print('smoothFilter completed')


                # 保存模型
                self.vtk_polydata = smoothFilter.GetOutput()
                self.save_model_as_vtk(self.vtk_polydata,self.project_folder,self.vtkname)
                end_3d = time.time()
                print(f'建模完成！总耗时：{end_3d - start_3d:.4f} 秒')

        # 创建指示探针位置的薄片
        box = vtkCylinderSource()
        box.SetHeight(7)  # 设置圆柱的高度
        box.SetRadius(512 * 2)  # 设置圆柱的半径
        box.SetResolution(50)  # 设置圆柱的分辨率（越高越圆滑）
        box.Update() # 确保生成输出!!!!!!!!!!!!
        self.model_probe = box.GetOutput()
        self.save_model_as_vtk(self.model_probe, self.project_folder, "probe.vtk")
        print("指针模型建模成功")
        print()

class Rendering:
    def __init__(self, vtk_polydata, probe):
        self.polydata_vessel = vtk_polydata
        self.polydata_probe = probe
        self.actor_vessel = None
        self.actor_probe = None
        self.render = None  # 存储处理后的render渲染对象
        self.colors = vtkNamedColors()  # 颜色列表
        self.setup_colors()
        self.colors_array = vtkUnsignedCharArray()  # 存储血管颜色的数组
        self.colors_array.SetNumberOfComponents(3)  # 3 分量表示 RGB
        self.colors_array.SetName("whole")  # 设置数组名称

    # 自定义颜色
    def setup_colors(self):
        color_vessel = list(map(lambda x: x / 255.0, [204, 0, 0, 255]))
        self.colors.SetColor("Vessel", *color_vessel)  # 血管红色
        color_branch = list(map(lambda x: x / 255.0, [8, 79, 184, 255]))
        self.colors.SetColor("Branch", *color_branch)  # 分支蓝色
        color_probe = list(map(lambda x: x / 255.0, [192, 192, 192, 255]))
        self.colors.SetColor("Probe", *color_probe)  # 探针银色
        self.colors.SetColor("Background", self.colors.GetColor3d('CornflowerBlue'))  # 背景蓝色

    # 渲染主方法
    def process(self):
        print("开始渲染")
        # ————————————————探针渲染————————————————————
        property_probe = vtkProperty()
        property_probe.SetColor(self.colors.GetColor3d("Probe"))
        property_probe.SetOpacity(0.5)  # 透明度

        mapper_probe = vtkPolyDataMapper()
        mapper_probe.SetInputData(self.polydata_probe)
        self.actor_probe = vtkActor()
        self.actor_probe.SetMapper(mapper_probe)
        self.actor_probe.SetProperty(property_probe)
        self.actor_probe.RotateX(90)
        print("指针渲染成功")

        # ————————————————血管渲染————————————————————
        # 分段设置颜色
        points = self.polydata_vessel.GetPoints()
        for i in range(points.GetNumberOfPoints()):
            x, y, z = points.GetPoint(i)
            if 1582 <= z <= 1694 or 2177 <= z <= 2345 or 5208 <= z <= 5348 or 7021 <= z <= 7252:
                self.colors_array.InsertNextTuple3(*[int(c * 255) for c in self.colors.GetColor3d("Branch")])
            else:
                self.colors_array.InsertNextTuple3(*[int(c * 255) for c in self.colors.GetColor3d("Vessel")])
        self.polydata_vessel.GetPointData().SetScalars(self.colors_array)
        print("分支颜色设置完成")

        mapper_vessel = vtkPolyDataMapper()
        mapper_vessel.SetInputData(self.polydata_vessel)
        mapper_vessel.ScalarVisibilityOn()
        print("映射器创建完成")

        self.actor_vessel = vtkActor()
        self.actor_vessel.SetMapper(mapper_vessel)
        self.actor_vessel.SetPosition([-512*4/2,-512*4/2,0])
        print("演员创建完成")

        self.render = vtkRenderer()
        self.render.AddActor(self.actor_vessel)
        self.render.AddActor(self.actor_probe)
        self.render.SetBackground(self.colors.GetColor3d('Background'))
        print("渲染器创建完成")

        # 创建picker获取位置
        self.picker = vtkCellPicker()
        self.picker.SetTolerance(0.001)
        self.picker.AddPickList(self.actor_vessel)  # 将血管模型加入拾取列表中，只捕捉鼠标在模型上的坐标
        self.picker.PickFromListOn()  # 启用拾取列表
        print("picker创建成功")

        print("渲染完成")
        print()
        # 到这里，产生了一个ren
        # 下面是把ren加入到windows里

# 开始显示
if __name__ == '__main__':
    model_modeling = Modeling("interpolation_out_3d_in", 512, 512)
    model_modeling.process()
    render = Rendering(model_modeling.vtk_polydata)
    render.process()
