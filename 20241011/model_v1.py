'''
v1:分离了建模与渲染过程
'''
import os
import time
from vtk import (
    vtkJPEGReader,
    vtkPolyDataReader,
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

class Modeling:
    def __init__(self, jpg_folder: str, width: int, height: int, spacing=[4, 4, 7]):
        self.jpg_folder = jpg_folder    # jpg文件夹地址
        self.width = width   # 横向分辨率
        self.height = height    # 纵向分辨率
        self.spacing = spacing  # 空间间隔
        self.vtk_polydata = None  # 存储处理后的vtkPolyData对象

    # 读取已有模型
    def vtkReader(self, vtkfilepath: str, vtkfilename: str):
        vtk_reader = vtkPolyDataReader()
        vtk_reader.SetFileName(os.path.join(vtkfilepath, vtkfilename))  # 指定要读取的 .vtk 文件路径
        vtk_reader.Update()  # 执行读取操作
        print(f"模型已成功从{vtkfilename}文件读取！")
        return vtk_reader.GetOutput()  # 获取 PolyData 对象

    # 读取JPG序列
    def jpgReader(self):
        jpg_files = [file for file in os.listdir(self.jpg_folder) if file.lower().endswith('.jpg')]
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

    # 没有就建立文件夹
    def folder_preparation(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print('输出文件夹已建立')
        else: print('输出文件夹寻址成功')

    # 保存stl文件
    def save_model_as_stl(self):
        if self.vtk_polydata is None:
            print("错误: 需要先运行process方法生成vtkpolydata!")
            return
        self.folder_preparation(self.jpg_folder)
        stl_writer = vtkSTLWriter()
        stl_writer.SetFileName(os.path.join(self.jpg_folder, self.jpg_folder + ".stl"))
        stl_writer.SetInputData(self.vtk_polydata)
        stl_writer.Write()
        print(f"stl模型已成功保存为'{self.jpg_folder}.stl'，保存路径：'{self.jpg_folder}'")

    # 保存vtk文件到图片序列的原文件夹
    def save_model_as_vtk(self):
        if self.vtk_polydata is None:
            print("错误: 需要先运行process方法生成vtkpolydata!")
            return
        self.folder_preparation(self.jpg_folder)
        vtk_writer = vtkPolyDataWriter()
        vtk_writer.SetFileName(os.path.join(self.jpg_folder, self.jpg_folder + ".vtk"))
        vtk_writer.SetInputData(self.vtk_polydata)
        vtk_writer.Write()
        print(f"vtk模型已成功保存为'{self.jpg_folder}.vtk'，保存路径：'{self.jpg_folder}'")

    # 建模的主方法
    def process(self):
        start_3d = time.time()

        if self.jpg_folder+'.vtk' in os.listdir(self.jpg_folder):
            # 读取已有模型
            print("检测到已有模型")
            self.vtk_polydata = self.vtkReader(self.jpg_folder, self.jpg_folder+'.vtk')
        else:
            # 无已有模型，建模
            print("未检测到已有模型，开始建模")

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
            self.vtk_polydata = smoothFilter.GetOutput()
            # 保存模型
            self.save_model_as_vtk()

        # self.save_model_as_stl()
        end_3d = time.time()
        print(f'建模完成！总耗时：{end_3d - start_3d:.4f} 秒')


class Rendering:
    def __init__(self, vtk_polydata):
        self.vtk_polydata = vtk_polydata
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

    def process(self):

        # 分段设置颜色
        points = self.vtk_polydata.GetPoints()
        for i in range(points.GetNumberOfPoints()):
            x, y, z = points.GetPoint(i)
            if 1582 <= z <= 1694 or 2177 <= z <= 2345 or 5208 <= z <= 5348 or 7021 <= z <= 7252:
                self.colors_array.InsertNextTuple3(*[int(c * 255) for c in self.colors.GetColor3d("Branch")])
            else:
                self.colors_array.InsertNextTuple3(*[int(c * 255) for c in self.colors.GetColor3d("Vessel")])

        self.vtk_polydata.GetPointData().SetScalars(self.colors_array)

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(self.vtk_polydata)
        mapper.ScalarVisibilityOn()

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.RotateY(-90)
        print("演员创建完成")

        ren = vtkRenderer()
        ren.AddActor(actor)
        ren.SetBackground(self.colors.GetColor3d('Background'))

        window = vtkRenderWindow()
        window.AddRenderer(ren)
        window.SetSize(1024, 576)
        window.SetWindowName("IVOCT三维重建")

        interactor = vtkRenderWindowInteractor()
        interactor.SetRenderWindow(window)
        style = vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(style)

        window.Render()
        interactor.Initialize()
        interactor.Start()

# 开始显示
if __name__ == '__main__':
    model_modeling = Modeling("interpolation_out_3d_in", 512, 512)
    model_modeling.process()
    render = Rendering(model_modeling.vtk_polydata)
    render.process()
