'''
v6:尝试分支开孔
reconstruction更新到此结束，后续转为model_v1，分离了建模与渲染过程，更适配GUI操作
v5:优化代码结构；
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
    vtkImageGaussianSmooth,
    vtkWindowedSincPolyDataFilter,
    vtkFlyingEdges3D,
    vtkSTLWriter,
    vtkPolyDataWriter,
    vtkUnsignedCharArray,
    vtkAppendPolyData,
    vtkTransform,
    vtkTransformPolyDataFilter,
    vtkPlane,
    vtkClipPolyData,
    vtkExtractVOI,
)
from vtkmodules.vtkCommonColor import vtkNamedColors  # 颜色
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import (
    vtkPolyDataMapper,
    vtkActor,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor
)

class Reconstruction:
    def __init__(self, jpg_folder: str, width: int, height: int, spacing=[4, 4, 7]):
        self.jpg_folder = jpg_folder  # jpg文件夹地址
        self.width = width  # 横向分辨率
        self.height = height  # 纵向分辨率
        self.spacing = spacing  #
        self.colors = vtkNamedColors()  # 颜色列表
        self.setup_colors()
        self.colors_array = vtkUnsignedCharArray()  # 存储血管颜色的数组
        self.colors_array.SetNumberOfComponents(3)  # 3 分量表示 RGB
        self.colors_array.SetName("whole")  # 设置数组名称
        self.vtk_polydata = None  # 存储处理后的vtkPolyData对象

    # 自定义颜色
    def setup_colors(self):
        color_vessel = list(map(lambda x: x / 255.0, [204, 0, 0, 255]))
        self.colors.SetColor("Vessel", *color_vessel)  # 血管红色
        color_branch = list(map(lambda x: x / 255.0, [8, 79, 184, 255]))
        self.colors.SetColor("Branch", *color_branch)  # 分支蓝色
        color_probe = list(map(lambda x: x / 255.0, [192, 192, 192, 255]))
        self.colors.SetColor("Probe", *color_probe)  # 探针银色
        self.colors.SetColor("Background", self.colors.GetColor3d('CornflowerBlue'))  # 背景蓝色

    # 读取已有的vtk模型
    def vtkReader(self, vtkfilepath: str, vtkfilename: str):
        vtk_reader = vtkPolyDataReader()
        vtk_reader.SetFileName(os.path.join(vtkfilepath, vtkfilename))  # 指定要读取的 .vtk 文件路径
        vtk_reader.Update()  # 执行读取操作
        print(f"模型已成功从{vtkfilename}.vtk文件读取！")
        return vtk_reader.GetOutput()  # 获取 PolyData 对象

    # 读取JPG序列
    def jpgReader(self,start,end):
        jpg_files = [file for file in os.listdir(self.jpg_folder) if file.lower().endswith('.jpg')]
        jpg_count = len(jpg_files)
        jpg_reader = vtkJPEGReader()
        jpg_reader.SetNumberOfScalarComponents(1)
        jpg_reader.SetFileDimensionality(2)
        jpg_reader.SetFilePattern(os.path.join(self.jpg_folder, "%06d.jpg"))
        jpg_reader.SetFileNameSliceOffset(start)
        jpg_reader.SetDataExtent(0, self.width - 1, 0, self.height - 1, 0, end-start)
        jpg_reader.Update()
        jpg_reader.GetOutput().SetSpacing(self.spacing)
        print('JPG reading completed')
        return jpg_reader.GetOutput()

    # 没有就建立文件夹
    def folder_preparation(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print('输出文件夹已建立')
        else:
            print('输出文件夹寻址成功')

    def save_model_as_stl(self, filepath: str, filename: str):
        if self.vtk_polydata is None:
            print("错误: 需要先运行process方法生成vtkpolydata!")
            return
        self.folder_preparation(filepath)
        stl_writer = vtkSTLWriter()
        stl_writer.SetFileName(os.path.join(filepath, filename + ".stl"))
        stl_writer.SetInputData(self.vtk_polydata)
        stl_writer.Write()
        print(f"stl模型已成功保存为'{filename}'，保存路径：'{filepath}'")

    def save_model_as_vtk(self, filepath: str, filename: str):
        if self.vtk_polydata is None:
            print("错误: 需要先运行process方法生成vtkpolydata!")
            return
        self.folder_preparation(filepath)
        vtk_writer = vtkPolyDataWriter()
        vtk_writer.SetFileName(os.path.join(filepath, filename + ".vtk"))
        vtk_writer.SetInputData(self.vtk_polydata)
        vtk_writer.Write()
        print(f"vtk模型已成功保存为'{filename}'，保存路径：'{filepath}'")

    def process(self):
        start_3d = time.time()

        # 读取jpg序列
        jpgmodel = self.jpgReader(0, 500)

        # 高斯平滑
        gauss = vtkImageGaussianSmooth()
        gauss.SetInputData(jpgmodel)
        gauss.SetStandardDeviations(1, 1, 1)  # 标准差
        gauss.SetRadiusFactors(1, 1, 1)  # 半径
        gauss.Update()
        print('gauss completed')

        # 分割
        extractor_1 = vtkExtractVOI()
        extractor_1.SetInputData(gauss.GetOutput())  # 假设你有一个 vtkImageData 对象
        extractor_1.SetVOI(0, self.width - 1, 0, self.height - 1, 0, 315)  # 定义你感兴趣的区域
        extractor_1.Update()
        extractor_1.GetOutput().SetSpacing(self.spacing)  # 设置空间间隔
        print("切割完毕")

        extractor_2 = vtkExtractVOI()
        extractor_2.SetInputData(gauss.GetOutput())  # 假设你有一个 vtkImageData 对象
        extractor_2.SetVOI(0, self.width - 1, 0, self.height - 1, 316, 500)  # 定义你感兴趣的区域
        extractor_2.Update()
        extractor_2.GetOutput().SetSpacing(self.spacing)  # 设置空间间隔
        print("切割完毕")

        # 创建 vtkFlyingEdges3D 对象
        flying_edges_1 = vtkFlyingEdges3D()
        flying_edges_1.SetInputConnection(extractor_1.GetOutputPort())
        flying_edges_1.SetValue(0, 128)  # 设置边缘提取的阈值
        flying_edges_1.Update()

        flying_edges_2 = vtkFlyingEdges3D()
        flying_edges_2.SetInputConnection(extractor_2.GetOutputPort())
        flying_edges_2.SetValue(0, 128)  # 设置边缘提取的阈值
        flying_edges_2.Update()
        print('flying_edges completed')

        # 拼接
        append_filter = vtkAppendPolyData()
        append_filter.AddInputData(flying_edges_1.GetOutput())
        append_filter.AddInputData(flying_edges_2.GetOutput())
        append_filter.Update()

        '''
        # 创建 vtkFlyingEdges3D 对象
        flying_edges_1 = vtkFlyingEdges3D()
        flying_edges_1.SetInputConnection(gauss.GetOutputPort())
        flying_edges_1.SetValue(0, 128)  # 设置边缘提取的阈值
        flying_edges_1.Update()

        # 剪切面
        plane = vtkPlane()
        plane.SetOrigin(0, 0, 315*self.spacing[2])  # 平面的原点
        plane.SetNormal(0, 0, 1)  # 平面的法向量，决定保留的方向

        # 创建 vtkClipPolyData
        clipper = vtkClipPolyData()
        clipper.SetInputConnection(flying_edges_1.GetOutputPort())
        clipper.SetClipFunction(plane)  # 设置剪切平面
        # clipper.InsideOutOn()  # 如果启用此行，将保留剪切掉的部分，而不是默认的保留部分
        clipper.Update()
        # 获取剪切后的数据
        # clipped_polydata = clipper.GetOutput().SetSpacing(self.spacing)
        
        # 高斯平滑
        gauss_2 = vtkImageGaussianSmooth()
        gauss_2.SetInputData(jpgmodel_2)
        gauss_2.SetStandardDeviations(1, 1, 1)  # 标准差
        gauss_2.SetRadiusFactors(1, 1, 1)  # 半径
        gauss_2.Update()
        gauss_2.GetOutput().SetSpacing(self.spacing)

        # 创建 vtkFlyingEdges3D 对象
        flying_edges_2 = vtkFlyingEdges3D()
        flying_edges_2.SetInputConnection(gauss_2.GetOutputPort())
        flying_edges_2.SetValue(0, 128)  # 设置边缘提取的阈值
        flying_edges_2.Update()
        print('flying_edges completed')
        
        # 平移
        transform = vtkTransform()
        z_translation = 315*self.spacing[2]  # 在 Z 方向上平移的距离
        transform.Translate(0, 0, z_translation)

        transform_filter = vtkTransformPolyDataFilter()
        transform_filter.SetInputData(flying_edges_2.GetOutput())
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        # 拼接
        append_filter = vtkAppendPolyData()
        append_filter.AddInputData(flying_edges_1.GetOutput())
        append_filter.AddInputData(transform_filter.GetOutput())
        append_filter.Update()
        '''

        # 平滑
        smoothFilter = vtkWindowedSincPolyDataFilter()
        smoothFilter.SetInputConnection(append_filter.GetOutputPort())
        smoothFilter.SetNumberOfIterations(150)
        smoothFilter.SetPassBand(0.1)
        smoothFilter.BoundarySmoothingOn()
        smoothFilter.FeatureEdgeSmoothingOff()
        smoothFilter.NonManifoldSmoothingOn()
        smoothFilter.NormalizeCoordinatesOn()
        smoothFilter.Update()

        # 保存生成的 vtkPolyData
        self.vtk_polydata = smoothFilter.GetOutput()

        '''
        # 分段设置颜色
        points = smoothFilter.GetOutput().GetPoints()
        for i in range(points.GetNumberOfPoints()):
            x, y, z = points.GetPoint(i)
            if 1582 <= z <= 1694 or 2177 <= z <= 2345 or 5208 <= z <= 5348 or 7021 <= z <= 7252:
                self.colors_array.InsertNextTuple3(*[int(c * 255) for c in self.colors.GetColor3d("Branch")])
            else:
                self.colors_array.InsertNextTuple3(*[int(c * 255) for c in self.colors.GetColor3d("Vessel")])

        smoothFilter.GetOutput().GetPointData().SetScalars(self.colors_array)
        '''

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(smoothFilter.GetOutputPort())
        mapper.ScalarVisibilityOn()

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.RotateY(-90)
        print("演员创建完成")

        # 创建渲染器
        ren = vtkRenderer()
        ren.AddActor(actor)
        ren.SetBackground(self.colors.GetColor3d('Background'))

        # 创建渲染窗口
        window = vtkRenderWindow()
        window.AddRenderer(ren)
        window.SetSize(1024, 576)
        window.SetWindowName("IVOCT三维重建")

        interactor = vtkRenderWindowInteractor()
        interactor.SetRenderWindow(window)
        style = vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(style)

        # 开始显示
        window.Render()

        '''
        # 保存模型
        self.save_model_as_stl(smoothFilter.GetOutput(), "3dmodel", "3dstl")
        self.save_model_as_vtk(smoothFilter.GetOutput(), "3dmodel", "3dvtk")
        '''

        end_3d = time.time()
        print(f'进程总耗时：{end_3d - start_3d:.4f} 秒')
        interactor.Initialize()
        interactor.Start()


# 开始显示
if __name__ == '__main__':
    model_reconstruction = Reconstruction("123456", 512, 512)
    model_reconstruction.process()

    # model_reconstruction.save_model_as_vtk("3dmodel", "3dvtk")
