'''
v8:尝试生成点云,生成实体，成功！
v7：使用不同的三维方法，从点云生成实体
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
from tqdm import tqdm # 终端进度条

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
    vtkImageThreshold,
    vtkGeometryFilter,
    vtkImageDataGeometryFilter,
    vtkImageSobel2D,
    vtkContourFilter,
    vtkMarchingSquares,
    vtkLinearExtrusionFilter,
    vtkProperty,
    vtkMarchingCubes,
    vtkPoints,
    vtkPolyData,
vtkGlyph3D,
vtkSphereSource,
vtkCubeSource,
vtkSurfaceReconstructionFilter,
vtkDelaunay3D,


)
from vtkmodules.vtkCommonColor import vtkNamedColors  # 颜色
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import (
    vtkPolyDataMapper,
    vtkActor,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
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

    # 由图片序列生成点云
    def create_point_cloud_from_images(self):
        points = vtkPoints()
        jpg_files = [file for file in os.listdir(self.jpg_folder) if file.lower().endswith('.jpg')]
        # 读取图像序列
        i = 0
        for jpg_file in tqdm(jpg_files):  # num_images 是你的图像序列数量
            file_path = os.path.join(self.jpg_folder, jpg_file)  # 根据你的文件命名方式调整
            # 读取一张图片
            reader = vtkJPEGReader()
            reader.SetFileName(file_path)
            reader.Update()

            # 二值化处理(还是非常必要的)
            threshold = vtkImageThreshold()
            threshold.SetInputData(reader.GetOutput())
            threshold.ThresholdByUpper(128)  # 提取白色区域（值为255）
            threshold.SetInValue(255)  # 白色区域保留
            threshold.SetOutValue(0)  # 其他区域设为0
            threshold.Update()

            # 提取白色像素
            white_pixels = threshold.GetOutput()
            dims = white_pixels.GetDimensions()
            scalars = white_pixels.GetPointData().GetScalars()  # 获取标量数据

            z_thickness = 7/4  # 设置厚度
            for y in range(dims[1]):  # 遍历每一行
                for x in range(dims[0]):  # 遍历每一列
                    pixel_value = scalars.GetComponent(x + y * dims[0], 0)  # 获取对应的像素值
                    if pixel_value == 255:  # 如果是白色
                        points.InsertNextPoint(x, y, i*z_thickness)  # 在 z=0 的平面上插入点
            i += 1
            # print(f"{jpg_file}读取完毕！")

        # 创建点云
        polydata = vtkPolyData()
        polydata.SetPoints(points)
        print("坐标点生成完毕！")

        # 创建 Delaunay 3D 对象
        delaunay = vtkDelaunay3D()
        delaunay.SetInputData(polydata)
        delaunay.SetAlpha(7)  # 可以调整这个值
        delaunay.SetTolerance(10)  # 设置适当的公差值
        delaunay.Update()
        print(type(delaunay.GetOutput()))

        geometry_filter = vtkGeometryFilter()
        geometry_filter.SetInputData(delaunay.GetOutput())  # 输入 vtkUnstructuredGrid
        geometry_filter.Update()  # 执行过滤

        '''                
        # 创建小球作为点的几何体
        sphere_source = vtkSphereSource()
        sphere_source.SetRadius(0.5)  # 小球的半径
        sphere_source.SetPhiResolution(10)  # 经度分辨率
        sphere_source.SetThetaResolution(10)  # 纬度分辨率
        sphere_source.Update()
        
        # 创建立方体源
        cube_source = vtkCubeSource()
        cube_source.SetXLength(1)  # 设置立方体的宽度
        cube_source.SetYLength(1)  # 设置立方体的高度
        cube_source.SetZLength(7 / 4)  # 设置立方体的厚度（与之前的小球厚度一致）

        # 使用Glyph3D将每个点转换为小球
        glyph = vtkGlyph3D()
        glyph.SetSourceConnection(cube_source.GetOutputPort())
        glyph.SetInputData(polydata)
        # glyph.SetScaleModeToDataScalingOff()  # 使用点的标量值缩放（可选）
        glyph.Update()
        '''
        return geometry_filter.GetOutput()

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

        PointCloud = self.create_point_cloud_from_images()

        # 对 Delaunay 结果进行平滑处理
        smoothFilter = vtkWindowedSincPolyDataFilter()
        smoothFilter.SetInputData(PointCloud)
        smoothFilter.SetNumberOfIterations(50)
        smoothFilter.SetPassBand(0.1)
        smoothFilter.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(smoothFilter.GetOutput())
        mapper.ScalarVisibilityOff()

        property_vessel = vtkProperty()
        property_vessel.SetColor(self.colors.GetColor3d("Vessel"))

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.SetProperty(property_vessel)
        actor.GetProperty().SetRepresentationToSurface()  # 只显示表面
        # actor.RotateY(-90)
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
    model_reconstruction = Reconstruction(r"C:\Users\19398\Desktop\cloud_test", 512, 512)
    model_reconstruction.process()

    # model_reconstruction.save_model_as_vtk("3dmodel", "3dvtk")
