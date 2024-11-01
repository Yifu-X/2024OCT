import os
import vtk


def create_point_cloud_from_images(image_folder):
    points = vtk.vtkPoints()

    # 获取文件夹中所有的 JPG 文件
    jpg_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]

    for jpg_file in jpg_files:
        file_path = os.path.join(image_folder, jpg_file)

        # 读取图像
        reader = vtk.vtkJPEGReader()
        reader.SetFileName(file_path)
        reader.Update()

        image_data = reader.GetOutput()

        # 二值化处理
        threshold = vtk.vtkImageThreshold()
        threshold.SetInputData(image_data)
        threshold.ThresholdByUpper(128)  # 提取白色区域（值为255）
        threshold.SetInValue(255)  # 白色区域保留
        threshold.SetOutValue(0)  # 其他区域设为0
        threshold.Update()

        # 提取白色像素
        white_pixels = threshold.GetOutput()
        dims = white_pixels.GetDimensions()
        scalars = white_pixels.GetPointData().GetScalars()  # 获取标量数据

        for z in range(dims[2]):  # 遍历每一层
            for y in range(dims[1]):  # 遍历每一行
                for x in range(dims[0]):  # 遍历每一列
                    pixel_value = scalars.GetComponent(x + y * dims[0] + z * dims[0] * dims[1], 0)
                    if pixel_value == 255:  # 如果是白色
                        points.InsertNextPoint(x, y, z)

    # 创建点云
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    # 可选：创建点的几何图形（如球体）
    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(vtk.vtkSphereSource().GetOutputPort())
    glyph.SetInputData(polydata)
    glyph.SetScaleModeToDataScaling()
    glyph.Update()

    # 渲染
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # 创建渲染器和渲染窗口
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.1, 0.1)  # 背景颜色
    render_window.Render()
    render_window_interactor.Start()


# 使用示例
create_point_cloud_from_images(r"C:\Users\19398\Desktop\cloud_test")