import vtk


# 定义图片读取接口
PNG_Reader = vtk.vtkPNGReader()
PNG_Reader.SetNumberOfScalarComponents(1)
PNG_Reader.SetFileDimensionality(3)

# 定义图像大小
PNG_Reader.SetDataExtent(0, 255, 0, 255, 0, 269)

# 设置图像的存放位置
name_prefix = ['after/']
name1_prefix = ['D:/Project/python/2024OCT/output/']
PNG_Reader.SetFilePrefix(name_prefix[0])

# 设置图像前缀名字
PNG_Reader.SetFilePattern("%s%06d.png")
PNG_Reader.Update()
PNG_Reader.SetDataByteOrderToLittleEndian()
spacing = [0.1, 0.1, 2.8]  # x, y 方向上的间距为 0.1 像素，z 方向上的间距为 2.8 像素
PNG_Reader.GetOutput().SetSpacing(spacing)

# 计算轮廓的方法
contour = vtk.vtkMarchingCubes()
PNG_Reader.GetOutput().SetSpacing(spacing)
contour.SetInputConnection(PNG_Reader.GetOutputPort())
contour.ComputeNormalsOn()
contour.SetValue(0, 100)

# 添加切割平面
plane = vtk.vtkPlane()
plane.SetOrigin(96.0, 111.0, 378.0)  # 设置切割平面的原点
plane.SetNormal(1, 0, 0)  # 设置切割平面的法向量

clipper = vtk.vtkClipPolyData()
clipper.SetInputConnection(contour.GetOutputPort())
clipper.SetClipFunction(plane)
clipper.SetValue(0)  # 0代表保留切割平面法线方向上的部分

# 创建Mapper
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(clipper.GetOutputPort())
mapper.ScalarVisibilityOff()

# 创建Actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(1, 0, 0)  # 设置Actor的颜色为红色

# 创建Renderer
renderer = vtk.vtkRenderer()
renderer.SetBackground([1.0, 1.0, 1.0])
renderer.AddActor(actor)

# 创建RenderWindow
window = vtk.vtkRenderWindow()
window.SetSize(512, 512)
window.AddRenderer(renderer)

# 创建Interactor
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(window)

# 定义图片显示接口
imageActor = vtk.vtkImageActor()
imageActor.GetProperty().SetOpacity(0.5)  # 设置图片透明度


# 开始显示
if __name__ == '__main__':
    window.Render()
    interactor.Initialize()
    interactor.Start()
