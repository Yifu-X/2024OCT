'''
v0:学长原版代码
'''
import vtk
import xlrd
import pandas as pd
# 定义渲染窗口、交互模式
aRender = vtk.vtkRenderer()
Renwin = vtk.vtkRenderWindow()
Renwin.SetSize(512, 512)
Renwin.AddRenderer(aRender)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(Renwin)

# 和poly一样是source
# 定义图片读取接口
PNG_Reader = vtk.vtkPNGReader()
PNG_Reader.SetNumberOfScalarComponents(1)
PNG_Reader.SetFileDimensionality(2)

# 定义图像大小
PNG_Reader.SetDataExtent(0, 255, 0, 255, 0, 269)

# 设置图像的存放位置
name_prefix = ['D:/Project/python/2024OCT/3D/after/']
name1_prefix = ['D:/Project/python/2024OCT/3D/after/']
name2_prefix = ['./Datasets2/test_patch/1\\']
#name2_prefix = [path + '//' for path in name2_prefix]
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

# 创建Mapper
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(contour.GetOutputPort())
mapper.ScalarVisibilityOff()

# 创建Actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)

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

# 创建矩形方框
box = vtk.vtkCubeSource()
box.SetXLength(120)
box.SetYLength(120)
box.SetZLength(2.8)
boxMapper = vtk.vtkPolyDataMapper()
boxMapper.SetInputConnection(box.GetOutputPort())
boxActor = vtk.vtkActor()
boxActor.SetMapper(boxMapper)
boxActor.GetProperty().SetColor(1, 1, 0)  # 设置黄色

# 添加鼠标事件处理函数
def OnMouseMove(obj, event):
    # 获取鼠标位置
    x, y = obj.GetEventPosition()

    # 将窗口坐标转换为世界坐标
    picker = vtk.vtkCellPicker()
    picker.SetTolerance(0.001)
    picker.Pick(x, y, 0, renderer)
    point = picker.GetPickPosition()

    # 根据鼠标位置获取对应的图片索引
    image_index = int(point[2] / spacing[2])  # 假设 z 方向上的间距为 2.8
    print("Image index:", image_index)
    # 加载并显示对应的原始数据集图片
    file_name = name1_prefix[0] + "%06d.png" % image_index
    # 数据表格
    file_name1 = name2_prefix[0] + "%06d.jpg" % image_index


    print("Loading image:", file_name)
    reader = vtk.vtkPNGReader()
    reader.SetFileName(file_name)
    reader.Update()

    # 设置图片Actor的输入数据
    imageActor.SetInputData(reader.GetOutput())
    # 更新矩形方框的位置
    boxActor.SetPosition(96, 111, point[2])
    aRender.AddActor(imageActor)
    window.Render()
    Renwin.Render()

    # 读取Excel文件
    df = pd.read_excel('D:\\Project\\python\\2024OCT\\new_test_results.xls')

    # 检索第一列中与file_name1内容相同的某一行
    row = df[df['图片名'] == file_name1]

    # 若行存在
    if not row.empty:
        # 获取该行的第二列内容
        value = row.iloc[0]['预测标签']

        # 根据第二列内容输出信息
        if value == 0.0:
            print("无分支")
        elif value == 1.0:
            print("有分支")
# 将鼠标事件处理函数与交互器绑定
interactor.AddObserver("MouseMoveEvent", OnMouseMove)

# 开始显示
if __name__ == '__main__':
    renderer.AddActor(boxActor)  # 添加矩形方框到渲染器
    window.Render()
    Renwin.Render()
    interactor.Initialize()
    interactor.Start()
