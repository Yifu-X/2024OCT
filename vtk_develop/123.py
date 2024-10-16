
import sys
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QVBoxLayout
import vtk
from vtk import vtkRenderWindow, vtkRenderer, vtkRenderWindowInteractor
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from MainWin import Ui_MainWindow  # 替换为你的 UI 文件名

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 创建 VTK 渲染窗口
        self.vtkWidget = QVTKRenderWindowInteractor(self.centralwidget)
        self.horizontalLayout.addWidget(self.vtkWidget)

        self.renderer = vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)

        # 进行 VTK 初始化
        self.vtkWidget.Initialize()
        self.vtkWidget.Start()

        # 添加一个简单的立方体
        self.add_cube()

    def add_cube(self):
        cube = vtk.vtkCubeSource()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        self.renderer.AddActor(actor)
        self.renderer.SetBackground(0.1, 0.1, 0.1)  # 设置背景颜色
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

