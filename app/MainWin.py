# Form implementation generated from reading ui file 'MainWin.ui'
#
# Created by: PyQt6 UI code generator 6.7.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt6 import QtCore, QtGui
from PyQt6.QtGui import (
    QPalette,
    QColor,
)
from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
    QStatusBar,
    QMenuBar,
    QFileDialog,
    QLabel,
    QPushButton,
    QProgressBar,
    QSpacerItem,
    QApplication,
    QMainWindow,
)

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from modeler import Modeling, Rendering
import interpolator
import os
import shutil
from tqdm import tqdm # 终端进度条
from functools import partial


class Ui_MainWindow(object):

    def __init__(self,):
        self.project_folder = None    # 工程文件夹地址
        self.original_folder = None     # 原始图片序列文件夹
        self.interpolation_folder = None    # 插值图片序列文件夹
        self.phtonum = None # 记录原始图片数

    def set_light_theme(self, MainWindow):
        # 定义一个亮色调色板
        light_palette = QPalette()

        # 基本颜色
        light_palette.setColor(QPalette.ColorRole.Window, QColor(255, 255, 255))  # 窗口背景色
        light_palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))  # 窗口文本颜色
        light_palette.setColor(QPalette.ColorRole.Base, QColor(240, 240, 240))  # 文本框背景色
        light_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(255, 255, 255))  # 备用背景色
        light_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
        light_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 0, 0))

        # 按钮
        light_palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))  # 按钮颜色
        light_palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))  # 按钮文本颜色

        # 文本
        light_palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))  # 常规文本颜色
        light_palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))

        # 链接
        light_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))

        # 选中的文本和背景
        light_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))  # 选中的背景色
        light_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))  # 选中的文本颜色

        # 应用调色板
        MainWindow.setPalette(light_palette)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        # MainWindow.showMaximized()  # 窗口最大化显示，但保留任务栏
        MainWindow.resize(1024, 576)

        # 设置明亮主题
        # self.set_light_theme(MainWindow)

        # 菜单栏
        self.menubar = QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 803, 33))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        # “文件”
        file_menu = self.menubar.addMenu("文件")
        new_action = QtGui.QAction("新建工程", MainWindow)
        open_action = QtGui.QAction("打开工程", MainWindow)
        file_menu.addAction(new_action)
        file_menu.addAction(open_action)

        # 连接动作的槽函数（可选）
        new_action.triggered.connect(self.new_file)
        open_action.triggered.connect(self.open_file)

        # 添加“编辑”菜单
        edit_menu = self.menubar.addMenu("编辑")
        undo_action = QtGui.QAction("撤销", MainWindow)
        redo_action = QtGui.QAction("重做", MainWindow)
        edit_menu.addAction(undo_action)
        edit_menu.addAction(redo_action)

        # 添加“设置”菜单
        settings_menu = self.menubar.addMenu("设置")
        preferences_action = QtGui.QAction("偏好设置", MainWindow)
        settings_menu.addAction(preferences_action)

        # 状态栏，用于在窗口底部显示状态信息的控件
        self.statusbar = QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        self.statusbar.setStyleSheet("QStatusBar::item { border: 0px solid black; }")
        MainWindow.setStatusBar(self.statusbar)  # 将状态栏添加到主窗口
        progress = QProgressBar()
        progress.setMaximum(100)
        progress.setValue(50)  # 设置进度条为50%
        self.statusbar.addPermanentWidget(progress)
        # 状态栏-工程名
        self.status_projectname = QLabel("Project: None")
        self.statusbar.addWidget(self.status_projectname)  # 添加到状态栏的左侧
        # 状态栏-间隔
        self.status_spacer = QLabel("   ")
        self.statusbar.addWidget(self.status_spacer)  # 添加到状态栏的左侧
        # 状态栏-图片数
        self.status_photonum = QLabel("Photo: None")
        self.statusbar.addWidget(self.status_photonum)  # 添加到状态栏的左侧

        self.centralwidget = QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # 使用 QHBoxLayout 将窗口分为左右两部分
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")

        # 左侧的控件
        self.left_widget = QWidget(parent=self.centralwidget)
        self.left_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # 创建垂直布局
        self.left_v_layout = QVBoxLayout(self.left_widget)
        # 创建一个占位符，使用 QSpacerItem
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum,QSizePolicy.Policy.Expanding)
        # 添加占位符到布局中
        self.left_v_layout.addItem(spacer)

        # 创建一个水平布局用于右对齐
        self.left_v_h_layout = QHBoxLayout()
        self.left_v_h_layout.addStretch()  # 添加弹性空间，使其右对齐

        # 创建一个切片文字和图片的布局
        self.slice = QVBoxLayout()

        # 创建一个标签用于显示文字
        self.slice_label = QLabel("切片横截面", self.left_widget)
        font = QtGui.QFont("Microsoft YaHei", 12)  # 设置字体为 Arial，大小为 12
        self.slice_label.setFont(font)  # 假设你的标签叫 text_label
        self.slice_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)  # 文字居中对齐
        # 将文字标签添加到左侧布局
        self.slice.addWidget(self.slice_label)

        # 添加一个标签用于显示切片图片
        self.image_label = QLabel(self.left_widget)
        self.image_label.setFixedSize(256, 256)  # 设置固定大小
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)  # 居中对齐
        self.slice.addWidget(self.image_label)# 添加图片标签到右对齐布局

        # 在左侧 widget 中添加按钮
        self.button_jpg = QPushButton("导入图片序列", self.left_widget)  # 假设左侧的widget叫left_widget
        self.button_jpg.clicked.connect(self.import_jpg)  # 连接到你的函数
        self.left_v_layout.addWidget(self.button_jpg)  # 将按钮添加到布局中

        # 在左侧 widget 中添加按钮
        self.button_interpolation = QPushButton("插值", self.left_widget)  # 假设左侧的widget叫left_widget
        self.button_interpolation.clicked.connect(self.interpolation)  # 连接到你的函数
        self.left_v_layout.addWidget(self.button_interpolation)  # 将按钮添加到布局中

        # 在左侧 widget 中添加按钮
        self.button_modeling = QPushButton("建模", self.left_widget)  # 假设左侧的widget叫left_widget
        self.button_modeling.clicked.connect(self.modeling)  # 连接到你的函数
        self.left_v_layout.addWidget(self.button_modeling)  # 将按钮添加到布局中

        self.left_v_h_layout.addLayout(self.slice)
        # 将底部布局添加到左侧布局中
        self.left_v_layout.addLayout(self.left_v_h_layout)
        # 设置布局
        self.left_widget.setLayout(self.left_v_layout)
        self.horizontalLayout.addWidget(self.left_widget)

        '''
        以下部分是将vtk窗口放入ui的关键代码
        '''
        self.vtkwidget = QVTKRenderWindowInteractor(parent=self.centralwidget)
        self.vtkwidget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.vtkwidget.setStyleSheet("QWidget { border: 2px solid rgb(255, 255, 255); }")  # 黑边框，宽度为2
        self.horizontalLayout.addWidget(self.vtkwidget)     # 右侧的 vtk 渲染窗口
        self.vtkwidget.setAutoFillBackground(False)
        self.vtkwidget.setObjectName("vtkwidget")
        # 设置交互方式
        style = vtkInteractorStyleTrackballCamera()
        self.vtkwidget.GetRenderWindow().GetInteractor().SetInteractorStyle(style)

        # 开始渲染
        self.vtkwidget.Initialize()
        self.vtkwidget.Start()
        '''
        以上部分是将vtk窗口放入ui的关键代码
        '''

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "IVOCT图像处理"))

    # 新建工程
    def new_file(self):
        # 选择文件夹
        self.project_folder = self.select_directory()
        self.original_folder = os.path.join(self.project_folder, os.path.basename(self.project_folder) + "_original")
        self.interpolation_folder = os.path.join(self.project_folder,os.path.basename(self.project_folder) + "_interpolation")
        self.status_projectname.setText(f"Project:{os.path.basename(self.project_folder)}") # 工程名显示在左下角状态栏
        pass

    # 打开工程
    def open_file(self):
        # 选择文件夹
        self.project_folder = self.select_directory()
        self.original_folder = os.path.join(self.project_folder, os.path.basename(self.project_folder) + "_original")
        self.interpolation_folder = os.path.join(self.project_folder,os.path.basename(self.project_folder) + "_interpolation")
        self.status_projectname.setText(f"Project: {os.path.basename(self.project_folder)}") # 工程名显示在左下角状态栏
        if os.path.exists(self.original_folder):  # 检测到已导入图片序列，提示
            self.phtonum = len([f for f in os.listdir(self.original_folder) if f.endswith('.jpg') and os.path.isfile(os.path.join(self.original_folder, f))])
            self.status_photonum.setText(f"Photo: {self.phtonum}")
            print("检测到已导入图片序列")
            if os.path.exists(self.interpolation_folder):  # 检测到已有插值结果，提示
                print("检测到已完成插值")
                if os.path.exists(os.path.join(self.project_folder, os.path.basename(self.project_folder) + ".vtk")):   # 检测到已有模型，直接建模
                    print("检测到已有模型")
                    self.modeling()
                else:
                    print("当前进度：未运行建模")
            else:
                print("当前进度：未运行插值算法")
        else:
            print("当前进度：未导入图片序列")
        pass

    # 按键，导入图片序列
    def import_jpg(self):
        # 确保保存图片的文件夹存在
        if self.project_folder is None :
            print("请先打开或新建一个工程！")
        else:
            self.original_folder = os.path.join(self.project_folder,os.path.basename(self.project_folder) + "_original")
            os.makedirs(self.original_folder, exist_ok=True)    # 没有就创建
            print(self.original_folder)
            source_folder = self.select_directory()
            self.copy_jpg_files(source_folder, self.original_folder)
            self.phtonum = len([f for f in os.listdir(self.original_folder) if f.endswith('.jpg') and os.path.isfile(os.path.join(self.original_folder, f))])
            self.status_photonum.setText(f"Photo: {self.phtonum}")
        pass

    # 按键调用插值算法
    def interpolation(self):
        # 保证已导入图片序列
        if os.path.exists(self.original_folder):
            # 确保输出文件夹存在
            self.interpolation_folder = os.path.join(self.project_folder, os.path.basename(self.project_folder) + "_interpolation")
            os.makedirs(self.interpolation_folder, exist_ok=True)   # 没有就创建
            print(self.interpolation_folder)
            interpolator.process(self.original_folder, self.interpolation_folder)  # 输入文件夹。在源文件夹内新建xxx_interpolation文件夹用于输出。插值数默认3
        else:
            print("未检测到图片序列，插值前请先导入图片序列。")
    # 按键调用建模
    def modeling(self):
        # 进行建模和渲染
        myModeling = Modeling(self.project_folder, 512, 512)
        myModeling.process()
        myRendering = Rendering(myModeling.vtk_polydata, myModeling.model_probe)    # 传入模型
        myRendering.process()
        # 设置鼠标点击触发
        self.vtkwidget.GetRenderWindow().GetInteractor().AddObserver("LeftButtonPressEvent",partial(self.left_button_press_callback, myModeling, myRendering))
        # 将渲染器放入UI窗口
        self.vtkwidget.GetRenderWindow().AddRenderer(myRendering.render)
        # 开始渲染
        self.vtkwidget.Initialize()
        self.vtkwidget.Start()

    # 选择目标文件夹
    def select_directory(self):
        options = QFileDialog.Option.ShowDirsOnly  # 仅显示文件夹
        directory = QFileDialog.getExistingDirectory(None, "选择目标文件夹", "", options)
        if directory:
            print(f"选择的文件夹是: {directory}")
        else:
            print("未选择任何文件夹")
        return directory

    # 复制图片序列
    def copy_jpg_files(self, source_folder:str, destination_folder:str):
        # 遍历源文件夹中的所有文件
        for filename in tqdm(os.listdir(source_folder)):
            if filename.lower().endswith('.jpg'):
                # 构建完整的源和目标路径
                source_path = os.path.join(source_folder, filename)
                destination_path = os.path.join(destination_folder, filename)
                # 复制文件
                shutil.copy2(source_path, destination_path)
                print(f"已复制: {filename}")
        # print("复制完毕")

    def left_button_press_callback(self, Modeling, Rendering, obj, event):
        # 获取鼠标位置
        print("点击！")
        x, y = obj.GetEventPosition()
        if Rendering.picker.Pick(x, y, 0, Rendering.render):
            # 将窗口坐标转换为世界坐标
            point = Rendering.picker.GetPickPosition()
            image_index = int(point[2] / Modeling.spacing[2])  # 根据鼠标位置获取对应的图片索引,假设 z 方向上的间距
            print(image_index)
            # 加载切片图片
            file_name = self.interpolation_folder+"/%06d.jpg" % image_index
            self.update_image(file_name)
            # 更新指针的位置
            Rendering.actor_probe.SetPosition(0, 0, point[2])
            Rendering.render.AddActor(Rendering.actor_probe)
            print(f"切片加载成功")
        # 更新渲染结果
        self.vtkwidget.GetRenderWindow().AddRenderer(Rendering.render)
        self.vtkwidget.Initialize()
        self.vtkwidget.Start()

    # 更新横切面图片
    def update_image(self, image_path):
        pixmap = QtGui.QPixmap(image_path)
        self.image_label.setPixmap(pixmap.scaled(256, 256, QtCore.Qt.AspectRatioMode.KeepAspectRatio,QtCore.Qt.TransformationMode.SmoothTransformation))

# 主程序
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    MainWindow.show()
    sys.exit(app.exec())
