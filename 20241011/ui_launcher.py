# 逻辑文件
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow
from MainWin import *


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    my_win = MyMainWindow()
    my_win.show()
    sys.exit(app.exec())
  