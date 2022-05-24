from PyQt5.QtWidgets import QWidget
from PyQt5 import uic
class setModelParameters(QWidget):
    def __init__(self, path):
        super(setModelParameters, self).__init__()
        uic.loadUi(path, self)
