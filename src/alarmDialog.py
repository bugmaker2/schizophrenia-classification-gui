from PyQt5.QtWidgets import QDialog
from PyQt5 import uic

class alarmDialog(QDialog):
    def __init__(self, path):
        super(alarmDialog, self).__init__()
        uic.loadUi(path, self)