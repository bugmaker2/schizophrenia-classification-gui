import time
from PyQt5.Qt import QThread
# from load_combined_dataset import load_combined_dataset

class Thread_test(QThread):
    def __init__(self):
        super().__init__()

    def run(self):
        while True:
            print(2)
            time.sleep(1)