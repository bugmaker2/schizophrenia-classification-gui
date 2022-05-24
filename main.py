from contextlib import nullcontext
import sys
import os
from threading import Thread

from PyQt5.QtWidgets import QMainWindow, QApplication, QDialog, QAction, QWidget
from PyQt5 import QtWidgets
from PyQt5 import uic
from sklearn import svm

from xgboost import XGBClassifier
from src.alarmDialog import alarmDialog
from src.setParameters import setModelParameters
from src.DataThread import DataThread

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier



class mainWindow(QMainWindow):
    def __init__(self):
        super(mainWindow, self).__init__()
        # 载入主页面的ui页面
        self.ui = uic.loadUi('./ui/main.ui', self)
        self.progressBar.setRange(0,int(192/5))
        self.steps = []

    def showAuthor(self):
        # 载入展示作者的页面
        author = alarmDialog("./ui/author.ui")
        author.show()
        author.exec_()

    def showLibraries(self):
        # 载入展示库的页面
        library = alarmDialog("./ui/libraries.ui")
        library.show()
        library.exec_()

    """
        有关导入数据的函数
    """
    def loadData(self):
        # 加载数据
        # 利用多线程技术让加载过程在后台运行
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(), 
        "CSV Files(*.csv);;Text Files(*.txt);;Mat Files(*.mat)")

        self.dataClass = DataThread(fileName)
        self.dataClass.start()
        self.dataClass.progress.connect(self.progressDisplay)
        
    def progressDisplay(self,val):
        # 只是个进度条显示模块，不重要
        self.progressBar.setValue(val)

    def preprocessData(self):
        # 预处理数据
        try:
            self.dataBunch = self.dataClass.getData()
            data = self.dataBunch['data'].copy()
            target = self.dataBunch['target'].ravel().copy()
            target[target==2]=0
            print("数据大小为：",data.shape)
            print(self.dataBunch['DESCR'])

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, target, test_size=0.25, shuffle=True)
            print("X_train",self.X_train.shape,"y_train",self.y_train.shape)
            print("X_test",self.X_test.shape,"y_test",self.y_test.shape)
            self.testNum = self.y_test.shape[0]
        except:
            pass
        
        
        # 清空
        if len(self.steps)!=0:
            for step in self.steps:
                if step[0] == "RS":
                    self.steps.remove(step)
                if step[0] == "SC":
                    self.steps.remove(step)
                if step[0] == "DR":
                    self.steps.remove(step)
        # 重新赋值
        if self.checkDimensionReduction.isChecked():
            dimRed = self.chooseDRMethod.currentText()
            if dimRed == "PCA":
                self.steps.insert(0, ('DR', PCA(n_components=2)))
            if dimRed == "NMF":
                self.steps.insert(0, ('DR', NMF(n_componnets=2, init='random', random_state=0)))
        if self.checkStandardScaler.isChecked():
            self.steps.insert(0, ('SC', StandardScaler()))
        if self.checkBox.isChecked():
            pass

        print(self.steps)

    """
        模型选择
    """
    def setParameters(self):
        if len(self.steps)!=0:
            for step in self.steps:
                if step[0] == "MODEL":
                    self.steps.remove(step)
        else:
            print("empty")
        # 重新赋值
        modelName = self.chooseModel.currentText()
        if modelName == "Decision Tree":
            model = DecisionTreeClassifier(
                max_depth=4,
                min_samples_split = 0.7,
                min_samples_leaf = 0.3
                )
        elif modelName == "Random Forest":
            model = RandomForestClassifier(
                max_depth = 4,
                min_samples_split = 0.2,
                max_leaf_nodes = 5,
                random_state = 2
                )
        elif modelName == "SVM":
            model = svm.SVC()
        elif modelName == "kNN":
            model = KNeighborsClassifier()
        elif modelName == "XGBoost":
            model = XGBClassifier(
                objective ='reg:squarederror', 
                colsample_bytree = 0.3, 
                learning_rate = 0.1,
                max_depth = 5, 
                alpha = 10, 
                n_estimators = 10
                )
        elif modelName == "DNN":
            model = MLPClassifier(
                hidden_layer_sizes=(200,50,3),
                activation='logistic',
                max_iter=100
                )
        self.steps.append(('MODEL', model))
        print(self.steps)


    def trainData(self):

        self.pipeline = Pipeline(self.steps)
        print("Pipeline: ", self.pipeline)
        self.pipeline.fit(self.X_train,self.y_train)
        print("train")

    def predictData(self):
        result = self.pipeline.predict(self.X_test)
        prob = sum(result == self.y_test)/self.testNum
        print(result)
        final = round(round(prob,4)*100,2)
        with open("output.txt","w") as fout:
            fout.write(str(result))

        self.ui.trainState.setText("预测结果已输出")
        self.ui.predictResult.setText(str(final)+"%")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = mainWindow()
    win.show()
    sys.exit(app.exec())