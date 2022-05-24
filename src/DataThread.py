from asyncio.windows_events import NULL
import pandas as pd
import numpy as np
from sklearn.utils import Bunch
from PyQt5.Qt import QThread
from tqdm import tqdm
from PyQt5.Qt import pyqtSignal


class DataThread(QThread):
    #进度信号
    progress = pyqtSignal(int)

    def __init__(self,fileName) -> None:
        super().__init__()
        self.fileName = fileName
        self.Combined = None

    def run(self):
        """
        load Combined_94
        """
        def _read_and_form_combined_dataframe(data):
            loop = True
            chunksize = 5
            chunks = []
            total_num = int(192/chunksize)
            num = 0
            with tqdm(total=total_num) as pbar:
                while loop:
                    try:
                        chunk = data.get_chunk(chunksize)
                        chunks.append(chunk)
                        # print("Progress {:.1f} %".format(100*num/total_num))
                        pbar.update(1)
                        num += 1
                        self.progress.emit(int(num))
                    except StopIteration:
                        loop = False
                        print("Iteration is stopped")

            df = pd.concat(chunks,ignore_index = True)
            return df
        
        def _get_Combineddata(data):
            data_r = data.iloc[:,0:140422]
            data_np = np.array(data_r)
            return data_np

        def _get_Combinedtarget(data):
            data_b = data.iloc[:,140422:140423]
            data_np = np.array(data_b)
            return data_np

        def _get_Combineddescr(data):
            text = "本数据集为Combined数据, 样本数量: {};" \
                "特征数量: {}; 目标值数量: {}; \n" \
                "其中EPSZV 21个样本，标记为0；HC 95个样本，标记为1；SZ 76个样本，标记为2\n" \
                "将五个数据集合并得到了这个数据集, 且每个数据集中每个数据只取下三角" \
                "".format(data.index.size, data.columns.size - 1, 1)
            return text
        
        def _get_feature_names(data_df):
            fnames = data_df.columns.to_list()
            return fnames

        def _get_target_names():
            tnames = ["EPSZV","NC","SZ"]
            return tnames

        data_csv = pd.read_csv(self.fileName,iterator=True)
        data_df = _read_and_form_combined_dataframe(data_csv)
        self.Combined = Bunch()
        self.Combined.data = _get_Combineddata(data_df)
        self.Combined.target = _get_Combinedtarget(data_df)
        self.Combined.DESCR = _get_Combineddescr(data_df)
        self.Combined.feature_names = _get_feature_names(data_df)
        self.Combined.target_names = _get_target_names()

    def getData(self):
        return self.Combined