import pyqtgraph as pg
import pandas as pd
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
from Myfunc import DistanceMatrix,Prim,KMeans,AlgorithmList,Coopcheckdiv
from PyQt5 import QtWidgets
import time
import copy
# from validation import validation

file = "datas.csv"
divnum = 5
pg.setConfigOptions(antialias=True)
class Runthread(QtCore.QThread):
    #  通过类成员对象定义信号对象
    _signal = QtCore.pyqtSignal(str)
    stop_flag = False
    def __init__(self,func):
        super(Runthread, self).__init__()
        self.func = func
        # self.func=func
        # self.qtlock = QtCore.QMutex()

    def __del__(self):
        self.wait()

    def run(self):
        self.stop_flag=False
        num=0
        # self.qtlock.tryLock()
        # self.qtlock.unlock()
        while not self.stop_flag:
            # self.qtlock.lock()
            self.func()
            # self.qtlock.unlock()
            num+=1
            # time.sleep(0.01)

    # def changefunc(self,func):
    #     self.func = func

    def stop(self):
        self.stop_flag=True
        self.exit(0)

class MyWidget(QtGui.QWidget):
    function = None
    default_para=[]
    point_size=0.1
    file=file
    maxLine=10
    infotext=[]
    auto_try_flag=False
    if_best_flag=False
    div=[]
    sin = QtCore.pyqtSignal()
    rate=0
    nowDimen=None
    color_set =[tuple(list(i) + [255/3]) for i in np.random.randint(64, 256, size=[30, 3])]
    pdf=None
    # 第一个颜色是噪声的颜色，在聚类中标为-1
    def __init__(self):
        super(MyWidget, self).__init__()
        #  用来表示点阵的区域
        w = pg.GraphicsLayoutWidget(show=True)


        #   标签
        title = QtWidgets.QLabel(self)
        title.setText("概率密度估计")
        title.setAlignment(QtCore.Qt.AlignCenter)
        lb1 = QtWidgets.QLabel(self)
        lb1.setText("文件名:")
        lb2 = QtWidgets.QLabel(self)
        lb2.setText("估计方法:")
        lb3 = QtWidgets.QLabel(self)
        lb3.setText("输入参数:")
        lb4 = QtWidgets.QLabel(self)
        lb4.setText("文本反馈:")
        lb7 = QtWidgets.QLabel(self)
        lb7.setText("展示维度：")
        #  参数控制
        self.para = [QtWidgets.QLabel(self) for i in range(3)]
        self.para_value=[QtWidgets.QLineEdit() for i in range(3)]
        for i in range(3):
            self.para[i].setText(" -- :")
            self.para_value[i].setText("")
        #  函数选择
        self.func_text = QtWidgets.QComboBox()
        self.func_text.addItems(["--"] + list(AlgorithmList.keys()))
        self.func_text.activated.connect(self.changefunc)
        #  编写文件名
        self.file_name=QtWidgets.QLineEdit()
        self.file_name.setText(self.file)
        self.file_name.editingFinished.connect(self.changefile)
        self.load_data(self.file)
        self.__SetPlotWidget(w)

        # 维度选择：
        self.chooseDimention = QtWidgets.QComboBox()
        self.chooseDimention.addItems(self.Dementions[:-1])
        self.chooseDimention.activated.connect(self.changedimen)

        #  准确率显示
        # self.Accuracy = QtWidgets.QLabel(self)
        # self.Accuracy.setText("--")
        #  用时显示
        # self.Time = QtWidgets.QLabel(self)
        # self.Time.setText("--")

        # 按钮组
        self.btn1 = QtWidgets.QPushButton('Run', self)
        self.btn1.clicked.connect(self.button_Retry)

        self.btn2 = QtWidgets.QPushButton('Check', self)
        self.btn2.clicked.connect(self.button_Check)
        self.info = QtWidgets.QTextEdit()


        ## 创建一个栅格Grid布局作为窗口的布局
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)
        ## 细节的控制每个区域所属的格子
        layout.addWidget(w, 0, 0, 0, 1)  # plot goes on right side, spanning all rows
        layout.addWidget(title, 0, 1, 1, 2)
        layout.addWidget(lb1, 1, 1)
        layout.addWidget(self.file_name, 1, 2)
        layout.addWidget(lb2, 2, 1)
        layout.addWidget(self.func_text, 2, 2)
        layout.addWidget(lb7, 3, 1)
        layout.addWidget(self.chooseDimention, 3, 2)
        layout.addWidget(lb3, 4, 1)
        for i in range(3):
            layout.addWidget(self.para[i], 5+i, 1)
            layout.addWidget(self.para_value[i], 5+i, 2)

        layout.addWidget(self.btn1, 8, 1)   # button goes in upper-left
        layout.addWidget(self.btn2, 8, 2)   # button goes in upper-left
        layout.addWidget(lb4, 9, 1)
        layout.addWidget(self.info, 10, 1, 1, 2)  # list widget goes in bottom-left

        # layout.addWidget(lb5, 11, 1)
        # layout.addWidget(self.Accuracy, 11, 2)
        # layout.addWidget(lb6, 12, 1)
        # layout.addWidget(self.Time, 12, 2)
        # layout.addWidget(self.ifbest, 13, 1)
        # layout.addWidget(self.auto_retry, 13, 2)

    def load_data(self,filename):
        train = pd.read_csv(filename, sep=',', header=0,index_col=0)
        newer = pd.read_csv("std.csv",sep=',', header=0,index_col=0)
        self.train_data = train
        self.check_data = newer
        self.Dementions = train.columns

    def Onetry(self):
        if self.ChangeForzenPdf():
            self.plot_update(0,0,0)
        return True
    #
    def plot_update(self, point, div, m):

        for key in self.Dementions[:-1]:
            data = self.train_data[key]
            label = self.train_data.iloc[:, -1]
            tmphist=[]
            seem = self.function(self.train_data.loc[:,key], self.train_data.iloc[:, -1],*self.para_now)
            for j in self.train_data.iloc[:,-1].unique():
                minnum = np.min(data.loc[label == j])-10
                maxnum = np.max(data.loc[label == j])+10
                x=np.linspace(minnum,maxnum,300)
                qq = np.histogram(data.loc[label == j],bins=maxnum-minnum,range=[minnum,maxnum])
                # print(qq)
                self.hists[(key,j)].setOpts(x=(qq[1][1:]+qq[1][:-1])/2,height=qq[0]/len(data),width=qq[1][1]-qq[1][0],
                                            brush=self.color_set[j],pen=None)
                # x = np.linspace(qq[1][0]-10,qq[1][-1]+10,100)

                self.estimation[(key,j)].setData(x=x, y=seem[j](x),pen=pg.mkPen(width=5, color=[i for i in self.color_set[j][:-1]]+[255*0.9]))



                # tmphist.extend([[i,qq[1][i]+qq[1][i+1],self.color_set[j]] for i in qq[0]])
            # tmphist_draw = np.array(tmphist).T
            # self.hists[key].setOpts(x=tmphist_draw[0],height=tmphist_draw[1],color=tmphist_draw[2],width=2)


            # print(j,key)

        # pointsize = self.point_size
        # divnum = len(set(div))
        # if divnum>30:
        #     self.show_info(f"类型数达到{divnum},请调参后尝试")
        #     return 8
        # pos = np.array(point)
        # # color_set=[tuple(list(i)+[255]) for i in np.random.randint(64,256,size=[divnum,3])]
        # color_set = np.array(self.color_set[0:divnum+1],
        #                      dtype=[('red', np.ubyte), ('green', np.ubyte), ('blue', np.ubyte), ('alpha', np.ubyte)])
        #
        # color = np.array([color_set[i+1] for i in div],
        #                  dtype=[('red', np.ubyte), ('green', np.ubyte), ('blue', np.ubyte), ('alpha', np.ubyte)])
        # symbol = np.array(["o" if i >=0 else "t" for i in div])
        #
        #
        # pos_m = np.array(m).reshape(-1, 2)
        # color_m = np.array([color_set[i+1] for i in range(len(m))],
        #                    dtype=[('red', np.ubyte), ('green', np.ubyte), ('blue', np.ubyte), ('alpha', np.ubyte)])
        # symbol_m = ['+']*len(m)
        # symbols = np.hstack([symbol, symbol_m])
        # # symbols[np.where()]
        # sizes = [pointsize] * len(div) + [pointsize * 5] * len(m)

        # self.g.setData(pos=np.vstack([pos, pos_m]), size=sizes, symbol=symbols, symbolBrush=np.hstack([color, color_m]),
        #                pxMode=False)

    def button_Retry(self):
        self.Onetry()

    def button_Check(self):
        str=""
        if not self.ChangeForzenPdf():
            return False
        str += "分析：\n对样本数据分析可以获得先验概率有:\n"
        label=self.train_data.iloc[:,-1]
        priori={}
        for key in label.unique():
            priori[key]=np.sum(self.train_data.iloc[:,-1]==key)/len(label)
            str+="\t'{}':{:.3}\n".format(key,np.sum(self.train_data.iloc[:,-1]==key)/len(label))

        str+="按照{}方法对新的数据进行分析，可知其似然函数值与后验概率分别为:".format(self.func_text.currentText())

        pd = {}
        for key in label.unique():
            str += "\nkey:{:<20}{:^20}{:^20}\n".format(key,"似然","先验*似然")

            pd[key]=self.pdf[key](self.check_data)
            num=0
            for j in self.check_data.index:
                str+="{:<20}{:^20g}{:^20g}\n".format(j,pd[key][num],pd[key][num]*priori[key])
                num+=1
            # pass
        print(pd)
        self.infotext.clear()
        self.show_info(str)

        # pass
        # if len(self.div)==0:
        #     self.show_info("暂未分类")
        #     return 0
        # # self.rate = Coopcheckdiv(np.array(self.div),np.array(self.answer))
        # self.Accuracy.setText(f"{self.rate}")
        # print(f"function name:{self.func_text.currentText()}")
        # dic = validation(self.answer,self.div,True)
        # self.show_info(str(dic))
        # print(self.answer)

    def __SetPlotWidget(self, w):
        w.setWindowTitle('A Window')
        self.hists={}
        self.estimation={}
        num=0
        for i in self.Dementions[:-1]:
            # print(i)
            tmp=w.addPlot()
            tmp.setTitle(i)
            # x=pg.AxisItem(orientation='top')
            # tmp.setAxisItems(x)
            for j in self.train_data.iloc[:,-1].unique():
                self.hists[(i,j)]=pg.BarGraphItem(x=np.arange(100), height=0, width=0.8, brush='y')
                tmp.addItem(self.hists[(i,j)])
                self.estimation[(i,j)]=tmp.plot(np.abs(np.zeros(100)))

            num+=1
            if num%2==0:
                w.nextRow()

    def show_info(self, text):
        self.infotext.append(text)
        if len(self.infotext)>self.maxLine:
            self.infotext.pop(0)
        self.info.setPlainText("\n".join(self.infotext))

    def ChangeForzenPdf(self):
        if self.function == None:
            self.show_info("请选择方法！")
            return False
        else:
            try:
                self.para_now = [float(i) for i in self.default_para]
                for i in range(self.para_num):
                    text = self.para_value[i].text()
                    if text != '':
                        self.para_now[i] = float(text)
            except Exception as z:
                self.show_info("请输入正确的参数值")
                print(z)
                return False
        self.pdf = self.function(self.train_data.iloc[:,:-1], self.train_data.iloc[:,-1], *self.para_now)
        self.show_info("似然函数修改完成")
        return True

    def changefunc(self):
        if self.func_text.itemText(0)== "--":
            self.func_text.removeItem(0)
        currect = self.func_text.currentText()
        temp = AlgorithmList[currect]
        self.function=temp["func"]
        self.para_num=0

        for i in temp["para"]:
            self.para[self.para_num].setText(i)
            self.para_value[self.para_num].setText(str(temp["para"][i]))
            self.para_num+=1
        for i in range(self.para_num,3):
            self.para[i].setText(" -- :")
            self.para_value[i].setText("")

        self.default_para=list(temp["para"].values())
        self.show_info("函数修改完成")
        # self.ChangeForzenPdf()

    def changedimen(self):
        text=self.chooseDimention.text()
        self.nowDimen=text

    def changefile(self):
        text = self.file_name.text()
        try:
            self.load_data(text)
            self.file = text
            self.show_info("文件已加载")
        except Exception as z:
            self.show_info("找不到该文件")

if __name__ == '__main__':
    filename=["spiral_unbalance.txt","three_cluster.txt","two_cluster.txt"]
    app = QtGui.QApplication([])
    w = MyWidget()
    w.show()
    ## Start the Qt event loop
    app.exec_()


