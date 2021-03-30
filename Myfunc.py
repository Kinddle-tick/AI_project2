import pyqtgraph as pg
import pandas as pd
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
from collections import Counter
import time
from sklearn.cluster import Birch
from scipy import stats
from matplotlib import pyplot as plt
'''
在该目录下可以编写和注册自己的函数，根据GUI的特点，函数有以下约束：
1.函数的第一个参数必须是点的np.array类型数据
2.函数可以有三个可以在GUI窗口上直接控制的float类型的数值，如果实际使用时不是float形，需要在函数内部转换（如int)
3.函数的返回值应当是一个元祖，第一个元素为所有点按照输入顺序所给出的聚类结果，第二个元素为聚类中心，若无聚类中心可以为空列表[]
为了方便函数的展示，在AlgorithmList中进行注册，格式为：{"显示函数名": {"func":函数的程序内名, "para":{"显示的第一个参数名":默认值,}},}
其中，para可以为空字典，para内部元素的定义顺序必须和函数要求的相同
'''
prime_list=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103,
            107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
            227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]

def DistanceMatrix(point:[[2,11],[3,5],]):
    G=np.dot(point,point.T)
    H=np.tile(np.diag(G),(len(G),1))
    D=H+H.T-2*G
    return D

def CoopDistanceMatrix(point:[[2,11],[3,5],],point2):
    G=np.dot(point,point2.T)
    H=np.expand_dims(np.array(np.sum(point**2,axis=1)),axis=1)
    K=np.expand_dims(np.array(np.sum(point2**2,axis=1)),axis=0)
    D=H+K-2*G
    return D

def Coopcheckdiv(x,y):
    '''
    :param x: 一行分类数据 narray
    :param y: 一行分类数据,与x等长 narray
    :return: 两份数据的"乐观准确率"
    '''
    divx=set(x)
    xdivnum=len(divx)
    divy=set(y)
    ydivnum=len(divy)
    num=2
    # 防止串号
    while prime_list[num]<len(divx)+len(divy):
        num+=1

    for i in divx:
        if i<0:  # 噪声不属于任何一类
            x[np.where(x == i)] = 0
            xdivnum-=1
            continue
        x[np.where(x == i)] = prime_list[num]
        num+=1
    for i in divy:
        if i<0:
            x[np.where(x == i)] = 0
            ydivnum-=1
            continue
        y[np.where(y == i)] = prime_list[num]
        num += 1

    # T = np.dot(x.reshape(-1,1),y.reshape(1,-1))
    # rightlist = Counter(T.reshape(-1)).most_common(min(len(divx),len(divy)))
    # print(rightlist)
    rightlist = Counter((x*y)[x*y!=0]).most_common(min(xdivnum,ydivnum))
    right = sum(np.array(rightlist)[:,1])
    all =len(x)
    return right/all


    print(x,y)
    return 0

def Prim(D_M):
    D_df = pd.DataFrame(D_M)
    lens = len(D_M)
    V= {0}
    ALL = set([i for i in range(lens)])
    E=[]
    while len(V)<lens:
        tmp= D_df.iloc[list(V),list(ALL-V)]
        x,y = np.where(D_M == np.min(np.array(tmp)))
        line = [x[0],y[0]]
        E.append(line)
        V=V|set(line)
    return E

def KMeans(point,div=2):
    div=int(div)
    m=np.array([point[i] for i in np.random.choice(range(len(point)),size=div)])
    # time=0
    point=np.array(point)
    old_m = m.copy()+1
    while not np.all(old_m==m):
        # time+=1
        DM=CoopDistanceMatrix(point,m)
        raw_div=np.argmin(DM,axis=1)
        old_m=m.copy()
        for i in range(div):
            m[i]=np.average(point[np.where(raw_div==i)[0]],axis=0)
            if np.any(np.isnan(m)):
                m[np.where(np.any(np.isnan(m), axis=1))] = \
                    np.array([point[i] for i in np.random.choice(range(len(point)),
                                                                 size=np.sum(np.any(np.isnan(m), axis=1)))])
                div=len(m)
    # print(time)
    return raw_div,m

def DB_scan(point,Eps=0.05,MinPts=5):
    # t=time.time_ns()
    D = DistanceMatrix(point)
    # print((time.time_ns()-t)/1e6)
    Dlink = D<Eps
    heart = np.where(np.sum(Dlink, axis=1) > MinPts + 1)
    div = np.array([-1]*len(point))
    divnum=0
    tmp=set(list(heart[0]))  # 等待聚类的核心点
    while tmp:
        i = tmp.pop()
        # print(i)
        div[i] = divnum
        xlist=[i]  # 所有可能的可达核心点
        for i in xlist:
            tmp_link_index = np.where((Dlink)[i] * div == -1)[0].tolist()  # 选出尚未聚类的可达点
            xlist.extend(set(tmp_link_index) & tmp)  # 将尚未聚类的可达核心点加入到xlist中
            div[tmp_link_index] = divnum  # 将所有可达点标号
        tmp = tmp - set(xlist)  #从等待聚类的核心点中 去除已经聚类过的所有核心点
        divnum+=1
    return div,[]

def DPCA(point,Eps=1):
    # D = DistanceMatrix(point)
    lens = len(point)
    div = np.array([-1] * lens)

    D = DistanceMatrix(point)
    p = np.sum(D < Eps, axis=1)
    deta = np.empty(lens)

    link = np.empty(lens)
    for i in range(lens):
        x = D[i][np.where(p > p[i])]  # 取出密度比他大的点
        if len(x) == 0:
            deta[i] = np.max(D)
            link[i] = -1
        else:
            deta[i] = np.min(x)
            link[i] = np.where(D[i] == deta[i])[0][0]
    # pdeta=np.array([p,deta])

    pdeta = p * deta
    gama = np.sort(pdeta)
    game_arg = np.argsort(pdeta)
    # 主观猜测 那些点是中心点-- #
    # figs = plt.figure()
    # plt.scatter(p,deta)
    # plt.bar(np.arange(lens), poss, width=1.0, color=[(i[0] / 255, i[1] / 255, i[2] / 255) for i in c])
    # plt.plot(np.arange(lens), [x] * lens, c="r")
    # plt.ylim(0, 3)
    # plt.show()

    may_heart = np.where((gama[1:] - gama[:-1]) / gama[:-1] > 1)
    heart_line_index = game_arg[may_heart[0][np.where(may_heart[0] > lens * 0.8)]][0]
    heart_line = pdeta[heart_line_index]
    heart = np.where(pdeta > heart_line)[0]
    m = point[heart]
    # print(heart)
    # print(m)
    # print(DistanceMatrix(m))
    check = np.array(np.where(DistanceMatrix(m) < Eps)).transpose()
    # print(check)
    fix = check[np.where((check[:, 0] - check[:, 1]) > 0)]
    # print(fix)

    link[heart] = -1
    for i in fix:
        link[heart[i[0]]] = heart[i[1]]
        heart[i[0]] = heart[i[1]]
    divnum = 0
    heart = list(set(heart))
    m = point[heart]
    # print(heart)
    for center in heart:
        div[center] = divnum
        xlist = [center]
        for i in xlist:
            neighbor = np.where(link == i)
            xlist.extend(neighbor[0][np.where(div[neighbor] == -1)[0]])
            div[neighbor] = divnum
        divnum += 1

    return div,m

def OPTICS(point,Eps=0,Minpts=5):
    Minpts=int(Minpts)
    D = DistanceMatrix(point)
    lens = len(point)
    div = np.ones(lens, dtype=np.int) * -1
    if Eps == 0:
        Eps = np.inf

    core = np.where(np.sum(D < Eps, axis=1) > Minpts)[0]
    core_distance = np.sort(D, axis=0)[Minpts + 1]
    rd_yx = np.max(np.dstack([np.tile(core_distance, [lens, 1]), D]), axis=2)
    P = []
    I = set(np.arange(lens))
    r = np.ones(lens) * np.inf
    while I:
        i = I.pop()
        P.append(i)
        if i in core:
            tmp_rd = rd_yx[i].copy()
            tmp_rd[(list(set(P)),)] = np.inf
            seedlist = list(np.where(tmp_rd != np.inf)[0])

            while seedlist:
                # print(len(seedlist))
                j = seedlist[np.argmin(r[(seedlist,)])]
                seedlist.remove(j)
                P.append(j)
                if j in core:
                    tmp_rd_2 = rd_yx[j].copy()
                    r[(seedlist,)] = np.min(np.vstack([r[(seedlist,)], tmp_rd_2[(seedlist,)]]), axis=0)

                    tmp_rd_2[(list(set(P)),)] = np.inf
                    tmp_rd_2[(seedlist,)] = np.inf
                    seedlist.extend(list(np.where(tmp_rd_2 != np.inf)[0]))
            I = I - set(P)


    poss = r[(P,)].copy()
    x = np.average(poss[np.where(poss != np.inf)])
    # y = (poss[1:] - poss[:-1]) / x

    color_set = [tuple([40, 64, 64, 255])] + [tuple(list(i) + [255]) for i in
                                              np.random.randint(64, 256, size=[30, 3])]
    color_set = np.array(color_set[0:10 + 10],
                         dtype=[('red', np.ubyte), ('green', np.ubyte), ('blue', np.ubyte), ('alpha', np.ubyte)])
    n = 0
    x = np.average(poss[np.where(poss != np.inf)])
    c = [tuple([40, 64, 64, 255])] * lens
    poss[2:] = (poss[2:] + poss[1:-1] + poss[:-2] * 0.1) / 2.1
    for i in range(len(poss)):
        if poss[i] != np.inf and poss[i] - poss[i - 1] < -0.52 * x:
            n += 1
        elif poss[i] > 2 * x:
            div[P[i]] = -1
            c[i] = color_set[0]
            continue
        div[P[i]] = n
        c[i]=color_set[n+1]

    figs = plt.figure()
    plt.bar(np.arange(lens), poss, width=1.0, color=[(i[0] / 255, i[1] / 255, i[2] / 255) for i in c])
    plt.plot(np.arange(lens), [x*2] * lens, c="r")
    plt.ylim(0, 3)
    plt.show()
    return div,[]

def Birch_lff(point,cluster_num):
    # lff
    cluster_num = int(cluster_num)
    X = point
    # g_truth = DataMat[:, 0]
    # for 'five_cluser.txt':threshold=1.5,branching_factor=20
    # for 'spiral.txt':不适用
    # for 'ThreeCircles.txt':不适用
    # for 'Twomoons.txt':不适用
    y_pred = Birch(n_clusters=cluster_num, threshold=1.5, branching_factor=20).fit_predict(X)
    return y_pred,[]

#  所有算法的名称与函数地址映射表
# AlgorithmList= {"K-mean": {"func":KMeans, "para":{"prediv":5,}},
#                 "DBscan": {"func":DB_scan, "para":{"Eps":2.0,"MinPts":2,}},
#                 "DPCA": {"func":DPCA, "para":{"Eps":1.5,}},
#                 "Birch": {"func":Birch_lff, "para":{"prediv":5,}},
#                 "OPTICS_beta": {"func": OPTICS, "para": {"Eps": 0,"Minpts":5, }},
#                 }


def moment(train,label):
    rtndic={}
    if train.ndim==1:
        dimen=1
    else:
        dimen=len(train.columns)
    for key in label.unique():
        index = label[label==key].index
        meta= np.array(train.loc[index]).transpose().reshape([dimen,-1])
        rtndic[key] = stats.multivariate_normal(np.mean(meta,axis=1),np.cov(meta)).pdf

    return rtndic


def rect(train, label, width=10):
    rtndic={}
    if train.ndim==1:
        dimen=1
    else:
        dimen=len(train.columns)
    class func():
        def __init__(self,data,h):
            self.data=data
            self.h=h
        def pdf(self,x):
            x = np.array(x).reshape([-1, dimen])
            if type(self.h) == float or type(self.h) == int:
                self.h = np.array([self.h] * dimen)
            assert len(self.h) == dimen
            V = np.prod(self.h)

            tmp_ = (np.tile(self.data, [len(x), 1, 1]) - x.reshape(len(x), -1, dimen)) / self.h
            return np.sum(np.all(np.abs(tmp_) < 0.5, axis=2), axis=1) / len(self.data) / V

    for key in label.unique():
        index = label[label == key].index
        meta = np.array(train.loc[index]).reshape([-1,dimen])
        tmps=func(meta.copy(),width)
        rtndic[key] = tmps.pdf
    return rtndic


def exponent(train, label, width=10):
    rtndic={}
    if train.ndim==1:
        dimen=1
    else:
        dimen=len(train.columns)
    class func():
        def __init__(self,data,h):
            self.data=data
            self.h=h
        def pdf(self,x):
            x = np.array(x).reshape([-1, dimen])
            if type(self.h) == float or type(self.h) == int:
                self.h = np.array([self.h] * dimen)
            assert len(self.h) == dimen

            V = np.prod(self.h)

            tmp_ = (np.tile(self.data, [len(x), 1, 1]) - x.reshape(len(x), -1, dimen)) / self.h
            return np.sum(np.prod(np.exp(-np.abs(tmp_)), axis=2), axis=1) / len(self.data) / V

    for key in label.unique():
        index = label[label == key].index
        meta = np.array(train.loc[index]).reshape([-1,dimen])
        tmps=func(meta.copy(),width)
        rtndic[key] = tmps.pdf
    return rtndic


def triangle(train, label, width=10):
    rtndic={}
    if train.ndim==1:
        dimen=1
    else:
        dimen=len(train.columns)
    class func():
        def __init__(self,data,h):
            self.data=data
            self.h=h
        def pdf(self,x):
            x = np.array(x).reshape([-1, dimen])
            if type(self.h) == float or type(self.h) == int:
                self.h = np.array([self.h] * dimen)
            assert len(self.h) == dimen

            V = np.prod(self.h)

            tmp_ = (np.tile(self.data, [len(x), 1, 1]) - x.reshape(len(x), -1, dimen)) / self.h
            tmp_[np.where(np.abs(tmp_)>1)]=0
            # print(tmp_)
            return np.sum(np.prod(np.abs(tmp_), axis=2), axis=1) / len(self.data) / V

    for key in label.unique():
        index = label[label == key].index
        meta = np.array(train.loc[index]).reshape([-1,dimen])
        tmps=func(meta.copy(),width)
        rtndic[key] = tmps.pdf
    return rtndic


def norm(train, label, width=10):
    # print(width)
    rtndic={}
    if train.ndim==1:
        dimen=1
    else:
        dimen=len(train.columns)

    class func():
        def __init__(self,data,h):
            self.data=data
            self.h=h
        def pdf(self,x):
            x = np.array(x).reshape([-1, dimen])
            if type(self.h) == int or type(self.h) == float:
                self.h = np.array([self.h] * dimen)
            assert len(self.h) == dimen
            V = np.prod(self.h)

            tmp_ = (np.tile(self.data, [len(x), 1, 1]) - x.reshape(len(x), -1, dimen)) / self.h
            return np.sum(np.prod(stats.norm.pdf(tmp_), axis=2), axis=1) / len(self.data) / V

            # 并不知道为什么要prod累乘 毕竟不一定独立 这里有疑点

    for key in label.unique():
        index = label[label == key].index
        meta = np.array(train.loc[index]).reshape([-1,dimen])
        tmps=func(meta.copy(),width)
        rtndic[key] = tmps.pdf
    return rtndic


def K_near(train,label,Kn=3):
    rtndic={}
    if train.ndim==1:
        dimen=1
    else:
        dimen=len(train.columns)

    class func():
        def __init__(self,data,Kn,dimen):
            self.data=data
            self.Kn=Kn
            self.dimen = dimen
        def pdf(self,x):
            dimen = self.dimen
            x = np.array(x).reshape([-1, dimen])
            # if type(self.Kn) == int or type(self.Kn) == float:
            #     self.Kn = np.array([self.Kn] * dimen)
            # assert len(self.Kn) == dimen

            tmp_ = np.abs(np.tile(self.data, [len(x), 1, 1]) - x.reshape(len(x), -1, dimen))
            h = np.sort(tmp_,axis=1)[:,int(self.Kn-1),:]
            V = np.prod(h,axis=1)

            return self.Kn / len(self.data) / V

            # 并不知道为什么要prod累乘 毕竟不一定独立 这里有疑点

    for key in label.unique():
        index = label[label == key].index
        meta = np.array(train.loc[index]).reshape([-1,dimen])
        tmps=func(meta.copy(),Kn,dimen)
        rtndic[key] = tmps.pdf
    return rtndic

AlgorithmList= {"parameter": {"func":moment, "para":{}},
                "rect-win": {"func":rect, "para":{"width":14}},
                "Gaussian-win": {"func":norm, "para":{"width":3}},
                "exponent-win": {"func": exponent, "para": {"width": 3}},
                "triangle-win": {"func": triangle, "para": {"width": 5}},
                "K-near": {"func": K_near, "para": {"K_n": 6}},
                }

if __name__ == '__main__':

    import pyqtgraph as pg
    import pandas as pd
    import numpy as np
    from pyqtgraph.Qt import QtCore, QtGui
    # from Myfunc import DistanceMatrix
    from collections import Counter
    from matplotlib import pyplot as plt
    from scipy.integrate import nquad
    from scipy.misc import derivative
    from scipy.integrate import quad
    def moment(train,label):
        rtndic={}
        for key in label.unique():
            index = label[label==key].index
            meta= np.array(train.loc[index]).transpose()
            rtndic[key] = stats.multivariate_normal(np.mean(meta,axis=1),np.cov(meta)).pdf

        return rtndic

    def rect(train, label, width=10):
        rtndic={}
        def func(x,data,h=width):
            rtn = []
            dimen=data.shape[1]
            x=np.array(x).reshape([-1,dimen])
            if type(h)==int:
                h=np.array([h]*dimen)
            assert len(h)==dimen
            V=np.prod(h)

            tmp_=(np.tile(data,[len(x),1,1])-x.reshape(len(x),-1,dimen))/h
            return np.sum(np.all(np.abs(tmp_) < 0.5, axis=2), axis=1)/len(data)/V

        for key in label.unique():
            index = label[label == key].index
            meta = np.array(train.loc[index])

            rtndic[key] = lambda x:func(x,meta,width)
        return rtndic

    def norm(train, label, width=10):
        rtndic={}
        def func(x,data,h=width):
            rtn = []
            dimen=data.shape[1]
            x=np.array(x).reshape([-1,dimen])
            if type(h)==int:
                h=np.array([h]*dimen)
            assert len(h)==dimen
            V=np.prod(h)

            tmp_=(np.tile(data,[len(x),1,1])-x.reshape(len(x),-1,dimen))/h
            return np.sum(np.prod(stats.norm.pdf(tmp_),axis=2),axis=1)/len(data)/V

            # 并不知道为什么要prod累乘 毕竟不一定独立
        for key in label.unique():
            index = label[label == key].index
            meta = np.array(train.loc[index])

            rtndic[key] = lambda x:func(x,meta,width)
        return rtndic

    # def edge(train,label,func):
    #
    #
    #     pass

    data = pd.read_csv("datas.csv", sep=',', header=0, index_col=0)
    newer = pd.read_csv("std.csv", sep=',', header=0, index_col=0)

    train = data.iloc[:,:-1]
    label = data.iloc[:,-1]
    new = newer

    # train_d = np.array(train).transpose()
    # Themean = np.mean(train_d, axis=1)
    # Thecov = np.cov(train_d)  # 无偏
    # func = stats.multivariate_normal(Themean,Thecov)
    # c = lambda z:derivative(lambda x: func.cdf([np.inf, np.inf, np.inf, x]), z, dx=1e-6)
    rtn= moment(train,label)

    rtn1 = rect(train,label,10)
    rtn2 = norm(train,label,10)
    ddd = rtn1[0](np.mean(np.array(train), axis=0))
    ccc = rtn2[0](np.mean(np.array(train), axis=0))


    # pg.setConfigOptions(antialias=True)
    #
    # w = pg.GraphicsLayoutWidget(show=True)
    # w.setWindowTitle('pyqtgraph example: GraphItem')
    # v = w.addViewBox()
    # v.setAspectLocked()
    #
    # g = pg.GraphItem()
    # v.addItem(g)
    #
    # # file_list = ["five_cluster.txt", "spiral.txt",
    # #              "ThreeCircles.txt", "Twomoons.txt"]
    # file_list = ["datas.txt"]
    # color_set = [tuple([40, 64, 64, 255])] + [tuple(list(i) + [255]) for i in np.random.randint(64, 256, size=[20, 3])]
    #
    # # Eps=0.03
    # # MinPts=5
    #
    # for file in file_list:
    #     train = pd.read_csv(file, sep=' ', header=None)
    #     answer = train.iloc[:, 0]
    #     point=np.array(train.iloc[:, 1:3].copy())
    #
    #
    #     div = np.array([-1]*len(point))
    #     m=[]
    #     Eps=0
    #     Minpts = 5
    #
    #     #  test function
    #     # inf = np.inf
    #     D=DistanceMatrix(point)
    #     lens=len(point)
    #     div=np.ones(lens,dtype=np.int)*-1
    #     if Eps == 0:
    #         Eps = np.inf
    #
    #     core=np.where(np.sum(D<Eps,axis=1)>Minpts)[0]
    #     core_distance = np.sort(D,axis=0)[Minpts+1]
    #     rd_yx = np.max(np.dstack([np.tile(core_distance, [lens, 1]), D]),axis=2)
    #     # fix=np.array([np.inf]*lens)
    #     # fix[core]=1
    #     # rd_yx = rd_yx_raw*fix #  去除矩阵中不是核心点的部分的数据
    #     # rd_yx=rd_yx_raw
    #     rd = np.ones(lens)*np.inf
    #     # P = np.zeros(lens,dtype=np.int)
    #     P=[]
    #     seedlist=[]
    #     I = set(np.arange(lens))
    #     r= np.ones(lens)*np.inf
    #     while I:
    #         i=I.pop()
    #         P.append(i)
    #         if i in core:
    #             tmp_rd=rd_yx[i].copy()
    #             tmp_rd[(list(set(P)),)]=np.inf
    #             seedlist=list(np.where(tmp_rd!=np.inf)[0])
    #             # index1=np.where(np.sort(tmp_rd)!=np.inf)
    #             # insert_seed_arg = np.argsort(tmp_rd)[index1]
    #             # insert_seed = np.sort(tmp_rd)[index1]
    #             # seedlist=list(insert_seed_arg)
    #
    #             while seedlist:
    #                 # print(len(seedlist))
    #                 j=seedlist[np.argmin(r[(seedlist,)])]
    #                 seedlist.remove(j)
    #                 P.append(j)
    #                 if j in core:
    #                     tmp_rd_2=rd_yx[j].copy()
    #                     r[(seedlist,)]=np.min(np.vstack([r[(seedlist,)], tmp_rd_2[(seedlist,)]]),axis=0)
    #
    #                     tmp_rd_2[(list(set(P)),)]=np.inf
    #                     tmp_rd_2[(seedlist,)] = np.inf
    #                     seedlist.extend(list(np.where(tmp_rd_2!=np.inf)[0]))
    #                     # index2 = np.where(np.sort(tmp_rd_2) != np.inf)
    #                     # insert_seed_2 = np.argsort(tmp_rd_2)[np.where(np.sort(tmp_rd_2)!=np.inf)]
    #                     # # print(insert_seed_2)
    #                     # seedlist.extend(insert_seed_2)
    #                     # seedlist=list(insert_seed_2)
    #                     # seedlist.extend(insert_seed_2)
    #             I=I-set(P)
    #     # div+=2
    #     # div[(P[:1000],)]=2
    #
    #     figs = plt.figure()
    #     poss=r[(P,)].copy()
    #     x = np.average(poss[np.where(poss!=np.inf)])
    #     y = (poss[1:]-poss[:-1])/x
    #
    #     color_set = [tuple([40, 64, 64, 255])] + [tuple(list(i) + [255]) for i in
    #                                               np.random.randint(64, 256, size=[30, 3])]
    #     color_set=np.array(color_set[0:10 + 10],
    #                              dtype=[('red', np.ubyte), ('green', np.ubyte), ('blue', np.ubyte), ('alpha', np.ubyte)])
    #     n=0
    #     x = np.average(poss[np.where(poss!=np.inf)])
    #     c = [tuple([40, 64, 64, 255])] * lens
    #     poss[2:]=(poss[2:]+poss[1:-1]+poss[:-2]*0.1)/2.1
    #     for i in range(len(poss)):
    #         # div[P[i]]=n
    #         # if poss[i] > 2*x:
    #         #     div[P[i]]=-1
    #
    #         if poss[i]!=np.inf and poss[i]-poss[i-1]<-0.52*x:
    #             n+=1
    #         elif poss[i] > 2*x:
    #             div[P[i]]=-1
    #             c[i] = color_set[0]
    #             continue
    #         div[P[i]]=n
    #         c[i]=color_set[n+1]
    #
    #     # plt.plot(np.arange(lens-1),y)
    #     # plt.ylim(-4, 1)
    #     plt.bar(np.arange(lens),poss,width=1.0,color=[(i[0]/255,i[1]/255,i[2]/255) for i in c])
    #     plt.plot(np.arange(lens),[x]*lens,c="r")
    #     plt.ylim(0, 3)
    #     #
    #     plt.show()
    #
    #
    #     # testfunc end
    #     DRAW = True
    #     if DRAW:
    #         pointsize=0.1
    #         divnum=len(set(div))
    #         pos = np.array(point)
    #         color_set = [tuple([40, 64, 64, 255])] + [tuple(list(i) + [255]) for i in
    #                                                   np.random.randint(64, 256, size=[30, 3])]
    #         color_set = np.array(color_set[0:divnum + 10],
    #                              dtype=[('red', np.ubyte), ('green', np.ubyte), ('blue', np.ubyte),
    #                                     ('alpha', np.ubyte)])
    #
    #         color = np.array([color_set[i + 1] for i in div],
    #                          dtype=[('red', np.ubyte), ('green', np.ubyte), ('blue', np.ubyte), ('alpha', np.ubyte)])
    #         symbol = np.array(["o" if i >= 0 else "t" for i in div])
    #
    #         pos_m = np.array(m).reshape(-1, 2)
    #         color_m = np.array([color_set[i + 1] for i in range(len(m))],
    #                            dtype=[('red', np.ubyte), ('green', np.ubyte), ('blue', np.ubyte), ('alpha', np.ubyte)])
    #         symbol_m = ['+'] * len(m)
    #         symbols = np.hstack([symbol, symbol_m])
    #         # symbols[np.where()]
    #         sizes = [pointsize] * len(div) + [pointsize * 5] * len(m)
    #
    #         g.setData(pos=np.vstack([pos, pos_m]), size=sizes, symbol=symbols,
    #                        symbolBrush=np.hstack([color, color_m]),
    #                        pxMode=False)
    #         # pointsize=0.1
    #         # divnum=len(set(div))
    #         # pos = np.array(point)
    #         # # color_set=[tuple(list(i)+[255]) for i in np.random.randint(64,256,size=[divnum,3])]
    #         # color_set = np.array(color_set[0:divnum + 10],
    #         #                      dtype=[('red', np.ubyte), ('green', np.ubyte), ('blue', np.ubyte), ('alpha', np.ubyte)])
    #         #
    #         # color = np.array([color_set[i + 1] for i in div],
    #         #                  dtype=[('red', np.ubyte), ('green', np.ubyte), ('blue', np.ubyte), ('alpha', np.ubyte)])
    #         # symbol = np.array(["o" if i >= 0 else "t" for i in div])
    #         #
    #         # pos_m = np.array(m).reshape(-1, 2)
    #         # color_m = np.array([color_set[i + 1] for i in range(len(m))],
    #         #                    dtype=[('red', np.ubyte), ('green', np.ubyte), ('blue', np.ubyte), ('alpha', np.ubyte)])
    #         # symbol_m = ['+'] * len(m)
    #         # symbols = np.hstack([symbol, symbol_m])
    #         # # symbols[np.where()]
    #         # sizes = [pointsize] * len(div) + [pointsize * 5] * len(m)
    #         #
    #         # g.setData(pos=np.vstack([pos, pos_m]), size=sizes, symbol=symbols, symbolBrush=np.hstack([color, color_m]),
    #         #                pxMode=False)
    #
    #     # g.setData(pos=pos, adj=None, size=0.01, pxMode=False)
    #
    #
    #         import sys
    #         if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #             QtGui.QApplication.instance().exec_()



