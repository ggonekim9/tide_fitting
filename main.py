import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mytide import tideQ
from mycons import kconstants,atand,sind,cosd

if __name__ == "__main__":
    pass
    data = pd.read_csv(r'test.csv',encoding='gbk').values[:,-12:]
    Num = data.shape[0]*data.shape[1]
    year = 1996
    cons = [Num,year]

    Sigma,V0,u,f = kconstants(cons)

    Y_obs,Y_13,Y_other,y_13 = tideQ(data,Sigma,V0,u,f)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10,5), dpi=100)
    plt.plot(Y_13,'r',label='预测')
    plt.plot(Y_obs,'b',label='真实')
    plt.plot(Y_other,'y',label='误差')
    plt.xlim((0, Num))
    plt.ylim((-800, 9000))
    plt.xticks(np.arange(0, Num, 1000),fontsize=10)
    plt.yticks(np.arange(-800, 9000, 1000),fontsize=10)
    plt.ylabel('h/mm',fontsize=12)
    plt.xlabel('t/hour',fontsize=12)
    plt.legend()

    plt.title('TIDE',fontsize=12)
    plt.show()

tit = ['M2','S2','N2','K2','K1','O1','P1','Q1','M4','MS4','M6','Sa','Ssa']
plt.figure(figsize=(13,10), dpi=80)
for i in range(1,9):
    plt.subplot(4,2,i)
    plt.plot(y_13[0:30,i-1])
    plt.ylabel(tit[i-1],fontsize=13,rotation=0)
plt.show()
plt.figure(figsize=(13,2), dpi=80)
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.plot(y_13[0:30,i+7])
    plt.ylabel(tit[i+7],fontsize=13,rotation=0)
plt.show()
plt.figure(figsize=(13,2), dpi=80)
plt.plot(y_13[:,11])
plt.ylabel(tit[11],fontsize=13,rotation=0)
plt.show()
plt.figure(figsize=(13,2), dpi=80)
plt.plot(y_13[:,12])
plt.ylabel(tit[12],fontsize=13,rotation=0)
plt.show()

