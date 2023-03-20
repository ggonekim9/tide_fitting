import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mycons import kconstants, atand, sind, cosd


def tideQ(data, Sigma, V0, u, f):
    m, n = data.shape
    Y_obs = data.reshape((m * n, 1))
    ## 迭代运算
    phase0 = np.zeros((m * n, 13))
    for t in range(m * n):
        for i in range(13):
            phase0[t, i] = Sigma[:, i] * t + V0[:, i] + u[t, i];  # 相角
    e1 = cosd(phase0);
    e2 = sind(phase0);
    C = f * e1;
    S = f * e2;
    Q = np.ones((1, m * n))
    for i in range(13):
        Q = np.concatenate((Q, C[:, i].reshape(1, m * n)), axis=0)
        Q = np.concatenate((Q, S[:, i].reshape(1, m * n)), axis=0)
        # 矩阵乘法np.dot 矩阵求逆np.linalg.inv
    X = np.dot(np.linalg.inv(np.dot(Q, Q.T)), np.dot(Q, Y_obs))
    length = X.shape[0]
    h0 = X[0];  # 平均水位
    A = X[1::2];
    B = X[2::2];
    # --------------------------------------------------------------------------
    H = np.sqrt(np.float64(A * A + B * B))  # 振幅
    g = np.zeros((1, 13))  # 迟角
    for i in range(13):
        if A[i, 0] > 0:
            g[:, i] = atand(np.float64(B[i, 0] / A[i, 0]))
        elif A[i, 0] == 0:
            if B[i, 0] > 0:
                g[:, i] = 90
            elif B == 0:
                g[:, i] = 0
            else:
                g[:, i] = -90
        else:
            g[:, i] = atand(np.float64(B[i, 0] / A[i, 0])) - 180

    g = g + 8 * Sigma;  # 转换东八区
    g = (g + 360) % 360;  # 角度转至0-360

    ## 分潮展开式
    y_13 = np.zeros((13, m * n))
    for t in range(m * n):
        for i in range(13):
            y_13[i, t] = f[t, i] * H[i, 0] * cosd(Sigma[0, i] * (t + 8) + V0[0, i] + u[t, i] - g[0, i]);

    # 四个主要半日潮M2、S2、N2、K2
    y_M2 = y_13[0, :];
    y_S2 = y_13[1, :];
    y_N2 = y_13[2, :];
    y_K2 = y_13[3, :];
    # 四个主要全日潮K1、O1、P1、Q1
    y_K1 = y_13[4, :];
    y_O1 = y_13[5, :];
    y_P1 = y_13[6, :];
    y_Q1 = y_13[7, :];
    # 浅水分潮M4、MS4、M6
    y_M4 = y_13[8, :];
    y_MS4 = y_13[9, :];
    y_M6 = y_13[10, :];
    # 长周期分潮Sa、Ssa
    y_Sa = y_13[11, :];
    y_Ssa = y_13[12, :];
    Y_13 = h0[0] + y_M2 + y_S2 + y_N2 + y_K2 + y_K1 + y_O1 + y_P1 + y_Q1 + y_M4 + y_MS4 + y_M6 + y_Sa + y_Ssa
    Y_other = Y_obs - Y_13.reshape(n * m, 1)
    y_13 = y_13.T
    return Y_obs, Y_13, Y_other, y_13