# -*- coding = utf-8 -*-
# Check and process data for class building and drawing
# See "Diao G L, Yu L M, Li Q Z. 1992. Hierarchical clustering analysis of the focal mechanism solution——Taking the
# Haicheng earthquake sequences for example. Earthquake research in China (in Chinese), 8(3): 86-92."
# 2023.03.14,edit by Lin

import os
from math import pi, radians, degrees, fabs, hypot, sin, cos, tan, sqrt, atan, atan2, acos
import pandas as pd
import numpy as np


def check_fms(data_path):
    """
    Check the input data, loaded by "scv" file.
    :param data_path: Input data path
    :return: Return the original data, removed the head index; If the input data contains the label column, the labels
                values could also be returned.
    """
    name = os.path.basename(data_path).split('.')[0]
    data = pd.read_csv(data_path)
    data_col = data.columns[0]
    if data_col.isdigit() or "".join(data_col.split(".")).isdigit() and data_col.count('.') == 1:
        data = pd.read_csv(data_path, header=None)
        data = data.values
    else:
        data = data.values

    if data.shape[1] == 3:
        labels = None
        result_sc = None
    elif data.shape[1] == 4:
        labels = data[:, -1]
        result_sc = data
        data = data[:, [0, 1, 2]]
    else:
        log = 'The focal mechanism data only need "strike", "rake", "dip" and "labels" ("labels" may not be imported)' \
              ', please validate your data.'
        raise ValueError(log)

    for i in range(data.shape[0]):
        if data[i][0] < 0 or data[i][0] >= 360:
            log = 'Wrong data for strike(line {}), please validate your data.'.format(i + 1)
            raise ValueError(log)
        if data[i][1] < -180 or data[i][1] > 180:
            log = 'Wrong data for rake(line {}), please validate your data.'.format(i + 1)
            raise ValueError(log)
        if data[i][2] < 0 or data[i][2] > 90:
            log = 'Wrong data for dip(line {}), please validate your data.'.format(i + 1)
            raise ValueError(log)

    log = 'Focal mechanism solutions number is {}.'.format(data.shape[0])
    print(log)

    return data, labels, result_sc, name


def sdr2pt(strike, rake, dip):
    """
    Convert the strike, rake and dip values to the P-axis and T-axis values.
    """
    strike = radians(strike)
    dip = radians(dip)
    rake = radians(rake)

    a = np.array([cos(rake) * cos(strike) + sin(rake) * cos(dip) * sin(strike),
                  cos(rake) * sin(strike) - sin(rake) * cos(dip) * cos(strike),
                  -sin(rake) * sin(dip)])
    n = np.array([-sin(strike) * sin(dip), cos(strike) * sin(dip), -cos(dip)])

    vt = sqrt(2) * (a + n)
    vp = sqrt(2) * (a - n)

    p = v2pt(vp)
    t = v2pt(vt)

    return p, t


def v2pt(v):
    """
    Convert the vectors to P-axis and T-axis values.
    """
    pt = np.full((2,), np.nan)
    for i in range(3):
        if fabs(v[i]) <= 1e-4:
            v[i] = 0.0
        if fabs(fabs(v[i]) - 1.0) < 1e-4:
            v[i] = v[i] / fabs(v[i])
    if fabs(v[2]) == 1.0:
        if v[2] < 0.0:
            pt[0] = 180.0
        else:
            pt[0] = 0.0
        pt[1] = 90
        return pt
    if fabs(v[0]) < 1e-4:
        if v[1] > 0:
            pt[0] = 90.0
        elif v[1] < 0:
            pt[0] = 270.0
        else:
            pt[0] = 0.0
    else:
        pt[0] = degrees(atan2(v[1], v[0]))
    hy_v = hypot(v[0], v[1])
    pt[1] = degrees(atan2(v[2], hy_v))
    if pt[1] < 0.0:
        pt[1] = -1.0 * pt[1]
        pt[0] = pt[0] - 180.0
    if pt[0] < 0.0:
        pt[0] = pt[0] + 360.0
    return pt


def pt_mean(result_sc):
    """
    Calculate the average of each cluster of the focal mechanism solutions.
    :param result_sc: Contains labels data, from spectral clustering.
    :return: Average values and P-axis and T-axis values.
    """
    idx = np.around(np.unique(result_sc[:, 3])).astype(int)
    n_idx = idx.shape[0]
    f_labels = []
    for i in idx:
        f_labels.append(result_sc[result_sc[:, 3] == i])

    result_pt = []
    result_average = np.full((n_idx, 10), np.nan)
    for i in range(n_idx):
        n_fms = f_labels[i].shape[0]
        p, t = sdr2pt(f_labels[i][0][0], f_labels[i][0][1], f_labels[i][0][2])
        result_pt.append([p[0], p[1], t[0], t[1], i + 1])
        p_v, t_v = pt2v(p, t)
        if len(f_labels[i]) == 1:
            a = sqrt(2) * (t_v + p_v)
            n = sqrt(2) * (t_v - p_v)
            strike1, rake1, dip1 = an2srd(a, n)
            strike2, rake2, dip2 = an2srd(n, a)
            p, t = sdr2pt(strike1, rake1, dip1)
            result_average[i][0] = strike1
            result_average[i][1] = rake1
            result_average[i][2] = dip1
            result_average[i][3] = strike2
            result_average[i][4] = rake2
            result_average[i][5] = dip2
            result_average[i][6:8] = p
            result_average[i][8:10] = t
            continue

        for j in range(1, n_fms):
            p2, t2 = sdr2pt(f_labels[i][j][0], f_labels[i][j][1], f_labels[i][j][2])
            result_pt.append([p2[0], p2[1], t2[0], t2[1], i + 1])
            p_v2, t_v2 = pt2v(p2, t2)
            pp1 = p_v + p_v2
            pp2 = p_v - p_v2
            tt1 = t_v + t_v2
            tt2 = t_v - t_v2

            if np.linalg.norm(pp1) <= np.linalg.norm(pp2):
                p_v = pp2
            else:
                p_v = pp1
            if np.linalg.norm(tt1) <= np.linalg.norm(tt2):
                t_v = tt2
            else:
                t_v = tt1

        p_v = p_v / np.linalg.norm(p_v)
        t_v = t_v / np.linalg.norm(t_v)

        a = sqrt(2) * (t_v + p_v)
        n = sqrt(2) * (t_v - p_v)

        strike1, rake1, dip1 = an2srd(a, n)
        strike2, rake2, dip2 = an2srd(n, a)
        p, t = sdr2pt(strike1, rake1, dip1)
        result_average[i][0] = strike1
        result_average[i][1] = rake1
        result_average[i][2] = dip1
        result_average[i][3] = strike2
        result_average[i][4] = rake2
        result_average[i][5] = dip2
        result_average[i][6:8] = p
        result_average[i][8:10] = t

    return result_average, result_pt


def pt2v(p, t):
    """
    Convert the P-axis and T-axis values to vectors.
    """
    p_tr = radians(p[0])
    p_pl = radians(p[1])
    t_tr = radians(t[0])
    t_pl = radians(t[1])

    p_v = np.array([cos(p_tr) * cos(p_pl), sin(p_tr) * cos(p_pl), sin(p_pl)])
    t_v = np.array([cos(t_tr) * cos(t_pl), sin(t_tr) * cos(t_pl), sin(t_pl)])

    return p_v, t_v


def an2srd(a, n):
    """
    Convert the slip vector and nodal plane normal vector to strike, rake and dip values.
    """
    if n[2] == -1.0:
        strike = atan2(a[1], a[0])
        dip = 0.0
    else:
        strike = atan2(-n[0], n[1])
        if n[2] == 0.0:
            dip = 0.5 * pi
        elif fabs(sin(strike)) >= 0.1:
            dip = atan2(-n[0] / sin(strike), -n[2])
        else:
            dip = atan2(n[1] / cos(strike), -n[2])
    a1 = a[0] * cos(strike) + a[1] * sin(strike)
    if fabs(a1) < 1e-4:
        a1 = 0.0
    if a[2] != 0.0:
        if dip != 0.0:
            rake = atan2(-a[2] / sin(dip), a1)
        else:
            rake = atan2(-1e6 * a[2], a1)
    else:
        if a1 > 1:
            a1 = 1.0
        if a1 < -1:
            a1 = -1.0
        rake = acos(a1)
    if dip < 0.0:
        dip = dip + pi
        rake = pi - rake
        if rake > pi:
            rake -= 2 * pi
    if dip > 0.5 * pi:
        dip = pi - dip
        strike += pi
        rake = -rake
        if strike >= 2 * pi:
            strike -= 2 * pi
    if strike < 0.0:
        strike += 2 * pi
    strike = degrees(strike)
    dip = degrees(dip)
    rake = degrees(rake)

    return strike, rake, dip


def plot_fm(srd_pt):
    """
    Calculates the coordinates used to plot the focal mechanism.
    :param srd_pt: The input data contained strike, rake, dip and P-axis, T-axis values.
    :return: The coordinates of base map, nodal plane, P and T.
    """
    srd = np.radians(srd_pt[0:6])
    pt = srd_pt[6:10]

    x1, y1 = coordinate_plane(srd[0], srd[2])
    x2, y2 = coordinate_plane(srd[3], srd[5])
    xp, yp = coordinate_pt(pt[0], pt[1])
    xt, yt = coordinate_pt(pt[2], pt[3])
    base_x = np.cos(np.arange(0, 2 * pi + 2 * pi / 200, 2 * pi / 200))
    base_y = np.sin(np.arange(0, 2 * pi + 2 * pi / 200, 2 * pi / 200))

    return base_x, base_y, x1, y1, x2, y2, xp, yp, xt, yt


def plot_pt(p, t):
    """
    Calculates the P and T coordinates.
    """
    xp, yp = coordinate_pt(p[0], p[1])
    xt, yt = coordinate_pt(t[0], t[1])

    return xp, yp, xt, yt


def coordinate_plane(strike, dip):
    """
    Calculates the nodal planes coordinates.
    """
    xp = (tan(pi / 4 - dip / 2) * sin(strike + pi / 2) + tan(pi / 4 + dip / 2) * sin(strike - pi / 2)) / 2
    yp = (tan(pi / 4 - dip / 2) * cos(strike + pi / 2) + tan(pi / 4 + dip / 2) * cos(strike - pi / 2)) / 2
    rp = 1 / cos(dip)
    po = sqrt(xp ** 2 + yp ** 2)
    if po <= 1e-3:
        tp = pi
    else:
        tp = 2 * atan(1 / po)
    t1 = strike + pi / 2 - tp / 2
    t = [i for i in np.arange(t1, t1 + tp + tp / 200, tp / 200)]
    x = xp + rp * np.sin(t)
    y = yp + rp * np.cos(t)

    return x, y


def coordinate_pt(tr, pl):
    """
    Used fro "plot_pt" function.
    """
    rp = tan(radians((90 - pl) / 2))
    xp = rp * sin(radians(tr))
    yp = rp * cos(radians(tr))
    return xp, yp
