# -*- coding = utf-8 -*-
# Calculate the distance (rotation angle) for creating similarity matrix.
# See "Kagan Y Y. 1991. 3-D rotation of double-couple earthquake sources. Geophys J Int, 106(3): 709~716".
# 2022.08.10,edit by Lin

import math


def srd2tpb(fms):
    """
    Convert strike, rake, dip to P, T, B
    :param fms: focal mechanism solutions (n*3), sorted by strike, rake, dip. e.g.:fms=[strike, rake, dip]
    :return: P, T, B
    """
    fms = [math.radians(fms[0]), math.radians(fms[1]), math.radians(fms[2])]
    T1 = (- math.sin(fms[0]) * math.sin(fms[2]) + math.cos(fms[0]) * math.cos(fms[1]) + math.sin(fms[0]) * math.cos(fms[2])
          * math.sin(fms[1])) / math.sqrt(2)
    T2 = (math.cos(fms[0]) * math.sin(fms[2]) + math.sin(fms[0]) * math.cos(fms[1]) - math.cos(fms[0]) * math.cos(fms[2])
          * math.sin(fms[1])) / math.sqrt(2)
    T3 = (- math.cos(fms[2]) - math.sin(fms[2]) * math.sin(fms[1])) / math.sqrt(2)
    P1 = (- math.sin(fms[0]) * math.sin(fms[2]) - math.cos(fms[0]) * math.cos(fms[1]) - math.sin(fms[0]) * math.cos(fms[2])
          * math.sin(fms[1])) / math.sqrt(2)
    P2 = (math.cos(fms[0]) * math.sin(fms[2]) - math.sin(fms[0]) * math.cos(fms[1]) + math.cos(fms[0]) * math.cos(fms[2])
          * math.sin(fms[1])) / math.sqrt(2)
    P3 = (- math.cos(fms[2]) + math.sin(fms[2]) * math.sin(fms[1])) / math.sqrt(2)
    B1 = math.cos(fms[0]) * math.sin(fms[1]) - math.sin(fms[0]) * math.cos(fms[2]) * math.cos(fms[1])
    B2 = math.sin(fms[0]) * math.sin(fms[1]) + math.cos(fms[0]) * math.cos(fms[2]) * math.cos(fms[1])
    B3 = math.sin(fms[2]) * math.cos(fms[1])
    T = [T1, T2, T3]
    P = [P1, P2, P3]
    B = [B1, B2, B3]
    return T, P, B


def quatfps(T, P):
    """
    The rotation quaternion.
    :param T: T-axis
    :param P: P-axis
    :return: quaternion
    """
    err = 1.0e-15
    ic = 1
    perp = T[0] * P[0] + T[1] * P[1] + T[2] * P[2]
    if perp > 0.02:
        raise ValueError('T- and P-axes are not orthogonal!')
    v = [T[0] + P[0], T[1] + P[1], T[2] + P[2]]
    s = [T[0] - P[0], T[1] - P[1], T[2] - P[2]]
    anormv = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    v = [v[0] / anormv, v[1] / anormv, v[2] / anormv]
    anorms = math.sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2])
    s = [s[0] / anorms, s[1] / anorms, s[2] / anorms]
    an = [s[1] * v[2] - v[1] * s[2], v[0] * s[2] - s[0] * v[2], s[0] * v[1] - v[0] * s[1]]
    t = [(v[0] + s[0]) * 1.0 / math.sqrt(2.0), (v[1] + s[1]) * 1.0 / math.sqrt(2.0), (v[2] + s[2]) * 1.0 / math.sqrt(2.0)]
    p = [(v[0] - s[0]) * 1.0 / math.sqrt(2.0), (v[1] - s[1]) * 1.0 / math.sqrt(2.0), (v[2] - s[2]) * 1.0 / math.sqrt(2.0)]
    u = [(t[0] + p[1] + an[2] + 1.0) / 4.0, (t[0] - p[1] - an[2] + 1.0) / 4.0, (-t[0] + p[1] - an[2] + 1.0) / 4.0,
         (-t[0] - p[1] + an[2] + 1.0) / 4.0]
    um = max(u[0], max(u[1], max(u[2], u[3])))
    if um == u[0]:
        icod = 1 * ic
        u0 = math.sqrt(u[0])
        u = [u0, (p[2] - an[1]) / (4.0 * u0), (an[0] - t[2]) / (4.0 * u0), (t[1] - p[0]) / (4.0 * u0)]
    elif um == u[1]:
        icod = 2 * ic
        u1 = math.sqrt(u[1])
        u = [(p[2] - an[1]) / (4.0 * u1), u1, (t[1] + p[0]) / (4.0 * u1), (an[0] + t[2]) / (4.0 * u1)]
    elif um == u[2]:
        icod = 3 * ic
        u2 = math.sqrt(u[2])
        u = [(an[0] - t[2]) / (4.0 * u2), (t[1] + p[0]) / (4.0 * u2), u2, (p[2] + an[1]) / (4.0 * u2)]
    else:
        icod = 4 * ic
        u3 = math.sqrt(u[3])
        u = [(t[1] - p[0]) / (4.0 * u3), (an[0] + t[2]) / (4.0 * u3), (p[2] + an[1]) / (4.0 * u3), u3]

    if math.fabs(u[0] * u[0] + u[1] * u[1] + u[2] * u[2] + u[3] * u[3] - 1.0) > err:
        raise ValueError('Computation error!')
    quat = [u[1], u[2], u[3], u[0]]
    return quat


def quatp(Q1, Q2):
    """
    Calculates product of two quaternions.
    """
    Q3 = [Q1[3] * Q2[0] + Q1[2] * Q2[1] - Q1[1] * Q2[2] + Q1[0] * Q2[3],
          -Q1[2] * Q2[0] + Q1[3] * Q2[1] + Q1[0] * Q2[2] + Q1[1] * Q2[3],
          Q1[1] * Q2[0] - Q1[0] * Q2[1] + Q1[3] * Q2[2] + Q1[2] * Q2[3],
          -Q1[0] * Q2[0] - Q1[1] * Q2[1] - Q1[2] * Q2[2] + Q1[3] * Q2[3]]
    return Q3


def boxtest(Q1, icode):
    """
    For icode=0, finds minimal rotation quaternion;
    For icode=N finds rotation quaternion Q2 = Q1*(i,j,k,1), for N=1,2,3,4
    """
    Q2 = []
    quatt = []
    quat = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
    if icode == 0:
        icode = 1
        qm = math.fabs(Q1[0])
        for i in [1, 2, 3]:
            if qm < math.fabs(Q1[i]):
                qm = math.fabs(Q1[i])
                icode = i + 1
    for i in range(4):
        Q2.append(Q1[i])
    if icode != 4:
        Q2 = []
        for i in range(4):
            quatt.append(quat[i][icode - 1])
        Q2 = quatp(quatt, Q1)
    if Q2[3] <= 0.0:
        for i in range(4):
            Q2[i] = -1.0 * Q2[i]
    qm = Q2[3]
    return Q2, qm


def quatd(Q1, Q2):
    """
    Quaternion division Q3 = Q2*(Q1)**(-1), or Q2 = Q3*Q1
    """
    qc1 = []
    for i in range(3):
        qc1.append(-1.0 * Q1[i])
    qc1.append(Q1[3])
    Q3 = quatp(qc1, Q2)
    return Q3


def f4r1(Q1, Q2, icode):
    """
    Q = Q2*(Q1*(I,J,K,1))**(-1)
    """
    qr1, qm = boxtest(Q1, icode)
    Q = quatd(qr1, Q2)
    return Q


def sphcoor(quat):
    """
    for the rotation quaternion QUAT the subroutine finds the rotation angle (angl) of a
    counterclockwise rotation and spherical coordinates (colatitude theta, and azimuth phi)
    of the rotation pole (intersection of the axis with reference sphere); theta=0 corresponds
    to the vector pointing down.
    """
    if quat[3] < 0:
        for i in range(4):
            quat[i] = -1.0 * quat[i]
    if math.fabs(quat[3] - 1) <= 1e-15:
        quat[3] = 1.0
    q4n = math.sqrt(1.0 - math.pow(quat[3], 2))
    costh = 1.0
    if math.fabs(q4n) > 1.0e-10:
        costh = quat[2]/q4n
    if math.fabs(costh) > 1.0:
        if costh > 0:
            costh = math.floor(costh)
        else:
            costh = math.ceil(costh)
    theta = math.acos(costh) * 180.0 / math.pi
    angl = 2.0 * math.acos(quat[3]) * 180.0 / math.pi
    phi = 0.0
    if math.fabs(quat[0]) > 1.0e-10 or math.fabs(quat[1]) > 1.0e-10:
        phi = math.atan2(quat[1], quat[0]) * 180.0 / math.pi
    if phi < 0:
        phi += 360.0
    return angl, theta, phi


def fps4r(T1, P1, T2, P2):
    """
    Calculate the angle, theta, phi using T and P
    """
    quat1 = quatfps(T1, P1)
    quat2 = quatfps(T2, P2)

    q1a = f4r1(quat1, quat2, 1)
    angl1, theta1, phi1 = sphcoor(q1a)
    angl = [angl1]
    theta = [theta1]
    phi = [phi1]
    q2a = f4r1(quat1, quat2, 2)
    angl2, theta2, phi2 = sphcoor(q2a)
    angl.append(angl2)
    theta.append(theta2)
    phi.append(phi2)
    q3a = f4r1(quat1, quat2, 3)
    angl3, theta3, phi3 = sphcoor(q3a)
    angl.append(angl3)
    theta.append(theta3)
    phi.append(phi3)
    q4a = f4r1(quat1, quat2, 4)
    angl4, theta4, phi4 = sphcoor(q4a)
    angl.append(angl4)
    theta.append(theta4)
    phi.append(phi4)

    return angl, theta, phi


def calkagan(fms1, fms2):
    """
    Main function for rotation angle
    """
    t1, p1, b1 = srd2tpb(fms1)
    t2, p2, b2 = srd2tpb(fms2)
    angl, theta, phi = fps4r(t1, p1, t2, p2)
    kagan = min(angl)

    return kagan


def ckdist(x1, x2):
    """
    Main function for two fms‘ distance.
    """
    dist = []
    for i in range(len(x2)):
        t1, p1, b1 = srd2tpb(x1)
        t2, p2, b2 = srd2tpb(x2[i])
        angl, theta, phi = fps4r(t1, p1, t2, p2)
        dist.append(min(angl))
    return dist


