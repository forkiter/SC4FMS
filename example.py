# -*- coding = utf-8 -*-
# EXAMPLE
# 具体操作请详见"/docs/UserGuide.pdf"，或直接查看程序代码注释
# 2023.03.30,edit by Lin

from core.sc4fms import ScFms

''' 创建实例 '''
sc = ScFms('data/example_data.csv')
print(sc.data)  # 查看实例是否创建成功，查看data属性


''' 谱聚类 '''
sc.spectral_cluster_com(k=8)  # 谱聚类计算
print(sc.labels)  # 查看谱聚类结果
print(sc.result_sc)


''' 计算平均解 '''
sc.pt_average()  # 平均解计算
print(sc.result_average)  # 查看平均解结果


''' Gap-statistics（计算量大，请选择合适时间运行） '''
sc.gap_eva(n_inspect=15, b_num=100)  # Gap-statistics计算
print(sc.gap_values)  # 查看Gap-statistics计算结果
print(sc.se)
print(sc.op_k)


''' 绘制3d分布图 '''
sc.plot_3d(sc.user_labels('sc'), show_legend=1)  # 绘制谱聚类结果3d图
sc.plot_3d(sc.user_labels('user', n=4))  # 绘制用户所需的聚类结果图


''' 绘制gap曲线图 '''
sc.plot_gap()


''' 绘制平均解图 '''
sc.plot_average(n=6, show_legend=1)
