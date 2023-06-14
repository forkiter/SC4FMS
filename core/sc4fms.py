# -*- coding = utf-8 -*-
# Main class for the program running.
# 2023.03.17,edit by Lin

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from core.data_process import check_fms, pt_mean, plot_fm, plot_pt
from core.spectral_cluster import spectral_cluster
from core.gap_evaluation import find_optimal


class ScFms:
    """
    The main class for program running.

    Read more in "User Guide".

    Parameters
    ----------

    path: Input data path. The data, which must be saved in "scv" file, should contain the "strike", "rake" and "dip"
            values. The "labels" data can also be contained.

    Attributes
    ----------

    data: array of shape (n_samples, 3)
            The original data, removed the head index.

    labels: array of shape (n_samples, )
            The result for spectral clustering, labels of each focal mechanism solution.

    file_name: str
                The name of input data file.

    gap_values: array of shape (n_inspect, )
                The criterion values for each clustering, calculated by gap statistics.

    se: array of shape (n_inspect, )
        The standard errors for each clustering, calculated by gap statistics.

    op_k: int
            The optimal number of clusters suggested, calculated by gap statistics.

    op_idx: array of shape (n_samples, )
            The optimal clustering solution (labels, when k=op_k), calculated by gap statistics.

    result_sc: array of shape (n_samples, 4)
                The result for spectral clustering. The first three columns contain the original data, the last column
                contains the labels.

    result_average: array of shape (k, 10)
                    The result of average solutions for each cluster of focal mechanisms. The columns are
                    ['strike1', 'rake1', 'dip1', 'strike2', 'rake2', 'dip2', 'ptr', 'ppl', 'ttr', 'tpl']

    result_pt: array of shape (n_samples, 5)
                The original data, converted to P-axis (P trend, P plunge) and T (T trend, T plunge) values. The last
                column contains the labels.

    result_op_sc: array of shape (n_samples, 4)
                    The original data, add the optimal clustering solution (labels, when k=op_k), calculated by
                    gap statistics.

    result_gap: array of shape (n_inspect, 2)
                    It contains two rows: One is gap values, the other is se.

    result_labels: array of shape (n_samples, n_inspect)
                    It contains all the labels, calculated by gap statistics.

    """

    def __init__(self, path):
        data, labels, result_sc, file_name = check_fms(path)

        self.data = data
        self.file_name = file_name
        self.labels = labels
        self.gap_values = None
        self.se = None
        self.op_k = None
        self.op_idx = None
        self.result_sc = result_sc
        self.result_average = None
        self.result_pt = None
        self.result_op_sc = None
        self.result_gap = None
        self.result_labels = None

    def spectral_cluster_com(self, k, numN=150, kscale=40, saved=True):
        """
        Method for spectral cluster.

        :param k: int type. Cluster number, which you want to cluster.
        :param numN: int type, default=150. Number of neighbors to use when constructing the affinity matrix using
                        the nearest neighbors method.
        :param kscale: int type, default=40. Scale factor for the kernel.
        :param saved: bool type, default=True. Whether to save the results.
        """
        n_row = self.data.shape[0]
        data_list = self.data.tolist()

        labels = spectral_cluster(self.data, k, numN=numN, kscale=kscale, s_log=True)
        labels += 1
        labels_data = [data_list[i] + [labels[i]] for i in range(n_row)]
        labels_data = pd.DataFrame(columns=['strike', 'rake', 'dip', 'label'], data=labels_data)
        self.result_sc = labels_data.values

        if saved:
            labels_data.to_csv(os.path.join('data', 'result_sc_' + self.file_name + '.csv'), index=False)

        self.labels = labels
        print('Spectral cluster complete.')

    def pt_average(self, saved=True):
        """ Method for average solutions for each cluster of focal mechanisms. """
        if self.result_sc is None:
            result_sc_path = os.path.join('data', 'result_sc_' + self.file_name + '.csv')
            if os.path.exists(result_sc_path):
                result_sc = pd.read_csv(result_sc_path).values
                if result_sc.shape[1] != 4:
                    raise ValueError('Please validate your data matched with pre-data!')
            else:
                raise ValueError('Please check the result_sc parameters, or the "result_sc_'
                                 + self.file_name + '.csv" in the "data" folder!')
        else:
            result_sc = self.result_sc

        result_average, result_pt = pt_mean(result_sc)
        self.result_average = result_average
        self.result_pt = result_pt

        if saved:
            columns = ['strike1', 'rake1', 'dip1', 'strike2', 'rake2', 'dip2', 'ptr', 'ppl', 'ttr', 'tpl']
            result_average = pd.DataFrame(columns=columns, data=result_average)
            result_average.to_csv(os.path.join('data', 'result_average_' + self.file_name + '.csv'), index=False)

            columns = ['ptr', 'ppl', 'ttr', 'tpl', 'label']
            result_pt_save = pd.DataFrame(columns=columns, data=result_pt)
            result_pt_save.to_csv(os.path.join('data', f'result_pt_' + self.file_name + '.csv'), index=False)

    def gap_eva(self, n_inspect=20, b_num=100, numN=150, kscale=40, saved=True):
        """
        Method for gap statistics.

        :param n_inspect: int type, default=20. Using k that in range(n_inspect) for gap statistics.
        :param b_num: int type, default=100. The number of reference data sets (bootstrap) used for computing
                        gap values.
        """
        start_time = time.time()
        optimal_k, optimal_idx, criterion_values, idx, se = find_optimal(self.data, n_inspect=n_inspect, b_num=b_num,
                                                                         numN=numN, kscale=kscale)
        optimal_idx += 1
        idx += 1
        result_gap = np.vstack((criterion_values, se))
        n_row = self.data.shape[0]
        data_list = self.data.tolist()
        result_op_sc = [data_list[i] + [optimal_idx[i]] for i in range(n_row)]
        result_op_sc = pd.DataFrame(columns=['strike', 'rake', 'dip', 'label'], data=result_op_sc)
        self.result_op_sc = result_op_sc.values

        if saved:
            columns = ['label_' + str(i + 1) for i in range(n_inspect)]
            result_gap = pd.DataFrame(columns=columns, data=result_gap)
            result_labels = pd.DataFrame(columns=columns, data=idx.T)
            result_gap.to_csv(os.path.join('data', 'result_gap_' + self.file_name + '.csv'), index=False)
            result_labels.to_csv(os.path.join('data', 'result_labels_' + self.file_name + '.csv'), index=False)
            result_op_sc.to_csv(os.path.join('data', 'result_op_sc_' + self.file_name + '.csv'), index=False)

        com_time = time.time() - start_time
        print('Total time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(com_time))))

        self.gap_values = criterion_values
        self.se = se
        self.result_gap = result_gap
        self.result_labels = idx.T
        self.op_k = optimal_k
        self.op_idx = optimal_idx

    def user_labels(self, label_type='sc', n=None):
        """
        Method for user label selecting, which used for spectral clustering plotting.

        :param label_type: {'sc', 'op', 'user'}, default='sc'. 'sc' is used for spectral clustering result plotting,
                            'op' is used for optimal result plotting, 'user' is used for spectral clustering result
                            plotting (when k=n).
        :param n: int type. The result selected by user.
        """
        if label_type == 'sc':
            if self.labels is None:
                data_path = os.path.join('data', 'result_sc_' + self.file_name + '.csv')
                if os.path.exists(data_path):
                    data = pd.read_csv(data_path)
                    if data.shape[0] != self.data.shape[0]:
                        raise ValueError('Please validate your data matched with "result_sc_' + self.file_name
                                         + '.csv"!')
                    labels = data.values[:, 3]
                    labels = np.array(labels, dtype=np.int32)
                else:
                    raise ValueError(
                        'Please check the labels attribute, or the "result_sc_' + self.file_name
                        + '.csv" in the "data" folder!')
            else:
                labels = self.labels
        elif label_type == 'op':
            if self.op_idx is None:
                data_path = os.path.join('data', 'result_op_sc_' + self.file_name + '.csv')
                if os.path.exists(data_path):
                    data = pd.read_csv(data_path)
                    if data.shape[0] != self.data.shape[0]:
                        raise ValueError('Please validate your data matched with "result_op_sc_' + self.file_name
                                         + '.csv"!')
                    labels = data.values[:, 3]
                    labels = np.array(labels, dtype=np.int32)
                else:
                    raise ValueError(
                        'Please check the op_idx attribute, or the "result_op_sc_' + self.file_name
                        + '.csv" in the "data" folder!')
            else:
                labels = self.op_idx
        elif label_type == 'user':
            if n is None:
                raise TypeError('Please input the number of clusters (computed by GS) you want.')
            if self.result_labels is None:
                data_path = os.path.join('data', 'result_labels_' + self.file_name + '.csv')
                if os.path.exists(data_path):
                    data = pd.read_csv(data_path)
                    labels = data.values[:, n - 1]
                    labels = np.array(labels, dtype=np.int32)
                else:
                    raise ValueError(
                        'Please check the result_labels attribute, or the "result_labels_' + self.file_name
                        + '.csv" in the "data" folder!')
            else:
                labels = self.result_labels[:, n - 1]
        else:
            raise UnboundLocalError('The "label_type" you selected is wrong.')

        return labels

    def plot_3d(self, labels, saved=True, gui=False, show_legend=None, save_name='result_sc.png'):
        """
        Method for plotting a three-dimensional distribution map of focal mechanism solutions.

        :param labels: The clusters, selected from "user_labels" method.
        :param gui: bool type, default=False. Determine whether for GUI or not.
        :param show_legend: If None, the legend will not show in the image.
        """
        fig = plt.figure(figsize=(5.6, 3.8), dpi=100)
        ax = plt.axes(projection='3d')
        idx = np.array([labels[i] for i in range(labels.shape[0])])
        sca = ax.scatter3D(self.data[:, 2], self.data[:, 0], self.data[:, 1], c=idx)
        ax.set_xlabel('Dip')
        ax.set_ylabel('Strike')
        ax.set_zlabel('Rake')
        ax.set_title('Distribution of spectral cluster results for FMS (3D)')
        if show_legend is not None:
            plt.legend(*sca.legend_elements(), loc="upper left", title='Cluster', bbox_to_anchor=(-0.25, 0.85),
                       fontsize=8)

        if saved:
            if save_name == 'result_sc.png':
                save_name = 'result_sc_' + self.file_name + '.png'
            plt.savefig(os.path.join('img', save_name), dpi=300)
        if not gui:
            plt.show()

        return fig

    def plot_gap(self, saved=True, gui=False, save_name='result_gap.png'):
        """ Method for plotting the gap graph with error bar for se. """
        if self.result_gap is None:
            result_gap_path = os.path.join('data', 'result_gap_' + self.file_name + '.csv')
            if os.path.exists(result_gap_path):
                result_gap = pd.read_csv(result_gap_path).values
                if result_gap.shape[0] != 2:
                    raise ValueError('Please validate your data matched with pre-data!')
                gap_values = result_gap[0]
                se_values = result_gap[1]
            else:
                raise ValueError('Please check the gap_values and se_values parameters, or the "result_gap_'
                                 + self.file_name + '.csv" in the "data" folder!')
        else:
            gap_values = self.gap_values
            se_values = self.se

        n = gap_values.shape[0]
        x = np.array([int(i + 1) for i in range(n)])
        op_k = self.op_k
        if op_k is None:
            result_op_path = os.path.join('data', 'result_op_sc_' + self.file_name + '.csv')
            if os.path.exists(result_op_path):
                result_op = pd.read_csv(result_op_path).values
                if result_op.shape[1] != 4:
                    raise ValueError('Please validate your data matched with pre-data!')
                labels_values = np.unique(result_op[:, 3])
                op_k = labels_values.shape[0]
            else:
                raise ValueError('Please check the op_k parameters, or the "result_op_sc_'
                                 + self.file_name + '.csv" in the "data" folder!')

        fig = plt.figure(figsize=(5.6, 3.8), dpi=100)
        plt.errorbar(x, gap_values, marker='o', markeredgecolor='k', markerfacecolor='none', markersize=6,
                     yerr=se_values,
                     elinewidth=2, ecolor='k', capsize=4)
        plt.scatter(op_k, gap_values[op_k - 1], color='red', marker='o')
        plt.xticks([i for i in range(0, n + 1, 2)])
        plt.grid(linestyle='--', color='lightgray')
        plt.xlabel('Cluster')
        plt.ylabel('Gap value')
        plt.title('Gap Statistics')

        if saved:
            if save_name == 'result_gap.png':
                save_name = 'result_gap_' + self.file_name + '.png'
            plt.savefig(os.path.join('img', save_name), dpi=300)
        if not gui:
            plt.show()

        return fig

    def plot_average(self, n, saved=True, gui=False, show_legend=None, save_name='result_average.png'):
        """
        Method for plotting the focal mechanisms, calculated by method "pt_average".

        :param n: int type. Selected by user for which focal mechanisms wanted to plot.
        """
        if self.result_average is None:
            result_average_path = os.path.join('data', 'result_average_' + self.file_name + '.csv')
            result_pt_path = os.path.join('data', 'result_pt_' + self.file_name + '.csv')
            if os.path.exists(result_average_path) and os.path.exists(result_pt_path):
                result_average = pd.read_csv(result_average_path).values
                result_pt = pd.read_csv(result_pt_path).values
                if result_average.shape[1] != 10:
                    raise ValueError('Please validate your result_average data matched with pre-data!')
                if result_average.shape[0] < n:
                    raise ValueError('Please validate the parameter n less than clusters!')
                if result_pt.shape[0] != self.data.shape[0] or result_pt.shape[1] != 5:
                    raise ValueError('Please validate your result_pt data matched with pre-data!')
            else:
                raise ValueError('Please check the result_average and result_pt parameters, or the "result_average_'
                                 + self.file_name + '.csv"  and the "result_pt_' + self.file_name +
                                 '.csv" in the "data" folder!')
        else:
            result_average = self.result_average
            result_pt = self.result_pt

        base_x, base_y, x1, y1, x2, y2, xp_aver, yp_aver, xt_aver, yt_aver = plot_fm(result_average[n - 1])
        result_pt = np.array(result_pt)
        result_pt_n = result_pt[result_pt[:, 4] == n]

        fig = plt.figure(figsize=(5.6, 3.8), dpi=100)
        plt.plot(base_x, base_y, color='k', linewidth=1.0)
        plt.plot(x1, y1, color='r', linewidth=1.0)
        plt.plot(x2, y2, color='r', linewidth=1.0)
        plt.axis("scaled")
        plt.axis('off')

        for i in range(result_pt_n.shape[0]):
            xp, yp, xt, yt = plot_pt(result_pt_n[i][0:2], result_pt_n[i][2:4])
            plt.plot(xp, yp, alpha=0.5, color=[0.4940, 0.1840, 0.5560], marker='.', markersize=5)
            plt.plot(xt, yt, alpha=0.5, color=[0.4660, 0.6740, 0.1880], marker='.', markersize=5)

        plt.plot(xp_aver, yp_aver, color=[0.4940, 0.1840, 0.5560], marker='.', markersize=15)
        plt.plot(xt_aver, yt_aver, color=[0.4660, 0.6740, 0.1880], marker='.', markersize=15)

        if show_legend is not None:
            plt.text(0.8, 0.8, f'Cluster {n}', fontsize=14, family='Times New Roman')
            plt.plot(0.9, -0.7, color=[0.4940, 0.1840, 0.5560], marker='.', markersize=15)
            plt.plot(0.9, -0.9, color=[0.4660, 0.6740, 0.1880], marker='.', markersize=15)
            plt.text(1.0, -0.75, 'P-axis', fontsize=14, family='Times New Roman')
            plt.text(1.0, -0.95, 'T-axis', fontsize=14, family='Times New Roman')

        if saved:
            if save_name == 'result_average.png':
                save_name = 'result_average_' + self.file_name + '.png'
            plt.savefig(os.path.join('img', save_name), dpi=300)

        if not gui:
            plt.show()

        return fig
