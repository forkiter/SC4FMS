# -*- coding = utf-8 -*-
# GUI for SC4FMS
# 2023.03.22,edit by Lin

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from tkinter.messagebox import askyesno
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core.sc4fms import ScFms


class ScGui(object):
    def __init__(self, init_window_name):
        self.file_input_dirs = None
        self.sc = None  # 由类"ScFms"创建的实例
        self.canvas = None  # 画图对象
        self.init_window_name = init_window_name
        self.init_window_name.title('SC4FMS v0.1.0')
        self.init_window_name.geometry('800x600')
        self.init_window_name.iconbitmap(os.path.join('docs', 'fig', 'head.ico'))

        self.init_window_name.protocol('WM_DELETE_WINDOW', self.close_window)  # 点击右上角关闭窗体弹窗事件

        """ 组件容器 """
        self.left_frame = tk.Frame(self.init_window_name)  # 左侧布局
        self.left_frame.pack(side=tk.LEFT, anchor=tk.N, padx=5, pady=5)
        self.input_data_frame = tk.LabelFrame(self.left_frame, text='数据导入', padx=5, pady=5)
        self.input_data_frame.pack()
        self.sc_frame = tk.LabelFrame(self.left_frame, text='谱聚类', padx=5, pady=5)
        self.sc_frame.pack()
        self.gap_frame = tk.LabelFrame(self.left_frame, text='Gap Statistic', padx=5, pady=5)
        self.gap_frame.pack()
        self.plot_frame = tk.LabelFrame(self.left_frame, text='绘图', padx=5, pady=5)
        self.plot_frame.pack()

        self.right_frame = tk.Frame(self.init_window_name)  # 右侧布局
        self.right_frame.pack(side=tk.TOP, padx=5, pady=5)
        self.show_data_frame = tk.Frame(self.right_frame)
        self.show_data_frame.pack()
        tk.Label(self.show_data_frame, text="数据显示窗口").pack(anchor=tk.W)
        self.data_pad = tk.Text(self.show_data_frame, width=80, height=30)
        self.data_pad.config(state='normal')
        self.data_pad.pack(side=tk.LEFT, fill=tk.X)

        """ 数据显示窗和日志显示窗 """
        self.listbox = ttk.Treeview(self.data_pad, show='headings', height=18)  # 数据显示窗
        self.data_bar = ttk.Scrollbar(self.show_data_frame, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=self.data_bar.set)
        heading = ['ID', 'Strike', 'Rake', 'Dip', 'Label']
        self.listbox.configure(columns=heading)
        for col in heading:
            if col == 'ID' or col == 'Label':
                self.listbox.column(col, width=40, anchor='center')
                self.listbox.heading(col, text=col)
            else:
                self.listbox.column(col, width=160, anchor='center')
                self.listbox.heading(col, text=col)
        self.listbox.pack()
        self.data_bar.pack(side=tk.RIGHT, fill=tk.Y)

        self.show_log_frame = tk.Frame(self.right_frame)  # 日志显示窗
        self.show_log_frame.pack()
        tk.Label(self.show_log_frame, text="日志显示窗口").pack(anchor=tk.W)
        self.log_pad = ScrolledText(self.show_log_frame, width=80, height=11)
        self.log_pad.pack(side=tk.LEFT, fill=tk.X)
        self.log_pad.tag_configure('stderr', foreground='#b22222')

        """ 重定向输出到log """
        sys.stdout = TextRedirector(self.log_pad, 'stdout')
        sys.stderr = TextRedirector(self.log_pad, 'stderr')

        """ 文本输入及文件选择组件 """
        self.file_input_title = tk.Label(self.input_data_frame, text="文件路径：")  # 导入数据
        self.file_input_title.grid(padx=0, pady=0, row=0, column=0, sticky=tk.W)
        self.file_input_entry = tk.Entry(self.input_data_frame, width=25)
        self.file_input_entry.grid(padx=0, pady=0, row=1, column=0)
        self.file_input_button = tk.Button(self.input_data_frame, text="导入", command=self.file_input_path)
        self.file_input_button.grid(padx=20, pady=3, row=2, column=0, sticky=tk.W)
        self.file_confirm_button = tk.Button(self.input_data_frame, text="确定", command=lambda: self.load_data())
        self.file_confirm_button.grid(padx=30, pady=3, row=2, column=0, sticky=tk.E)

        """ 谱聚类参数输入及计算组件 """
        self.k_input_title = tk.Label(self.sc_frame, text="聚类数k =")  # 谱聚类
        self.k_input_title.grid(padx=0, pady=0, row=0, column=0, sticky=tk.W)
        self.k_input_entry = tk.Entry(self.sc_frame, width=9)
        self.k_input_entry.grid(padx=0, pady=0, row=0, column=1, sticky=tk.E)
        self.numN_input_title = tk.Label(self.sc_frame, text="近邻数numN =")
        self.numN_input_title.grid(padx=0, pady=3, row=1, column=0, sticky=tk.W)
        self.numN_input_entry = tk.Entry(self.sc_frame, width=9)
        self.numN_input_entry.insert(0, '150')
        self.numN_input_entry.grid(padx=0, pady=3, row=1, column=1, sticky=tk.E)
        self.kscale_input_title = tk.Label(self.sc_frame, text="核比例因子kscale =")
        self.kscale_input_title.grid(padx=0, pady=3, row=2, column=0, sticky=tk.W)
        self.kscale_input_entry = tk.Entry(self.sc_frame, width=9)
        self.kscale_input_entry.insert(0, '40')
        self.kscale_input_entry.grid(padx=0, pady=3, row=2, column=1, sticky=tk.E)
        self.sc_confirm_button = tk.Button(self.sc_frame, text="开始计算", command=lambda: self.sc_run())
        self.sc_confirm_button.grid(padx=0, pady=3, row=3, column=0, sticky=tk.W)
        self.aver_confirm_button = tk.Button(self.sc_frame, text="计算平均解", command=lambda: self.aver_run())
        self.aver_confirm_button.grid(padx=0, pady=3, row=3, column=1, sticky=tk.W)

        """ Gap-statistics参数输入及计算组件 """
        self.n_input_title = tk.Label(self.gap_frame, text="聚类个数n =")  # 间隔统计
        self.n_input_title.grid(padx=0, pady=0, row=0, column=0, sticky=tk.W)
        self.n_input_entry = tk.Entry(self.gap_frame, width=15)
        self.n_input_entry.grid(padx=0, pady=0, row=0, column=1, sticky=tk.E)
        self.b_input_title = tk.Label(self.gap_frame, text="抽样数b =")
        self.b_input_title.grid(padx=0, pady=3, row=1, column=0, sticky=tk.W)
        self.b_input_entry = tk.Entry(self.gap_frame, width=15)
        self.b_input_entry.insert(0, '100')
        self.b_input_entry.grid(padx=0, pady=3, row=1, column=1, sticky=tk.E)
        self.gap_confirm_button = tk.Button(self.gap_frame, text="开始计算", command=lambda: self.gap_run())
        self.gap_confirm_button.grid(padx=0, pady=5, row=2, column=0, sticky=tk.W)

        """ 画图组件 """
        self.plot_choose_type = ttk.Combobox(self.plot_frame, width=23)
        self.plot_choose_type['values'] = ['谱聚类结果三维分布图', '最佳谱聚类结果三维分布图', 'k=n时谱聚类结果三维分布图',
                                           'Gap-Statistic曲线图', '震源机制平均解图']
        self.plot_choose_type.current(0)
        self.plot_choose_type.grid(padx=0, pady=3, row=1, column=0, sticky=tk.W)
        self.plot_n_input_title = tk.Label(self.plot_frame, text="n =")
        self.plot_n_input_title.grid(padx=0, pady=3, row=2, column=0, sticky=tk.W)
        self.plot_n_input_entry = tk.Entry(self.plot_frame, width=3)
        self.plot_n_input_entry.insert(0, '1')
        self.plot_n_input_entry.grid(padx=30, pady=3, row=2, column=0, sticky=tk.W)
        self.plot_show_legend_label = tk.Label(self.plot_frame, text="显示图例：")  # 复选框
        self.plot_show_legend_label.grid(padx=40, pady=3, row=2, column=0, sticky=tk.E)
        self.plot_show_legend_value = tk.IntVar()
        self.plot_show_legend_value.set(0)  # 设置默认值 0
        self.plot_show_legend_entry = tk.Checkbutton(self.plot_frame, variable=self.plot_show_legend_value, onvalue=1,
                                                     offvalue=0)
        self.plot_show_legend_entry.grid(padx=20, pady=3, row=2, column=0, sticky=tk.E)
        self.plot_save_name_title = tk.Label(self.plot_frame, text="图片名：")
        self.plot_save_name_title.grid(padx=0, pady=3, row=3, column=0, sticky=tk.W)
        self.plot_save_name_entry = tk.Entry(self.plot_frame, width=17)
        self.plot_save_name_entry.insert(0, '')
        self.plot_save_name_entry.grid(padx=10, pady=3, row=3, column=0, sticky=tk.E)
        self.plot_confirm_button = tk.Button(self.plot_frame, text="开始绘图", command=lambda: self.plot_run())
        self.plot_confirm_button.grid(padx=20, pady=3, row=4, column=0, sticky=tk.W)
        self.plot_destroy_button = tk.Button(self.plot_frame, text="清除绘图", command=lambda: self.plot_des())
        self.plot_destroy_button.grid(padx=30, pady=3, row=4, column=0, sticky=tk.E)

        """ 版本信息及联系方式 """
        self.version = tk.Label(self.left_frame, text='SC4FMS made by Lin, Gdsin\nEmail:  forkiter@163.com')
        self.version.pack(side=tk.LEFT, padx=5, pady=5)

    def file_input_path(self):
        """ 上传文件路径选择 """
        path_ = filedialog.askopenfilename(title='打开csv文件', filetypes=[('csv', '*.csv')])
        self.file_input_dirs = path_
        self.file_input_entry.delete(0, tk.END)
        self.file_input_entry.insert(tk.END, path_)

    def load_data(self):
        """ 数据导入及生成sc实例 """
        if len(self.file_input_entry.get().strip()) < 1 and self.file_input_dirs is None:
            messagebox.showwarning(message='必须输入或选择文件地址!', title='警告')
            return False
        if len(self.file_input_entry.get().strip()) > 1:
            file_path = self.file_input_entry.get().strip()
            self.file_input_dirs = file_path
        else:
            file_path = self.file_input_dirs

        file_path_content = f"文件地址为：{file_path}"
        self.run_log_print(file_path_content)

        try:
            sc = ScFms(self.file_input_dirs)
        except ValueError as e:
            messagebox.showerror('Error:', str(e))
            return False
        self.sc = sc
        if sc.result_sc is None:
            data_from = 'ori'
        else:
            data_from = 'sc_data'
        self.show_data(data_from)

    def show_data(self, data_from='ori'):
        """ 右上数据窗口显示数据 """
        if data_from == 'ori':
            show_data = self.sc.data
        elif data_from == 'sc_data':
            show_data = self.sc.result_sc
        elif data_from == 'op_data':
            show_data = self.sc.result_op_sc
        else:
            messagebox.showwarning(message='数据显示失败。', title='警告')
            return False

        self.listbox.delete(*self.listbox.get_children())
        for i, row in enumerate(show_data, start=1):
            if data_from == 'ori':
                self.listbox.insert("", "end", values=(i, row[0], row[1], row[2], ''))
            else:
                self.listbox.insert("", "end", values=(i, row[0], row[1], row[2], int(row[3])))

        self.data_pad.config(state='disabled')

    def run_log_print(self, message):
        """ 实时更新日志，固定用法 """
        self.log_pad.config(state=tk.NORMAL)
        self.log_pad.insert(tk.END, "\n" + message + "\n")
        self.log_pad.see(tk.END)
        self.log_pad.update()
        self.log_pad.config(state=tk.DISABLED)

    def close_window(self):
        """ 退出/关闭窗体 固定方法 """
        ans = askyesno(title='退出警告', message='是否确定退出程序？\n是则退出，否则继续！')
        if ans:
            self.init_window_name.destroy()
            sys.exit()
        else:
            return None

    def sc_run(self):
        """ 谱聚类计算 """
        if self.sc is None:
            messagebox.showwarning(message='请导入数据。', title='警告')
        elif self.k_input_entry.get().isdigit() and self.numN_input_entry.get().isdigit() and \
                self.kscale_input_entry.get().isdigit():
            self.run_log_print('开始震源机制解的谱聚类计算...')
            k = int(self.k_input_entry.get())
            numN = int(self.numN_input_entry.get())
            kscale = int(self.kscale_input_entry.get())
            self.sc.spectral_cluster_com(k=k, numN=numN, kscale=kscale, saved=True)
            self.show_data(data_from='sc_data')
        else:
            messagebox.showwarning(message='请确认输入的k, numN, kscale值是否正确。', title='警告')

    def aver_run(self):
        """ 平均解计算 """
        if self.sc is None:
            messagebox.showwarning(message='请导入数据。', title='警告')
        else:
            self.run_log_print('开始计算每类平均解...')
            self.sc.pt_average()
            self.run_log_print('平均解计算完成.')

    def gap_run(self):
        """ Gap-statistics计算 """
        if self.sc is None:
            messagebox.showwarning(message='请导入数据。', title='警告')
        elif self.numN_input_entry.get().isdigit() and self.kscale_input_entry.get().isdigit() and \
                self.n_input_entry.get().isdigit() and self.b_input_entry.get().isdigit():
            self.run_log_print('开始Gap-Statistic计算...')
            n_inspect = int(self.n_input_entry.get())
            b_num = int(self.b_input_entry.get())
            numN = int(self.numN_input_entry.get())
            kscale = int(self.kscale_input_entry.get())
            self.sc.gap_eva(n_inspect=n_inspect, b_num=b_num, numN=numN, kscale=kscale, saved=True)
            self.show_data(data_from='op_data')
        else:
            messagebox.showwarning(message='请确认输入的n, b, numN, kscale值是否正确。', title='警告')

    def plot_run(self):
        """ 绘图 """
        if self.sc is None:
            messagebox.showwarning(message='请导入数据。', title='警告')
            return False

        show_legend = None if self.plot_show_legend_value.get() == 0 else self.plot_show_legend_value.get()
        save_name = 'result_' + self.sc.file_name + '.png' if self.plot_save_name_entry.get() == ''\
            else self.plot_save_name_entry.get()
        plot_type = self.plot_choose_type.get()
        if plot_type == '谱聚类结果三维分布图':
            try:
                fig = self.sc.plot_3d(self.sc.user_labels(), gui=True, show_legend=show_legend, save_name=save_name)
            except Exception as e:
                messagebox.showerror('Error:', str(e))
                return False
        elif plot_type == '最佳谱聚类结果三维分布图':
            try:
                fig = self.sc.plot_3d(self.sc.user_labels(label_type='op'), gui=True, show_legend=show_legend,
                                      save_name=save_name)
            except Exception as e:
                messagebox.showerror('Error:', str(e))
                return False
        elif plot_type == 'k=n时谱聚类结果三维分布图':
            if self.plot_n_input_entry.get().isdigit():
                pass
            else:
                messagebox.showwarning(message='请输入n整数值。', title='警告')
                return False
            n_value = int(self.plot_n_input_entry.get())
            try:
                fig = self.sc.plot_3d(self.sc.user_labels(label_type='user', n=n_value), gui=True,
                                      show_legend=show_legend, save_name=save_name)
            except Exception as e:
                messagebox.showerror('Error:', str(e))
                return False
        elif plot_type == 'Gap-Statistic曲线图':
            try:
                fig = self.sc.plot_gap(gui=True, save_name=save_name)
            except Exception as e:
                messagebox.showerror('Error:', str(e))
                return False
        elif plot_type == '震源机制平均解图':
            if self.plot_n_input_entry.get().isdigit():
                pass
            else:
                messagebox.showwarning(message='请输入n整数值。', title='警告')
                return False
            n_value = int(self.plot_n_input_entry.get())
            try:
                fig = self.sc.plot_average(n=n_value, gui=True, show_legend=show_legend, save_name=save_name)
            except Exception as e:
                messagebox.showerror('Error:', str(e))
                return False
        else:
            messagebox.showwarning(message='请选择绘图类型。', title='警告')
            return False

        canvas = FigureCanvasTkAgg(fig, master=self.init_window_name)
        canvas.get_tk_widget().place(x=216, y=32)
        canvas.draw()
        self.canvas = canvas

    def plot_des(self):
        self.canvas.get_tk_widget().destroy()


class TextRedirector(object):
    """ 控制台输出重定向只日志显示窗口 """
    def __init__(self, widget, tag='stdout'):
        self.widget = widget
        self.tag = tag

    def write(self, str_):
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, str_, (self.tag,))
        self.widget.see(tk.END)
        self.widget.update()
        self.widget.configure(state='disabled')
