import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

plt.rcParams['font.sans-serif']=['Arial'] #显示中文
plt.rcParams['axes.unicode_minus'] = False
plt.rc("font",family="Arial",size="10.5")

def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# 读取数据
dir = '../test/'
c = np.loadtxt(dir + 'time_complete.txt')
d = np.loadtxt(dir + 'time_uncomplete.txt')

small_five_size = 10.5
time_complete = c.tolist()
time_un_complete = d.tolist()

x = np.array([20, 40, 60, 80, 100])
width = 5.0  # 每个柱子的宽度
group_gap = width * 0.2  # 每组柱子之间的间隙比例

# 调整图形的宽高比
fig, ax = plt.subplots()

# 设置柱状图，使用好看的颜色搭配
bar1 = ax.bar(x - width / 2, time_complete, width, label='Complete Graph', color='#3498db')  # 浅蓝色
bar2 = ax.bar(x + width / 2, time_un_complete, width, label='Incomplete Graph', color='#e74c3c')  # 鲜红色

# 设置刻度和标签
y_major_locator = MultipleLocator(0.001)
ax.yaxis.set_major_locator(y_major_locator)
ax.set_ylim(0.004, 0.008)  # 设置 y 轴的范围
ax.set_xlabel('Number of Participating Vehicles', fontsize=small_five_size)
ax.set_ylabel('Decision Time (s)', fontsize=small_five_size)
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in x])
ax.grid(linestyle=':')

# 设置图例位置
ax.legend(fontsize=10.5, loc='upper right', bbox_to_anchor=(1, 1))

# 设置刻度字体大小
plt.xticks(fontsize=small_five_size)
plt.yticks(fontsize=small_five_size)

# 保存和显示图形
dir2 = '../结果图/'
plt.savefig(dir2 + 'Fig_4_time1', dpi=600)
plt.show()