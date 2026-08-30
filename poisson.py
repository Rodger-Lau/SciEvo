# -*- coding: utf-8 -*-
"""
Poisson process sample paths.

@author: administer
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

# 当前环境没有安装 seaborn，因此使用 matplotlib 自带的 seaborn 风格。
plt.style.use("seaborn-v0_8")

# 固定随机种子，方便每次运行得到相同结果。
rng = default_rng(2026)

# n = 20 表示生成 20 条独立的泊松过程样本轨迹。
n = 20

# lambd 是泊松过程强度，T 是观察时长。
# 在 [0, T] 内，每条轨迹的事件总数 N(T) 服从 Poisson(lambd * T)。
# n 只表示样本轨迹条数，不改变单条轨迹的分布参数。
lambd, T = 10, 1


def one_poisson_path(lambd, T):
    # times 是事件时刻数组。第一个元素 0.0 表示观察起点；
    # 中间元素是不超过 T 的事件发生时刻；最后一个元素 T 用于画到观察终点。
    times = [0.0]

    # Tk 表示当前累计到达时刻，也就是第 k 次事件发生的时间。
    Tk = 0.0

    while True:
        # rng.standard_exponential() 生成均值为 1 的指数分布随机数。
        # 除以 lambd 后，等待时间服从均值为 1/lambd 的指数分布，
        # 这正是强度为 lambd 的泊松过程的相邻事件间隔分布。
        Tk = Tk + rng.standard_exponential() / lambd

        # 上面语句的功能：生成下一次事件间隔，并累加得到下一次事件时刻 Tk。
        # 如果下一次事件已经超过观察终点 T，就不把它计入 [0, T] 内的事件。
        if Tk > T:
            break

        times.append(Tk)

    # 追加 T 是为了画阶梯图时横轴能延伸到观察终点。
    # T 本身不是新增事件，因此后面统计事件数时要排除起点 0 和终点 T。
    times.append(T)
    return np.array(times)


# paths 保存 n 条独立样本轨迹；每条轨迹都是一个事件时刻数组。
# 注意这里应传入 (lambd, T)，原代码 one_poisson_path(n, lambd) 把参数传反了。
paths = [one_poisson_path(lambd, T) for _ in range(n)]


def count(size):
    # count 函数生成阶梯图的纵轴累计事件数 N(t)。
    # size = len(path) - 1，表示画图用的横坐标点数量减 1；
    # 返回值长度与 path 长度一致，最后重复 size - 1，让阶梯保持到 T。
    return np.append(np.arange(size), size - 1)


fig, ax = plt.subplots(layout="tight")
for path in paths:
    ax.step(path, count(len(path) - 1), where="post")

ax.set_xlabel("Time $t$")
ax.set_ylabel("Count $N(t)$")
ax.set_title(
    r"Poisson Process ($\lambda$ = %d, $T$ = %d) with %d sample paths"
    % (lambd, T, n)
)
ax.set_yticks(2 * np.arange(0, 11))

plt.savefig("AApoisson.png", dpi=300, bbox_inches="tight")
plt.show()

# 上面程序生成了 n=20 条阶梯状轨迹。
# 每条轨迹在时间 T=1 内的事件数服从 Poisson(lambd * T) = Poisson(10)。
# 图上的纵轴代表累计事件数 N(t)，即截至时间 t 已发生多少次事件。
# 阶梯跳变位置发生在事件到达时刻，也就是每个 path 中除 0 和 T 以外的时刻。
event_counts = np.array([len(path) - 2 for path in paths])
print("20条轨迹的事件数:", event_counts)
print("20条轨迹的平均事件数:", event_counts.mean())
print("理论平均事件数 lambd * T:", lambd * T)

# 增加轨迹数量后，大数定律会使平均事件数更接近理论值 lambd * T。
n_large = 10000
large_paths = [one_poisson_path(lambd, T) for _ in range(n_large)]
large_event_counts = np.array([len(path) - 2 for path in large_paths])
print(f"{n_large}条轨迹的平均事件数:", large_event_counts.mean())
