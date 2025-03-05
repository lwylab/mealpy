#!/usr/bin/env python
# Created by "Trae AI" at 16:05, 27/02/2024 ----------%                                                                               
#       Email: traeai@example.com                      %                                                    
#       Github: https://github.com/traeai             %                         
# --------------------------------------------------%

"""
比较HHO、PSO和GA算法在CEC2022测试函数上的性能
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from opfunu.cec_based.cec2022 import *
from mealpy import FloatVar
from mealpy.swarm_based import HHO, PSO
from mealpy.evolutionary_based import GA
from mealpy import Multitask

# 创建结果目录
results_dir = "results_cec2022"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
# 创建图表目录
plots_dir = os.path.join(results_dir, "plots")
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# 定义问题维度和算法参数
problem_size = 20
pop_size = 50
epoch = 500
n_trials = 3  # 每个算法在每个函数上运行的次数

# 定义CEC2022测试函数
functions = [
    F12022(ndim=problem_size), F22022(ndim=problem_size), 
    F32022(ndim=problem_size), F42022(ndim=problem_size),
    F52022(ndim=problem_size), F62022(ndim=problem_size),
    F72022(ndim=problem_size), F82022(ndim=problem_size),
    F92022(ndim=problem_size), F102022(ndim=problem_size),
    F112022(ndim=problem_size), F122022(ndim=problem_size)
]

# 创建问题字典列表
problems = []
for i, func in enumerate(functions):
    problems.append({
        "bounds": FloatVar(lb=func.lb, ub=func.ub),
        "obj_func": func.evaluate,
        "minmax": "min",
        "name": f"F{i+1}",
        "log_to": None,
        "save_population": False
    })

# 定义算法模型
hho_model = HHO.OriginalHHO(epoch=epoch, pop_size=pop_size)
pso_model = PSO.OriginalPSO(epoch=epoch, pop_size=pop_size)
ga_model = GA.BaseGA(epoch=epoch, pop_size=pop_size)

# 定义终止条件
term = {
    "max_epoch": epoch  # 直接使用最大迭代次数作为终止条件
}

# 创建并执行多任务
if __name__ == "__main__":
    # 创建多任务对象
    multitask = Multitask(
        algorithms=(hho_model, pso_model, ga_model), 
        problems=problems, 
        terminations=(term,), 
        modes=("thread",), 
        n_workers=4
    )
    
    # 执行多任务
    multitask.execute(
        n_trials=n_trials, 
        n_jobs=None, 
        save_path=os.path.join(results_dir, "history"), 
        save_as="csv", 
        save_convergence=True, 
        verbose=True
    )
    
    # # 绘制收敛曲线
    # for i, func in enumerate(functions):
    #     func_name = f"F{i+1}"
    #     plt.figure(figsize=(10, 6))
    #
    #     # 为每个算法绘制收敛曲线
    #     algorithms = ["HHO", "PSO", "GA"]
    #     colors = ['r', 'g', 'b']
    #
    #     for j, algo in enumerate(algorithms):
    #         # 读取所有试验的收敛数据
    #         convergence_data = []
    #         for trial in range(1, n_trials + 1):
    #             file_path = os.path.join(results_dir, "history", f"{algo}-{func_name}-trial-{trial}-convergence.csv")
    #             if os.path.exists(file_path):
    #                 data = pd.read_csv(file_path)
    #                 convergence_data.append(data['Global Best Fitness'].values)
    #
    #         if convergence_data:
    #             # 计算平均收敛曲线
    #             avg_convergence = np.mean(convergence_data, axis=0)
    #             epochs = np.arange(1, len(avg_convergence) + 1)
    #
    #             # 绘制收敛曲线
    #             plt.plot(epochs, avg_convergence, color=colors[j], label=algo)
    #
    #     plt.title(f'收敛曲线比较 - {func_name}')
    #     plt.xlabel('迭代次数')
    #     plt.ylabel('适应度值 (越小越好)')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.yscale('log')  # 使用对数刻度更好地显示收敛过程
    #
    #     # 保存图表
    #     plt.savefig(os.path.join(plots_dir, f"{func_name}_convergence.png"), dpi=300, bbox_inches='tight')
    #     plt.close()
    
    print("所有实验完成，结果已保存到", results_dir)
    # print("收敛曲线已保存到", plots_dir)