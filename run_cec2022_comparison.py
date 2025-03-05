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
import glob
import datetime
from opfunu.cec_based.cec2022 import *
from mealpy import FloatVar
from mealpy.swarm_based import HHO, PSO
from mealpy.evolutionary_based import GA
from mealpy import Multitask

# 创建带时间戳的结果目录
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_cec2022_{timestamp}"
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

def process_convergence_data(base_path, algorithms, output_dir=None):
    """
    处理收敛曲线数据，计算每个算法在每个测试函数上的平均性能
    
    参数:
        base_path: 收敛曲线CSV文件的基础路径
        algorithms: 算法名称列表
        output_dir: 输出目录，默认为base_path
    """
    if output_dir is None:
        output_dir = base_path
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有测试函数文件名
    first_algo_path = os.path.join(base_path, algorithms[0])
    function_files = [os.path.basename(f) for f in glob.glob(os.path.join(first_algo_path, "F*_convergence.csv"))]
    function_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x.split('_')[0]))))
    
    # 处理每个测试函数
    for func_file in function_files:
        func_name = func_file.split('_')[0]  # 提取函数名称 (F1, F2, ...)
        
        # 创建一个字典来存储每个算法的平均收敛数据
        algo_avg_data = {}
        
        # 处理每个算法
        for algo in algorithms:
            file_path = os.path.join(base_path, algo, func_file)
            
            if os.path.exists(file_path):
                # 读取CSV文件
                df = pd.read_csv(file_path)
                
                # 计算每行的平均值
                if len(df.columns) > 1:  # 确保有多个试验列
                    avg_values = df.mean(axis=1).values
                    algo_avg_data[algo] = avg_values
                else:
                    print(f"警告: {file_path} 没有足够的列来计算平均值")
            else:
                print(f"警告: 文件不存在 {file_path}")
        
        # 确定最大迭代次数（最长的收敛曲线）
        max_iterations = max([len(data) for data in algo_avg_data.values()]) if algo_avg_data else 0
        
        # 创建结果DataFrame
        result_df = pd.DataFrame(index=range(max_iterations))
        
        # 填充数据
        for algo, data in algo_avg_data.items():
            # 如果数据长度小于最大迭代次数，用最后一个值填充
            if len(data) < max_iterations:
                padded_data = np.pad(data, (0, max_iterations - len(data)), 'edge')
                result_df[algo] = padded_data
            else:
                result_df[algo] = data
        
        # 保存结果
        output_file = os.path.join(output_dir, f"{func_name}_avg_convergence.csv")
        result_df.to_csv(output_file, index=False)
        print(f"已保存平均收敛数据到 {output_file}")
    
    return function_files

def plot_convergence_curves(avg_data_dir, function_files, output_dir):
    """
    绘制收敛曲线
    
    参数:
        avg_data_dir: 平均收敛数据目录
        function_files: 函数文件名列表
        output_dir: 输出目录
    """
    for func_file in function_files:
        func_name = func_file.split('_')[0]  # 提取函数名称 (F1, F2, ...)
        avg_file = os.path.join(avg_data_dir, f"{func_name}_avg_convergence.csv")
        
        if os.path.exists(avg_file):
            # 读取平均收敛数据
            df = pd.read_csv(avg_file)
            
            # 绘制收敛曲线
            plt.figure(figsize=(10, 6))
            
            # 为每个算法绘制收敛曲线
            algorithms = df.columns
            colors = ['r', 'g', 'b']
            
            for i, algo in enumerate(algorithms):
                epochs = np.arange(1, len(df) + 1)
                plt.plot(epochs, df[algo], color=colors[i % len(colors)], label=algo)
            
            plt.title(f'收敛曲线比较 - {func_name}')
            plt.xlabel('迭代次数')
            plt.ylabel('适应度值 (越小越好)')
            plt.legend()
            plt.grid(True)
            plt.yscale('log')  # 使用对数刻度更好地显示收敛过程
            
            # 保存图表
            plt.savefig(os.path.join(output_dir, f"{func_name}_convergence.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"已保存收敛曲线图表到 {os.path.join(output_dir, f'{func_name}_convergence.png')}")

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
    
    print("所有实验完成，结果已保存到", results_dir)
    
    # 处理收敛曲线数据
    convergence_dir = os.path.join(results_dir, "history", "convergence")
    avg_data_dir = os.path.join(convergence_dir, "avg_data")
    if not os.path.exists(avg_data_dir):
        os.makedirs(avg_data_dir)
    
    algorithms = ['BaseGA', 'OriginalHHO', 'OriginalPSO']
    function_files = process_convergence_data(convergence_dir, algorithms, avg_data_dir)
    
    # 绘制收敛曲线
    plot_convergence_curves(avg_data_dir, function_files, plots_dir)
    
    print("所有数据处理和绘图完成")
