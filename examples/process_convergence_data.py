#!/usr/bin/env python
# Created by "Trae AI" at 16:05, 27/02/2024 ----------%                                                                               
#       Email: traeai@example.com                      %                                                    
#       Github: https://github.com/traeai             %                         
# --------------------------------------------------%

"""
处理收敛曲线CSV文件，计算每个算法在每个测试函数上的平均性能
"""

import os
import pandas as pd
import numpy as np
import glob

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

if __name__ == "__main__":
    # 在这里直接设置参数
    base_path = r"c:\Users\L\PycharmProjects\20250227\mealpy\examples\results_cec2022\history\convergence"
    algorithms = ['BaseGA', 'OriginalHHO', 'OriginalPSO']
    output_dir = base_path  # 输出到同一目录，如需更改请修改此行
    
    # 调用处理函数
    process_convergence_data(base_path, algorithms, output_dir)