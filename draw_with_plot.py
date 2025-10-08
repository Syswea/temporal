import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

def plot_data_and_predicted(plot_name, data: pd.DataFrame, predicted: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    
    # 绘制原始数据（蓝色点）
    plt.scatter(data['t'], data['y'], 
               color='blue', 
               label='Original Data', 
               alpha=0.7, 
               s=30)
    
    # 绘制预测结果（红色点）
    plt.scatter(predicted['t'], predicted['y'], 
               color='red', 
               label='Predicted', 
               alpha=0.7, 
               s=30)
    
    # 设置图表属性
    plt.title(plot_name, fontsize=16)
    plt.xlabel('Time (t)', fontsize=12)
    plt.ylabel('Position (y)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 调整布局并显示
    plt.tight_layout()
    plt.show()

# %%
def plot_dataFrame(plot_name, data: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    
    # 绘制预测结果（红色点）
    plt.scatter(data['x'], data['prob'], 
               color='red', 
               label='probs', 
               alpha=0.7, 
               s=30)
    
    # 设置图表属性
    plt.title(plot_name, fontsize=16)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('prob', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 调整布局并显示
    plt.tight_layout()
    plt.show()

# %%
def draw_t_distribution(model, t, scaler, device):
    t = torch.tensor(t, dtype=torch.float32, device=device)
    x_range = (-4, 4)
    num_points = 1000
    x_search = torch.linspace(x_range[0], x_range[1], num_points, dtype=torch.float32, device=device).view(-1, 1)
    t_search = t.view(-1, 1).expand(num_points, -1)

    print(x_search.shape, t_search.shape)

    probs = model.get_probability(t_search, x_search)

    x = scaler.inverse_transform(x_search.cpu().numpy().reshape(-1, 1)).reshape(-1)
    p = probs.detach().cpu().numpy().reshape(-1)
    data = pd.DataFrame({'x': x, 'prob': p})

    plot_dataFrame(f"distribution in t {t}", data)