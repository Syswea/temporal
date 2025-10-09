import pandas as pd
import numpy as np

L = 10 # 过去10天的波动
thate = 0.75 # 75%分位数

# 1. 读取数据（确保按日期排序）
df = pd.read_csv('finance.csv')
df = df.sort_values('t').reset_index(drop=True)

# 2. 计算对数收益率
df['return'] = np.log(df['y'] / df['y'].shift(1))

# 3. 去掉第一行（NaN）
df = df.dropna().reset_index(drop=True)

# 计算过去 L 天的滚动标准差（波动率）
df['rolling_vol'] = df['return'].rolling(window=L).std()

# E_t 表示：在时间 t，是否知道过去 L 天（t-L 到 t-1）高波动
df['vol_label'] = df['rolling_vol'].shift(1)

# 取 vol_label 的 75% 分位数作为阈值
tau = df['vol_label'].quantile(thate)
df['E_t'] = (df['vol_label'] > tau).astype(int)

# 去掉前 L+1 行（因 rolling 和 shift 产生 NaN）
df = df.dropna(subset=['E_t']).reset_index(drop=True)

# 检查标签分布
print("High volatility days:", df['E_t'].sum())
print("Proportion:", df['E_t'].mean())

df.to_csv('finance_with_Et.csv', index=False)