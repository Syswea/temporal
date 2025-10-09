import baostock as bs
import pandas as pd


#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

#### 获取沪深A股历史K线数据 ####
# 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
# 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
# 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
rs = bs.query_history_k_data_plus("sh.600000",
    "close",
    start_date='2012-07-01', end_date='2024-12-31',
    frequency="d", adjustflag="3")
# 浦发银行沪交所默认不复权的日线数据
print('query_history_k_data_plus respond error_code:'+rs.error_code)
print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)

#### 打印结果集 ####
data_list = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    data_list.append(rs.get_row_data())
result = pd.DataFrame(data_list, columns=rs.fields)

result.rename(columns={'close': 'y'}, inplace=True)
result['t'] = range(1, len(result) + 1)

#### 结果集输出到csv文件 ####   
result.to_csv("finance.csv", index=False)
print(result)

#### 登出系统 ####
bs.logout()
