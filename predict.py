import math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
import time
import tushare as ts
import pandas as pd


# 格式化成2016-03-20 11:45:39形式
print ()


#获取股票简码列表
def get_stock_code():
    stock_info = ts.get_stock_basics()
    return stock_info.index
      
#获取历史数据
#返回值说明：date：日期 open：开盘价 high：最高价 close：收盘价 low：最低价 volume：成交量 price_change：价格变动 p_change：涨跌幅 ma5：5日均价 ma10：10日均价 ma20:20日均价 v_ma5:5日均量 v_ma10:10日均量 v_ma20:20日均量
#turnover:换手率[注：指数无此项]
def get_history_price(code):
  df=ts.get_k_data(code, start='2015-05-10', end=time.strftime("%Y-%m-%d ", time.localtime()))
  # 构造两个新的列
  # HL_PCT为股票最高价与最低价的变化百分比
  df['HL_PCT'] = (df['high'] - df['close']) / df['close'] * 100.0
  # HL_PCT为股票收盘价与开盘价的变化百分比
  df['PCT_change'] = (df['close'] - df['open']) / df['open'] * 100.0
  # 返回真正用到的特征字段
  return df[['close', 'HL_PCT', 'PCT_change', 'volume','date']].sort_values(by = 'date',axis = 0,ascending = True) 
df=get_history_price('600808')
df['date'] = pd.to_datetime(df['date'])
df=df.set_index('date')
# 定义预测列变量，它存放研究对象的标签名
forecast_col = 'close'
# 定义预测天数，这里设置为所有数据量长度的1%
forecast_out = 1


#数据预处理
# 因为scikit-learn并不会处理空数据，需要把为空的数据都设置为一个比较难出现的值，这里取-9999，
df.fillna(-99999, inplace=True)
# 用label代表该字段，是预测结果
# 通过让与Close列的数据往前移动1%行来表示
df['label'] = df[forecast_col].shift(-forecast_out)
X = np.array(df.drop(['label'], 1))
#中心均值和分量尺度到单位方差。
X = preprocessing.scale(X)
X_lately = X[-1:]
X = X[:-1]
# 抛弃label列中为空的那些行
df.dropna(inplace=True)
y = np.array(df['label'])

X_train,X_test,y_train,y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1) # LinearRegression 97.7%
#clf = svm.SVR() # Support vector machine 79.5%
#clf = svm.SVR(kernel='poly') # Support vector machine 68.5%   'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
clf.fit(X_train, y_train) # Here, start training
accuracy = clf.score(X_test, y_test) # Test an get a score for the classfier
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy)

#24小时second
one_day = 86400
# 在df中新建Forecast列，用于存放预测结果的数据
df['Forecast'] = np.nan

# 取df最后一行的时间索引
last_date = df.iloc[-1].name
print(last_date)
#datetime为时间戳
last_unix = last_date.timestamp()
next_unix= last_unix + one_day
# 遍历预测结果，用它往df追加行
# 这些行除了Forecast字段，其他都设为np.nan
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    # 这段[np.nan for _ in range(len(df.columns) - 1)]代码生成不包含Forecast字段的列表
    # 而[i]为只包含Forecast值的列表
    # 上述两个列表拼接在一起就组成了新行，按日期追加到df的下面
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
# 开始绘图
#print(df.index['date'])
print(df)
df['close'].plot()
df['Forecast'].plot()

# 修改matplotlib样式
style.use('ggplot') 
plt.legend(loc=4)
plt.rcParams['font.sans-serif']=['SimHei']
plt.title('股票预测')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

