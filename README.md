# CNN_TA
## 方案
* 连续N日的股价信息组成N*1维张量
* 每日开盘、收盘、最高、最低、MA5...组成不同channel
* 再输入数据上进行L层卷积操作，每层包含BN，RELU
* 叠加全连接网络对数据逐渐进行降维
* 最后输出目标为class={1，-1}对应牛、熊、震荡，score为涨跌幅预测
* 拟合的target为输入数据之后T日内最高涨跌幅是否超出阈值S，超出则为正负样本，否则为ignore
