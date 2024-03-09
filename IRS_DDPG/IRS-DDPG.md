# DDPG算法

[Johnson-Nyquist noise](https://en.wikipedia.org/wiki/Johnson–Nyquist_noise#Noise_power_in_decibels)

为什么要用这个噪声？



DDPG算法优化相移矩阵的思考：

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h15ulhq9irj20to0auta6.jpg" alt="截屏2022-04-01 10.35.41" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h15ulfphn6j20sg0ikt9g.jpg" alt="截屏2022-04-01 10.36.14" style="zoom:50%;" />

以上为智能反射面随机生成相位的情况下信噪比的分布，可以看出大部分集中在平均值附近，

episode=100, number_slots=1000

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h15ulg7p6qj20u00va0w6.jpg" alt="截屏2022-03-30 22.55.15" style="zoom:50%;" />

不同的episode、num_slots、动作噪声情况下，snr变动不大，考虑修改奖励：

将原先的奖励——snr减去随机生成相移情况下的snr平均值

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h15ulglpurj20u00wkmzr.jpg" alt="截屏2022-04-01 10.47.28" style="zoom:50%;" />



episode=100, number_slots=2000

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h15ulh3ptij20u00u7aeb.jpg" alt="截屏2022-03-30 23.34.09" style="zoom:50%;" />

效果仍然不理想，但可以看到波动的范围增大了，下一步考虑增大number_slots来放大探索到最优情况的几率。