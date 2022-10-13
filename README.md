# DQN

#### 介绍
DQN及DQN相关的改进算法

#### 代码结构
dqn
----weights (存放权重文件)
----dqn_Nature_eval.py (验证训练得到的模型效果)
----dqn_Nature_gpu.py (训练模型，CPU或GPU均可训练)


#### 运行代码
没有main主入口。
如果要训练模型，则运行dqn_Nature_gpu.py，训练过程中的权重文件会存储到weights文件夹下；
如果要验证模型效果，则运行dqn_Nature_eval.py。可以在weights文件夹中选择指定的权重进行加载，注意修改代码中的权重文件路径。

#### 注意事项

运行代码如果出现 **AttributeError: module 'gym.envs.atari' has no attribute 'AtariEnv'** 问题，请检查自己的gym版本。如果gym版本 > 0.19.0，请按下述步骤改为0.19.0版本：

```
# 先切换到所在anaconda环境
conda activate xxx
pip install --upgrade gym==0.19.0
pip install gym[atari]
```

