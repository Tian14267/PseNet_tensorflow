# PseNet_tensorflow
# 说明:
本代码是为PSENET检测网络的tensorflow版本。代码主要参考了以下链接的代码：<br>
链接一：https://github.com/whai362/PSENet <br>
链接二：https://github.com/looput/PSENet-Tensorflow <br>
链接一为原始链接，主要使用pytorch实现；链接二为tensorflow版本的代码。相比较两个链接的版本，通过本人使用和测试发现，链接一pytorch的效果略优于链接二tensorflow。<br>
对于链接二，本人感觉其代码写的较为复杂，阅读学习性不太高（主要还是本人太菜），于是在这里改了一个简易版本的 Psenet_tensorflow

## 注意：
本代码目前只能实现训练模块(psenet_train.py)，在测试模块(psenet_test.py)一直有 Bug。
测试模块的主要问题为，训练的模型无法很好的识别出box，对于该问题，本人目前也很懵逼。但是训练模块一切正常。
对于这个问题，诸位大佬也可以帮忙看看是什么问题，并加以改正，小弟不胜感激（哭）

## Data
以下为代码训练所使用的训练数据：
链接：https://pan.baidu.com/s/1IRc-FAhb7cXmsycjAmO1cw  提取码：k8je 

## 模型
链接：https://pan.baidu.com/s/1zk6P5-LUGCB66vJPfouJGQ  提取码：l1ls 
注：kernaels = 3

## 训练细节

![image](https://github.com/Tian14267/PseNet_tensorflow/blob/master/Images/qqq.png)
