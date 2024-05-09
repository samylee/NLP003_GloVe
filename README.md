# NLP003_GloVe
NLP003: glove using pytorch

## 使用说明
### 要求
> Python == 3.6.13 \
> PyTorch == 1.10.1
### 训练
```shell script
python train.py
```
### 测试
```shell script
python predict.py
```
### 结果
```shell script
加拿大            男人
加拿大 : 1.000    男人 : 1.000
美国 : 0.385      女人 : 0.578
队长 : 0.355      喜欢 : 0.387
澳洲 : 0.346      一个 : 0.368
成都 : 0.337      的 : 0.347
移民 : 0.312      不 : 0.315
莫斯科 : 0.307    结婚 : 0.307
兰州 : 0.304      是 : 0.305
东京 : 0.302      爱 : 0.305
迪士尼 : 0.300    会 : 0.305
```
## 参考
https://github.com/gdtydm/pytorch-glove-word2vec  
https://blog.csdn.net/samylee  
