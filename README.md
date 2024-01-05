---
frameworks:
- Pytorch
license: Apache License 2.0
tasks:
- face-emotion
---

#### Clone with HTTP
```bash
 git clone https://www.modelscope.cn/wanbiao/FER-wanbiao.git
```

# 模型介绍

本模型使用CREMA-D数据集，使用CNN-BiLSTM进行训练。识别视频中的人脸情绪。

本模型用于复旦大学

本库提供了5个基本模型
1. ResNet18 用于提取图像空间特征
2. ResNet18_BiLSTM 用于处理空间和时序特征
3. ResNet181D 用于提取MFCC特征
4. ResNet181D_BiLSTM 用于处理MFCC和时序特征
5. 融合ResNet18_BiLSTM和ResNet181D_BiLSTM的多模态模型

# 本模型使用例子

https://modelscope.cn/studios/wanbiao/split-video-emotion/summary

# 本模型微调方法

1. 将视频保存在一个文件夹中，命名为：序号_名字_类别_XX.flv
2. 运行CNNBiLSTM_SortVideo.py对其进行重新排序
3. 运行CNNBiLSTM_Extraxx.py对其进行帧抽取
4. 运行CNNBiLSTM_Imag2Pickle.py将抽取的帧进行打包，可以进行分包操作
5. 运行ipynb训练相应的模型即可
