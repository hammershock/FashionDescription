# FashionDescription
服饰图像描述生成模型

图像描述问题，即对于一个图像输入，输出一段文字描述信息，这通常被称作一个`Image to Text`问题。

我们使用不同的方式处理图像信息，并产生描述文本。
------
### 1.CNN+LSTM
这里使用普遍的图像处理方法，使用卷积神经网络将图像编码为固定长度的特征向量，然后将其作为LSTM解码器在每一个时间步的输入的一部分（即作为条件嵌入）。

------
### 2.ViT Encoder + Transformer Decoder
根据Vision Transformer(ViT)的思路，这里将图像视作一个序列。首先对图像分隔为若干个相等大小的图像块，使用相同的卷积运算对于每个图像块分别提取特征，并串联成为序列。
利用Transformer Decoder的跨模态注意力机制建模图像序列与文本序列的依赖关系，循环解码为目标序列。
