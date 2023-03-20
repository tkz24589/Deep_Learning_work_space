# original net：https://github.com/mouse1998mouse/RRunet
#将证书中的串改位置进行掩码标记，示例如下
## 输入
<img width=400, src="2.jpg" />

## 输出
 <img width=400, src="2.png" />
 
 目前训练无法达到这个效果，模型标记的篡改部分不全，咱训练不起来就是说。。

## 依赖
- Python 3.6 +
- PyTorch 1.0 +
- cudatoolkit 10.0 +
- mmcv

## 运行
 - './model_seg/*': it includes detailed implementations of 'ConvNext', 'Swin', 'uperHead'
 - 'train.py': you can use it to train your model
 - 'predict.py': you can use it to test
##
