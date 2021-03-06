# <img src="https://i.loli.net/2021/05/22/TkJWnd7fQvsoqhR.png" alt="微信图片_20210515110227" style="zoom: 50%;" />   " 广工暖男团(毕业啦)"的《动漫视频片头片尾定位》方案

## 方案介绍

#### 模型设计

我们的基础模型主要有两个：

* AVTNet：audio、video和text_ROI的多模态神经网络模型，如下图所示。

![image-20210522193338469](https://i.loli.net/2021/05/22/DT6xiBXa2mOApHs.png)

1) 对音频提取mfcc特征，并送入我们设计的audioNet；

2) 对视频每一秒抽一帧，并放缩到（66, 88）的尺寸，同时使用easyocr来提取每一帧图像中的文字边界框，称为Text ROI，将视频帧序列和对应的Text ROI送入videoNet；

3) 由Backend将audioNet和videoNet的特征图进行融合（SE layer）并完成时间点位的回归（平滑L1损失）。

* AVTNet_YOLT：YOLT即you only look twice，修改了AVTNet的backend结构

1）对200个时刻的特征向量划分为20个网格；

2）先由SE_layer、conv1d和Linear得到大回归结果（L1损失）；

3）在大回归结果所在的网格上，使用conv1d + Linear进行小回归（L1损失）。

#### 模型融合

基于上述的AVTNet和AVTNet_YOLT，我们训练了3次，分别为：

* AVTNet训练。（*选取best_model.pth*）
* AVTNet + 数据标准化 + 2层的backend_conv1d训练。（*选取best_model.pth和epoch63.pth*）
* AVTNet_YOLT训练，数据划分种子args.data_seed设为2。（*选取best_model.pth和epoch63.pth*）

从以上3次训练中挑选出所述的5个模型进行K-means融合。

## 源代码说明

### 环境

```
python3
pytorch 1.6
torchvision 0.7
libsora
imageio
moviepy
numpy
opencv-python
easyocr
```

（PS. 训练需三卡2080ti）

### 步骤

#### 数据预处理

* 先用utils/extract_audio.py提取语音特征(其中data_root指定数据集路径, save_root指定处理数据集的保存路径)
* 用utils/extract_video.py从视频中抽帧(其中data_root指定数据集路径, save_root指定处理数据集的保存路径)
* 用datatools/text_extract.py从抽帧后的视频中提取文本信息(其中main()的data_root指定为上面的处理数据集的保存路径)(ps: 这一部分提取大概需要一天左右时间，已经在代码中提供了提取结果，如需要使用其他数据集进行验证需要把save_root和data_dir改成对应的数据集路径进行提取)
* 用utils/extract_textROI.py生成文本ROI并保存(dataset的data_root参数指定为上面的处理数据集的保存路径，直接运行则生成B榜测试集的textROI，生成A榜数据集则取消对应代码的注释)
* (注意上述的数据处理生成文件的存放路径需要统一， 即save_root需要一样)

#### 训练

* AVTNet训练

```
python avt_main.py --gpus=0,1 --lr=0.0003 --batch_size=16 --data_root=/home/insomnia/Video_Position/data/process_data/ --save_prefix=./work_dir/AVTNet_smooth/
```

其中--data_root用于指定训练的数据集路径，即处理后的数据集保存路径。



* AVTNet + 数据标准化 + 2层的backend_conv1d训练

```
python avt_normal_main.py --gpus=0,1 --lr=0.0003 --batch_size=16 --data_root=/home/insomnia/Video_Position/data/process_data/ --save_prefix=./work_dir/AVTNet_normal/ --if_normal
```



* AVTNet_YOLT训练，数据划分种子args.data_seed设为2

```
python avt_yolt_main.py --gpus=0,1,2 --lr=0.0003 --batch_size=21 --data_root=/home/insomnia/Video_Position/data/process_data/ --save_prefix=./work_dir/AVTNet_YOLT/ --data_seed 2
```

#### 测试

* AVTNet

```
python avt_main.py --gpus=0 --data_root=/home/insomnia/Video_Position/data/process_data/ --save_prefix=./work_dir/AVTNet_smooth/ --weights ./work_dir/AVTNet_smooth/avnet_bestmodel.pth
```

其中--weights用于指定对应权重所在路径

推理结果复制到fuse/submission_B_3文件夹下，用于融合

```
mkdir ./fuse/submission_B_3
cp ./work_dir/AVTNet_smooth/submission.csv ./fuse/submission_B_3/avt_smooth.csv
```



* AVTNet + 数据标准化 + 2层的backend_conv1d

best model 推理

```
python avt_normal_main.py --gpus=0 --data_root=/home/insomnia/Video_Position/data/process_data/ --save_prefix=./work_dir/AVTNet_normal/ --weights ./work_dir/AVTNet_normal/avnet_bestmodel.pth
```

推理结果用于融合

```
cp ./work_dir/AVTNet_normal/submission.csv ./fuse/submission_B_3/avt_normalize_best.csv
```



epoch63推理

```
python avt_normal_main.py --gpus=0 --data_root=/home/insomnia/Video_Position/data/process_data/ --save_prefix=./work_dir/AVTNet_normal/ --weights ./work_dir/AVTNet_normal/avnet_epoch63.pth
```

推理结果复制用于融合

```
cp ./work_dir/AVTNet_normal/submission.csv ./fuse/submission_B_3/avt_normalize_epoch63.csv
```



* AVTNet_YOLT

best model 推理

```
python avt_yolt_main.py --gpus=0 --data_root=/home/insomnia/Video_Position/data/process_data/ --save_prefix=./work_dir/AVTNet_YOLT/ --weights ./work_dir/AVTNet_YOLT/avnet_bestmodel.pth
```

推理结果复制用于融合

```
cp ./work_dir/AVTNet_YOLT/submission.csv ./fuse/submission_B_3/avt_YOLT_best.csv
```



epoch63 推理

```
python avt_yolt_main.py --gpus=0 --data_root=/home/insomnia/Video_Position/data/process_data/ --save_prefix=./work_dir/AVTNet_YOLT/ --weights ./work_dir/AVTNet_YOLT/avnet_epoch63.pth
```

推理结果复制用于融合

```
cp ./work_dir/AVTNet_YOLT/submission.csv ./fuse/submission_B_3/avt_YOLT_epoch63.csv
```



#### 融合

运行fuse/fuse.py进行上述5个模型推理结果的融合。

```
cd ./fuse
python fuse.py
```

