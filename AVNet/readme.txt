环境：
python3
pytorch 1.6
torchvision 0.7
libsora
imageio
moviepy
numpy
opencv-python

# 训练需要三卡2080ti

步骤：
1. 先用utils/extract_audio.py提取语音特征(其中data_root指定数据集路径, save_root指定处理数据集的保存路径)
   用utils/extract_video.py从视频中抽帧(其中data_root指定数据集路径, save_root指定处理数据集的保存路径)
2. 运行训练代码：
    python av_main.py --gpus='0' --lr=0.0003 --batch_size=24 --num_workers=8 --max_epoch=64 --use_amp=True --accumulation_step=1 --show_freq=10 --data_root='/home/insomnia/Video_Position/data/procrss_data/' --save_prefix='./work_dir/AVNet/' --test=False
    其中--data_root用于指定数据集路径
3. 验证测试集：
    python av_main.py --gpus='0' --lr=0.0003 --batch_size=24 --num_workers=8 --max_epoch=64 --use_amp=True --accumulation_step=1 --show_freq=10 --data_root='/home/insomnia/Video_Position/data/process_data/' --save_prefix='./work_dir/AVNet'/ --test=True --weights='./work_dir/AVNet/avnet_bestmodel.pth'
    其中--weights用于指定对应权重所在路径