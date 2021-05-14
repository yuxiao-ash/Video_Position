import os
import torch
import imageio
import numpy as np
from tqdm import tqdm

from model import VideoNet
from utils import read_video


data_root = '/home/insomnia/Video_Position/data/data_A/test/'
audio_predict_file = './log/2submission.csv'
video_predict_file = './log2/submission.csv'
video_weight_path = './log2/videonet_bestmodel_epoch2.pth'


img_size = (224, 224)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model = VideoNet().cuda()
model.load_state_dict(torch.load(video_weight_path)['model'])

with open(audio_predict_file, 'r') as file:
    lines = file.readlines()

new_lines = []
for line in tqdm(lines):
    video_path = os.path.join(data_root, line.split(',')[0])
    second = float(line.strip().split(',')[2])
    position = int(second * 25)
    images = read_video(video_path, img_size)
    if position - 250 < 0:
        videos = images[0:position+250]
        for i in range(500 - (position + 250)):
            videos.insert(0, np.zeros((img_size[0], img_size[1], 3), dtype=float))
    elif position + 250 > len(images)-1:
        videos = images[position-250:]
        for i in range(500 - len(videos)):
            videos.append(np.zeros((img_size[0], img_size[1], 3), dtype=float))
    else:
        videos = images[position-250: position+250]
    videos = videos[::2]
    videos = np.stack(videos, axis=0)
    videos = torch.FloatTensor(videos).unsqueeze(0)
    with torch.no_grad():
        videos = videos.cuda()
        prediction = model(videos)
        model.eval()
        prediction = prediction.cpu().numpy()[0][0]
        final_second = second - 10 + prediction * 20
        print(second, final_second, prediction)
        new_lines.append('{},{},{}\n'.format(line.split(',')[0], line.split(',')[1], final_second))

with open(video_predict_file, 'w') as file:
    file.writelines(new_lines)

print('success')

