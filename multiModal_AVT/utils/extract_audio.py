import os
import cv2
import time
import glob
import random
import imageio
import skimage
import librosa
import numpy as np
from tqdm import tqdm
import multiprocessing
from moviepy.editor import AudioFileClip
from python_speech_features import mfcc, delta, logfbank

def extract_fbank(y, sr, nfilt):
    feat_fbank = logfbank(y, sr, nfilt=nfilt)
    feat_fbank_d = delta(feat_fbank, 2)
    feat_fbank_dd = delta(feat_fbank_d, 2)
    wav_feature = np.column_stack((feat_fbank, feat_fbank_d, feat_fbank_dd))
    return wav_feature

data_root='/home/insomnia/Video_Position/data/'
save_root='/home/insomnia/Video_Position/data/process_data/'
label_file1 = os.path.join(data_root, 'data_A/train/必选数据.txt')
label_file2 = os.path.join(data_root, 'data_B/补充数据.txt')
with open(label_file1, 'r') as file:
    lines = file.readlines()
    label_1 = []
    for line in lines:
        label_1.append([os.path.join('data_A/train/', line.split(' ')[0]+'.mp4'), float(line.split(' ')[1])])
with open(label_file2, 'r') as file:
    lines = file.readlines()
    label_2 = []
    for line in lines:
        label_2.append([os.path.join('data_B/', line.split(' ')[0]+'.mp4'), float(line.split(' ')[1])])
labels = label_1 + label_2

# 多进程子函数
def work(work_id, datas):
    print('work{} start!'.format(work_id))
    for i, data in enumerate(datas):
        t_start = time.time()
        video_path = os.path.join(data_root, data[0])
        audio_path = os.path.join(save_root, data[0][:-4] + '.wav')
        feature_path = os.path.join(save_root, data[0][:-4] + '_fbank.npy')
        my_audio_clip = AudioFileClip(video_path)
        my_audio_clip.write_audiofile(audio_path, verbose=False, logger=None)
        audio, freq = librosa.load(audio_path)
        # feature = librosa.feature.chroma_stft(audio, sr=freq, n_chroma=20)
        feature = logfbank(audio, freq, nfilt=20, nfft=551)
        np.save(feature_path, feature)
        t_end = time.time()
        if (i+1) % 10 == 0:
            print('work_id:{}, {:.2f}%, eta:{:.2f}h'.format(work_id, i / len(datas) * 100, (t_end - t_start)*(len(datas)-i-1)/3600))


num_workers = 8
step = len(labels) // num_workers
process = []
for i in range(num_workers):
    if i == num_workers - 1:
        datas = labels[i*step:]
    else:
        datas = labels[i*step: (i+1)*step]
    p = multiprocessing.Process(target = work, args = (i,datas,))
    p.daemon = True
    p.start()
    process.append(p)

for p in process:
    p.join()


# 测试集部分

# data_root = "/home/insomnia/Video_Position/data/data_A/test/"
# files = glob.glob(data_root + '*.mp4')

# def work(work_id, datas):
#     print('work{} start!'.format(work_id))
#     for i, data in enumerate(datas):
#         t_start = time.time()
#         video_path = data
#         audio_path = data[:-4] + '.wav'
#         mfcc_path = data[:-4] + '_mfcc.npy'
#         my_audio_clip = AudioFileClip(video_path)
#         my_audio_clip.write_audiofile(audio_path, verbose=False, logger=None)
#         audio, freq = librosa.load(audio_path)
#         mfcc = librosa.feature.mfcc(audio, freq)
#         np.save(mfcc_path, mfcc)
#         t_end = time.time()
#         if (i+1) % 10 == 0:
#             print('work_id:{}, {:.2f}%, eta:{:.2f}h'.format(work_id, i / len(datas) * 100, (t_end - t_start)*(len(datas)-i-1)/3600))


# num_workers = 8
# step = len(files) // num_workers
# process = []
# for i in range(num_workers):
#     if i == num_workers - 1:
#         datas = files[i*step:]
#     else:
#         datas = files[i*step: (i+1)*step]
#     p = multiprocessing.Process(target = work, args = (i,datas,))
#     p.daemon = True
#     p.start()
#     process.append(p)

# for p in process:
#     p.join()
