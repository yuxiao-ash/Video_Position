import os
import time
import librosa
import youtube_dl
import numpy as np
import multiprocessing
from pydub import AudioSegment


label_path = './eval_segments.csv'
label_indices_path = './class_labels_indices.csv'
save_root = './eval/'
new_label_file = './audioset_eval_label.csv'

# 读取标签映射文件
label_indices = {}
with open(label_indices_path, 'r') as file:
    lines = file.readlines()[1:]
    for line in lines:
        label_indices.update({line.strip().split(',')[1]: line.strip().split(',')[0]})

# 读取标签文件
with open(label_path, 'r') as file:
    lines = file.readlines()[3:]
dataset = []
for line in lines:
    dataset.append(line.strip().split(','))


def GetAudio(line, save_path, f='140'):
    name = line.split('=')[1]
    # format setting
    ydl_opts = {
        'format': f,   # save as m4a
        'outtmpl': save_path + ".%(ext)s",
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([line])


def work(work_id, datas):
    print('work{} start!'.format(work_id))
    t_start = time.time()
    for i, data in enumerate(datas):
        # print('https://www.youtube.com/watch?v='+data[0])
        try:
            if os.path.exists('./temp{}.m4a'.format(work_id)):
                os.remove('./temp{}.m4a'.format(work_id))
            if os.path.exists('./temp{}.m4a.part'.format(work_id)):
                os.remove('./temp{}.m4a.part'.format(work_id))
            mfcc_path = os.path.join(save_root, data[0]+'_mfcc.npy')
            wav_path = os.path.join(save_root, data[0]+'.wav')
            GetAudio('https://www.youtube.com/watch?v='+data[0], './temp{}'.format(work_id))
            m4a_version = AudioSegment.from_file("./temp{}.m4a".format(work_id), "m4a")
            m4a_version = m4a_version[float(data[1])*1000 : float(data[2]) * 1000]
            m4a_version.export(wav_path, format="wav")
            # audio, freq = librosa.load(wav_path)
            # feature = librosa.feature.mfcc(audio, sr=freq)
            # np.save(mfcc_path, feature)
            label_file_part = './audioset_eval_label{}.csv'.format(work_id)
            with open(label_file_part, 'a') as file:
                data[3] = data[3].strip()[1:]
                data[-1] = data[-1].strip()[:-1] 
                labels = [label_indices[d.strip()] for d in data[3:]]
                file.write(','.join([data[0], '-'.join(labels)]) + '\n')
            os.remove('./temp{}.m4a'.format(work_id)) 
            t_end = time.time()
            if (i+1) % 1 == 0 or i == len(datas) - 1:
                print('work_id:{}, {:.2f}%, eta:{:.2f}h'.format(work_id, (i + 1) / len(datas) * 100, (t_end - t_start)/(i+1)*len(datas)/3600))
        except KeyboardInterrupt:
            break
        except:
            continue


if(__name__ == '__main__'):
    multiprocessing.freeze_support()

    # work(0, dataset)

    num_workers = 4
    step = len(dataset) // num_workers
    process = []
    for i in range(num_workers):
        if i == num_workers - 1:
            datas = dataset[i*step:]
        else:
            datas = dataset[i*step: (i+1)*step]
        p = multiprocessing.Process(target = work, args = (i, datas,))
        p.daemon = True
        p.start()
        process.append(p)

    for p in process:
        p.join()
