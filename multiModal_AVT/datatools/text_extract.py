import multiprocessing
import time

import cv2
import os
import shutil
from glob import glob

import imageio
import easyocr
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def main_2nd(data_root='/home/insomnia/Video_Position/data/process_data/'):
    mkdir_or_exist('ocr_text_trn_2nd')
    ocr_finish = load_json('ocr_text_trn.json')
    print('上次已经识别完成的样本是 {} 个'.format(len(ocr_finish)))

    # dirs = ['data_A/train', 'data_A/test', 'data_B']
    dirs = ['data_A/train', 'data_B']
    paths = []
    for dir in dirs:
        video_paths = glob(data_root + dir + '/*.mp4')
        paths.extend(video_paths)
    print('数据集文件夹中总共有样本 {} 个'.format(len(paths)))

    # 去除已经识别过的样本
    temp = []
    for item in paths:
        name = item.split('/')[-1].split('.')[0]
        if name not in ocr_finish.keys():
            temp.append(item)
    paths = temp
    print('该次ocr待识别的样本数量为 {} 个'.format(len(paths)))

    num_workers = 6
    step = len(paths) // num_workers
    process = []
    for i in range(num_workers):
        if i == num_workers - 1:
            datas = paths[i * step:]
        else:
            datas = paths[i * step: (i + 1) * step]
        p = multiprocessing.Process(target=work_2nd, args=(i, datas,))
        p.daemon = True
        p.start()
        process.append(p)

    for p in process:
        p.join()

    return


def work_2nd(work_id, datas):
    print('work{} start!'.format(work_id))
    for i, path in enumerate(datas):
        t_start = time.time()
        # print(path)
        name = path.split('/')[-1].split('.')[0]
        res = get_text_from_video(path)
        write_json(res, 'ocr_text_trn_2nd/' + name + '.json')
        t_end = time.time()
        if (i + 1) % 1 == 0:
            print('work_id:{}, {:.2f}%, eta:{:.2f}h'.format(work_id, i / len(datas) * 100,
                                                            (t_end - t_start) * (len(datas) - i - 1) / 3600))


def main(data_root='/home/insomnia/Video_Position/data/process_data/', data_dir=None, save_name='ocr_text_trn'):
    mkdir_or_exist(save_name)
    if data_dir is None:
        dirs = ['data_A/train', 'data_B']
        paths = []
        for dir in dirs:
            video_paths = glob(data_root + dir + '/*.mp4')
            paths.extend(video_paths)
    else:
        paths = glob(data_root + data_dir + '/*.mp4')#‘data_A/test’


    num_workers = 7
    step = len(paths) // num_workers
    process = []
    for i in range(num_workers):
        if i == num_workers - 1:
            datas = paths[i * step:]
        else:
            datas = paths[i * step: (i + 1) * step]
        p = multiprocessing.Process(target=work, args=(i, datas, save_name))
        p.daemon = True
        p.start()
        process.append(p)

    for p in process:
        p.join()

    return


def work(work_id, datas, save_name):
    print('work{} start!'.format(work_id))
    for i, path in enumerate(datas):
        t_start = time.time()
        # print(path)
        name = path.split('/')[-1].split('.')[0]
        res = get_text_from_video(path)
        write_json(res, save_name + '/' + name + '.json')
        t_end = time.time()
        if (i + 1) % 1 == 0:
            print('work_id:{}, {:.2f}%, eta:{:.2f}h'.format(work_id, i / len(datas) * 100,
                                                            (t_end - t_start) * (len(datas) - i - 1) / 3600))


def get_text_from_video(video_name, bs=8, save_dir=None):
    """
    :param video_name: 视频路径
    """
    # 开始读视频
    video_capture = cv2.VideoCapture(video_name)
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
    result = {}
    i = 0
    while True:
        success, frame = video_capture.read()
        if not success:
            # print('video is all read')
            break

        text = reader.readtext(frame)

        # 变换边界框
        if len(text) != 0:
            img_size = (frame.shape[0], frame.shape[1])
            for index, t in enumerate(text):
                # trans_text = (box_transform([t[0][0], t[0][2]], img_size), t[1], t[2])  # t[0]是ocr
                trans_text = (float_transform(t[0]),
                              [float(img_size[0]), float(img_size[1])],
                              t[1],
                              t[2])
                # 的边界框，给出了四个坐标值，这里只拿左上角和右下角两个坐标
                text[index] = trans_text
        # print(text)
        result[i] = text# [([ [x1, y1],[x2,y2] ], text, prob), (...)]
        # print(len(result))
        # print('='*20)
        i += 1

    return result


def float_transform(bbox):
    """
    字典中的int64类型无法写入json，用于转换easyocr的bbox的原始类型
    """
    float_bbox = []
    for box in bbox:
        f_box = [float(box[0]), float(box[1])]
        float_bbox.append(f_box)
    return float_bbox


def box_transform(box, img_size):
    """
    :param box: 原始边界框 list[list[int]]
    :param img_size: 原始图片尺寸 (h, w)
    :return: 中心比例边界框，相对于图片中心点的偏移量, list[list[float]]
    """
    h, w = img_size[0], img_size[1]
    center_y, center_x = h // 2, w // 2
    new_box = []
    for cord in box:
        new_cord = [0, 0]
        new_cord[0] = round((cord[0] - center_x) / w, 4)
        new_cord[1] = round((cord[1] - center_y) / h, 4)
        new_box.append(new_cord)
    return new_box


def box_invTrans(new_box, tgt_size):
    """
    :param new_box: 中心比例边界框
    :param tgt_size: 目标图像尺寸
    :return: 在目标图像上的边界框位置
    """
    h, w = tgt_size[0], tgt_size[1]
    center_y, center_x = h // 2, w // 2
    box = []
    for new_cord in new_box:
        cord = [0, 0]
        cord[0] = int(new_cord[0] * w) + center_x
        cord[1] = int(new_cord[1] * h) + center_y
        box.append(cord)
    return box


def write_json(data, save_path):
    with open(save_path, 'w') as fw:
        json.dump(data, fw)
    return


def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def merge_json(dir_path, save_path='ocr_text.json'):
    """
    把多进程下生成的所有json合并为一个总的json
    :param dir_path: 子json路径
    :param save_path: 保存路径
    """
    res = {}
    json_paths = glob(dir_path + '/*.json')
    for path in tqdm(json_paths):
        sub_data = load_json(path)
        name = path.split('/')[-1].split('.')[0]
        res[name] = sub_data
    write_json(res, save_path)
    return


def mkdir_or_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)


if __name__ == '__main__':
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    # video = '/home/insomnia/Video_Position/data/process_data/data_A/test/0rmGV3Va_start.mp4'
    # res = get_text_from_video(video)
    # print(res)
    # write_json(res, save_path='ttt.json')

    # 运行以下函数
    # main(data_dir='data_A/test', save_name='ocr_text_tst')
    # merge_json(dir_path='./ocr_text_tst', save_path='ocr_text_tst.json')
    # main(save_name='ocr_text_trn')
    # merge_json(dir_path='./ocr_text_trn', save_path='ocr_text_trn.json')


    # 若多进程下有进程中断，就运行下面代码来将因中断而未能识别的那部分样本补全识别
    # 示例，trn有进程中断
    main_2nd()
    merge_json(dir_path='./ocr_text_trn_2nd', save_path='ocr_text_trn_2nd.json')
    ocr1 = load_json('ocr_text_trn.json')
    ocr2 = load_json('ocr_text_trn_2nd.json')
    print(len(ocr1))
    print(len(ocr2))
    for name in ocr2.keys():
        ocr1[name] = ocr2[name]
    write_json(ocr1, 'ocr_text_trn_full.json')
    print(len(ocr1))

    # data = load_json('ocr_text.json')
    # print(data)
