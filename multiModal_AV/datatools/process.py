import cv2
import os
import shutil
from glob import glob
from tqdm import tqdm


def get_frame_from_video(video_name, interval, save_dir=None):
    """
    :param interval: 抽帧间隔
    :param video_name: 视频路径
    :param save_dir: 抽帧保存路径
    """

    # 保存图片的路径
    save_path = video_name.split('.mp4')[0] + '/'
    if save_dir is not None:
        save_path = save_path.split('/')[-2]
        save_path = os.path.join(save_dir, save_path)
    is_exists = os.path.exists(save_path)
    if not is_exists:
        os.makedirs(save_path)
        print('path of %s is build' % save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
        print('path of %s already exist and rebuild' % save_path)

    # 开始读视频
    video_capture = cv2.VideoCapture(video_name)
    i = 0
    j = 0

    while True:
        success, frame = video_capture.read()
        i += 1
        if i % interval == 0:
            # 保存图片
            j += 1
            save_name = os.path.join(save_path, str(j) + '.jpg')
            frame = resizeImage(frame, 112)
            cv2.imwrite(save_name, frame)
            # print('image of %s is saved' % save_name)
        if not success:
            print('video is all read')
            break

    return save_path


def resizeImage(frame, size=88):
    (h, w, _) = frame.shape
    if h >= w:
        # resize的新维度是 (w, h)
        return cv2.resize(frame, (int(size * w / h), size))
    else:
        return cv2.resize(frame, (size, int(size * h / w)))


def print_size_of_file(path):
    print('Size (MB):', os.path.getsize(path) / 1e6)


def print_size_of_dir(path):
    size = 0
    for root, dirs, files in os.walk(path):
        size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
    print('Size (MB):', size / 1e6)


def mkdir_or_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)


def split_train_val(root_path='/home/insomnia/Video_Position/data'):
    """
    :param root_path: 数据集文件夹位置
    :return: 生成划分的训练集和验证集标签文件
    """
    part_A = root_path + '/data_A/train/必选数据.txt'
    part_B = root_path + '/data_B/补充数据.txt'
    label_file = open(part_A, 'r', encoding='utf-8')
    label_lines = label_file.readlines()
    train = open('train.txt', 'w', encoding='utf-8')
    val = open('val.txt', 'w', encoding='utf-8')
    i = 0
    for line in tqdm(label_lines):
        if i % 7 != 0:
            train.write(line)
        else:
            val.write(line)
        i += 1

    label_file = open(part_B, 'r', encoding='utf-8')
    label_lines = label_file.readlines()
    i = 0
    for line in tqdm(label_lines):
        if i % 7 != 0:
            train.write(line)
        else:
            val.write(line)
        i += 1


if __name__ == '__main__':
    # video = './0Gv5nPPa_start.mp4'
    # print_size_of_file(video)
    # frame_path = get_frame_from_video(video, 25)#9.6M => 1.09M
    # print_size_of_dir(frame_path)
    # pass

    # root_path = '/home/insomnia/Video_Position/data'
    # save_dir = './frames_data/train'
    # mkdir_or_exist(save_dir)
    # for datadir in ['data_A/train', 'data_B']:
    #     path = os.path.join(root_path, datadir)
    #     videos = glob(path + '/*.mp4')
    #     for video in videos:
    #         _ = get_frame_from_video(video, 25, save_dir)
    #
    # save_dir = './frames_data/test'
    # mkdir_or_exist(save_dir)
    # path = os.path.join(root_path, 'data_A/test')
    # videos = glob(path + '/*.mp4')
    # for video in videos:
    #     _ = get_frame_from_video(video, 25, save_dir)
    # pass

    split_train_val()
