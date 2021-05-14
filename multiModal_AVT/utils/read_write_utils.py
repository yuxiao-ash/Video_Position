import cv2
import imageio


def resize_img_keep_ratio(image, target_size):
    old_size = image.shape[0:2]
    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
    new_size = tuple([int(i * ratio) for i in old_size])
    image = cv2.resize(image, (new_size[1], new_size[0]))
    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    img_new = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    return img_new


def read_video(filename, img_size):
        vid = imageio.get_reader(filename, 'ffmpeg')
        video = []
        for im in vid:
            video.append(resize_img_keep_ratio(im, img_size)/255.)
        return video

