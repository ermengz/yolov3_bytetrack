
import cv2
import numpy as np

__all__=['resize','normalize','preprocess','draw_box']

def resize(img, target_size):
    """resize to target size"""
    if not isinstance(img, np.ndarray):
        raise TypeError('image type is not numpy.')
    im_shape = img.shape
    # print(im_shape)
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale_x = float(target_size) / float(im_shape[1])
    im_scale_y = float(target_size) / float(im_shape[0])
    img = cv2.resize(img, None, None, fx=im_scale_x, fy=im_scale_y)
    return img


def normalize(img, mean, std):
    img = img / 255.0
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    std = np.array(std)[np.newaxis, np.newaxis, :]
    img -= mean
    img /= std
    return img

def preprocess(img, img_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = resize(img, img_size)
    img = img[:, :, ::-1].astype('float32')  # bgr -> rgb
    img = normalize(img, mean, std)
    img = img.transpose((2, 0, 1))  # hwc -> chw
    img = img[np.newaxis, :]
    # print(img.shape)
    return img

def draw_box(img,result,threshold=0.5):
    
    for res in result[0]:
        cat_id, score, bbox = res[0], res[1], res[2:]
        if score < threshold:
            continue
        xmin, ymin, xmax, ymax = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
        print('category id is {},score={}, bbox is {}'.format(int(cat_id),score, bbox))
        cv2.rectangle(img,(xmin,ymax),(xmax,ymin),(255,0,0),2)
    cv2.namedWindow("detection",cv2.WINDOW_NORMAL)
    cv2.imshow("detection",img)
    cv2.waitKey(0)
