
"""
    paddle框架下的yolov3模型引用示例
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import time
import os
from tqdm import tqdm

from infer import init_predictor, run
from utils.prepare import preprocess
from utils.prepare import draw_box
from utils.prepare import resize

def show_detection():
    """
        paddle inference 模型推理
    """
    img = cv2.imread('test_images\\boat_1.jpg')

    # model_file =f"yolov3_r50vd_dcn_270e_coco\model.pdmodel"
    # params_file = f"yolov3_r50vd_dcn_270e_coco\model.pdiparams"
    # im_size = 608

    model_file =f"inference_model\\yolov3_r50vd_dcn_270e_coco_official\\model.pdmodel"
    params_file = f"inference_model\\yolov3_r50vd_dcn_270e_coco_official\\model.pdiparams"
    im_size = 608
    
    predictor = init_predictor(model_file, params_file)
    
    # warmup
    if 0:
        for i in range(10):
            prepared_image = preprocess(img, im_size)
        
    iteration=1
    infer_start = time.time()
    for i in range(iteration):
        prepared_image = preprocess(img, im_size)
        # 给输入data增加一个batch维度
        # data = data[np.newaxis, :]
        # 缩放倍数
        scale_factor = np.array([im_size * 1. / img.shape[0], im_size * 1. / img.shape[1]]).reshape((1, 2)).astype(np.float32)
        # 输入data尺寸
        im_shape = np.array([im_size, im_size]).reshape((1, 2)).astype(np.float32)
        input_list = [im_shape, prepared_image, scale_factor]
        
        result = run(predictor, input_list)
        
    infer_end = time.time()
    avg_latency = (infer_end - infer_start) / iteration * 1000 # ms    # 时延
    print(f"avg_latency: {avg_latency} ms")
    
    draw_box(img, result, 0.5)

def make_label():
    """
        图片标注
    """
    
    model_file =f"inference_model\\yolov3_r50vd_dcn_270e_coco_512\\model.pdmodel"
    params_file = f"inference_model\\yolov3_r50vd_dcn_270e_coco_512\\model.pdiparams"
    im_size = 512
    
    predictor = init_predictor(model_file, params_file)
    
    img_path=R"E:\829\merge\froend"
    
    def get_imaegs(images_path):
        if not os.path.isdir(images_path):
            raise 
        files = os.listdir(images_path)
        images_list = [f for f in files if str(f).endswith(".jpg")]
        return images_list  

    imgs = get_imaegs(img_path)
    for imgf in tqdm(imgs):
        imgpath = os.path.join(img_path, imgf)
        img = cv2.imread(imgpath)
        height,width,c = img.shape
        prepared_image = preprocess(img, im_size)
        scale_factor = np.array([im_size * 1. / img.shape[0], im_size * 1. / img.shape[1]]).reshape((1, 2)).astype(np.float32)
        im_shape = np.array([im_size, im_size]).reshape((1, 2)).astype(np.float32)
        input_list = [im_shape, prepared_image, scale_factor]
        
        result = run(predictor, input_list)
        with open(imgpath.replace(".jpg",".txt"),"w") as f:
            for res in result[0]:
                cat_id, score, bbox = res[0], res[1], res[2:]
                if score < 0.3 :#or int(cat_id) !=8 :
                    continue
                xmin, ymin, xmax, ymax = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
                # print('category id is {},score={}, bbox is {}'.format(int(cat_id),score, bbox))
                
                w = 1.0*(xmax - xmin) / width
                h = 1.0*(ymax - ymin) / height
                cx = 1.0*xmin/width + w/2.
                cy = 1.0*ymin/height + h/2.
                tline = f"0 {cx:.2} {cy:.2} {w:.2} {h:.2}\n"
                f.write(tline)

def analysis_video():
    """
        model inference 视频
    """
    video_path = R"E:\829\误检\4DC8C92F.mp4"
    
    model_file =f"inference_model\\yolov3_r50vd_dcn_270e_coco_512\\model.pdmodel"
    params_file = f"inference_model\\yolov3_r50vd_dcn_270e_coco_512\\model.pdiparams"
    im_size = 512
    predictor = init_predictor(model_file, params_file)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileExistsError("文件打开错误")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))  # 30
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 1826
    print(height, width, fps, frameCount)


    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 'X','V','I','D' 简写为 *'XVID'
    vedioWrite = "output_video.avi"  # 写入视频文件的路径
    capWrite = cv2.VideoWriter(vedioWrite, fourcc, fps, (width, height), True)
    frameCount = 0
    ret, frame = cap.read() 
    while ret ==True:
        # frame
        # height,width,c = frame.shape
        prepared_image = preprocess(frame, im_size)
        scale_factor = np.array([im_size * 1. / frame.shape[0], im_size * 1. / frame.shape[1]]).reshape((1, 2)).astype(np.float32)
        im_shape = np.array([im_size, im_size]).reshape((1, 2)).astype(np.float32)
        input_list = [im_shape, prepared_image, scale_factor]
        result = run(predictor, input_list)
        for res in result[0]:
            cat_id, score, bbox = res[0], res[1], res[2:]
            if score < 0.3 :#or int(cat_id) !=8 :
                continue
            xmin, ymin, xmax, ymax = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
            print('category id is {},score={}, bbox is {}'.format(int(cat_id),score, bbox))
            cv2.rectangle(frame,(xmin,ymax),(xmax,ymin),(255,0,0),2)
        
        capWrite.write(frame)
        ret, frame = cap.read()
        frameCount+=1
        if frameCount >25*10:break # 10s
        print(frameCount) 
    
    cap.release()
    capWrite.release() 


def track_demo():
    """
        paddle inference + ByteTrack的行人跟踪
    """
    from tracker.byte_tracker import BYTETracker
    from tracker.visualize import plot_tracking
    from tracker.timer import Timer
    
    video_path = R"E:\ermengz\ns_project\code\ByteTrack-main\videos\palace.mp4"
    
    model_file =f"inference_model\\yolov3_r50vd_dcn_270e_coco_official\\model.pdmodel"
    params_file = f"inference_model\\yolov3_r50vd_dcn_270e_coco_official\\model.pdiparams"
    im_size = 608
    predictor = init_predictor(model_file, params_file)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileExistsError("文件打开错误")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))  # 30
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 1826
    print(height, width, fps, frameCount)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 'X','V','I','D' 简写为 *'XVID'
    vedioWrite = "person_track2.avi"  # 写入视频文件的路径
    capWrite = cv2.VideoWriter(vedioWrite, fourcc, fps, (width, height), True)
    
    # tracker
    class track_args:
        track_thresh=0.5
        track_buffer = 30
        match_thresh = 0.8
        mot20 = False
        aspect_ratio_thresh = 1.6
        min_box_area=10

    tracker = BYTETracker(track_args)
    timer = Timer()
    
    frameCount = 0
    ret, frame = cap.read() 
    while ret ==True:
        # frame
        if frameCount % 20 == 0:
            print('Processing frame {} ({:.2f} fps)'.format(frameCount, 1. / max(1e-5, timer.average_time)))
        prepared_image = preprocess(frame, im_size)
        scale_factor = np.array([im_size * 1. / frame.shape[0], im_size * 1. / frame.shape[1]]).reshape((1, 2)).astype(np.float32)
        im_shape = np.array([im_size, im_size]).reshape((1, 2)).astype(np.float32)
        input_list = [im_shape, prepared_image, scale_factor]
        result = run(predictor, input_list)
        if len(result[0]) >0:
            dets = []
            for res in result[0]:
                cat_id, score, bbox = res[0], res[1], res[2:]
                if int(cat_id) !=0 : # score < 0.3 or 
                    continue
                xmin, ymin, xmax, ymax = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
                dets.append([xmin, ymin, xmax, ymax,score])

            if len(dets)>0:
                dets = np.array(dets)
                print(dets.shape)
                online_targets = tracker.update(dets,img_info=[height,width],img_size=[height,width])
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > track_args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > track_args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                timer.toc()
                online_im = plot_tracking(
                        frame, online_tlwhs, online_ids, frame_id=frameCount + 1, fps=1. / timer.average_time
                    )   
            else:
                timer.toc()
                online_im = frame
        else:
            timer.toc()
            online_im = frame
         
        capWrite.write(online_im)
        ret, frame = cap.read()
        frameCount+=1

    
    cap.release()
    capWrite.release() 

        
if __name__ == '__main__':
    # make_label()
    # analysis_video()
    track_demo()


