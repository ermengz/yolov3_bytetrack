import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import time

from paddle.inference import Config
from paddle.inference import create_predictor

class yolox_detector():
    def __init__(self, modelfile, params_file) -> None:
        if not os.path.exists(os.path.abspath(modelfile)): raise FileExistsError
        if not os.path.exists(os.path.abspath(params_file)): raise FileExistsError
        try:
            config = Config(modelfile, params_file)
            config.enable_memory_optim()
            config.set_cpu_math_library_num_threads(4)
            config.enable_mkldnn()
            self.predictor = create_predictor(config)
        except Exception as e:
            print(f"error: {e}")
            exit(-1)

    def predict(self, img, input_size):
        
        # pre_process image
        prepared, scale_factor = self.pre_process(img, input_size)
        # print(f"scale_factor={scale_factor}")

        inputs = [prepared, scale_factor]
        
        
        # prepare input data
        input_names = self.predictor.get_input_names()
        # print(f"input_names: {input_names},output_name={self.predictor.get_output_names()}")
        for i, name in enumerate(input_names):
            input_tensor = self.predictor.get_input_handle(name)
            # print(f"input[{i}] shape: {inputs[i].shape}")
            input_tensor.reshape(inputs[i].shape)
            # print("input_tensor:",img[i].shape)
            input_tensor.copy_from_cpu(inputs[i].copy())
        
        self.predictor.run()
        
        # get output data
        results = []
        output_names = self.predictor.get_output_names()
        for i, name in enumerate(output_names):
            # print(f"output[{i}] name: {name}")
            output_tensor = self.predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            # print(f"output shape: {output_data.shape}")
            results.append(output_data)
            
        return  results
        
    def pre_process(slef, img, input_size):
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(img, 
                                 (int(img.shape[1] * r), int(img.shape[0] * r)), 
                                 interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        padded_img = padded_img.transpose((2, 0, 1))  # hwc -> chw
        padded_img = padded_img[np.newaxis, :].astype(np.float32)
        
        scale_factor = [r,r]
        scale_factor = np.array(scale_factor).reshape((1, 2)).astype(np.float32)
        return padded_img, scale_factor
    
    def post_process(self,img,results,threshold=0.5):
        # print(f"results.shape={results}")
        for res in results[0]:
            cat_id, score, bbox = res[0], res[1], res[2:]
            if score < threshold:
                continue
            xmin, ymin, xmax, ymax = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
            print('category id is {},score={}, bbox is {}'.format(int(cat_id),score, bbox))
            cv2.rectangle(img,(xmin,ymax),(xmax,ymin),(255,0,0),2)
        cv2.namedWindow("detection",cv2.WINDOW_NORMAL)
        cv2.imshow("detection",img)
        cv2.waitKey(0)


if __name__ == "__main__":
    # model_file = f"E:\\ermengz\\ns_project\\code\\PaddleDetection\\output_inference\\yolox_m_300e_coco_official\\model.pdmodel"
    # params_file = f"E:\\ermengz\\ns_project\code\\PaddleDetection\\output_inference\\yolox_m_300e_coco\\model.pdiparams"
    
    model_file = f"E:\\ermengz\\ns_project\\code\\PaddleDetection\\output_inference\\yolox_s_300e_coco_official\\model.pdmodel"
    params_file = f"E:\\ermengz\\ns_project\\code\\PaddleDetection\\output_inference\\yolox_s_300e_coco_official\\model.pdiparams"
     
    model_file = f"E:\\ermengz\\ns_project\\code\\PaddleDetection\\output_inference\\yolox_s_300e_coco_custom-trt\\model.pdmodel"
    params_file = f"E:\\ermengz\\ns_project\\code\\PaddleDetection\\output_inference\\yolox_s_300e_coco_custom-trt\\model.pdiparams"
 
    detector = yolox_detector(modelfile=model_file, params_file=params_file)
    
    img = cv2.imread('test_images\\boat_1.jpg')# 
    netsize=[640,640]
    
    # warmup
    if 1:
        for i in range(10):detector.predict(img, netsize )
    
    iteration=1
    infer_start = time.time()
    for i in range(iteration):  
        result = detector.predict(img, netsize )
    infer_end = time.time()
    avg_latency = (infer_end - infer_start) / iteration * 1000 # ms    # 时延
    print(f"avg_latency: {avg_latency} ms")
    
    detector.post_process(img, result)
    
        