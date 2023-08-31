# paddle下的yolov3+bytetrack行人跟踪

一个简单的检测 + 跟踪结合的使用示例。paddle框架下的inference模型推理，bytetrack的跟踪算法结合。两个算法解耦，可以使用不同的检测算法搭配跟踪算法。bytetrack具有使用方便、跟踪效率高，精确度高的优点。

## 环境

1. paddle inference的推理环境

    [Paddle Inference 简介](https://www.paddlepaddle.org.cn/inference/v2.5/guides/introduction/index_intro.html)

2. bytetrack的环境

    + [ByteTrack](https://github.com/ifzhang/ByteTrack/tree/main#combining-byte-with-other-detectors)
    + 安装cython_bbox, [安装方式](https://www.jb51.net/article/284250.htm)


### 代码

```python
from yolox.tracker.byte_tracker import BYTETracker
tracker = BYTETracker(args)
for image in images:
   dets = detector(image)               # 检测算法结果的格式为: [x1, y1, x2, y2, score]
   online_targets = tracker.update(dets, info_imgs, img_size)

# 2. 跟踪结果为
for t in online_targets:
    tlwh = t.tlwh       # 跟踪结果框 左上宽高
    tid = t.track_id    # 跟踪id
```

### result

<img src="assets/palace_demo.gif" width="600"/>