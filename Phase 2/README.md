# Lane Detection Project

## How to run the project
- Navigate to the project directory and open windows command prompt.
```cmd
> phase2.bat <input video> <output video> --debug <0/1> --phase <1/2>
```
- Example
```cmd
> phase2.bat yolo/test_videos/project_video.mp4 yolo/results/outbat.mp4 --debug 0 --phase 1
```
- Example (to open car detection with lane detection debuging)
```cmd
> phase2.bat yolo/test_videos/project_video.mp4 yolo/results/outbat.mp4 --debug 1 --phase 2
```
___________________________

## Pipeline Steps for lane detection
1) Combined thresholding:
    - Apply sobel in x direction and apply a threshold.
    - Calculate direction of the gradient and apply an angle threshold.
    - Apply threshold on R, G, S, L channels and apply the combined mask on the input image.
2) Perspective warp to get bird eye view for the lanes.
3) Sliding window algorithm to detect the lane lines pixels and fit a second order polynomial to them.
4) Calculate the radius of curvature for the lane lines.
5) Draw the lane lines and highlight the lane.

## Car Detection
- We use YOLOv4 algorithm. We downloaded the pre-trained [weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) and used this [config file](https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/cfg/yolov4.cfg).

___________________________

## Repo Link

https://github.com/amrehab-98/Lane-Detection