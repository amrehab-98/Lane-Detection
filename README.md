# Lane Detection Project

## How to run the project
- Navigate to the project directory and open windows command prompt.
```cmd
> laneDetection.bat input_path output_path --debug 1
```
- Example
```cmd
> laneDetection.bat ./input/project_video.mp4 ./output/out.mp4
```
- Example (to open the project video in debugging mode)
```cmd
> laneDetection.bat ./input/project_video.mp4 ./output/out.mp4 --debug 1
```
___________________________

## Pipeline Steps
1) Combined thresholding:
    - Apply sobel in x direction and apply a threshold.
    - Calculate direction of the gradient and apply an angle threshold.
    - Apply threshold on R, G, S, L channels and apply the combined mask on the input image.
2) Perspective warp to get bird eye view for the lanes.
3) Sliding window algorithm to detect the lane lines pixels and fit a second order polynomial to them.
4) Calculate the radius of curvature for the lane lines.
5) Draw the lane lines and highlight the lane.

___________________________

## Repo Link

https://github.com/amrehab-98/Lane-Detection