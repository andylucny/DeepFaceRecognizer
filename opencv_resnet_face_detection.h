#ifndef OPENCV_RESNET_FACE_DETECTION_H
#define OPENCV_RESNET_FACE_DETECTION_H

#include <vector>
#include <opencv2/core.hpp>

bool resnet_face_init();
int resnet_face_detect (cv::Mat &frame, std::vector<cv::Rect> &rects, std::vector<float> &confidences, double confidenceThreshold=0.5f);
int resnet_double_check (cv::Mat &frame);

#endif //OPENCV_RESNET_FACE_DETECTION_H