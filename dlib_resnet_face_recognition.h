#ifndef DLIB_RESNET_FACE_RECOGNITON_H
#define DLIB_RESNET_FACE_RECOGNITON_H

#include <vector>
#include <string>
#include <opencv2/core.hpp>

void resnet_descriptor_init ();
std::vector<float> resnet_descriptor (cv::Mat &image);
void write_resnet_descriptor (std::string path, int label, std::vector<float> &descriptor);
void read_resnet_descriptor (std::string path, int &label, std::vector<float> &descriptor);

#endif //DLIB_RESNET_FACE_RECOGNITON_H