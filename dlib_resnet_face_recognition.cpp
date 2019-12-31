#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <dlib/string.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include "dlib_resnet_face_recognition.h"

namespace dlib {

// ----------------------------------------------------------------------------------------
// The next bit of code defines a ResNet network. It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
    alevel0<
    alevel1<
    alevel2<
    alevel3<
    alevel4<
    max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
    input_rgb_image_sized<150>
    >>>>>>>>>>>>;

}

void resnet_load (dlib::anet_type &net)
{
    std::string imageRecognitionModelPath("dlib_face_recognition_resnet_model_v1.dat");
    dlib::deserialize(imageRecognitionModelPath) >> net;
}

void resnet_predict (dlib::anet_type &net, cv::Mat &im, std::vector<float> &descriptor)
{
    cv::Mat imRGB;
    cv::cvtColor(im, imRGB, cv::COLOR_GRAY2RGB);
    dlib::matrix<dlib::rgb_pixel> imDlib(dlib::mat(dlib::cv_image<dlib::rgb_pixel>(imRGB)));
    dlib::matrix<float,0,1> imageDescriptor = net(imDlib);
    std::vector<float> imageDescriptorVec(imageDescriptor.begin(), imageDescriptor.end());
    descriptor.swap(imageDescriptorVec);
}

void resnet_descriptor (dlib::anet_type &net, cv::Mat &image, std::vector<float> &descriptor)
{
    cv::Mat input;
    cv::resize(image,input,cv::Size(150,150));
    resnet_predict(net,input,descriptor);
}

static dlib::anet_type net;

void resnet_descriptor_init ()
{
	resnet_load(net);
}

std::vector<float> resnet_descriptor (cv::Mat &image) 
{
	std::vector<float> descriptor; 
	resnet_descriptor(net,image,descriptor);
	return descriptor;
}

void write_resnet_descriptor (std::string path, int label, std::vector<float> &descriptor) 
{
    std::ofstream out;
    out.open(path);
    out << descriptor.size() << " " << label << std::endl;
    for (size_t k=0; k<descriptor.size(); k++) out << descriptor[k] << " ";
    out << std::endl;
    out.close();
}

void read_resnet_descriptor (std::string path, int &label, std::vector<float> &descriptor) 
{
    std::ifstream in;
    in.open(path);
    int n=0;
    if (in >> n) {
        if (in >> label) {
            for (int k=0; k<n; k++) {
                float d;
                in >> d;
                descriptor.push_back(d);
            }
        }
    }
    in.close();
}
