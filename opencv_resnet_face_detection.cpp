#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#include "opencv_resnet_face_detection.h"

using namespace cv;
using namespace std;
using namespace cv::dnn;

bool resnet_face_init (dnn::Net &facenet)
{
	string modelConfiguration("deploy.prototxt");
	string modelBinary("res10_300x300_ssd_iter_140000.caffemodel");
    facenet = readNetFromCaffe(modelConfiguration, modelBinary);
	if (facenet.empty()) return false;
	return true;
}

int resnet_face_detect (dnn::Net &facenet, Mat &frame, std::vector<Rect> &rects, std::vector<float> &confidences, double confidenceThreshold=0.5)
{
	const size_t inWidth = 300;
	const size_t inHeight = 300;
	const double inScaleFactor = 1.0;
	const Scalar meanVal(104.0, 177.0, 123.0);
    Mat inputBlob = blobFromImage(frame, inScaleFactor, Size(inWidth, inHeight), meanVal, false, false);
    facenet.setInput(inputBlob, "data"); //set the network input
    Mat detection = facenet.forward("detection_out"); //compute output
    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	rects.clear();
	confidences.clear();
    Rect whole(0,0,frame.cols,frame.rows);
	int cnt = 0;
	for (int i = 0; i < detectionMat.rows; i++) {
		float confidence = detectionMat.at<float>(i, 2);
		if (confidence > confidenceThreshold) {
			int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
			int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
			int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
			int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
			Rect rect((int)xLeftBottom, (int)yLeftBottom,(int)(xRightTop - xLeftBottom),(int)(yRightTop - yLeftBottom));
            if ((rect & whole) == rect) {
			    rects.push_back(rect);
			    confidences.push_back(confidence);
    			cnt++;
            }
		}
	}
	return cnt;
}

bool resnet_double_check (dnn::Net &facenet, Mat &frame)
{
    Mat img; 
    cvtColor(frame,img,CV_GRAY2BGR);
    std::vector<Rect> faces;
    std::vector<float> confidences;
    int n = resnet_face_detect(facenet,img,faces,confidences);
    Rect whole(0,0,img.cols,img.rows);
    for (int i=0; i<n; i++) {
        float coverage = (100.0f*(faces[i]&whole).area())/img.total();
        if (coverage < 33) {
            //cout << "coverage " << coverage << endl;
            continue;
        }
        Point center((faces[i].tl()+faces[i].br())/2);
        Point wholeCenter(img.cols/2,img.rows/2);
        double distanceX = 100*fabs(center.x-wholeCenter.x)/img.cols;
        double distanceY = 100*fabs(center.y-wholeCenter.y)/img.rows;
        double distance = std::max(distanceX,distanceY);
        if (distance > 12.5) {
            //cout << "distance " << distance << endl;
            continue;
        }
        return true;
    }
    return false;
}

static dnn::Net facenet;

bool resnet_face_init ()
{
	return resnet_face_init(facenet);
}

int resnet_face_detect (Mat &frame, std::vector<Rect> &rects, std::vector<float> &confidences, double confidenceThreshold)
{
	return resnet_face_detect(facenet,frame,rects,confidences,confidenceThreshold);
}

int resnet_double_check (Mat &frame)
{
	return resnet_double_check(facenet,frame);
}
