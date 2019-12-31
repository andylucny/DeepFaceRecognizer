#include <opencv2/face.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

#include <dlib/opencv.h> 
#include <dlib/image_processing.h> 
#include <dlib/image_processing/frontal_face_detector.h> 

#include "dlib_resnet_face_recognition.h"
#include "opencv_resnet_face_detection.h"
#include "speak.h"
#include "lang.h"

#include "listdir.h"
#include "pathutils.h"

using namespace std;
using namespace cv;
using namespace cv::face;
using namespace dlib;

void alignFace(Mat &imFace, Mat &alignedImFace, std::vector<Point2f> &landmarks, bool alsoLandmarks=false) 
{
    float l_x = landmarks[39].x;
    float l_y = landmarks[39].y;
    float r_x = landmarks[42].x;
    float r_y = landmarks[42].y;
    float dx = r_x - l_x;
    float dy = r_y - l_y;
    double angle = atan2(dy, dx) * 180 / 3.1415927;
    Point2f eyesCenter;
    eyesCenter.x = (l_x + r_x) / 2.0;
    eyesCenter.y = (l_y + r_y) / 2.0;
    Mat M = Mat(2, 3, CV_32F);
    M = getRotationMatrix2D(eyesCenter, angle, 1);
    warpAffine(imFace, alignedImFace, M, imFace.size());
    if (alsoLandmarks)
        for (size_t i = 0; i < landmarks.size(); i++) {
            landmarks[i] = Point2f (
                M.at<double>(0, 0) * landmarks[i].x + M.at<double>(0, 1) * landmarks[i].y + M.at<double>(0,2),
                M.at<double>(1, 0) * landmarks[i].x + M.at<double>(1, 1) * landmarks[i].y + M.at<double>(1,2)
            );
        }
}

Mat getCroppedFaceRegion(Mat image, std::vector<Point2f> landmarks)
{
    int ytop = min(landmarks[19].y,landmarks[24].y);
    int ybottom = max(landmarks[6].y,landmarks[10].y);
    int xleft = min(landmarks[17].x,landmarks[4].x);
    int xright = min(landmarks[26].x,landmarks[12].x);
    Rect roi(xleft,ytop,xright-xleft+1,ybottom-ytop+1);
    return image(roi);
}

Mat getCroppedHeadRegion(Mat image, std::vector<Point2f> landmarks)
{
    int x1Limit = landmarks[0].x - (landmarks[36].x - landmarks[0].x);
    int x2Limit = landmarks[16].x + (landmarks[16].x - landmarks[45].x);
    int y1Limit = landmarks[27].y - 3*(landmarks[30].y - landmarks[27].y);
    int y2Limit = landmarks[8].y + (landmarks[30].y - landmarks[29].y);
    int x1 = max(x1Limit,0);
    int x2 = min(x2Limit, image.cols);
    int y1 = max(y1Limit, 0);
    int y2 = min(y2Limit, image.rows);
    cv::Rect selectedRegion = cv::Rect( x1, y1, x2-x1, y2-y1 );
    return image(selectedRegion);
}

std::vector<Point2f> getLandmarks (shape_predictor &landmarkDetector, Mat &img, Rect rect)
{
    std::vector<Point2f> points;
    dlib::cv_image<dlib::bgr_pixel> dlibIm(img);
    dlib::rectangle dlibRect(rect.x,rect.y,rect.x+rect.width,rect.y+rect.height);
    dlib::full_object_detection landmarks = landmarkDetector(dlibIm, dlibRect);
    for (size_t i = 0; i < landmarks.num_parts(); i++) {
        Point2f pt(landmarks.part(i).x(), landmarks.part(i).y());
        points.push_back(pt);
    }
    return points;
}

std::vector<Point2f> getFaceAndLandmarks (shape_predictor &landmarkDetector, Mat &img)
{
    std::vector<Rect> faces;
    std::vector<float> confidences;
    resnet_face_detect(img,faces,confidences);
    int kind = 0;
    if (faces.size() > 0) {
        size_t best = 0;
        for (size_t i=1; i<faces.size(); i++)
            if (abs(faces[best].x+faces[best].width/2-img.cols/2) > abs(faces[i].x+faces[i].width/2-img.cols/2)) best = i;
        Rect faceRect =  faces[best];
        return getLandmarks(landmarkDetector,img,faceRect);
    }
    else return std::vector<Point2f>();
}

int stabilize (int label, int n)
{
    static std::deque<int> labels;
    if (label < 0) {
        if (labels.size() > 0) labels.pop_front();
        return label;
    }
    else {
        labels.push_back(label);
        if (labels.size() > n) labels.pop_front();
        std::deque<int> values;
        std::copy(labels.begin(), labels.end(), std::back_inserter(values));
        std::nth_element(values.begin(), values.begin() + values.size()/2, values.end());
        return values[values.size()/2];
    }
}

int main(int argc,char** argv) {
    bool to_train = true;
    if (argc > 1) to_train = false;
	
    cout << "...started" << endl;

    resnet_face_init();
    cout << "...resnet face detector loaded" << endl;

    shape_predictor landmarkDetector;
    deserialize("shape_predictor_68_face_landmarks.dat") >> landmarkDetector;
    cout << "...face landmark detector loaded" << endl;

    
    resnet_descriptor_init();
    cout << "...resnet face recognition loaded" << endl;  

    std::vector<int> labels;
    std::vector<std::vector<float>> descriptors;
    std::vector<string> folders;
    string faceDatasetFolder = "faces";
    string descriptorDatasetFolder = "descriptors";
    if (to_train) {
		std::vector<std::vector<string>> filenames;
		listdirs(faceDatasetFolder, folders, filenames);
		printdirs(folders,filenames);

        std::vector<Mat> images;
        std::vector<string> imageFilenames;
		for (size_t i=0; i<folders.size(); i++) {
			for (size_t j=0; j<filenames[i].size(); j++) {
				Mat im = cv::imread(faceDatasetFolder+"/"+folders[i]+"/"+filenames[i][j],IMREAD_COLOR);
				std::vector<Point2f> landmarks = getFaceAndLandmarks(landmarkDetector, im);
				if (landmarks.size() < 68) continue;

				Mat gray, aligned;
				cvtColor(im, gray, COLOR_BGR2GRAY);
				alignFace(gray, aligned, landmarks, true);
				Mat imFace = getCroppedFaceRegion(aligned, landmarks);
				equalizeHist(imFace, imFace);

				printf("...preprocessed %s %s\n",folders[i].c_str(),filenames[i][j].c_str());

				cv::resize(imFace, imFace, Size(150, 150));

				images.push_back(imFace);    
				labels.push_back(i);
                imageFilenames.push_back(filenames[i][j].substr(0,filenames[i][j].find_last_of(".")));
			}
		}

		cout << "...faces loaded" << endl;

		for (size_t j = 0; j < images.size(); j++) {
			string path(descriptorDatasetFolder+"/"+folders[labels[j]]);
			createpath(path);
			imwrite(path+"/"+imageFilenames[j]+".png",images[j]);
			std::vector<float> descriptor = resnet_descriptor(images[j]);
			descriptors.push_back(descriptor);
			write_resnet_descriptor(path+"/"+imageFilenames[j]+".txt",labels[j],descriptor);
			printf("...descripted %s %d\n",folders[labels[j]].c_str(),labels[j]);
		}

		cout << "...trained" << endl;
    }
    else {
        std::vector<std::vector<string>> filenames;
        listdirs(descriptorDatasetFolder, folders, filenames, true);
        printdirs(folders,filenames);

        for (size_t i=0; i<folders.size(); i++) {
            for (size_t j=0; j<filenames[i].size(); j++) {
                int label;
                std::vector<float> descriptor;
                read_resnet_descriptor(descriptorDatasetFolder+"/"+folders[i]+"/"+filenames[i][j],label,descriptor);
                descriptors.push_back(descriptor);
                labels.push_back(label);
                cout << "... " << descriptorDatasetFolder+"/"+folders[i]+"/"+filenames[i][j] << " " << label << ". " << descriptor.size() << endl;
            }
        }

        cout << "...descriptors loaded" << endl;
    }

    int rightLabel = -1;
	speak_init(0,"en");
	speak("I am robot");
    int said = -1;
    int lastOptLabel = -1;
    int lastLabel = -1;
    int see = 0;
	
    VideoCapture camera(0);
    Mat frame;
    while (frame.empty()) camera >> frame;
    
	int tim=0, lasttim=0;
    for (;;) {

        camera >> frame;
        Mat img = frame.clone();

        std::vector<Rect> faces;
        std::vector<float> confidences;
        resnet_face_detect(frame,faces,confidences);

        int kind = 0;
		Rect2d faceRect;
        if (faces.size() > 0) {
            size_t best = 0;
            for (size_t i=1; i<faces.size(); i++)
                if (abs(faces[best].x+faces[best].width/2-img.cols/2) > abs(faces[i].x+faces[i].width/2-img.cols/2)) best = i;
            faceRect =  faces[best];
            cv::rectangle(img,faceRect,Scalar(255,0,0),2);
            kind = 1;
        }
        //cout << " " << faceRect << " " << kind;

		tim = time(NULL);
        Mat imFace, imHead;
        if (kind > 0) {
			
            std::vector<Point2f> landmarks = getLandmarks(landmarkDetector, frame, faceRect);
            if (landmarks.size() == 68) {
                for (size_t k=0;k<landmarks.size();k++)
                    cv::circle(img,landmarks[k],2,cv::Scalar(0,0,255),FILLED);

                Mat gray, aligned;
                cvtColor(frame, gray, COLOR_BGR2GRAY);
                alignFace(gray, aligned, landmarks, true);
                imHead = getCroppedHeadRegion(aligned, landmarks);
                imFace = getCroppedFaceRegion(aligned, landmarks);
                equalizeHist(imFace, imFace);
                cv::imshow("imFace",imFace);

                std::vector<float> descriptor = resnet_descriptor(imFace);
                float minDistance = 1e9f;
                float minId = -1;
                float limitDistance = 0.5; 
                for (size_t k = 0; k < descriptors.size(); k++) {
                    float distance = norm(descriptor, descriptors[k],NORM_L2);
                    if (distance < limitDistance && distance < minDistance) {
                        minDistance = distance;
                        minId = k;
                    }
                }
                //cout << " predicted: " << minId << " distance " << minDistance;
                if (minId >= 0) {
                    int minLabel = labels[minId];
                    if (minLabel != lastLabel) {
                        if (!resnet_double_check(imHead)) minId = -1;
                    }
                }
                //cout << " confirmed: " << minId;
                if (minId >= 0 || rightLabel != -1) {
                    int minLabel = (minId >=0) ? labels[minId] : rightLabel;
                    //cout << " " << minLabel << " " << folders[minLabel];
                    putText(img,folders[minLabel],faceRect.tl()-Point2d(0,12),0,1.0,Scalar(0,0,255),2);
                    lastLabel = minLabel;
                    if (rightLabel != -1 && (minLabel != rightLabel || minId == -1)) {
                        int id = tim;
                        cv::imwrite(faceDatasetFolder+"/"+folders[rightLabel]+"/"+to_string(id)+".png",frame);
                        cv::imwrite(descriptorDatasetFolder+"/"+folders[rightLabel]+"/"+to_string(id)+".png",imFace);
                        write_resnet_descriptor(descriptorDatasetFolder+"/"+folders[rightLabel]+"/"+to_string(id)+".txt",rightLabel,descriptor);
                        descriptors.push_back(descriptor);
                        labels.push_back(rightLabel);
                    }
                    int optLabel = stabilize(minLabel,5);
                    if (optLabel == lastOptLabel) {
                        see++;
                        if (see >= 5) {
                            if (optLabel != said) {
                                if (exists(descriptorDatasetFolder+"/"+folders[optLabel]+"/voice.txt")) {
									ifstream f(descriptorDatasetFolder+"/"+folders[optLabel]+"/voice.txt");
									string voice;
									if (f >> voice) speakYouAre(voice);
									else speakYouAre(folders[optLabel]);
									f.close();
                                }
                                else speakYouAre(folders[optLabel]);
                                said = optLabel;
                            }
                        }
                        else if (see % 2 == 0) speak("e...");
                    }
                    else {
                        lastOptLabel = optLabel;
                        see=0;
                    }
                }
            }
			lasttim = tim;
        }
        else {
            //cout << " Faces not detected.";
        }

        //cout << " (" << rightLabel << ")" << endl;
        cv::imshow("Detected_shape",img);
        int key = waitKey(1);
        if (key == 27) break;
        else if (key == 's') {
            int id = tim;
            cv::imwrite(to_string(id)+".png",frame);
            if (!imFace.empty()) cv::imwrite(to_string(id)+"_.png",imFace);
        }
		
		std::string quest;
		if (listen(quest)) {
            string voice;
			if (listenWho(quest)) {
				said = -1;
				//cout << "who?" << endl;
			}
			else if (listenIam(quest,voice)) {
				string folder = usa(voice); //filename cannot contain any characters
				rightLabel = -1;
				for (size_t k=0; k<folders.size(); k++) {
					if (folder == folders[k]) {
						rightLabel = k;
						break;
					}
				}
				if (rightLabel == -1) {
					createpath(faceDatasetFolder+"/"+folder);
					createpath(descriptorDatasetFolder+"/"+folder);
					ofstream out(descriptorDatasetFolder+"/"+folder+"/voice.txt");
					out << voice;
					out.close();
					rightLabel = (int) folders.size();
					folders.push_back(folder);
					cout << "... " << folder << " created and associated with label " << rightLabel << endl;
				}
				cout << "... " << folder << " logged in (" << rightLabel << ")" << endl;
			}
        }
		if (tim >= lasttim+3) {
            if (rightLabel != -1) {
                cout << "... " << folders[rightLabel] << " logged out" << endl;
                rightLabel = -1;
            }
        }
        if (tim >= lasttim + 20) {
            see = 0;
            said = -1;
        }

    }
    return 0;
}
