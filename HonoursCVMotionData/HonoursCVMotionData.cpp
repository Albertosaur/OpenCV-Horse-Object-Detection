// HonoursCVMotionData.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
//#include "opencv2/opencv.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/videoio.hpp>
#include "opencv2/video/background_segm.hpp"
#include <iostream>
#include <cmath>
#include <fstream>

using namespace cv;
using namespace std;

/// Function header
void detectAndDisplay(Mat frame);
float dot(float ax, float ay, float bx, float by);
float mag(float ax, float ay);
void CalcAngles(std::vector<std::vector<cv::Point>>& contours, bool& change, cv::Mat& drawing, std::vector<cv::Vec4i>& hierarchy, int& counter, vector<float>& angle, string fileprefix);
void PushToVector(float& temp_angle, std::vector<float>& angle, int& counter);
void writeToTxt(vector<float>& angle, string filename);

/** Global variables */
CascadeClassifier leg_cascade;

vector<float> FLangle;
vector<float> BLangle;

Mat frame;
Mat out_frame;

int thresh = 100;
int frameWidth, frameHeight;

/**
 * @function main
 */

int main(int argc, char** argv)
{
    //video input
    const string source = "C:/Users/romai/Desktop/honours/HonoursCVMotionData/HonoursCVMotionData/videoOG_trim.avi";
    VideoCapture cap(source);

    if (!cap.isOpened())  // check if we succeeded
        return -1;
      
    frameWidth = cap.get(CAP_PROP_FRAME_WIDTH);
    frameHeight = cap.get(CAP_PROP_FRAME_HEIGHT);

	CommandLineParser parser(argc, argv,
		"{help h||}"
		"{leg_cascade|C:/Users/romai/Desktop/honours/horse-or-human/FrontLegCascade/cascade.xml|Path to cascade.}");

    String leg_cascade_name = samples::findFile(parser.get<String>("leg_cascade"));

    //-- 1. Load the cascades
 
    if (!leg_cascade.load(leg_cascade_name))
    {
        cout << "--(!)Error loading eyes cascade\n";
        return -1;
    };

    while (cap.read(frame))
    {
        //take frame from video
        cap >> frame;
        if (frame.empty())
            break;

        //apply classifier to frame
        detectAndDisplay(frame);

        if (waitKey(10) == 27)
        {
            break; // escape
        }
    }

    //output angles to console
    cout << FLangle.size() << "\n";
    cout << BLangle.size() << "\n";

    //write to files
    writeToTxt(FLangle, "FrontLegAngles.txt");
    writeToTxt(BLangle, "BackLegAngles.txt");

    cap.release();
    destroyAllWindows();

    return 0;
}

void detectAndDisplay(Mat frame)
{
    //grey scale frame
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    //-- Detect legs
    
    std::vector<Rect> legs;
    leg_cascade.detectMultiScale(frame_gray, legs);
    int counterFL = -1;
    int counterBL = -1;

    //apply contour for every detected object by classifier
    for (size_t i = 0; i < legs.size(); i++)
    {
        //skip detected objects in top half of frame
        if (legs[i].y < frameHeight / 2)
        {
            continue;
        }

        //Canny edge detection
        blur(frame, frame, Size(3, 3));

        out_frame = frame(legs[i]);

        Mat canny_output;
        Canny(out_frame, canny_output, 150, thresh * 2);

        /// Find contours
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(canny_output, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
        
        /// Draw contours
        Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);

        float end = -1;
        bool change = false;

        //use different vectors depending if object is on the left or right
        if (legs[i].x > frameWidth/2)
        {
            CalcAngles(contours, change, drawing, hierarchy, counterFL, FLangle, "frontLegFace_");
            FLangle.push_back(end);
            counterFL++;
        }
        else
        {
            CalcAngles(contours, change, drawing, hierarchy, counterBL, BLangle, "backLegFace_");
            BLangle.push_back(end);
            counterBL++;
        }
        
    }
   
    //-- Show what you got
    imshow("Capture - Legs detection", frame);
}

void CalcAngles(std::vector<std::vector<cv::Point>>& contours, bool& change, cv::Mat& drawing, std::vector<cv::Vec4i>& hierarchy, int& counter, vector<float>& angle, string fileprefix)
{
    for (size_t i = 0; i < contours.size(); i++)
    {
        //alternate between colours for contours
        Scalar color;
        if (change == true)
            color = Scalar(256, 0, 0);
        else
            color = Scalar(0, 256, 0);

        drawContours(drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0);

        imshow("Contours", drawing);

        //skip over contours vectors that have too few contours or if contour is too small 
        if (contours.size() <= 3 || contours.at(i).size() <= 5)
        {
            continue;
        }

        //get points at start and end of contour
        float p1[2], p2[2], p3[2], p4[2];

        p1[0] = contours.at(i).at(0).x;
        p1[1] = contours.at(i).at(0).y;

        p2[0] = contours.at(i).back().x;
        p2[1] = contours.at(i).back().y;

        //calculate dot product between points
        float temp_angle = std::acos(dot(p1[0], p1[1], p2[0], p2[1]) / (mag(p1[0], p1[1]) * mag(p2[0], p2[1])));

        PushToVector(temp_angle, angle, counter);
        
        //if not at back of vector, get start point of one contour and start of next vector
        if (contours.at(i) != contours.back())
        {
            p3[0] = contours.at(i).at(0).x;
            p3[1] = contours.at(i).at(0).y;

            p4[0] = contours.at(i + 1).at(0).x;
            p4[1] = contours.at(i + 1).at(0).y;

            temp_angle = std::acos(dot(p3[0], p3[1], p4[0], p4[1]) / (mag(p3[0], p3[1]) * mag(p4[0], p4[1])));
            
            PushToVector(temp_angle, angle, counter);
        }

        change = !change;
    }
    //output frame as png
    std::ostringstream filename;
    filename << fileprefix << counter << ".png";

    imwrite(filename.str(), drawing);
}

void PushToVector(float& temp_angle, std::vector<float>& angle, int& counter) //pushes values to angle vector  
{
    // clamps values to between 90 - 5 degrees
    if (temp_angle > 0.174533 && temp_angle < 1.570796)
    {
        angle.push_back(temp_angle);
        counter++;

        cout << angle[counter] << "\n";
    }
}

float dot(float ax, float ay, float bx, float by)  //calculates dot product of a and b
{
    return  ax * bx + ay * by;
}

float mag(float ax, float ay)  //calculates magnitude of a
{
    return std::sqrt(ax * ax + ay * ay);
}

void writeToTxt(vector<float>& angle, string filename)
{
    ofstream outputfile;
    outputfile.open(filename);
    for (int i = 0; i < angle.size(); i++)
        outputfile << angle[i] << "\n";
}