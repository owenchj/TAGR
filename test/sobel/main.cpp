#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/video/tracking.hpp"
#include <vector>
#include <stdio.h>
#include <iostream>


using namespace cv;
using namespace std;



int main(int argc, const char** argv)
{

    // add your file name
    VideoCapture cap("/home/jchen/Pictures/TAGR/samples/g01s20.avi");


    Mat flow, frame;
    // some faster than mat image container
    UMat  flowUmat, prevgray;

    for (;;)
    {

        bool Is = cap.grab();
        if (Is == false) {
            // if video capture failed
            cout << "Video Capture Fail" << endl;
            break;
        }
        else {

            Mat src;
            Mat src_gray;
            Mat grad;

            const char* window_name = "Sobel Demo - Simple Edge Detector";
            int scale = 1;
            int delta = 0;
            int ddepth = CV_16S;
            //![variables]

            // capture frame from video file
            cap.retrieve(src, CV_CAP_OPENNI_BGR_IMAGE);
            //resize(img, img, Size(640, 480));

            //![reduce_noise]
            GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
            //![reduce_noise]

            //![convert_to_gray]
            cvtColor( src, src_gray, COLOR_BGR2GRAY );
            //![convert_to_gray]

            //![sobel]
            /// Generate grad_x and grad_y
            Mat grad_x, grad_y;
            Mat abs_grad_x, abs_grad_y;

            /// Gradient X
            //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
            Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );

            /// Gradient Y
            //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
            Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
            //![sobel]

            //![convert]
            convertScaleAbs( grad_x, abs_grad_x );
            convertScaleAbs( grad_y, abs_grad_y );
            //![convert]

            //![blend]
            /// Total Gradient (approximate)
            addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
            //![blend]

            //![display]
            namedWindow("soble", WINDOW_AUTOSIZE);
            imshow( window_name, grad );
            waitKey(20);
            //![display]
        }
    }
}
