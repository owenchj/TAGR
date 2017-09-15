#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/video/tracking.hpp"
#include <vector>
#include <stdio.h>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;



int main(int argc, const char** argv)
{
    stringstream ss;

    string name = "sobel_";
    string type = ".jpg";
    int cnt = 0;
    // add your file name
    //VideoCapture cap("/home/jchen/Pictures/TAGR/samples/g01s04.avi");
    VideoCapture cap("/home/jchen/Pictures/TAGR/samples/train_front.avi");


    Mat flow, frame;
    // some faster than mat image container
    UMat  flowUmat, prevgray;
    Mat src_pre;

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

            /**/
            Mat diff;
            if (cnt == 0)
            {
                src_gray.copyTo(src_pre);
                src_gray.copyTo(diff);
            }
            else
            {
                absdiff(src_gray, src_pre, diff);
                threshold( diff, diff, 10, 255, 0);
            }
            src_gray.copyTo(src_pre);


            /**/

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
            threshold( grad, grad, 10, 255, 0);

            Mat mix;

            grad.copyTo(mix);
            for (int i = 0; i < grad.cols; i++)
                for (int j = 0; j < grad.rows; j++)
                {
                    if (grad.at<unsigned char>(j, i) == diff.at<unsigned char>(i, j))
                        mix.at<unsigned char>(j, i) = 255;
                    else
                        mix.at<unsigned char>(j, i) = 0;
                }

            // Canny 0
            // int thresh = 50;
            // int max_thresh = 255;
            // RNG rng(12345);

            // Mat canny_output;
            // vector<vector<Point> > contours;
            // vector<Vec4i> hierarchy;
            // Canny( grad, canny_output, thresh, thresh*2, 3 );
            // findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
            // vector<Moments> mu(contours.size() );
            // for( size_t i = 0; i < contours.size(); i++ )
            // { mu[i] = moments( contours[i], false ); }
            // vector<Point2f> mc( contours.size() );
            // for( size_t i = 0; i < contours.size(); i++ )
            // { mc[i] = Point2f( static_cast<float>(mu[i].m10/mu[i].m00) , static_cast<float>(mu[i].m01/mu[i].m00) ); }
            // Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );

            // vector<Point> mass;

            // for( size_t i = 0; i< contours.size(); i++ )
            // {
            //     if (contourArea(contours[i]) < 1) continue;
            //     Point2f a ((int)(mu[i].m10/mu[i].m00), (int)(mu[i].m01/mu[i].m00));
            //     mass.push_back(a);

            //     // printf(" * Contour[%d] - Area (M_x) = %.2f - Area (M_y) = %.2f - Area OpenCV: %.2f - Length: %.2f \n", (int)i, mu[i].m10/mu[i].m00, mu[i].m01/mu[i].m00, contourArea(contours[i]), arcLength( contours[i], true ) );
            //     // Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            //     // drawContours( drawing, contours, (int)i, color, 2, 8, hierarchy, 0, Point() );
            //     // circle( drawing, mc[i], 4, color, -1, 8, 0 );
            // }

            // for( size_t i = 0; i < mass.size(); i++ )
            // {
            //     printf("Area (M_x) = %d - Area (M_y) = %d \n", mass[i].x, mass[i].y );

            //     circle(grad, mass[i], 1, 100, 5);
            // }

            // Canny 1
            // int thresh = 50;
            // int max_thresh = 255;
            // RNG rng(12345);
            // Mat canny_output;
            // vector<vector<Point> > contours;
            // vector<Vec4i> hierarchy;

            // /// Detect edges using canny
            // Canny( grad, canny_output, thresh, thresh*2, 3 );
            // /// Find contours
            // findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

            // // Draw contours
            // Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
            // for( int i = 0; i< contours.size(); i++ )
            // {
            //     Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            //     drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
            // }

            //![display]
            //namedWindow("soble", WINDOW_AUTOSIZE);
            imshow( window_name, diff );
            waitKey(20);
            //![display]

            // ss<< name << setprecision(3) <<cnt<<type;
            // string filename = ss.str();
            // ss.str("");
            // imwrite(filename, grad);
            cnt++;
        }
    }
}
