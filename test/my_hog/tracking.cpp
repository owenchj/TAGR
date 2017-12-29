#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/video/tracking.hpp"
#include <vector>
#include <memory>
#include <map>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>

using namespace cv;
using namespace std;

#define IMG_WIDTH  352
#define IMG_HEIGHT 288

#define min(x) (x < 0 ? 0 : x)
#define Wmax(x) (x > IMG_WIDTH ? IMG_WIDTH : x)
#define Hmax(x) (x > IMG_HEIGHT ? IMG_HEIGHT : x)


#define MY_WIDTH  40
#define MY_HEIGHT 40
#define CELL_WIN 8
#define VECT8_NUM MY_HEIGHT*MY_WIDTH/(CELL_WIN*CELL_WIN)
#define VECT16_NUM (MY_HEIGHT/CELL_WIN - 1) * (MY_WIDTH/CELL_WIN - 1)

float vect16_feature[4][4][36];

Mat roi_rectangle(Mat input, Point2i p, int w, int h) {
    Point2i a(min(p.y - w), min(p.x-h));
    Point2i b(Wmax(p.y + w), Hmax(p.x+h));

    return input( Rect(a, b) );
}

float HOG(Mat img, int feature) {
    float vect8[5][5][9];
    float vect16[4][4][36];

    //![reduce_noise]
    GaussianBlur(img, img, Size(3,3), 0, 0, BORDER_DEFAULT );
    //![reduce_noise]

    // Calculate gradients gx, gy
    Mat gx, gy;
    Sobel(img, gx, CV_32F, 1, 0, 1);
    Sobel(img, gy, CV_32F, 0, 1, 1);

    // C++ Calculate gradient magnitude and direction (in degrees)
    Mat mag, angle;
    cartToPolar(gx, gy, mag, angle, 1);

    // Display frame.
    // imshow("img", img);
    // imshow("gx", gx);
    // imshow("gy", gy);
    // imshow("mag", mag);
    // imshow("angle", angle);

    // Initialization
    for(int j = 0; j < 5; j++)
        for(int i = 0; i < 5; i++)
            for(int k = 0; k < 9; k++)
                vect8[j][i][k] = 0;

    for(int j = 0; j < 5; j++)
        for(int i = 0; i < 5; i++)
        {
            //Mat cell = mag( Rect(Point2i(8*j, 8*i), Point2i(8*(j+1), 8*(i+1))) );
            for(int m = 0; m < 8; m++)
                for(int n = 0; n < 8; n++)
                {
                    //cout << ((8*j+m)*40 + 8*i+n) <<' ' << mag.at<float>((8*j+m), 8*i+n) << endl;
                    int ang = (unsigned int)angle.at<float>((8*j+m), 8*i+n);
                    int rest = ang % 20;
                    int result = ang / 20;

                    vect8[j][i][result % 9] += mag.at<float>((8*j+m), 8*i+n) * ((float)1 - (float)rest / (float)20);
                    vect8[j][i][(result+1) % 9] += mag.at<float>((8*j+m), 8*i+n) * (float)rest / (float)20;
                }
        }

    // Print vector 8*8
    // for(int j = 0; j < 5; j++)
    //     for(int i = 0; i < 5; i++)
    //     {
    //         cout << '(' << j << ',' << i << ')'<< endl;
    //         for(int k = 0; k < 9; k++)
    //             cout << vect8[j][i][k] << ' ';
    //         cout << endl;
    //     }


    // Save to vector 16*16
    for(int j = 0; j < 4; j++)
        for(int i = 0; i < 4; i++)
        {
            for(int m = 0; m < 2; m++)
                for(int n = 0; n < 2; n++)
                {
                    for(int k = 0; k < 9; k++)
                        vect16[j][i][(2*m + n) * 9 + k] = vect8[j+m][i+n][k];
                }
        }

    // Normalization
    for(int j = 0; j < 4; j++)
        for(int i = 0; i < 4; i++)
        {
            float sum = 0;
            for(int k = 0; k < 36; k++)
            {
                sum+= vect16[j][i][k] * vect16[j][i][k];
            }

            float norm = sqrt(sum);
            for(int k = 0; k < 36; k++)
            {
                vect16[j][i][k] /= norm;
            }
        }

    cout << endl;
    // Print vector 16*16
    // for(int j = 0; j < 4; j++)
    //     for(int i = 0; i < 4; i++)
    //     {
    //         cout << '(' << j << ',' << i << ')'<< endl;
    //         for(int k = 0; k < 36; k++)
    //         {
    //             cout << vect16[j][i][k] << ' ';
    //             if (k != 0 && (k+1) % 9 ==0)
    //                 cout << endl;
    //         }
    //         cout << endl;
    //     }

    if(feature)
    {
        for(int j = 0; j < 4; j++)
            for(int i = 0; i < 4; i++)
                for(int k = 0; k < 36; k++)
                    vect16_feature[j][i][k] = vect16[j][i][k];

        return 0;
    } else {
        float sum = 0;
        for(int j = 0; j < 4; j++)
            for(int i = 0; i < 4; i++)
                for(int k = 0; k < 36; k++)
                {
                    sum += vect16_feature[j][i][k] * vect16[j][i][k];
                }

        return sum;
    }
}

int main(int argc, char **argv)
{
    // C++ gradient calculation.
    // Read image
    Mat img = imread("a0.jpg");
    //img.convertTo(img, CV_32F, 1/255.0);

    // Feature image
    HOG(img, 1);

    Mat test = imread("test.jpg");

    stringstream sname;
    string name = "result_";
    string type = ".jpg";
    int count = 0;

    for(int y = 20; y < test.rows - 20; y+=20)
        for(int x = 20; x < test.cols - 20; x+=20)
        {
            Mat patch88 = roi_rectangle(test, Point2i(y, x), 20, 20);

            float diff = HOG(patch88, 0);
            cout << "Count :=" << count << "    Viriation :=" << diff << endl;

            // Save patches
            sname<< name << setprecision(3) << count <<type;
            imwrite(sname.str(), patch88);
            sname.str("");
            count++;

        }


    // Exit if ESC pressed.
    waitKey(10000);
}
