#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"
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

float HOG_COLOR(Mat img, int feature) {
    float vect8[5][5][8];
    float vect16[4][4][32];
    Mat gray;

    //![reduce_noise]
    //GaussianBlur(img, img, Size(3,3), 0, 0, BORDER_DEFAULT );
    //![reduce_noise]
    cvtColor(img, gray, COLOR_BGR2GRAY );

    // Initialization
    for(int j = 0; j < 5; j++)
        for(int i = 0; i < 5; i++)
            for(int k = 0; k < 8; k++)
                vect8[j][i][k] = 0;

    for(int j = 0; j < 5; j++)
        for(int i = 0; i < 5; i++)
        {
            //Mat cell = mag( Rect(Point2i(8*j, 8*i), Point2i(8*(j+1), 8*(i+1))) );
            for(int m = 0; m < 8; m++)
                for(int n = 0; n < 8; n++)
                {
                    int ang = (unsigned int)gray.at<uchar>((8*j+m), 8*i+n);
                    int rest = ang % 32;
                    int result = ang / 32;

                    vect8[j][i][result % 8] += (float)gray.at<uchar>((8*j+m), 8*i+n) * ((float)1 - (float)rest / (float)32);
                    vect8[j][i][(result+1) % 8] += (float)gray.at<uchar>((8*j+m), 8*i+n) * (float)rest / (float)32;
                }
        }

    // Print vector 8*8
    // for(int j = 0; j < 5; j++)
    //     for(int i = 0; i < 5; i++)
    //     {
    //         cout << '(' << j << ',' << i << ')'<< endl;
    //         for(int k = 0; k < 8; k++)
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
                    for(int k = 0; k < 8; k++)
                        vect16[j][i][(2*m + n) * 8 + k] = vect8[j+m][i+n][k];
                }
        }

    // Normalization
    for(int j = 0; j < 4; j++)
        for(int i = 0; i < 4; i++)
        {
            float sum = 0;
            for(int k = 0; k < 32; k++)
            {
                sum+= vect16[j][i][k] * vect16[j][i][k];
            }

            float norm = sqrt(sum);
            for(int k = 0; k < 32; k++)
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
    //         for(int k = 0; k < 32; k++)
    //         {
    //             cout << vect16[j][i][k] << ' ';
    //             if (k != 0 && (k+1) % 8 ==0)
    //                 cout << endl;
    //         }
    //         cout << endl;
    //     }

    if(feature)
    {
        for(int j = 0; j < 4; j++)
            for(int i = 0; i < 4; i++)
                for(int k = 0; k < 32; k++)
                    vect16_feature[j][i][k] = vect16[j][i][k];

        return 0;
    } else {
        float sum = 0;
        for(int j = 0; j < 4; j++)
            for(int i = 0; i < 4; i++)
                for(int k = 0; k < 32; k++)
                {
                    sum += vect16_feature[j][i][k] * vect16[j][i][k];
                }

        return sum;
    }
}

int main(int argc, char **argv)
{
    HOGDescriptor hog;
    hog.winSize = Size(40, 40);
    Mat gray;
    vector< float > descriptors;
    vector< float > descriptors_sum;
    vector< float > descriptors_other;

    // C++ gradient calculation.
    // Read image
    Mat img = imread("a0.jpg");
    //img.convertTo(img, CV_32F, 1/255.0);

    // Feature image
    //HOG(img, 1);
    cvtColor(img, gray, COLOR_BGR2GRAY );
    hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ) );

    for(int i=0; i< descriptors.size();i++)
        descriptors_sum.push_back(descriptors[i]);

    img = imread("a1.jpg");
    cvtColor(img, gray, COLOR_BGR2GRAY );
    hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ) );

    for(int i=0; i< descriptors.size();i++)
        descriptors_sum[i] += descriptors[i];

    img = imread("a4.jpg");
    cvtColor(img, gray, COLOR_BGR2GRAY );
    hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ) );

    for(int i=0; i< descriptors.size();i++)
        descriptors_sum[i] += descriptors[i];

    img = imread("a10.jpg");
    cvtColor(img, gray, COLOR_BGR2GRAY );
    hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ) );

    for(int i=0; i< descriptors.size();i++)
        descriptors_sum[i] += descriptors[i];

    img = imread("a11.jpg");
    cvtColor(img, gray, COLOR_BGR2GRAY );
    hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ) );

    for(int i=0; i< descriptors.size();i++)
        descriptors_sum[i] += descriptors[i];

    img = imread("a12.jpg");
    cvtColor(img, gray, COLOR_BGR2GRAY );
    hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ) );

    for(int i=0; i< descriptors.size();i++)
        descriptors_sum[i] += descriptors[i];

    img = imread("a13.jpg");
    cvtColor(img, gray, COLOR_BGR2GRAY );
    hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ) );

    for(int i=0; i< descriptors.size();i++)
        descriptors_sum[i] += descriptors[i];

    img = imread("a14.jpg");
    cvtColor(img, gray, COLOR_BGR2GRAY );
    hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ) );

    HOG_COLOR(img, 1);

    for(int i=0; i< descriptors.size();i++)
    {
        descriptors_sum[i] += descriptors[i];
        descriptors_sum[i] /= (float)8;
    }

    cout << descriptors.size() << endl;

    Mat test = imread("test.jpg");

    stringstream sname;
    string name = "result_";
    string type = ".jpg";
    int count = 0;

    for(int y = 20; y < test.rows - 20; y+=20)
        for(int x = 20; x < test.cols - 20; x+=20)
        {
            Mat patch88 = roi_rectangle(test, Point2i(y, x), 20, 20);

            cvtColor(patch88, gray, COLOR_BGR2GRAY );
            hog.compute( gray, descriptors_other, Size( 8, 8 ), Size( 0, 0 ) );

            float sum = 0;
            for(int i = 0; i < descriptors.size(); i++)
                sum+= pow((descriptors_sum[i] - descriptors_other[i]), 2);

            //cout << "Count :=" << count << "    Viriation :=" << sum << endl;

            // Save patches
            sname<< name << setprecision(3) << count <<type;
            imwrite(sname.str(), patch88);
            sname.str("");
            count++;

        }

    // Exit if ESC pressed.
    waitKey(10000);
}
