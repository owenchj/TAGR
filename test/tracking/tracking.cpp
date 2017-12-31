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

#define MY_WIDTH  352
#define MY_HEIGHT 288

#define min(x) (x < 0 ? 0 : x)
#define Wmax(x) (x > MY_WIDTH ? MY_WIDTH : x)
#define Hmax(x) (x > MY_HEIGHT ? MY_HEIGHT : x)

// Const Parameters
const int thred_sobel = 50;
const int thred_diff = 10;

struct HOG_RES {
    Point2i local;
    Point2i global;
    float   distance;
    int     count;
};

//Point2i start(70, 100);
Point2i start(30, 30);

bool sortByDistance (const HOG_RES &i, const HOG_RES &j)
{
    return (i.distance < j.distance);
}

// Match in the first 5 min
int minMatch (const vector<HOG_RES> &gradient, const vector<HOG_RES> &color)
{
    for(int i = 0; i < color.size(); i++)
    {
        if(i < 5) {
            for(int j = 0; j < gradient.size(); j++)
            {
                if( j < 5)
                {
                    if (color[i].count  == gradient[j].count)
                        return j;
                }
            }
        } else {
            return -1;
        }
    }

    return -1;
}

Mat roi_rectangle(Mat input, Point2i p, int w, int h, Rect & rect) {
    Point2i a(min(p.y - w), min(p.x-h));
    Point2i b(Wmax(p.y + w), Hmax(p.x+h));

    rect.x = a.x;
    rect.y = a.y;
    rect.width = b.x - a.x;
    rect.height = b.y - a.y;

    return input(rect);
}

void draw_rectangle(Mat input, Point2i p, int w, int h) {
    Point2i a(p.y - w, p.x-h);
    Point2i b(p.y + w, p.x+h);
    rectangle(input, a, b, (255,255,255));
}

vector< float > descriptors_feature;
float HOG(Mat gray, int feature)
{
    HOGDescriptor hog;
    hog.winSize = Size(40, 40);
    vector< float > descriptors;

    if(feature)
    {
        hog.compute( gray, descriptors_feature, Size( 8, 8 ), Size( 0, 0 ) );
        return 0;
    }
    else
    {
        hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ) );
        float sum = 0;
        for(int i = 0; i < descriptors.size(); i++)
            sum+= pow((descriptors_feature[i] - descriptors[i]), 2);

        descriptors.clear();
        return sum;
    }
}

float descriptors_edge_feature[4][4][8];
float HOG_EDGE(Mat edge, int feature) {
    float vect8[5][5][2];
    float vect16[4][4][8];

    // Initialization
    for(int j = 0; j < 5; j++)
        for(int i = 0; i < 5; i++)
            for(int k = 0; k < 2; k++)
                vect8[j][i][k] = 0;

    for(int j = 0; j < 5; j++)
        for(int i = 0; i < 5; i++)
        {
            //Mat cell = mag( Rect(Point2i(8*j, 8*i), Point2i(8*(j+1), 8*(i+1))) );
            for(int m = 0; m < 8; m++)
                for(int n = 0; n < 8; n++)
                {
                    int ang = (unsigned int)edge.at<uchar>((8*j+m), 8*i+n);
                    int result = ang / 255;

                    vect8[j][i][result % 2]++;
                }
        }

    // Save to vector 16*16
    for(int j = 0; j < 4; j++)
        for(int i = 0; i < 4; i++)
        {
            for(int m = 0; m < 2; m++)
                for(int n = 0; n < 2; n++)
                {
                    for(int k = 0; k < 2; k++)
                        vect16[j][i][(2*m + n) * 2 + k] = vect8[j+m][i+n][k];
                }
        }

    // Normalization
    for(int j = 0; j < 4; j++)
        for(int i = 0; i < 4; i++)
        {
            float sum = 0;
            for(int k = 0; k < 8; k++)
            {
                sum+= vect16[j][i][k] * vect16[j][i][k];
            }

            float norm = sqrt(sum);
            for(int k = 0; k < 8; k++)
            {
                vect16[j][i][k] /= norm;
            }
        }

    if(feature)
    {
        for(int j = 0; j < 4; j++)
            for(int i = 0; i < 4; i++)
                for(int k = 0; k < 8; k++)
                    descriptors_edge_feature[j][i][k] = vect16[j][i][k];

        return 0;
    } else {
        float sum = 0;
        for(int j = 0; j < 4; j++)
            for(int i = 0; i < 4; i++)
                for(int k = 0; k < 8; k++)
                    sum += pow((descriptors_edge_feature[j][i][k] - vect16[j][i][k]), 2);
        return sum;
    }
}


float descriptors_color_feature[4][4][36];
float HOG_COLOR(Mat gray, int feature) {
    float vect8[5][5][8];
    float vect16[4][4][32];

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

    if(feature)
    {
        for(int j = 0; j < 4; j++)
            for(int i = 0; i < 4; i++)
                for(int k = 0; k < 32; k++)
                    descriptors_color_feature[j][i][k] = vect16[j][i][k];

        return 0;
    } else {
        float sum = 0;
        for(int j = 0; j < 4; j++)
            for(int i = 0; i < 4; i++)
                for(int k = 0; k < 32; k++)
                    sum += pow((descriptors_color_feature[j][i][k] - vect16[j][i][k]), 2);
        return sum;
    }
}


Point2i hog_head(Mat gray, Mat edge, Point2i center, string name) {
    //![segmentation]
    /// segmentation
    // Extra intresting segmention region
    Rect roi_rect;
    Mat roi_seg = roi_rectangle(gray, center, 40, 40, roi_rect);

    Rect edge_rect;
    Mat edge_seg = roi_rectangle(edge, center, 40, 40, edge_rect);

    //![HOG]
    /// HOG to find head
    stringstream sname;
    string patch_name = "hog_";
    string type = ".jpg";
    int count = 0;

    vector<HOG_RES> hog_gradient;
    vector<HOG_RES> hog_color;
    vector<HOG_RES> hog_edge;

    // cout << roi_seg.rows << ' ' << roi_seg.cols<< endl;
    // cout << roi_rect << endl;

    // Compute hog
    if (roi_seg.rows >= 40 && roi_seg.cols >= 40) {
        for(int y = 20; y <= roi_seg.rows - 20; y += 10)
            for(int x = 20; x <= roi_seg.cols - 20; x += 10)
            {
                Mat patch88 = roi_seg( Rect(Point2i(x - 20, y - 20), Point2i(x + 20, y + 20) ) );
                Mat patch88_edge = edge_seg( Rect(Point2i(x - 20, y - 20), Point2i(x + 20, y + 20) ) );

                HOG_RES gradient = {
                    .local = Point2i(y, x),
                    .global = Point2i(roi_rect.y + y, roi_rect.x + x),
                    .distance = HOG(patch88, 0),
                    .count = count
                };

                hog_gradient.push_back(gradient);

                HOG_RES color = {
                    .local = Point2i(y, x),
                    .global = Point2i(roi_rect.y + y, roi_rect.x + x),
                    .distance = HOG_COLOR(patch88, 0),
                    .count = count
                };

                hog_color.push_back(color);

                HOG_RES edge = {
                    .local = Point2i(y, x),
                    .global = Point2i(roi_rect.y + y, roi_rect.x + x),
                    .distance = HOG_EDGE(patch88_edge, 0),
                    .count = count
                };

                hog_edge.push_back(edge);

                // Save patches
                sname<< patch_name << setprecision(3) << count <<type;
                imwrite(sname.str(), patch88);
                imwrite("roi_seg.jpg", roi_seg);
                sname.str("");
                count++;
            }
    }
    std::sort(hog_gradient.begin(), hog_gradient.end(), sortByDistance);
    std::sort(hog_color.begin(), hog_color.end(), sortByDistance);
    std::sort(hog_edge.begin(), hog_edge.end(), sortByDistance);

    // Print result
    // cout << "G min is " << hog_gradient[0].local << ' '<< hog_gradient[0].count << endl;
    // cout << "C min is " << hog_color[0].local << ' '<< hog_color[0].count << endl;
    // cout << "E min is " << hog_edge[0].local << ' '<< hog_edge[0].count << endl;

    // for(auto patch : hog_gradient)
    //     cout << "G " << patch.distance << ' '<< patch.count << endl;
    // for(auto patch : hog_color)
    //     cout << "C " << patch.distance << ' '<< patch.count << endl;
    // for(auto patch : hog_edge)
    //     cout << "E " << patch.distance << ' '<< patch.count << endl;

    //![HOG Min reasult Match for gradient and color ]
    int min_index = minMatch(hog_gradient, hog_color);

    if( min_index != -1) {

        cout << "Head is tp " << hog_gradient[min_index].global << ' ' << hog_gradient[min_index].count << endl;
        draw_rectangle(gray, hog_gradient[min_index].global, 20, 20);

        // Show input
        string name_input = "HOG input" + name;
        // Create a window for display
        namedWindow( name_input, WINDOW_AUTOSIZE );
        // Show segment input image
        imshow( name_input, gray);

        return hog_gradient[min_index].global;
    } else {
        cout << "No people found " << endl;
        return Point2i(-1, -1);
    }
}

int main(int argc, char **argv)
{
    int cnt = 0;

    VideoCapture cap("/home/jchen/Pictures/TAGR/samples/real_train_seg.avi");
    Mat src_pre;

    // HOG feature Initialization
    Mat hog_img = imread("head.jpg");
    Mat hog_edge = imread("head_edge.jpg");
    Mat hog_gray ;
    //![reduce_noise]
    //GaussianBlur( hog_img, hog_img, Size(3,3), 0, 0, BORDER_DEFAULT );
    //![reduce_noise]

    //![convert_to_gray]
    cvtColor(hog_img, hog_gray, COLOR_BGR2GRAY );
    //![convert_to_gray]

    //![HOG_feature]
    HOG(hog_gray, 1);
    HOG_COLOR(hog_gray, 1);
    HOG_EDGE(hog_edge, 1);
    //![HOG_feature]

    for (;;)
    {
        bool Is = cap.grab();
        if (Is == false) {
            // if video capture failed
            cout << "Video Capture Fail" << endl;
            break;
        } else {
            Mat src;
            Mat src_gray;
            Mat grad;

            //![variables]
            int scale = 1;
            int delta = 0;
            int ddepth = CV_16S;

            // capture frame from video file
            cap.retrieve(src, CV_CAP_OPENNI_BGR_IMAGE);
            //resize(img, img, Size(640, 480));

            //![reduce_noise]
            GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
            //![reduce_noise]

            //![convert_to_gray]
            cvtColor( src, src_gray, COLOR_BGR2GRAY );
            //![convert_to_gray]

            //![Equalize Histogram ]
            //equalizeHist( src_gray, src_gray );
            //![Equalize Histogram ]

            /**/
            Mat diff;
            if (cnt == 0)
            {
                //![initial_first_diff]
                src_gray.copyTo(diff);
                //![initial_first_diff]
            }
            else
            {
                //![get_different_with_previous_image]
                absdiff(src_gray, src_pre, diff);
                //![get_different_with_previous_image]

                //![threshold_the_difference]
                threshold( diff, diff, thred_diff, 255, 0);
                //![threshold_the_difference]
            }
            //![save_present_image_as_previous_to_next]
            src_gray.copyTo(src_pre);
            //![save_present_image_as_previous_to_next]
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

            //![threshold]
            /// do threshold
            threshold( grad, grad, thred_sobel, 255, 0);
            //![threshold]

            //![Mix_edge_and_difference_and_delete_alone_points]
            /// Add movement and edge points together to Mat mix
            Mat mix = Mat::zeros( grad.size(), CV_8UC1 );

            for (int i = 0; i < grad.cols; i++)
                for (int j = 0; j < grad.rows; j++)
                {
                    if (diff.at<unsigned char>(j, i) == 255) {
                        // Initial empty neighbours value
                        unsigned int empty_neighbours = 0;

                        // Check neighbours' value
                        for (int n = -1; n <= 1; n++)
                            for (int m = -1; m <= 1; m++)
                            {
                                if ((j+n >= 0) && (j+n < src_gray.rows)
                                    && (i+m >= 0) && (i+m < src_gray.cols)) {
                                    if (grad.at<uchar>(j+n, i+m) == 255)
                                        mix.at<uchar>(j+n, i+m) = 255;
                                    else
                                        empty_neighbours++;
                                } else {
                                    empty_neighbours++;
                                }
                            }

                        // Check empty meighbours maxium 8,means noise points
                        if (empty_neighbours == 8)
                            mix.at<unsigned char>(j, i) = 0;
                    }
                }
            //![Mix_edge_and_difference]

            Point2i head_center = hog_head(src_gray, grad, start, "0");
            if( head_center != Point2i(-1, -1))
                start = head_center;

            // // Create a window for display
            // namedWindow( "diff", WINDOW_AUTOSIZE );
            // // Show segment input image
            // imshow( "diff", diff);

            waitKey(1000);
            cnt++;
        }
    }

}
