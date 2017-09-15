#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <stdarg.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include "People.h"

const unsigned int thred_sobel = 10;
const unsigned int thred_diff = 10;

People first_people;
People second_people;
vector<Point2i > total_points;

Point2i first_temp_center;
Point2i second_temp_center;

Point2i first_new_center;
Point2i second_new_center;

unsigned int points_distance(Point2iPtr first, Point2iPtr second) {
    return (abs(first->x-second->x) + abs(first->y-second->y));
}

void k_means_cluster(Mat input)
{
    total_points.clear();
    for (int j = 0; j < input.cols; j++)
        for (int i = 0; i < input.rows; i++)
        {
            if (input.at<unsigned char>(i, j) == 255)
                total_points.push_back(Point2i(i,j));
        }

    // Initialize random seed
    srand (time(NULL));
    // Generate secret number between 1 and 1000
    size_t first_index, second_index;
    first_index = rand() % total_points.size();
    first_people.center = total_points[first_index];
    first_people.contour.push_back(std::make_shared<Point2i>(total_points[first_index]));
    first_new_center.x = 0;
    first_new_center.y = 0;

    second_index = rand() % total_points.size();
    second_people.center = total_points[second_index];
    second_people.contour.push_back(std::make_shared<Point2i>(total_points[second_index]));
    second_new_center.x = 0;
    second_new_center.y = 0;
    cout << first_index << " "  << second_index << " "<< total_points.size() << endl;
    while ( points_distance(std::make_shared<Point2i>(first_new_center), std::make_shared<Point2i>(first_people.center)) > 10
            ||  points_distance(std::make_shared<Point2i>(second_new_center), std::make_shared<Point2i>(second_people.center)) > 10 ) {
    // while (first_people.center != first_new_center || second_people.center != second_new_center) {
        // Copy new to old
        if (first_new_center.x != 0 && first_new_center.y != 0) {
            first_people.center = first_new_center;
            second_people.center = second_new_center;

            first_people.contour.push_back(std::make_shared<Point2i>(first_new_center));
            second_people.contour.push_back(std::make_shared<Point2i>(second_new_center));
        }

        // Distribute points
        for (int i = 0; i < total_points.size(); i++)
        {
            Point2iPtr current_point = std::make_shared<Point2i>(total_points[i]);
            unsigned int first_distance =  points_distance(current_point, std::make_shared<Point2i>(first_people.center));
            unsigned int second_distance =  points_distance(current_point, std::make_shared<Point2i>(second_people.center));

            if (first_distance != 0 && second_distance != 0) {
                if (first_distance < second_distance)
                    first_people.contour.push_back(current_point);
                else
                    second_people.contour.push_back(current_point);
            }
        }


        // Calculate new center
        unsigned int x_sum = 0;
        unsigned int y_sum = 0;
        unsigned int min = 3000000;
        for (auto &point : first_people.contour)
        {
            x_sum += point->x;
            y_sum += point->y;
            input.at<unsigned char>(point->x, point->y) = 50;
        }
        first_temp_center = Point2i(x_sum / first_people.contour.size(), y_sum / first_people.contour.size());

        for (auto &point : first_people.contour)
        {
            unsigned int distance = points_distance(point, std::make_shared<Point2i>(first_temp_center));
            if ( distance < min)
            {
                min = distance;
                first_new_center.x = point->x;
                first_new_center.y = point->y;
            }
        }

        x_sum = 0;
        y_sum = 0;
        min = 3000000;
        for (auto &point : second_people.contour)
        {
            x_sum += point->x;
            y_sum += point->y;
            input.at<unsigned char>(point->x, point->y) = 50;
        }
        second_temp_center = Point2i(x_sum / second_people.contour.size(), y_sum / second_people.contour.size());

        for (auto &point : second_people.contour)
        {
            unsigned int distance = points_distance(point, std::make_shared<Point2i>(second_temp_center));
            if ( distance < min)
            {
                min = distance;
                second_new_center.x = point->x;
                second_new_center.y = point->y;
            }
        }

        first_people.contour.clear();
        second_people.contour.clear();

        // cout  << first_new_center.x << "," << first_new_center.y  << " ";
        // cout  << second_new_center.x << "," << second_new_center.y << " " << endl;
    }

    Point2i a(first_new_center.y, first_new_center.x);
    Point2i b(second_new_center.y, second_new_center.x);
    line(input, a, b, 255, 1);
    cout << "(" << first_new_center.x << "," << first_new_center.y  << ")" << endl;
    cout << "(" << second_new_center.x << "," << second_new_center.y  << ")" << endl;

    input.at<unsigned char>(first_new_center.x, first_new_center.y) = 255;
    input.at<unsigned char>(second_new_center.x, second_new_center.y) = 255;

}

///////////////////////////////////////////////////////////////////////*/

void ShowManyImages(string title, int nArgs, ...) {
    int size;
    int i;
    int m, n;
    int x, y;

// w - Maximum number of images in a row
// h - Maximum number of images in a column
    int w, h;

// scale - How much we have to resize the image
    float scale;
    int max;

// If the number of arguments is lesser than 0 or greater than 12
// return without displaying
    if(nArgs <= 0) {
        printf("Number of arguments too small....\n");
        return;
    }
    else if(nArgs > 14) {
        printf("Number of arguments too large, can only handle maximally 12 images at a time ...\n");
        return;
    }
// Determine the size of the image,
// and the number of rows/cols
// from number of arguments
    else if (nArgs == 1) {
        w = h = 1;
        size = 300;
    }
    else if (nArgs == 2) {
        w = 2; h = 1;
        size = 300;
    }
    else if (nArgs == 3 || nArgs == 4) {
        w = 2; h = 2;
        size = 300;
    }
    else if (nArgs == 5 || nArgs == 6) {
        w = 3; h = 2;
        size = 200;
    }
    else if (nArgs == 7 || nArgs == 8) {
        w = 4; h = 2;
        size = 200;
    }
    else {
        w = 4; h = 3;
        size = 150;
    }

// Create a new 3 channel image
    Mat DispImage = Mat::zeros(Size(100 + size*w, 60 + size*h), CV_8UC1);

// Used to get the arguments passed
    va_list args;
    va_start(args, nArgs);

// Loop for nArgs number of arguments
    for (i = 0, m = 20, n = 20; i < nArgs; i++, m += (20 + size)) {
        // Get the Pointer to the IplImage
        Mat img = va_arg(args, Mat);

        // Check whether it is NULL or not
        // If it is NULL, release the image, and return
        if(img.empty()) {
            printf("Invalid arguments");
            return;
        }

        // Find the width and height of the image
        x = img.cols;
        y = img.rows;

        // Find whether height or width is greater in order to resize the image
        max = (x > y)? x: y;

        // Find the scaling factor to resize the image
        scale = (float) ( (float) max / size );

        // Used to Align the images
        if( i % w == 0 && m!= 20) {
            m = 20;
            n+= 20 + size;
        }

        // Set the image ROI to display the current image
        // Resize the input image and copy the it to the Single Big Image
        Rect ROI(m, n, (int)( x/scale ), (int)( y/scale ));
        Mat temp; resize(img,temp, Size(ROI.width, ROI.height));
        temp.copyTo(DispImage(ROI));
    }

// Create a new window, and show the Single Big Image
    namedWindow( title, 1 );
    imshow( title, DispImage);
//waitKey();

// End the number of arguments
    va_end(args);
}

int main(int argc, const char** argv)
{
    stringstream ss;

    string name = "sobel_";
    string type = ".jpg";
    int cnt = 0;
    // add your file name
    //VideoCapture cap("/home/jchen/Pictures/TAGR/samples/g01s04.avi");
    //VideoCapture cap("/home/jchen/Pictures/TAGR/samples/train_front.avi");
    VideoCapture cap("/home/jchen/Pictures/TAGR/samples/real_train.avi");


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
                src_gray.copyTo(diff);
            }
            else
            {
                absdiff(src_gray, src_pre, diff);
                threshold( diff, diff, thred_diff, 255, 0);
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
            threshold( grad, grad, thred_sobel, 255, 0);

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
                                if ((j+n >= 0) && (i+m >= 0)) {
                                    if (grad.at<unsigned char>(j+n, i+m) == 255)
                                        mix.at<unsigned char>(j+n, i+m) = 255;
                                    else
                                        empty_neighbours++;
                                } else {
                                    empty_neighbours++;
                                }
                            }

                        // Check empty meighbours maxium 8,means noise points
                        // if (empty_neighbours > 6)
                        // {
                        //     mix.at<unsigned char>(j, i) = 0;
                        //     //cout << empty_neighbours <<" ";
                        // }
                    }
                }

            k_means_cluster(mix);
            //![display]
            //namedWindow("soble", WINDOW_AUTOSIZE);
            // imshow( "src", src_gray );
            // imshow( "sobel", grad);
            // imshow( "diff", diff );
            // imshow( "mix", mix );

            ShowManyImages("Image", 4, src_gray, grad, diff, mix);
            waitKey(200);
            //![display]

            // ss<< name << setprecision(3) <<cnt<<type;
            // string filename = ss.str();
            // ss.str("");
            // imwrite(filename, grad);
            cnt++;
        }
    }
}
