#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include "People.h"
#include "image_util.h"
#include "segment/image.h"
#include "segment/misc.h"
#include "segment/pnmfile.h"
#include "segment/segment-image.h"

#define MY_WIDTH  352
#define MY_HEIGHT 288

#define min(x) (x < 0 ? 0 : x)
#define Wmax(x) (x > MY_WIDTH ? MY_WIDTH : x)
#define Hmax(x) (x > MY_HEIGHT ? MY_HEIGHT : x)

using PeoplePtr = std::shared_ptr<People>;

// Const Parameters
const unsigned int thred_sobel = 50;
const unsigned int thred_diff = 10;
const float sigma = 0.5;
const float k_value = 500;
const float min_size = 50;

// Viriables
int change_rate = 0;
Point2i pre_real_center;

unsigned int people_num = 1;

vector<PeoplePtr > peoples;
vector<Point2i > total_points;

vector<Point2i > temp_center;
vector<Point2i > new_center;

vector<unsigned int > indexs;

int points_euclideanDist(Point2i &p, Point2i &q) {
    Point2i diff = p - q;
    return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
}

int points_distance(Point2iPtr p, Point2iPtr q) {
    return (abs(p->x - q->x) + abs(p->y - q->y));
}

void draw_rectangle(Mat input, Point2i p, int w, int h) {
    Point2i a(p.y - w, p.x-h);
    Point2i b(p.y + w, p.x+h);
    rectangle(input, a, b, (255,255,255));
}

Mat roi_rectangle(Mat input, Point2i p, int w, int h) {
    Point2i a(min(p.y - w), min(p.x-h));
    Point2i b(Wmax(p.y + w), Hmax(p.x+h));
    return input( Rect(a, b) );
}

vector<Point2i> &k_means_cluster(Mat &input)
{
    // Initialization
    total_points.clear();
    temp_center.clear();
    new_center.clear();
    indexs.clear();

    for (int j = 0; j < input.cols; j++)
        for (int i = 0; i < input.rows; i++)
        {
            if (input.at<unsigned char>(i, j) == 255)
                total_points.push_back(Point2i(i,j));
        }

    if (total_points.size() == 0) return new_center;

    // Initialize random seed
    srand (time(NULL));

    for (int i = 0; i < people_num; i++) {
        // Generate secret number between 1 and 1000
        size_t index = rand() % total_points.size();
        auto it = std::find(indexs.begin(), indexs.end(), index);

        if (it == indexs.end()) indexs.push_back(index);
        else continue;

        peoples[i]->center = total_points[index];
        peoples[i]->contour.push_back(std::make_shared<Point2i>(total_points[index]));

        temp_center.push_back(Point2i(0,0));
        new_center.push_back(Point2i(0,0));
        //cout << index << " "<< total_points.size() << endl;
    }

    // Start clustering
    while (1) {
        unsigned int stop_flag=0;
        for (int i = 0; i < people_num; i++) {
            if (points_distance(std::make_shared<Point2i>(new_center[i]), std::make_shared<Point2i>(peoples[i]->center)) > 7)
                stop_flag++;
        }

        if (stop_flag == 0) break;

        // Copy new to old
        for (int i = 0; i < people_num; i++) {
            if (new_center[0].x != 0 && new_center[0].y != 0) {
                peoples[i]->center = new_center[i];
                peoples[i]->contour.push_back(std::make_shared<Point2i>(new_center[i]));
            }
        }

        // Distribute points
        for (int i = 0; i < total_points.size(); i++)
        {
            unsigned int min = 300000;
            unsigned int index = 0;
            unsigned int drop_flag = 0;

            Point2iPtr current_point = std::make_shared<Point2i>(total_points[i]);
            for (int i = 0; i < people_num; i++) {
                unsigned int distance = points_distance(current_point, std::make_shared<Point2i>(peoples[i]->center));
                if (distance == 0) {
                    drop_flag++;
                    break;
                } else {
                    if (distance < min) {
                        min = distance;
                        index = i;
                    }
                }
            }
            if(drop_flag == 0)  peoples[index]->contour.push_back(current_point);
        }


        // Calculate new center
        for (int i = 0; i < people_num; i++) {
            unsigned int x_sum = 0;
            unsigned int y_sum = 0;
            unsigned int min = 3000000;

            for (auto &point : peoples[i]->contour)
            {
                x_sum += point->x;
                y_sum += point->y;
                input.at<unsigned char>(point->x, point->y) = 50;
            }

            temp_center[i] = Point2i(x_sum / peoples[i]->contour.size(), y_sum / peoples[i]->contour.size());

            // Find neareast point to be center
            for (auto &point : peoples[i]->contour)
            {
                unsigned int distance = points_distance(point, std::make_shared<Point2i>(temp_center[i]));
                if ( distance < min)
                {
                    min = distance;
                    new_center[i].x = point->x;
                    new_center[i].y = point->y;
                }
            }
            // Clear contours
            peoples[i]->contour.clear();
            // cout << new_center[i].y << "," << new_center[i].x << " " << endl;
        }
    }

    return new_center;
}


int main(int argc, const char** argv)
{
    int cnt = 0;

    stringstream sname;
    string name = "result_";
    string type = ".ppm";

    // add your file name
    VideoCapture cap("/home/jchen/Pictures/TAGR/samples/real_train.avi");

    Mat src_pre;
    Mat diff_all;

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
            Mat planes[3];

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

            // //![split planes]
            // split(src, planes);
            // planes[0].copyTo(src_gray);
            // //![split planes]

            //![convert_to_gray]
            cvtColor( src, src_gray, COLOR_BGR2GRAY );
            //![convert_to_gray]

            /**/
            Mat diff;
            if (cnt == 0)
            {
                src_gray.copyTo(diff);
                src_gray.copyTo(diff_all);
            }
            else
            {
                absdiff(src_gray, src_pre, diff);
                threshold( diff, diff, thred_diff, 255, 0);
            }
            src_gray.copyTo(src_pre);

            // if (cnt != 0 )
            // {
            //     if (cnt%4 == 0 )
            //     {
            //         diff.copyTo(diff_all);
            //     }
            //     else{
            //         for (int i = 0; i < diff.cols; i++)
            //             for (int j = 0; j < diff.rows; j++)
            //             {
            //                 if (diff.at<unsigned char>(j, i) == 255)
            //                     diff_all.at<unsigned char>(j, i) = 255;
            //             }
            //     }
            // }

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

            // Add movement and edge points together to Mat mix
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
                        if (empty_neighbours == 8)
                            mix.at<unsigned char>(j, i) = 0;
                    }
                }

            People first, second, third;
            peoples.push_back(std::make_shared<People>(first));
            peoples.push_back(std::make_shared<People>(second));
            peoples.push_back(std::make_shared<People>(third));

            vector<Point2i > center = k_means_cluster(mix);
            Point2i center_sum, real_center;

            // Draw in origin video
            if(center.begin() != center.end()) {
                for (auto &point : center)
                {
                    center_sum += point;
                    draw_rectangle(src_gray, point, 50, 50);
                    cout << "Center = (" << point.y << "," << point.x << ")" << endl;
                }
            }
            real_center = center_sum / (int)people_num;

            if(pre_real_center == Point2i(0,0)) {
                pre_real_center = real_center;
            } else {
                int center_bias = points_distance(std::make_shared<Point2i>(pre_real_center), std::make_shared<Point2i>(real_center));
                cout << change_rate << "center_bias = " << pre_real_center << " - " << real_center << " = " << center_bias << endl;

                // If center's bias is too large, it has the possibilities to be a noise
                // So do not update previous center, waiting
                if (center_bias > 100)
                    change_rate++;
                else
                {
                    if(change_rate > -10) change_rate--;
                    pre_real_center = real_center;
                }

                // If the new center is the truth, update previous center
                if(change_rate > 5)
                {
                    change_rate = 0;
                    pre_real_center = real_center;
                }
            }


            Mat roi_seg;
            if (change_rate){
                draw_rectangle(mix, pre_real_center, 60, 100);
                roi_seg = roi_rectangle(src, pre_real_center, 60, 100);
            } else {
                draw_rectangle(mix, real_center, 60, 100);
                roi_seg = roi_rectangle(src, real_center, 60, 100);
            }

            //![segmentation]
            /// segmentation
            // Extra intresting segmention region
            Mat aroi(roi_seg.rows, roi_seg.cols, CV_8UC3);

            // Create a window for display
            namedWindow( "Segment input", WINDOW_AUTOSIZE );
            // Show our image inside it.
            imshow( "Segment input", roi_seg);

            // Convert mat to image format
            image<rgb> *seg_input = new image<rgb>(roi_seg.cols, roi_seg.rows);

            for (int y = 0; y < roi_seg.rows; y++) {
                for (int x = 0; x < roi_seg.cols; x++) {
                    imRef(seg_input, x, y).b = roi_seg.at<cv::Vec3b>(y,x)[0];
                    imRef(seg_input, x, y).g = roi_seg.at<cv::Vec3b>(y,x)[1];
                    imRef(seg_input, x, y).r = roi_seg.at<cv::Vec3b>(y,x)[2];
                }
            }

            int num_ccs;
            image<rgb> *seg = segment_image(seg_input, sigma, k_value, min_size, &num_ccs);
            printf("got %d components\n", num_ccs);
            //![segmentation]

            //![display]
            ShowManyImages("Image", 4, src_gray, grad, diff, mix);
            waitKey(100);
            //![display]

            // Create file name and store image file
            sname<< name << setprecision(3) <<cnt <<type;
            string filename = sname.str();
            sname.str("");
            savePPM(seg, filename.c_str());
            // imwrite(filename, grad);
            cnt++;
        }
    }
}
