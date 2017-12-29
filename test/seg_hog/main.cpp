#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include "People.h"
#include "image_util.h"
#include "segment/image.h"
#include "segment/misc.h"
#include "segment/pnmfile.h"
#include "segment/segment-image.h"

// #define MY_WIDTH  320
// #define MY_HEIGHT 240

// Use color segmentation to distinguish people's blue clothes
#define USE_COLOR    0
#define MIX_EDGE_SEG 0

#define MY_WIDTH  352
#define MY_HEIGHT 288

#define min(x) (x < 0 ? 0 : x)
#define Wmax(x) (x > MY_WIDTH ? MY_WIDTH : x)
#define Hmax(x) (x > MY_HEIGHT ? MY_HEIGHT : x)

using PeoplePtr = std::shared_ptr<People>;

// Const Parameters
const int thred_sobel = 50;
const int thred_diff = 10;
const float sigma = 0.5;
const float k_value = 500;
const float min_size = 50;

// Inertial center
const int center_bias = 5000;
const int err_rate = 5;

// Viriables
vector<int > change_rates;
vector<Point2i > pre_real_centers;
vector<Point2i > ordered_centers;
vector<Point2i > final_centers;

unsigned int people_num = 2;
unsigned int one_people = 0;
vector<PeoplePtr > peoples;
vector<Point2i > total_points;

vector<Point2i > temp_center;
vector<Point2i > new_center;

vector<unsigned int > indexs;

struct HOG_RES {
    Point2i p;
    float   distance;
    int     count;
};

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

bool color_test(int color) {
    return (color >= 40 && color <= 60);
}

int points_xDist(Point2i &p, Point2i &q) {
    Point2i diff = p - q;
    return (diff.x * diff.x);
}

int points_yDist(Point2i &p, Point2i &q) {
    Point2i diff = p - q;
    return (diff.y * diff.y);
}

int points_normalDist(Point2i &p, Point2i &q) {
    return (points_xDist(p,q) + points_yDist(p,q));
}

int points_DistMove(Point2i &p, Point2i &q) {
    return ((float)points_xDist(p,q) - (float)points_yDist(p,q));
}

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

vector< float > descriptors_feature;
float HOG(Mat img, int feature, int count)
{
    HOGDescriptor hog;
    hog.winSize = Size(40, 40);
    Mat gray;
    vector< float > descriptors;

    cvtColor(img, gray, COLOR_BGR2GRAY );
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

//        cout << "Count :=" << count << "    Viriation :=" << sum << endl;
        return sum;
    }
}

float descriptors_color_feature[4][4][36];
float HOG_COLOR(Mat img, int feature, int count) {
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

    // cout << endl;
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
                    descriptors_color_feature[j][i][k] = vect16[j][i][k];

        return 0;
    } else {
        float sum = 0;
        for(int j = 0; j < 4; j++)
            for(int i = 0; i < 4; i++)
                for(int k = 0; k < 32; k++)
                    sum += pow((descriptors_color_feature[j][i][k] - vect16[j][i][k]), 2);

        //cout << "Count :=" << count << "    Color Viriation :=" << sum << endl;
        return sum;
    }
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

void center_inertial(vector<Point2i> & centers) {

    for (int i = 0; i < people_num; i++) {

        if(pre_real_centers[i] == Point2i(0,0)) {
            pre_real_centers[i] = centers[i];
        } else {
            int bias = points_normalDist(pre_real_centers[i], centers[i]);
            //float move_ratio = points_DistMove(pre_real_centers[i], centers[i]);
            cout << change_rates[i] << "-center_bias = " << pre_real_centers[i] << " - " << centers[i] << " = " << bias  << " with "<< i << endl;

            // If center's bias is too large, it has the possibilities to be a noise
            // So do not update previous center, waiting
            if (bias > center_bias) {
                change_rates[i]++;
            } else {
                if(change_rates[i] > -10) change_rates[i]--;
                pre_real_centers[i] = centers[i];
            }

            // If the new center is the truth, update previous center
            if(change_rates[i] > err_rate) {
                change_rates[i] = 0;
                pre_real_centers[i] = centers[i];
            }
        }
        if (change_rates[i]) final_centers[i] = pre_real_centers[i];
        else  final_centers[i] = centers[i];
    }

}

void seg_images(Mat src, Mat grad, Point2i center, string name) {
    //![segmentation]
    /// segmentation
    // Extra intresting segmention region
    Mat roi_seg = roi_rectangle(src, center, 50, 100);

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
    printf("SegC = %d\n", num_ccs);
    //![segmentation]

    string name_input = "Segment input" + name;
    // // Create a window for display
    // namedWindow( name_input, WINDOW_AUTOSIZE );
    // // Show segment input image
    // imshow( name_input, roi_seg);

    // Convert image format to mat
    Mat seg_result(roi_seg.rows, roi_seg.cols, CV_8UC3);
    for (int y = 0; y < roi_seg.rows; y++) {
        for (int x = 0; x < roi_seg.cols; x++) {
            seg_result.at<cv::Vec3b>(y,x)[0] = imRef(seg, x, y).b;
            seg_result.at<cv::Vec3b>(y,x)[1] = imRef(seg, x, y).g;
            seg_result.at<cv::Vec3b>(y,x)[2] = imRef(seg, x, y).r;

            #if MIX_EDGE_SEG == 1
            // Mix edge and segments
            if (grad.at<uchar>(y + center.x - (roi_seg.rows >> 1), x + center.y - (roi_seg.cols >> 1)) == 255) {
                seg_result.at<cv::Vec3b>(y,x)[0] = 255;
                seg_result.at<cv::Vec3b>(y,x)[1] = 255;
                seg_result.at<cv::Vec3b>(y,x)[2] = 255;
            }
            #endif
        }
    }

    circle(seg_result, Point2i(seg_result.cols>>1, seg_result.rows>>1), 3, Scalar(0, 0, 0), -1);

    string name_output = "Segment output" + name;
    // Create a window for display
    namedWindow( name_output, WINDOW_AUTOSIZE );
    // Show segment result image
    imshow( name_output, seg_result);

    //![HOG]
    /// HOG to find head
    stringstream sname;
    string patch_name = "hog_";
    string type = ".jpg";
    int count = 0;

    vector<HOG_RES> hog_gradient;
    vector<HOG_RES> hog_color;
    // Compute hog
    for(int y = 20; y < roi_seg.rows - 20; y+=10)
        for(int x = 20; x < roi_seg.cols - 20; x+=10)
        {
            Mat patch88 = roi_seg( Rect(Point2i(x - 20, y - 20), Point2i(x + 20, y + 20) ) );

            HOG_RES gradient = {
                .p = Point2i(y, x),
                .distance = HOG(patch88, 0, count),
                .count = count
            };

            hog_gradient.push_back(gradient);

            HOG_RES color = {
                .p= Point2i(y, x),
                .distance = HOG_COLOR(patch88, 0, count),
                .count = count
            };

            hog_color.push_back(color);

            // Save patches
            sname<< patch_name << setprecision(3) << count <<type;
            imwrite(sname.str(), patch88);
            imwrite("roi_seg.jpg", roi_seg);
            sname.str("");
            count++;
        }

    std::sort(hog_gradient.begin(), hog_gradient.end(), sortByDistance);
    std::sort(hog_color.begin(), hog_color.end(), sortByDistance);

    // Print result
    // cout << "G min is " << hog_gradient[0].p << ' '<< hog_gradient[0].count << endl;
    // cout << "C min is " << hog_color[0].p << ' '<< hog_color[0].count << endl;

    // for(auto patch : hog_gradient)
    //     cout << "G " << patch.distance << ' '<< patch.count << endl;
    // for(auto patch : hog_color)
    //     cout << "C " << patch.distance << ' '<< patch.count << endl;

    //![HOG Min reasult Match for gradient and color ]
    int min_index = minMatch(hog_gradient, hog_color);
    if( min_index != -1)
    {
        cout << "Head is " << hog_gradient[min_index].distance << ' '<< hog_gradient[min_index].count << endl;
        draw_rectangle(roi_seg, hog_gradient[min_index].p, 20, 20);
    }
    //![HOG Min reasult Match for gradient and color ]

    // Create a window for display
    namedWindow( name_input, WINDOW_AUTOSIZE );
    // Show segment input image
    imshow( name_input, roi_seg);
}

void center_with_order(vector<Point2i> cluster_center){

    for (auto &point : ordered_centers)
    {
        point.x = 0;
        point.y = 0;
    }

    //![Merge_to_one_people_if_the_two_center_is_too_close]
    // if(people_num  > 1) {
    //     if(points_normalDist(cluster_center[0], cluster_center[1]) < 50*50)
    //         one_people = 1;
    //     cout << "one_people !!!" << endl;
    // }

    for (int i = 0; i < people_num; i++)
    {
        int min = 1000000;
        int close_index = 0;
        for (int j = 0; j < people_num; j++)
        {
            int dis = points_normalDist(pre_real_centers[j], cluster_center[i]);
            if(dis < min) {
                min = dis;
                close_index = j;
            }
        }
        if(ordered_centers[close_index] == Point2i(0,0))
            ordered_centers[close_index] = cluster_center[i];
        else
            ordered_centers[1-close_index] = cluster_center[i];
    }
}


int main(int argc, const char** argv)
{
    int cnt = 0;

    stringstream sname;
    string name = "result_";
    string type = ".ppm";

    // add your file name
    VideoCapture cap("/home/jchen/Pictures/TAGR/samples/real_train_seg.avi");
    //VideoCapture cap("/home/jchen/Pictures/TAGR/samples/g01s20.avi");

    Mat src_pre;

    // Initialization
    for (int i = 0; i < people_num; i++) {
      People people;
      peoples.push_back(std::make_shared<People>(people));
      pre_real_centers.push_back(Point2i(0, 0));
      ordered_centers.push_back(Point2i(0, 0));
      final_centers.push_back(Point2i(0, 0));
      change_rates.push_back(0);
    }

    // HOG feature Initialization
    Mat hog_img = imread("head.jpg");
    HOG(hog_img, 1, 0);

    HOG_COLOR(hog_img, 1, 0);

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

            //![convert_to_gray]
            cvtColor( src, src_gray, COLOR_BGR2GRAY );
            //![convert_to_gray]

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
            //![Mix_edge_and_difference]

            //![Calculation_of_center_according_to_people_num]
            vector<Point2i > cluster_center = k_means_cluster(mix);
            Point2i real_center, center_sum = Point2i(0, 0);

            // Draw in origin video
            if(cluster_center.begin() != cluster_center.end()) {

                //![Reorder_cluster_center_to_match_each_cluster]
                center_with_order(cluster_center);
                //![Reorder_cluster_center_to_match_each_cluster]

                //![Calculat_all_cluster_real_average_center]
                for (auto &point : cluster_center)
                {
                    center_sum += point;
                    cout << "cluster Center = (" << point.y << "," << point.x << ")" << endl;
                }
                real_center = center_sum / (int)people_num;
                //![Calculat_all_cluster_real_average_center]

                //![Draw_cluster_center]
                for (auto &point : ordered_centers)
                {
                    draw_rectangle(src_gray, point, 50, 50);
                    cout << "order Center = (" << point.y << "," << point.x << ")" << endl;
                }
                //![Draw_cluster_center]

                //![Do_inertial_if_new_ordered_center_is_moving_dramatically-> final_center]
                center_inertial(ordered_centers);
                //![Do_inertial_if_new_ordered_center_is_moving_dramatically-> final_center]

                // Draw rectangle in mix
                for (auto &point : final_centers)
                {
                  draw_rectangle(mix, point, 50, 100);
                }
            }


            //![Test_one_people_mode_or_many_people_mode]
            if(people_num == 1)
                one_people = 1;

            //![Segmentation_in_cluster]
            if(one_people) {
                sname<< 0 <<type;
                seg_images(src, grad, real_center, sname.str());
                one_people = 0;
            } else {
                int show_cnt = 0;
                for (auto &point : final_centers)
                {
                    sname<< show_cnt++ <<type;
                    seg_images(src, grad, point, sname.str());
                }
            }
            sname.str("");

            //![display]
            ShowManyImages("Image", 4, src_gray, grad, diff, mix);
            //ShowManyImages("Image", 2, src_gray, grad);
            waitKey(1000);
            //![display]

            // Create file name and store image file
            #if SAVE_IMG == 1
            sname<< name << setprecision(3) <<cnt <<type;
            string filename = sname.str();
            sname.str("");
            savePPM(seg, filename.c_str());
            #endif
            //imwrite(filename, grad);

            cnt++;
        }
    }
}
