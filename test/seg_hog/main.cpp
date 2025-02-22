#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include "People.h"
#include "Model.h"
#include "image_util.h"
#include "segment/image.h"
#include "segment/misc.h"
#include "segment/pnmfile.h"
#include "segment/segment-image.h"

// #define MY_WIDTH  320
// #define MY_HEIGHT 240

// Use color segmentation to distinguish people's blue clothes
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
const int center_bias = 2500;
const int err_rate = 5;

// Viriables
vector<int > change_rates;
vector<Point2i > pre_real_centers;
vector<Point2i > ordered_centers;
vector<Point2i > final_centers;

// tracking
struct TRACK {
    Point2i head;
    int     still;
    Model   model;
    vector<float> descriptors;
};

vector<TRACK > tracker;

vector<Model> models;


vector<PeoplePtr > peoples;
unsigned int people_num = 1;


struct HOG_RES {
    Rect rect;
    Point2i local;
    Point2i global;
    float   distance;
    int     count;
};

enum FEATURE {
    HEAD_SAVE = 0,
    HEAD_COMPARE,
    BODY_SAVE,
    BODY_COMPARE,
    ARM_SAVE,
    ARM_COMPARE,
    HAND_SAVE,
    HAND_COMPARE
};

enum ACTION{
    FINDING =0,
    TRACKING = 1
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
        if(i < 5 && color[i].distance < 12.0) {
            for(int j = 0; j < gradient.size(); j++)
            {
                if( j < 5 &&  gradient[i].distance < 12.0)
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

// In the motion range
bool in_cluster(Point2i p, Point2i cluster ) {
    return (p.y >= cluster.y - 50
            && p.y <= cluster.y + 50
            && p.x <= cluster.x);
}

// In the head body range
bool in_body(Point2i b, Point2i head ) {
    return (b.y >= head.y - 30
            && b.y <= head.y + 30
            && b.x >= head.x + 40);
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

void draw_rectangle_body(Mat input, Point2i p, int w, int h) {
    Point2i a(p.y - w, p.x-h);
    Point2i b(p.y + w, p.x+h);
    rectangle(input, a, b, (255,255,255), -1);
}

Mat roi_rectangle(Mat input, Point2i p, int w, int h, Rect & rect, int half) {
    Point2i a(min(p.y - w), min(p.x-h));
    Point2i b(Wmax(p.y + w), Hmax(p.x+h));

    rect.x = a.x;
    rect.y = a.y;
    rect.width = b.x - a.x;
    if(half)
        rect.height = (b.y - a.y) / 2;
    else
        rect.height = b.y - a.y;

    return input( rect );
}

vector< float > descriptors_head;
vector< float > descriptors_head_sum;
int learn_count_g = 0;
float HOG(Mat gray, int feature, vector<float> descriptor_pre)
{
    HOGDescriptor hog;
    hog.winSize = Size(40, 40);
    vector< float > descriptors;

    if(feature % 2 == 0)
    {

        if(feature == HEAD_SAVE)
        {
            learn_count_g++;
            hog.compute( gray, descriptors_head, Size( 8, 8 ), Size( 0, 0 ) );

            for(int i =0; i < descriptors_head.size(); i++)
            {
                if(learn_count_g == 1)
                {
                    descriptors_head_sum.push_back(descriptors_head[i]);
                } else{
                    descriptors_head_sum[i] += descriptors_head[i];
                    descriptors_head[i] = descriptors_head_sum[i] / learn_count_g;
                }
            }
        } else if (feature == BODY_SAVE) {
            //hog.compute( gray, descriptors_body, Size( 8, 8 ), Size( 0, 0 ) );
        }
        return 0;
    }
    else
    {
        hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ) );

        float sum = 0;
        if (feature == HEAD_COMPARE) {
            for(int i = 0; i < descriptors.size(); i++)
                sum+= pow((descriptors_head[i] - descriptors[i]), 2);
        } else if (feature == BODY_COMPARE) {
            // for(int i = 0; i < descriptors.size(); i++)
            //     sum+= pow((descriptors_body[i] - descriptors[i]), 2);
        }
        descriptors.clear();
        return sum;
    }
}

float descriptors_color_feature[4][4][32];
float descriptors_color_feature_sum[4][4][32];
int learn_count_c = 0;
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

    if(feature % 2 == 0)
    {
        if(feature == HEAD_SAVE) {
            learn_count_c++;
            for(int j = 0; j < 4; j++)
                for(int i = 0; i < 4; i++)
                    for(int k = 0; k < 32; k++)
                    {
                        if(learn_count_c == 1)
                        {
                            descriptors_color_feature_sum[j][i][k] = vect16[j][i][k];
                            descriptors_color_feature[j][i][k] = descriptors_color_feature_sum[j][i][k];
                        } else {
                            descriptors_color_feature_sum[j][i][k] += vect16[j][i][k];
                            descriptors_color_feature[j][i][k] = descriptors_color_feature_sum[j][i][k] / learn_count_c;
                        }
                    }
        } else if(feature == BODY_SAVE) {
            for(int j = 0; j < 4; j++)
                for(int i = 0; i < 4; i++)
                    for(int k = 0; k < 32; k++)
                    {
                        descriptors_body_feature[j][i][k] = vect16[j][i][k];
                    }

        }
        return 0;
    } else {
        float sum = 0;

        if(feature == HEAD_COMPARE) {
            for(int j = 0; j < 4; j++)
                for(int i = 0; i < 4; i++)
                    for(int k = 0; k < 32; k++)
                        sum += pow((descriptors_color_feature[j][i][k] - vect16[j][i][k]), 2);
        } else if(feature == BODY_COMPARE) {
            for(int j = 0; j < 4; j++)
                for(int i = 0; i < 4; i++)
                    for(int k = 0; k < 32; k++)
                        sum += pow((descriptors_body_feature[j][i][k] - vect16[j][i][k]), 2);
        }

        return sum;
    }
}

Point2i hog_head(Mat gray, Point2i center, string name, ACTION act) {
    //![segmentation]
    /// segmentation
    // Extra intresting segmention region
    Rect roi_rect;
    Mat roi_seg;

    if(act == TRACKING) {
        roi_seg = roi_rectangle(gray, center, 40, 40, roi_rect, 0);
    } else {
        roi_seg = roi_rectangle(gray, center, 50, 100, roi_rect, 1);
    }

    //![HOG]
    /// HOG to find head
    int count = 0;

    vector<HOG_RES> hog_gradient;
    vector<HOG_RES> hog_color;

    // Compute hog
    if (roi_seg.rows >= 40 && roi_seg.cols >= 40) {
        for(int y = 20; y <= roi_seg.rows - 20; y += 10)
            for(int x = 20; x <= roi_seg.cols - 20; x += 10)
            {
                Rect rect = Rect(Point2i(x - 20, y - 20), Point2i(x + 20, y + 20) );
                Mat patch88 = roi_seg( rect );

                HOG_RES gradient = {
                    .rect = rect,
                    .local = Point2i(y, x),
                    .global = Point2i(roi_rect.y + y, roi_rect.x + x),
                    .distance = HOG(patch88, HEAD_COMPARE),
                    .count = count
                };

                hog_gradient.push_back(gradient);

                HOG_RES color = {
                    .rect = rect,
                    .local = Point2i(y, x),
                    .global = Point2i(roi_rect.y + y, roi_rect.x + x),
                    .distance = HOG_COLOR(patch88, HEAD_COMPARE),
                    .count = count
                };

                hog_color.push_back(color);
                count++;
            }
    }
    std::sort(hog_gradient.begin(), hog_gradient.end(), sortByDistance);
    std::sort(hog_color.begin(), hog_color.end(), sortByDistance);

    // Print result
    // cout << "G min is " << hog_gradient[0].local << ' '<< hog_gradient[0].count << endl;
    // cout << "C min is " << hog_color[0].local << ' '<< hog_color[0].count << endl;

    // for(auto patch : hog_gradient)
    //     cout << "G " << patch.distance << ' '<< patch.count << endl;
    // for(auto patch : hog_color)
    //     cout << "C " << patch.distance << ' '<< patch.count << endl;

    //![HOG Min reasult Match for gradient and color ]
    int min_index = minMatch(hog_gradient, hog_color);

    if( min_index != -1)
    {
        cout << "Head find is " << hog_gradient[min_index].global << ' ' << hog_gradient[min_index].count << endl;
        //draw_rectangle(gray, hog_gradient[min_index].global, 20, 20);

        return hog_gradient[min_index].global;
    } else {
        cout << "No people found " << endl;
        return Point2i(-1, -1);
    }
}

Point2i hog_tracking(Mat gray, Point2i center, vector<float>&descriptors) {
    //![segmentation]
    /// segmentation
    // Extra intresting segmention region
    Rect roi_rect;
    Mat roi_seg;

    roi_seg = roi_rectangle(gray, center, 40, 40, roi_rect, 0);

    //![HOG]
    /// HOG to find head
    int count = 0;

    vector<HOG_RES> hog_gradient;

    // Compute hog
    if (roi_seg.rows >= 40 && roi_seg.cols >= 40) {
        for(int y = 20; y <= roi_seg.rows - 20; y += 10)
            for(int x = 20; x <= roi_seg.cols - 20; x += 10)
            {
                Rect rect = Rect(Point2i(x - 20, y - 20), Point2i(x + 20, y + 20) );
                Mat patch88 = roi_seg( rect );

                HOG_RES gradient = {
                    .rect = rect,
                    .local = Point2i(y, x),
                    .global = Point2i(roi_rect.y + y, roi_rect.x + x),
                    .distance = HOG(patch88, HEAD_COMPARE, descriptors),
                    .count = count
                };

                hog_gradient.push_back(gradient);

                count++;
            }
    }
    std::sort(hog_gradient.begin(), hog_gradient.end(), sortByDistance);
    std::sort(hog_color.begin(), hog_color.end(), sortByDistance);

    // Print result
    // cout << "G min is " << hog_gradient[0].local << ' '<< hog_gradient[0].count << endl;
    // cout << "C min is " << hog_color[0].local << ' '<< hog_color[0].count << endl;

    // for(auto patch : hog_gradient)
    //     cout << "G " << patch.distance << ' '<< patch.count << endl;
    // for(auto patch : hog_color)
    //     cout << "C " << patch.distance << ' '<< patch.count << endl;

    //![HOG Min reasult Match for gradient and color ]
    int min_index = minMatch(hog_gradient, hog_color);

    if( min_index != -1)
    {
        cout << "Head find is " << hog_gradient[min_index].global << ' ' << hog_gradient[min_index].count << endl;
        //draw_rectangle(gray, hog_gradient[min_index].global, 20, 20);

        return hog_gradient[min_index].global;
    } else {
        cout << "No people found " << endl;
        return Point2i(-1, -1);
    }
}

void sobel(Mat src, Mat &src_gray, Mat &grad) {
    //![variables]
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

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

    //![threshold]
    /// do threshold
    threshold( grad, grad, thred_sobel, 255, 0);
    //![threshold]
}

Point2i body_points_sum[4];
Point2i *seg_images(Mat src, Mat mix, Point2i head, string name) {
    //![segmentation]
    /// segmentation
    // Extra intresting segmention region
    // Segment from head
    Rect roi_rect;
    Mat roi_seg = roi_rectangle(src, Point2i(head.x + 70, head.y), 50, 60, roi_rect, 0);

    Rect mix_rect;
    Mat  mix_seg = roi_rectangle(mix, Point2i(head.x + 70, head.y), 50, 60, mix_rect, 0);

    // Show input
    // string name_input = "Segment input" + name;
    // // Create a window for display
    // namedWindow( name_input, WINDOW_AUTOSIZE );
    // // Show segment input image
    // imshow( name_input, roi_seg);

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

    // Convert image format to mat
    Mat seg_result(roi_seg.rows, roi_seg.cols, CV_8UC3);
    for (int y = 0; y < roi_seg.rows; y++) {
        for (int x = 0; x < roi_seg.cols; x++) {
            {
                seg_result.at<cv::Vec3b>(y,x)[0] = imRef(seg, x, y).b;
                seg_result.at<cv::Vec3b>(y,x)[1] = imRef(seg, x, y).g;
                seg_result.at<cv::Vec3b>(y,x)[2] = imRef(seg, x, y).r;
            }
        }
    }

    //circle(seg_result, Point2i(seg_result.cols>>1, seg_result.rows>>1), 3, Scalar(0, 0, 0), -1);

    // Calculate body
    // Mat seg_gray, seg_grad;
    // sobel(seg_result, seg_gray, seg_grad);

    int count[4] = {0,0,0,0};

    for (int i = 0; i < 4; i++) body_points_sum[i] = Point2i(0,0);

    for (int y = 0; y < roi_seg.rows; y++)
        for (int x = 0; x < roi_seg.cols; x++) {
            if(mix_seg.at<uchar>(y,x) != 0) {
                if(y < (roi_seg.rows >> 1)) {
                    if(x < (roi_seg.cols >> 1)) {
                        body_points_sum[0] += Point2i(x,y);
                        count[0]++;
                    } else {
                        body_points_sum[1] += Point2i(x,y);
                        count[1]++;
                    }
                } else {
                    if(x < (roi_seg.cols >> 1)) {
                        body_points_sum[3] += Point2i(x,y);
                        count[3]++;
                    } else {
                        body_points_sum[2] += Point2i(x,y);
                        count[2]++;
                    }
                }
            }
        }

    for (int i = 0; i < 4; i++) {
        if(count[i]) {
            body_points_sum[i] /= count[i];
        }

        circle(seg_result, body_points_sum[i], 3, Scalar(255, 255, 255), -1);

        if(body_points_sum[i] == Point2i(0,0))
        {
            if(i == 0)
                body_points_sum[i] = Point2i(roi_seg.cols>>2, roi_seg.rows>>2);
            else if(i == 1)
                body_points_sum[i] = Point2i(roi_seg.cols - roi_seg.cols>>2, roi_seg.rows>>2);
            else if(i == 2)
                body_points_sum[i] = Point2i(roi_seg.cols - roi_seg.cols>>2, roi_seg.rows - roi_seg.rows>>2);
            else
                body_points_sum[i] = Point2i(roi_seg.cols>>2, roi_seg.rows - roi_seg.rows>>2);
        }
        cout << body_points_sum[i] << endl;

        body_points_sum[i] += Point2i(roi_rect.x, roi_rect.y);
    }

    string name_output = "Segment output" + name;
    // Create a window for display
    namedWindow( name_output, WINDOW_AUTOSIZE );
    // Show segment result image
    imshow( name_output, seg_result);

    return body_points_sum;
}

vector<Point2i> k_means_cluster(Mat input)
{
    // Initialization
    vector<Point2i > total_points;
    vector<Point2i > temp_center;
    vector<Point2i > new_center;
    vector<unsigned int > indexs;

    for (int j = 0; j < input.cols; j++)
        for (int i = 0; i < input.rows; i++)
        {
            if (input.at<unsigned char>(i, j) == 255)
                total_points.push_back(Point2i(i,j));
        }

    if (total_points.size() < 500) return new_center;

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
    for(int loop = 0; loop < 5; loop++) {
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

            //cout << change_rates[i] << "-center_bias = " << pre_real_centers[i] << " - " << centers[i] << " = " << bias  << " with "<< i << endl;

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

void center_with_order(vector<Point2i> cluster_center){

    for (auto &point : ordered_centers)
    {
        point.x = 0;
        point.y = 0;
    }

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

void add_into_tracker(TRACK track, Point2i center)
{
    if(tracker.size() == 4)
    {
        return;
    }
    else if(tracker.size() == 0)
    {
        cout << "Find one people ! = " << track.head << endl;
        tracker.push_back(track);
        return;
    }

    Point2i sum = Point2i(0,0);

    // New track head is long enough to all points
    for (auto &t : tracker)
    {
        if (points_normalDist(t.head, track.head) < center_bias)
            return;
    }

    // And it is in the motion range
    if(in_cluster(track.head, center))
    {
        cout << "Find one people ! = " << track.head << endl;
        tracker.push_back(track);
    }
}

void tracking(Mat gray, vector<TRACK> &tracker)
{
    cout << "Tracking size "<< tracker.size() << endl;

    // Tracking, update new point
    for (auto &t : tracker)
    {
        Point2i new_point = hog_head(gray, t.head, "Tracking", TRACKING);
    }

    // Delete wrong tracker
    for (int i = 0; i < tracker.size(); i++)
    {
        // If it is not in the motion range
        int is_in_cluser = 0;
        for (auto &p : final_centers) {
            if(in_cluster(tracker[i].head, p)) {
                is_in_cluser = 1;
                break;
            }
        }
        if (!is_in_cluser) tracker.erase(tracker.begin() + i);

        // If it is still too long
        if (tracker[i].still > 100) tracker.erase(tracker.begin() + i);

        // And it is too close to one tracker
        for (int j = i+1; j < tracker.size(); j++)
        {
            if(points_normalDist(tracker[i].head, tracker[j].head) <= 100)
                tracker.erase(tracker.begin() + i);
        }
    }

    cout << "Tracking size - "<< tracker.size() << endl;
}

void model_show(vector<TRACK> &track) {
    int count = 0;

    for (auto &t : track)
    {
        stringstream ss;
        ss.str("");
        ss << count++;
        t.model.setName("Model" + ss.str());
        t.model.update();
    }
}

int main(int argc, const char** argv)
{
    int cnt = 0;

    stringstream sname;
    string name = "result_";
    string type = ".ppm";

    // add your file name
    VideoCapture cap("/home/jchen/Pictures/TAGR/samples/HXD1C0828_20171009_033000.avi");
    // VideoCapture cap("/home/jchen/Pictures/TAGR/samples/real_train_seg.avi");
    // VideoCapture cap("/home/jchen/Pictures/TAGR/samples/output.avi");

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
    //Mat hog_img = imread("head.jpg");
    //Mat hog_gray ;
    //![reduce_noise]
    //GaussianBlur( hog_img, hog_img, Size(3,3), 0, 0, BORDER_DEFAULT );
    //![reduce_noise]

    //![convert_to_gray]
    //cvtColor(hog_img, hog_gray, COLOR_BGR2GRAY );
    //![convert_to_gray]

    //![HOG_feature]
    // HOG(hog_gray, 1);
    // HOG_COLOR(hog_gray, 1);
    //![HOG_feature]

    //![HOG_feature]
    Mat hog_gray = imread("head0.jpg", 0);
    HOG(hog_gray, HEAD_SAVE);
    HOG_COLOR(hog_gray, HEAD_SAVE);

    hog_gray = imread("head1.jpg", 0);
    HOG(hog_gray, HEAD_SAVE);
    HOG_COLOR(hog_gray, HEAD_SAVE);

    hog_gray = imread("head2.jpg", 0);
    HOG(hog_gray, HEAD_SAVE);
    HOG_COLOR(hog_gray, HEAD_SAVE);

    hog_gray = imread("head3.jpg", 0);
    HOG(hog_gray, HEAD_SAVE);
    HOG_COLOR(hog_gray, HEAD_SAVE);

    hog_gray = imread("head4.jpg", 0);
    HOG(hog_gray, HEAD_SAVE);
    HOG_COLOR(hog_gray, HEAD_SAVE);

    // BODY
    hog_gray = imread("body.jpg", 0);
    HOG_COLOR(hog_gray, BODY_SAVE);
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
            Mat src_grad;

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

            sobel(src, src_gray, src_grad);
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

            //![Mix_edge_and_difference_and_delete_alone_points]
            /// Add movement and edge points together to Mat mix
            Mat mix = Mat::zeros( src_gray.size(), CV_8UC1 );

            for (int i = 0; i < src_gray.cols; i++)
                for (int j = 0; j < src_gray.rows; j++)
                {
                    if (diff.at<uchar>(j, i) == 255) {
                        // Initial empty neighbours value
                        unsigned int empty_neighbours = 0;

                        // Check neighbours' value
                        for (int n = -1; n <= 1; n++)
                            for (int m = -1; m <= 1; m++)
                            {
                                if ((j+n >= 0) && (j+n < src_gray.rows)
                                    && (i+m >= 0) && (i+m < src_gray.cols)) {
                                    if (src_grad.at<uchar>(j+n, i+m) == 255)
                                        mix.at<uchar>(j+n, i+m) = 255;
                                    else
                                        empty_neighbours++;
                                } else {
                                    empty_neighbours++;
                                }
                            }

                        // Check empty meighbours maxium 8,means noise points
                        if (empty_neighbours == 8)
                            mix.at<uchar>(j, i) = 0;
                    }
                }
            //![Mix_edge_and_difference]

            //![Calculation_of_center_according_to_people_num]

            vector<Point2i > cluster_center;
            Point2i real_center, center_sum = Point2i(0, 0);

            cluster_center = k_means_cluster(mix);

            // Draw in origin video
            if(cluster_center.size() == people_num) {

                //![Reorder_cluster_center_to_match_each_cluster]
                center_with_order(cluster_center);
                //![Reorder_cluster_center_to_match_each_cluster]

                //![Calculat_all_cluster_real_average_center]
                for (auto &point : cluster_center)
                {
                    center_sum += point;
                }
                real_center = center_sum / (int)people_num;
                //![Calculat_all_cluster_real_average_center]

                //![Draw_cluster_center]
                // for (auto &point : ordered_centers)
                // {
                //     draw_rectangle(src_gray, point, 50, 50);
                //     cout << "order Center = (" << point.y << "," << point.x << ")" << endl;
                // }
                //![Draw_cluster_center]

                //![Do_inertial_if_new_ordered_center_is_moving_dramatically-> final_center]
                center_inertial(ordered_centers);
                //![Do_inertial_if_new_ordered_center_is_moving_dramatically-> final_center]

                // Draw rectangle in mix
                for (int i = 0; i < people_num; i++) {
                    //cout << "final_cluster Center = (" << final_centers[i].y << "," << final_centers[i].x << ")" << endl;
                    draw_rectangle(mix, final_centers[i], 128, 128);
                }

                //![Test_one_people_mode_or_many_people_mode]
                //![Segmentation_in_cluster]

                if(people_num == 1) {
                    sname << 0;
                    Point2i head_center = hog_head(src_gray, real_center, sname.str(), FINDING);

                    if(head_center != Point2i(-1, -1))
                    {
                        TRACK t = {.head = head_center,
                                   .still = 1};

                        add_into_tracker(t, real_center);
                    }
                    sname.str("");
                } else {
                    int show_cnt = 0;

                    for (int i = 0; i < final_centers.size(); i++)
                    {
                        sname << show_cnt++;

                        Point2i head_center = hog_head(src_gray, final_centers[i], sname.str(), FINDING);

                        if(head_center != Point2i(-1, -1))
                        {
                            TRACK t = {.head = head_center,
                                       .still = 1};

                            add_into_tracker(t, final_centers[i]);
                        }
                        sname.str("");
                    }
                }
            }

            tracking(src_gray, tracker);

            // Segmentation
            int show_cnt = 0;
            for (auto &t : tracker)
            {
                sname << show_cnt++;
                draw_rectangle(src_gray, t.head, 20, 20);
                // Point2i *p = seg_images(src, src_grad, t.head, sname.str());
                // t.model.setHead(t.head);

                // for (int i = 0; i < 4; i++)
                //     t.model.setBody(p[i],i);
                sname.str("");
            }

            //model_show(tracker);

            //![display]
            ShowManyImages("Image", 4, src_gray, src_grad, diff, mix);
            //ShowManyImages("Image", 2, src_gray, src_grad);
            waitKey(10000);
            //![display]

            cnt++;
        }
    }
}
