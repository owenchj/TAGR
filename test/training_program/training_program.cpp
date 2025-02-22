//opencv
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/video/tracking.hpp"
//C
#include <stdio.h>
#include "dtw.h"

#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
//C++
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
using namespace cv;
using namespace std;

#define K_MEANS   125
#define BOX_SIZE  256
// #define BOX_SIZE  256
#define PERSON_NUM   3
#define GESTURE_NUM  14

struct tree
{
    Mat data;
    int num;
    tree *left,*right;
    int left_num, right_num;
};

vector<tree* > Nodes;
vector<tree* > Binary_Tree;
vector<float > look_up_table;
vector<vector<int> > Sequences;

int node_count=0;

int sequence[PERSON_NUM * GESTURE_NUM][6] = {
    176,242, 260,331, 345,422,
    151,225, 240,322, 336,424,
    181,227, 247,291, 315,360,
    207,260, 272,327, 338,393,
    185,253, 275,357, 380,460,
    185,250, 260,336, 350,420,
    195,280, 290,375, 395,475,
    190,280, 295,385, 405,493,
    190,240, 250,310, 312,360,
    156,218, 220,293, 296,372,
    99,138, 140,174, 175,216,
    140,174, 180,215, 218,258,
    160,200, 210,250, 260,300,
    140,180, 181,214, 215,253,

    140,195, 199,254, 258,306,
    20,64, 75,130, 135,190,
    130,180, 183,225, 235,266,
    24,72, 80,128, 131,183,
    148,194, 200,245, 248,297,
    26,75, 80,140, 146,201,
    135,180, 185,240, 250,305,
    22,75, 83,132,140,192,
    120,158, 166,199, 206,238,
    16,78, 80,140, 141,205,
    122,162, 163,194, 195,222,
    17,52, 55,90, 92,136,
    155,188, 192,227, 231,260,
    55,77, 80,111, 114,148,

    116,162, 163,200, 202,240,
    2,39, 42,82, 84,122,
    136,182, 186,230, 234,274,
    20,65, 67,103, 106,142,
    125,167, 171,218, 220,266,
    1,43, 47,96, 100,150,
    103,143, 145,190, 192,234,
    2,51, 53,103, 105,150,
    130,170, 171,203, 205,244,
    9,51, 52,88, 89,125,
    115,141, 142,166, 167,190,
    15,40, 41,70, 71,90,
    14,38, 39,63, 65,84,
    6,30, 32,56, 57,82
};


// Global variables
Mat frame; //current frame
Rect target_box;
Mat target_frame; //current frame
Mat target_frame_gray;
Mat fgMaskMOG2; //fg mask fg mask generated by MOG2 method

Mat flow;
// some faster than mat image container
UMat flowUmat, prevgray;

Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
char keyboard; //input from keyboard

void help();

void processVideo(const char* videoFilename, int seq, vector<Mat > &SMD);

void k_means_cluster(vector<Mat > &SMD, vector<int > &prototypes, int k);

tree *k2_means_hiearchy_binary_tree(vector<Mat > &SMD, vector<int > &prototypes, int k);

void motion_sequence(vector<Mat > &SMD);

float smd_distance(Mat src1, Mat src2) {
    return norm(src1, src2, NORM_L2);
}

tree *fram_to_prototype(Mat smd);

void help()
{
    cout
    << "--------------------------------------------------------------------------" << endl
    << "This program shows how to use background subtraction methods provided by "  << endl
    << " OpenCV. You can process both videos (-vid) and images (-img)."             << endl
                                                                                    << endl
    << "Usage:"                                                                     << endl
    << "./bg_sub {-vid <video filename>|-img <image filename>}"                     << endl
    << "for example: ./bg_sub -vid video.avi"                                       << endl
    << "or: ./bg_sub -img /data/images/1.png"                                       << endl
    << "--------------------------------------------------------------------------" << endl
    << endl;
}

int main(int argc, char* argv[])
{
    vector<Mat > SMD;
    vector<int > prototypes;

    //print help information
    help();
    //check for the input parameter correctness
    if(argc != 3) {
        cerr <<"Incorret input list" << endl;
        cerr <<"exiting..." << endl;
        return EXIT_FAILURE;
    }
    //create GUI windows
    namedWindow("Frame");
    namedWindow("FG Mask MOG 2");
    //create Background Subtractor objects
    pMOG2 = createBackgroundSubtractorMOG2(); //MOG2 approach
    if(strcmp(argv[1], "-vid") == 0) {
        //input data coming from a video

        for(int person_num = 1; person_num <= PERSON_NUM; person_num++)
            for(int gesture_num = 1; gesture_num <= GESTURE_NUM; gesture_num++)
            {
                string video_name = std::string(argv[2]) + "person" + std::to_string(person_num) + "_gesture" + std::to_string(gesture_num) + "_com.avi";
                processVideo(video_name.c_str(), (person_num - 1) * GESTURE_NUM + gesture_num - 1, SMD);
            }

        k_means_cluster(SMD, prototypes, K_MEANS);

        for (int i = 0; i < K_MEANS; i++) {
            string title = "SMD" + to_string(i) + ".png";
            imwrite(title.c_str(), SMD[prototypes[i]]*256);
        }


        tree *root = k2_means_hiearchy_binary_tree(SMD, prototypes, 2);


        // Serializing struct to student.data
        ofstream look_up_file("look_up_table.data", ios::binary);
        look_up_file << node_count << endl;
        // Create look_up_table
        for(int i = 1; i < node_count; i++) {
            for(int j = 1; j < node_count; j++)
            {
                look_up_file << smd_distance(Nodes[i]->data, Nodes[j]->data) << ' ';
                look_up_table.push_back(smd_distance(Nodes[i]->data, Nodes[j]->data));
            }
            look_up_file <<endl;
        }

        look_up_file.close();


        // Serializing struct to student.data
        ofstream output_file("binary_tree.data", ios::binary);
        output_file << node_count << endl;

        for(int n = 0; n < node_count; n++)
        {
            // imshow("hello", Nodes[n]->data);
            // waitKey(2000);

            if(n) {
                for(int i=0; i<Nodes[n]->data.rows; i++)
                {
                    for(int j=0; j<Nodes[n]->data.cols; j++)
                    {
                        output_file<<Nodes[n]->data.at<float>(i,j)<<"\t";
                    }
                    output_file<<endl;
                }
            }

            output_file<< Nodes[n]->num << ' ';
            if(Nodes[n]->left)
                output_file<< Nodes[n]->left->num << ' ';
            else
                output_file<< -1 << ' ';

            if(Nodes[n]->right)
                output_file<< Nodes[n]->right->num << ' ';
            else
                output_file<< -1 << ' ';

            //delete Nodes[n];
        }

        output_file.close();

        motion_sequence(SMD);

        // Reading from it
        // ifstream input_file("binary_tree.data", ios::binary);
        // int total_node = 0;
        // input_file >> total_node;

        // for (int n = 0; n < total_node; n++) {
        //     Binary_Tree.push_back(new tree());

        //     if(n) {
        //         Mat smd(Size(BOX_SIZE >> 3, BOX_SIZE >> 2), CV_32FC1, Scalar(0));

        //         for(int i=0; i<BOX_SIZE >> 2; i++)
        //         {
        //             for(int j=0; j<BOX_SIZE >> 3; j++)
        //             {
        //                 input_file >> smd.at<float>(i,j);
        //             }
        //         }
        //         Binary_Tree[n]->data = smd;
        //     }

        //     input_file >> Binary_Tree[n]->num;
        //     input_file >> Binary_Tree[n]->left_num;
        //     input_file >> Binary_Tree[n]->right_num;
        // }
        // input_file.close();


        // for(int n = 1; n < node_count; n++)
        // {
        //     imshow("hello", Binary_Tree[n]->data);
        //     waitKey(2000);
        // }

    } else {
        //error in reading input parameters
        cerr <<"Please, check the input parameters." << endl;
        cerr <<"Exiting..." << endl;
        return EXIT_FAILURE;
    }
    //destroy GUI windows
    destroyAllWindows();
    return EXIT_SUCCESS;
}

Mat kernel = getStructuringElement(MORPH_ELLIPSE,Size(3,3));
vector<Mat> shape_descriptors;

void cal_shape_descriptor(Mat fgMaskMOG2, Mat &shape_descriptor)
{
    for (int y = 0; y < fgMaskMOG2.rows; y += 8) {
        for (int x = 0; x < fgMaskMOG2.cols; x += 8)
        {
            Mat patch = fgMaskMOG2(Rect(x, y, 8, 8));

            int count = 0;
            for (int n = 0; n < patch.rows; n ++) {
                for (int m = 0; m < patch.cols; m ++)
                {
                    if(patch.at<uchar>(n, m) != 0)
                        count++;
                }
            }
            //cout << count << ' ';
            shape_descriptor.at<float>(y >> 3, x >> 3) = count;
        }
    }

    // L2 normalize
    normalize(shape_descriptor, shape_descriptor, 1, 0, NORM_L2);
}

void cal_motion_descriptor(Mat flow, Mat &motion_descriptor) {
    Mat fxp(flow.size(), CV_32FC1, Scalar(0));
    Mat fxm(flow.size(), CV_32FC1, Scalar(0));
    Mat fyp(flow.size(), CV_32FC1, Scalar(0));
    Mat fym(flow.size(), CV_32FC1, Scalar(0));

    for (int y = 0; y < flow.rows; y ++) {
        for (int x = 0; x < flow.cols; x ++)
        {
            float fx = flow.at<Point2f>(y, x).x;
            if (fx > 0)
            {
                fxp.at<float>(y, x) =  fx;
                fxm.at<float>(y, x) =  0;
            } else {
                fxp.at<float>(y, x) =  0;
                fxm.at<float>(y, x) =  abs(fx);
            }

            float fy = flow.at<Point2f>(y, x).y;
            if (fy > 0)
            {
                fyp.at<float>(y, x) =  fy;
                fym.at<float>(y, x) =  0;
            } else {
                fyp.at<float>(y, x) =  0;
                fym.at<float>(y, x) =  abs(fy);
            }
        }
    }

    float THRESHOLD_M = 0.5;

    // Add threashold to eliminate noise
    threshold(fxp, fxp, THRESHOLD_M, 1000.0, 3);
    threshold(fxm, fxm, THRESHOLD_M, 1000.0, 3);
    threshold(fyp, fyp, THRESHOLD_M, 1000.0, 3);
    threshold(fym, fym, THRESHOLD_M, 1000.0, 3);

    // BF + -
    GaussianBlur( fxp, fxp, Size(3,3), 0, 0, BORDER_DEFAULT );
    GaussianBlur( fxm, fxm, Size(3,3), 0, 0, BORDER_DEFAULT );
    GaussianBlur( fyp, fyp, Size(3,3), 0, 0, BORDER_DEFAULT );
    GaussianBlur( fyp, fyp, Size(3,3), 0, 0, BORDER_DEFAULT );

    // QBF + -
    Mat fxp_bq(Size(BOX_SIZE >> 4, BOX_SIZE >> 4), CV_32FC1, Scalar(0));
    Mat fxm_bq(Size(BOX_SIZE >> 4, BOX_SIZE >> 4), CV_32FC1, Scalar(0));
    Mat fyp_bq(Size(BOX_SIZE >> 4, BOX_SIZE >> 4), CV_32FC1, Scalar(0));
    Mat fym_bq(Size(BOX_SIZE >> 4, BOX_SIZE >> 4), CV_32FC1, Scalar(0));

    // imshow("fxp_bq", fxp);
    // imshow("fxm_bq", fxm);
    // imshow("fyp_bq", fyp);
    // imshow("fym_bq", fym);

    for (int y = 0; y < flow.rows; y += BOX_SIZE >> 3) {
        for (int x = 0; x < flow.cols; x += BOX_SIZE >> 3)
        {
            Mat patch[4];
            patch[0] = fxp(Rect(x, y, BOX_SIZE >> 3, BOX_SIZE >> 3));
            patch[1] = fxm(Rect(x, y, BOX_SIZE >> 3, BOX_SIZE >> 3));
            patch[2] = fyp(Rect(x, y, BOX_SIZE >> 3, BOX_SIZE >> 3));
            patch[3] = fym(Rect(x, y, BOX_SIZE >> 3, BOX_SIZE >> 3));

            float sum[4] = {0,0,0,0};
            for (int n = 0; n < BOX_SIZE >> 3; n ++) {
                for (int m = 0; m < BOX_SIZE >> 3; m ++)
                {
                    sum[0] += patch[0].at<float>(n, m);
                    sum[1] += patch[1].at<float>(n, m);
                    sum[2] += patch[2].at<float>(n, m);
                    sum[3] += patch[3].at<float>(n, m);
                }
            }

            float area = ((BOX_SIZE * BOX_SIZE) >> 6);
            fxp_bq.at<float>(y / (BOX_SIZE >> 3), x / (BOX_SIZE >> 3)) = sum[0] / area;
            fxm_bq.at<float>(y / (BOX_SIZE >> 3), x / (BOX_SIZE >> 3)) = sum[1] / area;
            fyp_bq.at<float>(y / (BOX_SIZE >> 3), x / (BOX_SIZE >> 3)) = sum[2] / area;
            fym_bq.at<float>(y / (BOX_SIZE >> 3), x / (BOX_SIZE >> 3)) = sum[3] / area;
        }
    }

    // Normalize
    normalize(fxp_bq, fxp_bq, 1, 0, NORM_L2);
    normalize(fxm_bq, fxm_bq, 1, 0, NORM_L2);
    normalize(fyp_bq, fyp_bq, 1, 0, NORM_L2);
    normalize(fym_bq, fym_bq, 1, 0, NORM_L2);

    fxp_bq.copyTo(motion_descriptor(Rect(0,                0,             BOX_SIZE >> 4, BOX_SIZE >> 4)));
    fxm_bq.copyTo(motion_descriptor(Rect(BOX_SIZE >> 4,    0,             BOX_SIZE >> 4, BOX_SIZE >> 4)));
    fyp_bq.copyTo(motion_descriptor(Rect(0,                BOX_SIZE >> 4, BOX_SIZE >> 4, BOX_SIZE >> 4)));
    fym_bq.copyTo(motion_descriptor(Rect(BOX_SIZE >> 4,    BOX_SIZE >> 4, BOX_SIZE >> 4, BOX_SIZE >> 4)));
    normalize(motion_descriptor, motion_descriptor, 1, 0, NORM_L2);
}

void processVideo(const char* videoFilename, int seq, vector<Mat > &SMD) {
    cout << videoFilename << endl;
    //create the capture object
    VideoCapture capture(videoFilename);
    if(!capture.isOpened()){
        //error in opening the video input
        cerr << "Unable to open video file: " << videoFilename << endl;
        exit(EXIT_FAILURE);
    }
    //read input data. ESC or 'q' for quitting
    keyboard = 0;
    while( keyboard != 'q' && keyboard != 27 ){
        //read the current frame
        if(!capture.read(frame)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            return;
            //exit(EXIT_FAILURE);
        }
        //update the background model
        // target_box.x = 90;  target_box.y= 0;
        target_box.x = 220;  target_box.y= 50;
        target_box.width = BOX_SIZE; target_box.height = BOX_SIZE;

        target_frame = frame(target_box);
        cvtColor(target_frame, target_frame_gray, COLOR_BGR2GRAY );
        pMOG2->apply(target_frame_gray, fgMaskMOG2);
        morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_OPEN, kernel);

        Mat shape_descriptor(Size(BOX_SIZE >> 3, BOX_SIZE >> 3), CV_32FC1, Scalar(0));
        Mat motion_descriptor(Size(BOX_SIZE >> 3, BOX_SIZE >> 3), CV_32FC1, Scalar(0));

        int frame_num = capture.get(CAP_PROP_POS_FRAMES);
        if( (frame_num >= sequence[seq][0] && frame_num <= sequence[seq][1]) ||
            (frame_num >= sequence[seq][2] && frame_num <= sequence[seq][3]) ||
            (frame_num >= sequence[seq][4] && frame_num <= sequence[seq][5]) )
            cal_shape_descriptor(fgMaskMOG2, shape_descriptor);

        if (prevgray.empty() == false ) {
            // calculate optical flow
            calcOpticalFlowFarneback(prevgray, target_frame_gray, flowUmat, 0.4, 1, 12, 2, 8, 1.2, 0);
            // copy Umat container to standard Mat
            flowUmat.copyTo(flow);

            /*-------------------*/
            /* Draw optical flow */
            /*-------------------*/
            // for (int y = 0; y < target_frame_gray.rows; y += 8) {
            //     for (int x = 0; x < target_frame_gray.cols; x += 8)
            //     {
            //         // get the flow from y, x position * 10 for better visibility
            //         const Point2f flowatxy = flow.at<Point2f>(y, x) * 5;
            //         // draw line at flow direction
            //         line(target_frame, Point(x, y), Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar(255,0,0));
            //         // draw initial point
            //         circle(target_frame, Point(x, y), 1, Scalar(0, 0, 0), -1);
            //     }
            // }

            Mat smd(Size(BOX_SIZE >> 3, BOX_SIZE >> 2), CV_32FC1, Scalar(0));

            if( (frame_num >= sequence[seq][0] && frame_num <= sequence[seq][1]) ||
                (frame_num >= sequence[seq][2] && frame_num <= sequence[seq][3]) ||
                (frame_num >= sequence[seq][4] && frame_num <= sequence[seq][5]) )
            {
                cal_motion_descriptor(flow, motion_descriptor);

                shape_descriptor.copyTo(smd(Rect(0,               0,  BOX_SIZE >> 3, BOX_SIZE >> 3)));
                motion_descriptor.copyTo(smd(Rect(0,  BOX_SIZE >> 3,  BOX_SIZE >> 3, BOX_SIZE >> 3)));
                SMD.push_back(smd);

                // imshow("SMD", smd);
                // cout << SMD.size() << ' ';
            }

            target_frame_gray.copyTo(prevgray);
        } else {
            // fill previous image in case prevgray.empty() == true
            target_frame_gray.copyTo(prevgray);

        }


        //get the frame number and write it on the current frame
        // stringstream ss;
        // rectangle(frame, cv::Point(10, 2), cv::Point(100,20),
        //           cv::Scalar(255,255,255), -1);
        // ss << capture.get(CAP_PROP_POS_FRAMES);
        // string frameNumberString = ss.str();
        // putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
        //         FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));

        //show the current frame and the fg masks
        // imshow("Frame", frame);
        // imshow("Frame", target_frame);
        // imshow("FG Mask MOG 2", fgMaskMOG2);
        // imshow("Shape descriptor", shape_descriptor);
        // imshow("Motion descriptor", motion_descriptor);
        //get the input from the keyboard
        // keyboard = (char)waitKey( 30 );
    }

    //delete capture object
    capture.release();
}

void k_means_cluster(vector<Mat > &SMD, vector<int > &prototypes, int k) {
    // Initialization
    vector<int > new_center;
    vector<vector<int > > group;

    // Initialize random seed
    srand (time(NULL));

    for (int i = 0; i < k; i++) {
        // Generate secret number between 1 and 1000
        size_t index = rand() % SMD.size();
        auto it = std::find(new_center.begin(), new_center.end(), index);

        if (it == new_center.end()) {
            new_center.push_back(index);
        } else {
            i--;
            continue;
        }

        vector<int > g;
        group.push_back(g);
        prototypes.push_back(-1);
        // cout << index << " " << i << " "<< SMD.size() << endl;
    }

    // Start clustering
    for(int loop = 0; loop < 20; loop++) {
        cout << loop << " Iteration" << endl;

        unsigned int stop_flag = 0;

        for (int i = 0; i < k; i++) {
            if (new_center[i] != prototypes[i])
                stop_flag++;
        }

        if (stop_flag == 0) break;

        // Copy new to old
        for (int i = 0; i < k; i++) {
            prototypes[i] = new_center[i];
            //cout << prototypes[i] << " " << endl;
        }

        // Distribute points
        for (int i = 0; i < SMD.size(); i++)
        {
            float min = 100000;
            unsigned int index = 0;

            for (int j = 0; j < k; j++) {
                float distance = smd_distance(SMD[i], SMD[prototypes[j]]);

                if (distance < min) {
                    min = distance;
                    index = j;
                }
            }
            group[index].push_back(i);
        }

        // for (int i = 0; i < k; i++) {
        //     for (auto &p : group[i])
        //         cout << p << ' ';
        //     cout << endl;
        // }

        // Calculate new center
        for (int i = 0; i < k; i++) {
            Mat sum(Size(BOX_SIZE >> 3, BOX_SIZE >> 2), CV_32FC1, Scalar(0));
            float min = 100000.0;

            for (auto &p : group[i]) {
                sum += SMD[p];
            }

            Mat average = sum / group[i].size();

            // Find neareast point to be center
            for (auto &p : group[i]) {
                float distance = smd_distance(SMD[p], average);
                if (distance < min) {
                    min = distance;
                    new_center[i] = p;
                }
            }

            // Clear contours
            group[i].clear();

            //cout << new_center[i] << " ";
        }
        //cout << endl;
    }

    for (int i = 0; i < k; i++) {
        cout << prototypes[i] << ' ';
    }
    cout << endl;

}

tree *k2_means_hiearchy_binary_tree(vector<Mat > &SMD, vector<int > &prototypes, int k) {
    // Initialization
    vector<int > new_center;
    vector<Mat > average_SMD;
    vector<vector<int > > group;
    vector<int > real_center;
    vector<int > left, right;

    // for (int i = 0; i < prototypes.size(); i++) {
    //     cout << prototypes[i] << ' ';
    // }
    // cout << endl;

    Nodes.push_back(new tree());
    tree *root = Nodes[node_count];
    root->num  = node_count++;

    if(prototypes.size() == 1)
    {
        // Nodes.push_back(new tree());
        // tree *lnode = Nodes[node_count];

        // lnode->num  = node_count++;
        // lnode->data = SMD[prototypes[0]];
        // root->left = lnode;
        return root;
    } else if(prototypes.size() == 2) {
        Nodes.push_back(new tree());
        tree *lnode = Nodes[node_count];
        lnode->num  = node_count++;

        Nodes.push_back(new tree());
        tree *rnode = Nodes[node_count];
        rnode->num  = node_count++;

        lnode->data = SMD[prototypes[0]];
        rnode->data = SMD[prototypes[1]];
        root->left = lnode;
        root->right = rnode;
        return root;
    }

    // Initialize random seed
    srand (time(NULL));

    for (int i = 0; i < k; i++) {
        // Generate secret number between 1 and 1000
        size_t index = rand() % prototypes.size();
        auto it = std::find(new_center.begin(), new_center.end(), index);

        if (it == new_center.end()) {
            new_center.push_back(index);
        } else {
            i--;
            continue;
        }

        vector<int > g;
        group.push_back(g);
        real_center.push_back(-1);
        //cout << index << " " << i << " "<< prototypes.size() << endl;
    }

    // Start clustering
    for(int loop = 0; loop < 20; loop++) {
        // cout << loop << " Iteration" << endl;

        unsigned int stop_flag = 0;

        for (int i = 0; i < k; i++) {
            if (new_center[i] != real_center[i])
                stop_flag++;
        }

        if (stop_flag == 0) break;

        // Clear contours
        for (int i = 0; i < k; i++)
            group[i].clear();

        // Copy new to old
        for (int i = 0; i < k; i++) {
            real_center[i] = new_center[i];
            //cout << real_center[i] << " " << endl;
        }

        // Distribute points
        for (int i = 0; i < prototypes.size(); i++)
        {
            float min = 100000;
            unsigned int index = 0;

            for (int j = 0; j < k; j++) {
                float distance = smd_distance(SMD[prototypes[i]], SMD[prototypes[real_center[j]]]);

                if (distance < min) {
                    min = distance;
                    index = j;
                }
            }
            group[index].push_back(i);
        }

        // for (int i = 0; i < k; i++) {
        //     for (auto &p : group[i])
        //         cout << p << ' ';
        //     cout << endl;
        // }

        average_SMD.clear();
        // Calculate new center
        for (int i = 0; i < k; i++) {
            Mat sum(Size(BOX_SIZE >> 3, BOX_SIZE >> 2), CV_32FC1, Scalar(0));
            float min = 100000.0;

            for (auto &p : group[i]) {
                sum += SMD[prototypes[p]];
            }

            Mat average = sum / group[i].size();

            average_SMD.push_back(average);

            // Find neareast point to be center
            for (auto &p : group[i]) {
                float distance = smd_distance(SMD[prototypes[p]], average);
                if (distance < min) {
                    min = distance;
                    new_center[i] = p;
                }
            }

            for (int j = 0; j < group[i].size(); j++) {
                group[i][j] = prototypes[group[i][j]];
            }
            //cout << new_center[i] << " ";
        }
        //cout << endl;
    }

    // for (int i = 0; i < k; i++) {
    //     cout << real_center[i] << ' ';
    // }
    // cout << endl;

    tree *lnode = k2_means_hiearchy_binary_tree(SMD, group[0], 2);
    tree *rnode = k2_means_hiearchy_binary_tree(SMD, group[1], 2);
    lnode->data = average_SMD[0];
    rnode->data = average_SMD[1];
    root->left  = lnode;
    root->right = rnode;

    return root;
}

int frame_to_prototype(Mat smd)
{
    tree *node = Nodes[0];

    while(node->left || node->right)
    {
        if(node->left && node->right) {
            //cout << node->num << ' ' << node->left_num << ' ' << node->right_num << ' ' << smd_distance(smd, node->left->data) << ' ' << smd_distance(smd, node->right->data) << endl;

            if (smd_distance(smd, node->left->data) < smd_distance(smd, node->right->data))
                node = node->left;
            else
                node = node->right;
        } else if(node->left) {
            node = node->left;
        } else if(node->right) {
            node = node->right;
        }
    }

//    cout << node->num << endl;

    return node->num;
}

void motion_sequence(vector<Mat > &SMD) {
    int sum = 0;
    int index = 0;

    vector<vector<int> > SEQ;

    for (int i = 0; i < PERSON_NUM * GESTURE_NUM; i++) {
        for (int j = 0; j < 6; j += 2) {
            vector<int > s;
            SEQ.push_back(s);
            int count = (sequence[i][j+1] - sequence[i][j] + 1);
            for (int k = sum; k < sum + count; k++)
            {
                int num = frame_to_prototype(SMD[k]);
                SEQ[index].push_back(num);
            }
            sum += count;
            index++;
        }
    }

    sum = 0;
    for (auto &s : SEQ)
        sum +=s.size();

    if(sum != SMD.size())
        cout << "Error :" << sum << " != " << SMD.size();

    ofstream sequence_file("sequence.data", ios::binary);
    sequence_file << SEQ.size() << endl;
    // Create sequence file
    for(int i = 0; i < SEQ.size(); i++) {
        sequence_file << SEQ[i].size() << ' ';
        for(int j = 0; j < SEQ[i].size(); j++)
        {
            sequence_file << SEQ[i][j] << ' ';
        }
        sequence_file <<endl;
    }

    sequence_file.close();

}
