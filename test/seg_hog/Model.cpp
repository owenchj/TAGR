#include"Model.h"

#define MY_WIDTH  352
#define MY_HEIGHT 288


void Model::update() {
    Mat model = Mat::zeros(Size(352, 288), CV_8UC1);
    drawHead(model);
    drawBody(model);
    drawArm(model);
    drawHand(model);

    // Create a window for display
    namedWindow(name, WINDOW_AUTOSIZE );
    // Show result image
    imshow(name, model);
}

void Model::drawHead (Mat model)
{
    if(head != Point2i(-1, -1))
        circle(model, Point2i(head.y, head.x), 25, Scalar(255, 255, 255), -1);
}

void Model::drawBody (Mat model)
{


}

void Model::drawArm (Mat model)
{

}

void Model::drawHand (Mat model)
{

}
