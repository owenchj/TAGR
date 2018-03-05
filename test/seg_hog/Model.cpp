#include"Model.h"

#define MY_WIDTH  352
#define MY_HEIGHT 288


void Model::update() {
    Mat model = Mat::zeros(Size(352, 288), CV_8UC1);
    drawHead(model);
    drawBody(model);
    drawArm(model);
    drawHand(model);

    // // Create a window for display
    // namedWindow(name, WINDOW_AUTOSIZE );
    // Show result image
    imshow(name, model);
}

void Model::drawHead (Mat model)
{
    if(head != Point2i(-1, -1))
        circle(model, Point2i(head.y, head.x), 20, Scalar(255, 255, 255), -1);
}

void Model::drawBody (Mat model)
{
    // for( int j = 0; j < 4; j++ )
    // {
    //     if(head != Point2i(-1, -1))
    //         line( model, body[j],  body[(j+1)%4], Scalar( 255 ), 3, 8 );
    // }

    line( model, body[0],  Point2i(head.y, head.x), Scalar( 255 ), 3, 8 );
    line( model, Point2i(head.y, head.x),     body[1], Scalar( 255 ), 3, 8 );
    line( model, body[1],  body[2], Scalar( 255 ), 3, 8 );
    line( model, body[2],  body[3], Scalar( 255 ), 3, 8 );
    line( model, body[3],  body[0], Scalar( 255 ), 3, 8 );
}

void Model::drawArm (Mat model)
{

}

void Model::drawHand (Mat model)
{

}
