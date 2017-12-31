#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/video/tracking.hpp"
#include <vector>
#include <memory>
#include <map>

using namespace cv;
using namespace std;
using Point2iPtr = std::shared_ptr<Point2i>;

class Model {
public:
    using ModelPtr = std::shared_ptr<Model>;

    Point2i head;
    Point2i body[4];
    Point2i arms[2];
    Point2i hands[2];
    string name;

    Model(string name = "Model") : name(name) {
        head = Point2i(-1, -1);

        for(int i = 0; i < 4; i++)
            body[i] = Point2i(-1, -1);

        for(int i = 0; i < 2; i++)
            arms[i] = Point2i(-1, -1);

        for(int i = 0; i < 2; i++)
            hands[i] = Point2i(-1, -1);
    }

    Model (Point2i &p, string name = "Model") : name("Model" + name) {
        head = p;

        for(int i = 0; i < 4; i++)
            body[i] = Point2i(-1, -1);

        for(int i = 0; i < 2; i++)
            arms[i] = Point2i(-1, -1);

        for(int i = 0; i < 2; i++)
            hands[i] = Point2i(-1, -1);
    }

    virtual ~Model () {}


    void setHead (Point2i &p)            { head = p;}
    void setBody (Point2i &p, int index) { body[index] = p;}
    void setArm  (Point2i &p, int index) { arms[index] = p;}
    void setHand (Point2i &p, int index) { hands[index] = p;}

    void drawHead (Mat model);
    void drawBody (Mat model);
    void drawArm (Mat model);
    void drawHand (Mat model);

    void update();
};
