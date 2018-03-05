#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/video/tracking.hpp"
#include <opencv2/video.hpp>
#include <vector>
#include <memory>
#include <map>

using namespace cv;
using namespace std;
using Point2iPtr = std::shared_ptr<Point2i>;

class People {
public:
    using PeoplePtr = std::shared_ptr<People>;

    Point2i center;
    vector<Point2iPtr > contour;

    inline People () { center.x = 0; center.y = 0; }
    inline virtual ~People () {}
};
