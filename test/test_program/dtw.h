#include<vector>
#include <algorithm>    // std::reverse
#include<iostream>
#include <fstream>
using namespace std;

extern vector<vector<float> > look_up_table;


namespace SLOW {
#define MAX_VALUE 10000
    // Point(x,y)
    struct Point {
        int x;
        int y;

        Point() {x = 0; y=0;}
        Point(int i, int j) {x = i; y=j;}
        Point& operator=(const Point &rhs)
            {
                this->x = rhs.x;
                this->y = rhs.y;
                return *this;/* your comparison code goes here */
            }
        Point& operator+(const Point &rhs)
            {
                this->x += rhs.x;
                this->y +=  rhs.y;
                return *this;/* your comparison code goes here */
            }
        bool operator==(const Point &rhs)
            {
                return (this->x == rhs.x && this->y == rhs.y);
            }
    };

    ostream& operator<<(ostream& os, const Point & rhs)
    {
        os << '['<< rhs.x << ' ' << rhs.y << ']';
        return os;
    }

    struct D {
        float d;
        Point p;

        D() {
            d = MAX_VALUE;
            p = Point(0,0);
        }
        bool operator>(const D &rhs)
            {
                return this->d > rhs.d;/* your comparison code goes here */
            }
        bool operator==(const D &rhs)
            {
                return this->d == rhs.d;/* your comparison code goes here */
            }
        bool operator<(const D &rhs)
            {
                return this->d < rhs.d;/* your comparison code goes here */
            }
        bool operator<=(const D &rhs)
            {
                return this->d <= rhs.d;/* your comparison code goes here */
            }

        D& operator=(const D &rhs)
            {
                this->d = rhs.d;
                this->p = rhs.p;
                return *this;/* your comparison code goes here */
            }
    };

    ostream& operator<<(ostream& os, const D& rhs)
    {
        os << '('<< rhs.d << ", " << rhs.p <<')';
        return os;
    }

    float dist(int &a, int &b) {
        // cout << a << ' '<< b <<' ';
        //return abs(a - b);
        return look_up_table[a-1][b-1];
    }

    D& min(D &d0, D &d1, D &d2) {
        if(d0 <= d1 && d0 <= d2)
            return d0;
        else if(d1 <= d0 && d1 <= d2)
            return d1;
        else
            return d2;
    }

    struct DTW_RESULT {
        float distance;
        vector<Point> path;
        DTW_RESULT(float d, vector<Point>&p) {
            distance = d;
            for(auto &i:p)
                path.push_back(i);
        }
    };


    DTW_RESULT __dtw(vector<int>& x, vector<int> &y, Point &window) {
        int len_x = x.size();
        int len_y = y.size();

        if (window.x == -1) {
            window.x = len_x;
            window.y = len_y;
        }

        vector<vector<D> > d(window.y+1, vector<D>(window.x+1));

        d[0][0].d = 0;

        for (int i = 1; i <= window.x; i++)
            for (int j = 1; j <= window.y; j++)
            {
                float dt = dist(x[i-1], y[j-1]);
                D d0, d1, d2;

                if (d[j][i-1].d != MAX_VALUE)
                    d0.d = d[j][i-1].d + dt;

                d0.p = Point(i-1, j);

                if (d[j-1][i].d != MAX_VALUE)
                    d1.d = d[j-1][i].d + dt;

                d1.p = Point(i, j-1);

                if (d[j-1][i-1].d != MAX_VALUE)
                    d2.d = d[j-1][i-1].d + dt;

                d2.p = Point(i-1, j-1);

                d[j][i] = min (d0, d1, d2);

                //cout << d[j][i] << endl;
            }
        //cout << endl;

        vector<Point> path;
        int i = len_x;
        int j = len_y;
        while(!(i == j && j == 0)) {
            path.push_back(Point(i - 1, j - 1));

            int x = i, y = j;

            i = d[y][x].p.x;
            j = d[y][x].p.y;
        }

        std::reverse(path.begin(), path.end());

        /* for(auto &p :path) */
        /*     cout << p << " "; */

        /* cout << d[len_y][len_x].d << endl; */

        return DTW_RESULT(d[len_y][len_x].d, path);
    }

    DTW_RESULT dtw(vector<int>& x, vector<int> &y) {
        Point window = {-1, -1};
        return __dtw(x, y, window);
    }

    vector<int> __reduce_by_half(vector<int> &x) {
        vector<int> y;
        for (int i = 0; i < (x.size() - x.size() % 2); i += 2)
        {
            y.push_back((x[i] + x[i+1]) / 2);
        }
        return y;
    }

    bool point_in_window(Point &point, vector<Point> &window) {
        for (auto &p : window)
        {
            if (p == point)
                return true;
        }

        return false;
    }

    vector<Point> __expand_window(vector<Point> &path, int len_x, int len_y, int radius) {
        vector<Point> path_;
        for (auto &p : path)
            path_.push_back(p);

        for (auto &p : path)
        {
            for(int i = p.x - radius; i < p.x + radius + 1 ; i++)
                for(int j = p.y - radius; j < p.y + radius + 1 ; j++)
                {
                    Point point = Point(i,j);
                    if(!point_in_window(point, path_))
                        path_.push_back(point);
                }
        }

        vector<Point> window_;
        for (auto &p : path_)
        {
            Point point = Point(p.x * 2 , p.y * 2);
            if (!point_in_window(point, window_))
                window_.push_back(point);

            point = Point(p.x * 2 , p.y * 2 + 1);
            if (!point_in_window(point, window_))
                window_.push_back(point);

            point = Point(p.x * 2 + 1 , p.y * 2);
            if (!point_in_window(point, window_))
                window_.push_back(point);

            point = Point(p.x * 2 + 1 , p.y * 2 + 1);
            if (!point_in_window(point, window_))
                window_.push_back(point);
        }

        vector<Point> window;
        int start_j = 0;
        for(int i = 0; i < len_x; i++)
        {
            int new_start_j = -1;
            for(int j = start_j; j < len_y; j++)
            {
                Point p = Point(i, j);
                if(point_in_window(p, window_)) {
                    window.push_back(p);
                    if(new_start_j == -1)
                        new_start_j = j;
                } else if (new_start_j != -1) {
                    break;
                }
            }
            start_j = new_start_j;
        }

        return window;
    }

    DTW_RESULT __fastdtw(vector<int>& x, vector<int> &y, int radius) {
        int len_x = x.size();
        int len_y = y.size();

        int min_time_size = radius + 2;

        if (len_x < min_time_size || len_y < min_time_size)
            return dtw(x, y);

        vector<int> x_shrinked = __reduce_by_half(x);
        vector<int> y_shrinked = __reduce_by_half(y);

        DTW_RESULT r = __fastdtw(x_shrinked, y_shrinked, radius);
        vector<Point> window = __expand_window(r.path, len_x, len_y, radius);

        Point win = window[window.size()-1] + Point(1,1);

        return __dtw(x, y, win);
    }

    DTW_RESULT fastdtw(vector<int>& x, vector<int> &y, int radius) {
        if(radius == 0)
            radius = 1;
        return __fastdtw(x, y, radius);
    }

} // SLOW
