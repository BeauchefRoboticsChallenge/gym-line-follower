#ifndef GYMLINEFOLLOWER_IRSENSOR_IRSENSOR_H_
#define GYMLINEFOLLOWER_IRSENSOR_IRSENSOR_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <math.h>
#include <stdint.h>
#include <random>

namespace py = pybind11;

template <typename T>
struct point {
  T x,y;
  point() {}
  point(T a_x,T a_y): x(a_x), y(a_y) {}
  point operator + (const point p) { return point(x+p.x,y+p.y); }
  point operator - (const point p) { return point(x-p.x,y-p.y); }
  point operator * (T c) {return point(x*c,y*c);}
} ;

typedef point<double> dpoint;
typedef point<int> ipoint;

class IrSensor
{
public:
    IrSensor(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img,
             int track_ppm, double dx, double photo_heigth,
             int array_size, double photo_sep, double photo_fov);

    void update(double x, double y, double ang);
    py::array_t<int> read();
    py::array_t<int> get_photo_pos();

private:
    std::unique_ptr<uint8_t[]> track;
    int m_track_ppm;
    std::unique_ptr<ipoint[]> photo_pos;
    int m_track_width;
    int m_track_heigth;
    double m_ds;
    int m_array_size;
    int m_radius;
    double m_photo_sep;

    //Random number generator
    std::mt19937 gen;
    std::uniform_int_distribution<> dis;

    ipoint to_pixel(dpoint p);


};


#endif  // GYMLINEFOLLOWER_IRSENSOR_IRSENSOR_H_
