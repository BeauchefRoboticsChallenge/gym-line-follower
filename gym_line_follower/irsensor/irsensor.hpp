#ifndef GYMLINEFOLLOWER_IRSENSOR_IRSENSOR_H_
#define GYMLINEFOLLOWER_IRSENSOR_IRSENSOR_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <math.h>
#include <stdint.h>
#include <random>

namespace py = pybind11;

//2D point structure
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

//Simulated IR sensor class
class IrSensor
{
public:
    /*! Constuctor
        \param img Image of the track
        \param track_ppm  PPM of the track image
        \param ds  Distance from robot center to sensor array center
        \param photo_heigth  Distance in [m] from the floor to the ir sensor
        \param array_size Number of sensors in the array
        \param photo_sep  Separation in [m] between sensors of the array
        \param photo_fov Field of view of the sensors
        \param base_noise  Noise to be added to the sensor reading, stddev of normal dist.
    */
    IrSensor(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img,
             int track_ppm, double ds, double photo_heigth, int array_size,
             double photo_sep, double photo_fov, double base_noise=0.0);

    /*! Update the sensor position
        \param x x[m] pos of robot to the center of the track
        \param y y[m] pos of robot to the center of the track
        \param yaw  yaw angle[rad] of the robot
    */
    void update(double x, double y, double yaw);

    //Get the reading of each sensor in the array
    py::array_t<int> read();

    //Get the position[pixel] of each sensor of the array i the track image
    py::array_t<int> get_photo_pos();

    //Get the radius of the projection of the sensor in the track
    int get_sen_radius();

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
    std::mt19937 m_gen;
    std::normal_distribution<double> m_dis;

    //converts position in the track to position in the track image
    ipoint to_pixel(dpoint p);


};


#endif  // GYMLINEFOLLOWER_IRSENSOR_IRSENSOR_H_
