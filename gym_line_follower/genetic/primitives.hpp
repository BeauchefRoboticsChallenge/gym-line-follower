#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <math.h>
#include <cmath>

namespace py = pybind11;

/*! Get the end point of a rect
    \param x0  Start point of rect
    \param y0  Start point of rect
    \param cAng  Starting angle of the rect
    \param ds Distance displacement
*/
py::array_t<double> rect_p(double x0, double y0,
                             double cAng, double ds);

/*! Get the points of a rect
    \param x0  Start point of rect
    \param y0  Start point of rect
    \param cAng  Starting angle of the rect
    \param ds Distance displacement
    \param pd - Point distance, separation in mm
*/
py::array_t<double> get_rect(double x0, double y0,
                             double cAng, double ds, int pd);

/*! Get the end point of a curve
    \param x0 Start point of curve
    \param y0 Start point of curve
    \param cAng Starting angle of the curve
    \param da Angle displacement
    \param r Radius of the curve
*/
py::array_t<double> curve_p(double x0, double y0,
                             double cAng, double da, double r);

/*! Get the points of a curve
    \param x0 Start point of curve
    \param y0 Start point of curve
    \param cAng Starting angle of the curve
    \param da Angle displacement
    \param r Radius of the curve
    \param pd Point distance, separation in mm
*/
py::array_t<double> get_curve(double x0, double y0,
                              double cAng, double da, double ds,
                              int pd);

