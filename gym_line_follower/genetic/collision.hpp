#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <math.h>

namespace py = pybind11;

/*! Detect collison with L inf norm 
    \param seg segment to test
    \param prev_seg segment to add to the track
    \param track the points in the track
    \param th Threshold to use
*/
int collision_dect(py::array_t<double> seg,
                   py::array_t<double> track,
                   double th);

/*! Detect collison with L2 norm
    \param seg segment to test
    \param prev_seg segment to add to the track
    \param track the points in the track
    \param th Threshold to use
*/
int collision_dect2(py::array_t<double> seg,
                   py::array_t<double> track,
                   double th);