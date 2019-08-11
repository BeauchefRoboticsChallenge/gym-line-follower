#include <pybind11/pybind11.h>
#include "collision.hpp"
#include "primitives.hpp"

namespace py = pybind11;


PYBIND11_MODULE(trackutils, m) {
    m.doc() = "gym_line_follower track collisin detector"; // optional module docstring

    m.def("rect_p", &rect_p, "Function to calculate end point of a rect.",
    	py::arg("x0"),py::arg("y0"),py::arg("cAng"),py::arg("ds"));

    m.def("get_rect", &get_rect, "Function to calculate a rect.",
    	py::arg("x0"),py::arg("y0"),py::arg("cAng"),py::arg("ds"),py::arg("pd"));

    m.def("curve_p", &curve_p, "Function to calculate end point of a curve.",
    	py::arg("x0"),py::arg("y0"),py::arg("cAng"),py::arg("da"),py::arg("r"));

    m.def("get_curve", &get_curve, "Function to calculate a curve.",
    	py::arg("x0"),py::arg("y0"),py::arg("cAng"),py::arg("da"),py::arg("ds"),
    	py::arg("pd"));

    m.def("collision_dect", &collision_dect, "Function to detect collisions in track",
    	  py::arg("seg"),py::arg("track"),py::arg("th"));

    m.def("collision_dect2", &collision_dect2, "Function to detect collisions in track",
    	  py::arg("seg"),py::arg("track"),py::arg("th"));
}
