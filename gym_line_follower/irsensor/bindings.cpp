#include <pybind11/pybind11.h>
#include "irsensor.hpp"

namespace py = pybind11;


PYBIND11_MODULE(irsensor, m) {
    m.doc() = "gym_line_follower irsensor simulation"; // optional module docstring

    py::class_<IrSensor>(m, "IrSensor")
            .def(py::init<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>,
                 int, double, double,int, double, double, double>(),
                 py::arg("img"),py::arg("track_ppm"),py::arg("ds"),py::arg("photo_heigth"),
                 py::arg("array_size"),py::arg("photo_sep"),py::arg("photo_fov"),
                 py::arg("base_noise")= 0.0)
            .def("update", &IrSensor::update, py::arg("x"),py::arg("y"),py::arg("ang"))
            .def("read", &IrSensor::read)
            .def("get_photo_pos", &IrSensor::get_photo_pos);
}
