#include <pybind11/pybind11.h>
#include "irsensor.hpp"

namespace py = pybind11;


PYBIND11_MODULE(irsensor, m) {
    m.doc() = "gym_line_follower irsensor simulation"; // optional module docstring

    py::class_<IrSensor>(m, "IrSensor")
            .def(py::init<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>,
                 int, double, double,int, double, double>())
            .def("update", &IrSensor::update)
            .def("read", &IrSensor::read)
            .def("get_photo_pos", &IrSensor::get_photo_pos);
}
