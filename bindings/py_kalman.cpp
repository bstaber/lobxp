#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <kf_linear.hpp>

namespace py = pybind11;

static py::array_t<double>
filter_to_numpy(KFLinear& kf,
                const std::vector<std::optional<KFLinear::Vec>>& measurements)
{
    auto states = kf.filter(measurements);
    const py::ssize_t T = static_cast<py::ssize_t>(states.size());
    const py::ssize_t n = T ? static_cast<py::ssize_t>(states[0].size()) : 0;

    py::array_t<double> out({T, n});
    auto r = out.mutable_unchecked<2>();
    for (py::ssize_t t = 0; t < T; ++t)
        for (py::ssize_t i = 0; i < n; ++i)
            r(t, i) = states[t](static_cast<Eigen::Index>(i));
    return out;
}

PYBIND11_MODULE(lobxp_py, m) {
    py::class_<KFLinear>(m, "KFLinear")
        .def(py::init<
                 const KFLinear::Vec&,
                 const KFLinear::Mat&,
                 const KFLinear::Mat&,
                 const KFLinear::Mat&,
                 const KFLinear::Mat&,
                 const KFLinear::Mat&>(),
             py::arg("initial_state"),
             py::arg("initial_covariance"),
             py::arg("transition_matrix"),
             py::arg("observation_matrix"),
             py::arg("process_covariance"),
             py::arg("measurement_covariance"))
        .def("predict", &KFLinear::predict)
        .def("update",  &KFLinear::update, py::arg("measurement"))
        .def("step",    &KFLinear::step,   py::arg("measurement") = std::nullopt)
        .def_property_readonly("state",      &KFLinear::state)
        .def_property_readonly("covariance", &KFLinear::covariance)
        .def("filter",  [](KFLinear& self,
                           const std::vector<std::optional<KFLinear::Vec>>& zs) {
                return filter_to_numpy(self, zs);
            },
            py::arg("measurements"),
            "Run filter over a sequence of measurements (use None for missing). "
            "Returns NumPy array of shape (T, n).");
}
