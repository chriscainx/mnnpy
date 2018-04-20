#include <pybind11/pybind11.h>
#include <armadillo>

using namespace arma;

template<class M>
M cosine_norm(M in_matrix, int j) {
    return i + j;
}














namespace py = pybind11;

PYBIND11_MODULE(cmake_example, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: cmake_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    m.def("cosine_norm", &cosine_norm, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}