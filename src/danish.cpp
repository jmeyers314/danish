#include <pybind11/pybind11.h>
#include <vector>
#include <algorithm>

namespace py = pybind11;

void poly_grid_contains(
    size_t xp_ptr, size_t yp_ptr, size_t n_vertex,
    size_t x_ptr, size_t y_ptr, size_t out_ptr, size_t nx, size_t ny
) {
    // Polygon vertices
    double* xp = reinterpret_cast<double*>(xp_ptr);
    double* yp = reinterpret_cast<double*>(yp_ptr);
    // Grid points
    double* xgrid = reinterpret_cast<double*>(x_ptr);
    double* ygrid = reinterpret_cast<double*>(y_ptr);
    // Output
    bool* out = reinterpret_cast<bool*>(out_ptr);

    // Compute polygon y min/max
    double ymin = yp[0];
    double ymax = yp[0];
    for (size_t k=1; k<n_vertex; k++) {
        if (yp[k] < ymin)
            ymin = yp[k];
        if (yp[k] > ymax)
            ymax = yp[k];
    }

    // Proceed row by row
    std::vector<double> xinters;
    xinters.reserve(16);  // 2 is probably most common, but it's cheap to allocate 16
    for (size_t j=0; j<ny; j++) {
        double y = ygrid[j];
        if ((y < ymin) or (y > ymax)) {
            for (size_t i=0; i<nx; i++) {
                out[j*nx+i] = false;
            }
            continue;
        }
        xinters.clear();
        // Loop through edges to find all relevant x intercepts
        double x1 = xp[0];  // first point of segment
        double y1 = yp[0];
        for (size_t k=0; k<n_vertex; k++) {
            double x2 = xp[k % n_vertex];  // second point of segment
            double y2 = yp[k % n_vertex];
            if ((y > std::min(y1, y2)) && (y <= std::max(y1, y2)))
                xinters.push_back((y-y1)*(x2-x1)/(y2-y1)+x1);
            x1 = x2;
            y1 = y2;
        }
        std::sort(xinters.begin(), xinters.end());
        // All points to the left of first intercept are outside the polygon
        // Alternate after that.
        bool contained = false;
        auto xptr = xinters.begin();
        for (size_t i=0; i<nx; i++) {
            if (xptr != xinters.end()) {
                if (xgrid[i] > *xptr) {
                    contained = !contained;
                    xptr++;
                }
            }
            out[j*ny+i] = contained;
        }
    }
}

PYBIND11_MODULE(_danish, m) {
    m.def("poly_grid_contains", &poly_grid_contains);
}
