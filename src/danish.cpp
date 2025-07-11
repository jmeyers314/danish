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
        ymin = std::min(ymin, yp[k]);
        ymax = std::max(ymax, yp[k]);
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
        for (size_t k=1; k<n_vertex; k++) {
            double x2 = xp[k % n_vertex];  // second point of segment
            double y2 = yp[k % n_vertex];
            if ((y > std::min(y1, y2)) && (y <= std::max(y1, y2))) {
                double xinter = (y - y1) * (x2 - x1) / (y2 - y1) + x1;
                auto pos = std::lower_bound(xinters.begin(), xinters.end(), xinter);
                xinters.insert(pos, xinter);
            }
            x1 = x2;
            y1 = y2;
        }
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
            out[j*nx+i] = contained;
        }
    }
}

double pixel_frac_1(
    double u0, double v0, double sth0, double cth0,
    double u1, double v1,
    double x1, double y1,
    double dudx, double dudy,
    double dvdx, double dvdy
) {
    double cph = cth0 * dvdy - sth0 * dudy;
    double sph = sth0 * dudx - cth0 * dvdx;
    double norm = std::sqrt(sph*sph + cph*cph);
    cph /= norm;
    sph /= norm;

    // That takes care of the initial orientation, but we need the transformed point too.
    double det = dudx*dvdy - dvdx*dudy;
    double dxdu = dvdy/det;
    double dydu = -dvdx/det;
    double dxdv = -dudy/det;
    double dydv = dudx/det;
    double x0 = (u0-u1)*dxdu + (v0-v1)*dxdv + x1;
    double y0 = (u0-u1)*dydu + (v0-v1)*dydv + y1;

    // express x0, y0 wrt x1, y1
    x0 = x0 - x1;
    y0 = y0 - y1;

    bool flip = false;
    if (cph < 0) {
        cph = -cph;
        x0 = -x0;
        flip = !flip;
    }
    if (sph < 0) {
        sph = -sph;
        y0 = -y0;
        flip = !flip;
    }
    if (sph > cph) {
        std::swap(sph, cph);
        std::swap(x0, y0);
        flip = !flip;
    }

    double right = (0.5 - x0) * sph/cph + y0 + 0.5;  // wrt bottom
    double left = (-0.5 - x0) * sph/cph + y0 + 0.5;

    double frac = 0.0;

    if (left > 1) {
        frac = 1.0;
    } else if (right >= 1) {
        frac = 1.0 - 0.5 * cph / sph * (1 - left) * (1 - left);
    } else if (left > 0) {
        frac = 0.5 * (left + right);
    } else if (right > 0) {
        frac = 0.5 * cph / sph * right * right;
    } else {
        frac = 0.0;
    }

    return flip ? 1.0 - frac : frac;
}

void pixel_frac(
    double u0, double v0, double sth0, double cth0,
    size_t u1_ptr, size_t v1_ptr,
    size_t x1_ptr, size_t y1_ptr,
    size_t dudx_ptr, size_t dudy_ptr,
    size_t dvdx_ptr, size_t dvdy_ptr,
    size_t frac_ptr, size_t n_points
) {
    for (size_t i = 0; i < n_points; i++) {
        double u1 = reinterpret_cast<double*>(u1_ptr)[i];
        double v1 = reinterpret_cast<double*>(v1_ptr)[i];
        double x1 = reinterpret_cast<double*>(x1_ptr)[i];
        double y1 = reinterpret_cast<double*>(y1_ptr)[i];
        double dudx = reinterpret_cast<double*>(dudx_ptr)[i];
        double dudy = reinterpret_cast<double*>(dudy_ptr)[i];
        double dvdx = reinterpret_cast<double*>(dvdx_ptr)[i];
        double dvdy = reinterpret_cast<double*>(dvdy_ptr)[i];

        double frac = pixel_frac_1(
            u0, v0, sth0, cth0,
            u1, v1,
            x1, y1,
            dudx, dudy,
            dvdx, dvdy
        );

        reinterpret_cast<double*>(frac_ptr)[i] = frac;
    }
}

void pixel_frac(
    size_t u0_ptr, size_t v0_ptr,
    size_t sth0_ptr, size_t cth0_ptr,
    size_t u1_ptr, size_t v1_ptr,
    size_t x1_ptr, size_t y1_ptr,
    size_t dudx_ptr, size_t dudy_ptr,
    size_t dvdx_ptr, size_t dvdy_ptr,
    size_t frac_ptr, size_t n_points
) {
    for (size_t i = 0; i < n_points; i++) {
        double u0 = reinterpret_cast<double*>(u0_ptr)[i];
        double v0 = reinterpret_cast<double*>(v0_ptr)[i];
        double sth0 = reinterpret_cast<double*>(sth0_ptr)[i];
        double cth0 = reinterpret_cast<double*>(cth0_ptr)[i];
        double u1 = reinterpret_cast<double*>(u1_ptr)[i];
        double v1 = reinterpret_cast<double*>(v1_ptr)[i];
        double x1 = reinterpret_cast<double*>(x1_ptr)[i];
        double y1 = reinterpret_cast<double*>(y1_ptr)[i];
        double dudx = reinterpret_cast<double*>(dudx_ptr)[i];
        double dudy = reinterpret_cast<double*>(dudy_ptr)[i];
        double dvdx = reinterpret_cast<double*>(dvdx_ptr)[i];
        double dvdy = reinterpret_cast<double*>(dvdy_ptr)[i];

        double frac = pixel_frac_1(
            u0, v0, sth0, cth0,
            u1, v1,
            x1, y1,
            dudx, dudy,
            dvdx, dvdy
        );

        reinterpret_cast<double*>(frac_ptr)[i] = frac;
    }
}


PYBIND11_MODULE(_danish, m) {
    m.def("poly_grid_contains", &poly_grid_contains);
    m.def(
        "pixel_frac",
        py::overload_cast<
            double, double, double, double,
            size_t, size_t, size_t, size_t,
            size_t, size_t, size_t, size_t,
            size_t, size_t
        >(&pixel_frac)
    );
    m.def(
        "pixel_frac",
        py::overload_cast<
            size_t, size_t, size_t, size_t,
            size_t, size_t, size_t, size_t,
            size_t, size_t, size_t, size_t,
            size_t, size_t
        >(&pixel_frac)
    );
}
