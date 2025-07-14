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
    double* u1p = reinterpret_cast<double*>(u1_ptr);
    double* v1p = reinterpret_cast<double*>(v1_ptr);
    double* x1p = reinterpret_cast<double*>(x1_ptr);
    double* y1p = reinterpret_cast<double*>(y1_ptr);
    double* dudxp = reinterpret_cast<double*>(dudx_ptr);
    double* dudyp = reinterpret_cast<double*>(dudy_ptr);
    double* dvdxp = reinterpret_cast<double*>(dvdx_ptr);
    double* dvdyp = reinterpret_cast<double*>(dvdy_ptr);

    for (size_t i = 0; i < n_points; i++) {
        double u1 = u1p[i];
        double v1 = v1p[i];
        double x1 = x1p[i];
        double y1 = y1p[i];
        double dudx = dudxp[i];
        double dudy = dudyp[i];
        double dvdx = dvdxp[i];
        double dvdy = dvdyp[i];

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
    double* u0p = reinterpret_cast<double*>(u0_ptr);
    double* v0p = reinterpret_cast<double*>(v0_ptr);
    double* sth0p = reinterpret_cast<double*>(sth0_ptr);
    double* cth0p = reinterpret_cast<double*>(cth0_ptr);
    double* u1p = reinterpret_cast<double*>(u1_ptr);
    double* v1p = reinterpret_cast<double*>(v1_ptr);
    double* x1p = reinterpret_cast<double*>(x1_ptr);
    double* y1p = reinterpret_cast<double*>(y1_ptr);
    double* dudxp = reinterpret_cast<double*>(dudx_ptr);
    double* dudyp = reinterpret_cast<double*>(dudy_ptr);
    double* dvdxp = reinterpret_cast<double*>(dvdx_ptr);
    double* dvdyp = reinterpret_cast<double*>(dvdy_ptr);
    double* fracp = reinterpret_cast<double*>(frac_ptr);

    for (size_t i = 0; i < n_points; i++) {
        double u0 = u0p[i];
        double v0 = v0p[i];
        double sth0 = sth0p[i];
        double cth0 = cth0p[i];
        double u1 = u1p[i];
        double v1 = v1p[i];
        double x1 = x1p[i];
        double y1 = y1p[i];
        double dudx = dudxp[i];
        double dudy = dudyp[i];
        double dvdx = dvdxp[i];
        double dvdy = dvdyp[i];

        double frac = pixel_frac_1(
            u0, v0, sth0, cth0,
            u1, v1,
            x1, y1,
            dudx, dudy,
            dvdx, dvdy
        );

        fracp[i] = frac;
    }
}

double enclosed_circle_1(
    double x, double y, double u, double v,
    double u0, double v0, double radius,
    double dudx, double dudy,
    double dvdx, double dvdy
) {
    double du = u - u0;
    double dv = v - v0;

    double drhosq = du*du + dv*dv;
    double h1 = std::sqrt((dudx + dvdy)*(dudx + dvdy) + (dudy - dvdx)*(dudy - dvdx));
    double h2 = std::sqrt((dudx - dvdy)*(dudx - dvdy) + (dudy + dvdx)*(dudy + dvdx));
    double maxLinearScale = 0.5 * (h1 + h2);
    if (drhosq < (radius - maxLinearScale)*(radius - maxLinearScale))
        return 1.0;
    if (drhosq > (radius + maxLinearScale)*(radius + maxLinearScale))
        return 0.0;

    double norm = std::sqrt(drhosq);
    double lineu = u0 + radius * du / norm;
    double linev = v0 + radius * dv / norm;
    double sth = -du / norm;
    double cth = dv / norm;

    return pixel_frac_1(
        lineu, linev, sth, cth,
        u, v, x, y,
        dudx, dudy,
        dvdx, dvdy
    );
}


void enclosed_circle(
    size_t x_ptr, size_t y_ptr,
    size_t u_ptr, size_t v_ptr,
    double u0, double v0, double radius,
    size_t dudx_ptr, size_t dudy_ptr,
    size_t dvdx_ptr, size_t dvdy_ptr,
    size_t frac_ptr, size_t n_points
) {
    double* xp = reinterpret_cast<double*>(x_ptr);
    double* yp = reinterpret_cast<double*>(y_ptr);
    double* up = reinterpret_cast<double*>(u_ptr);
    double* vp = reinterpret_cast<double*>(v_ptr);
    double* dudxp = reinterpret_cast<double*>(dudx_ptr);
    double* dudyp = reinterpret_cast<double*>(dudy_ptr);
    double* dvdxp = reinterpret_cast<double*>(dvdx_ptr);
    double* dvdyp = reinterpret_cast<double*>(dvdy_ptr);
    double* fracp = reinterpret_cast<double*>(frac_ptr);

    for (size_t i = 0; i < n_points; i++) {
        double x = xp[i];
        double y = yp[i];
        double u = up[i];
        double v = vp[i];
        double dudx = dudxp[i];
        double dudy = dudyp[i];
        double dvdx = dvdxp[i];
        double dvdy = dvdyp[i];

        double frac = enclosed_circle_1(
            x, y, u, v,
            u0, v0, radius,
            dudx, dudy,
            dvdx, dvdy
        );

        fracp[i] = frac;
    }
}


double enclosed_strut_1(
    double x, double y, double u, double v,
    double length,
    double u1, double v1, double sth1, double cth1,
    double u2, double v2, double sth2, double cth2,
    double dudx, double dudy,
    double dvdx, double dvdy
) {
    // Center of the strut
    double cu = 0.5 * (u1 + u2);
    double cv = 0.5 * (v1 + v2);

    // Exclude points > length/2 from strut center
    double du0 = u - cu;
    double dv0 = v - cv;
    if (du0*du0 + dv0*dv0 >= (length/2)*(length/2))
        return 0.0;  // Outside the strut

    // Exclude points not close to either edge
    // Note this implies the strut is thin
    double h1 = std::sqrt((dudx + dvdy)*(dudx + dvdy) + (dudy - dvdx)*(dudy - dvdx));
    double h2 = std::sqrt((dudx - dvdy)*(dudx - dvdy) + (dudy + dvdx)*(dudy + dvdx));
    double maxLinearScale = 0.5 * (h1 + h2);

    // Points close to edge1
    double du1 = u - u1;
    double dv1 = v - v1;
    double d1 = std::abs(-du1*sth1 + dv1*cth1);
    bool wclose1 = d1 < 2*maxLinearScale;

    // Points close to edge2
    double du2 = u - u2;
    double dv2 = v - v2;
    double d2 = std::abs(-du2*sth2 + dv2*cth2);
    bool wclose2 = d2 < 2*maxLinearScale;

    if (!wclose1 && !wclose2)
        return 0.0;  // Outside the strut

    double frac = pixel_frac_1(
        u1, v1, sth1, cth1,
        u, v,
        x, y,
        dudx, dudy,
        dvdx, dvdy
    );
    frac -= pixel_frac_1(
        u2, v2, sth2, cth2,
        u, v,
        x, y,
        dudx, dudy,
        dvdx, dvdy
    );

    return frac;
}


void enclosed_strut(
    size_t x_ptr, size_t y_ptr,
    size_t u_ptr, size_t v_ptr,
    double length,
    double u1, double v1, double sth1, double cth1,
    double u2, double v2, double sth2, double cth2,
    size_t dudx_ptr, size_t dudy_ptr,
    size_t dvdx_ptr, size_t dvdy_ptr,
    size_t frac_ptr, size_t n_points
) {
    double* xp = reinterpret_cast<double*>(x_ptr);
    double* yp = reinterpret_cast<double*>(y_ptr);
    double* up = reinterpret_cast<double*>(u_ptr);
    double* vp = reinterpret_cast<double*>(v_ptr);
    double* dudxp = reinterpret_cast<double*>(dudx_ptr);
    double* dudyp = reinterpret_cast<double*>(dudy_ptr);
    double* dvdxp = reinterpret_cast<double*>(dvdx_ptr);
    double* dvdyp = reinterpret_cast<double*>(dvdy_ptr);
    double* fracp = reinterpret_cast<double*>(frac_ptr);

    for (size_t i = 0; i < n_points; i++) {
        double x = xp[i];
        double y = yp[i];
        double u = up[i];
        double v = vp[i];
        double dudx = dudxp[i];
        double dudy = dudyp[i];
        double dvdx = dvdxp[i];
        double dvdy = dvdyp[i];

        double frac = enclosed_strut_1(
            x, y, u, v,
            length,
            u1, v1, sth1, cth1,
            u2, v2, sth2, cth2,
            dudx, dudy,
            dvdx, dvdy
        );

        fracp[i] = frac;
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
    m.def("enclosed_circle", &enclosed_circle);
    m.def("enclosed_strut", &enclosed_strut);
}
