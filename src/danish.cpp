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

    if (!wclose1 && !wclose2) {
        // Pixel is far from both edges.  Use signed perpendicular distances
        // to decide whether the pixel is fully inside (between the edges) or
        // fully outside.  The signed distances have opposite signs when the
        // pixel lies between the two edges and the same sign when outside.
        // This handles the case where the strut is wider than ~4 pixels,
        // i.e., when the "thin strut" approximation above does not hold.
        double s1 = -du1*sth1 + dv1*cth1;
        double s2 = -du2*sth2 + dv2*cth2;
        return (s1 * s2 < 0) ? 1.0 : 0.0;
    }

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


// ---------------------------------------------------------------------------
// Circle-triangle clipping
// ---------------------------------------------------------------------------

// Solve |p1 + t*(p2-p1) - center|² = radius² for t ∈ [0, 1].
// Returns the number of solutions (0, 1, or 2), written to out_pts[2][2].
// Solutions are returned in ascending order of t.
static int circle_edge_isect(
    double p1x, double p1y, double p2x, double p2y,
    double cx, double cy, double radius,
    double out_pts[2][2]
) {
    double dx = p2x - p1x, dy = p2y - p1y;
    double fx = p1x - cx,  fy = p1y - cy;
    double a = dx*dx + dy*dy;
    if (a < 1e-30) return 0;
    double b = 2.0*(fx*dx + fy*dy);
    double c = fx*fx + fy*fy - radius*radius;
    double disc = b*b - 4.0*a*c;
    if (disc < 0.0) return 0;
    double sq = std::sqrt(disc > 0.0 ? disc : 0.0);
    int n = 0;
    double t1 = (-b - sq) / (2.0*a);
    double t2 = (-b + sq) / (2.0*a);
    if (t1 >= 0.0 && t1 <= 1.0) {
        out_pts[n][0] = p1x + t1*dx; out_pts[n][1] = p1y + t1*dy; n++;
    }
    if (t2 > t1 + 1e-12 && t2 >= 0.0 && t2 <= 1.0) {
        out_pts[n][0] = p1x + t2*dx; out_pts[n][1] = p1y + t2*dy; n++;
    }
    return n;
}

// Clip one triangle against the boundary of a circle.
// Returns -1 (keep as-is), 0 (discard), or nv ≥ 3 (clipped polygon in poly[]).
// poly must have at least 8 elements (a triangle clipped against a circle
// produces at most 5 output vertices; 8 gives comfortable margin).
static int clip_one_triangle_to_circle(
    const double tri[3][2],
    double cx, double cy, double radius, bool keep_inside, double tol,
    double poly[8][2]
) {
    double r[3];
    bool ins[3];
    for (int i = 0; i < 3; i++) {
        double dx = tri[i][0] - cx, dy = tri[i][1] - cy;
        r[i] = std::sqrt(dx*dx + dy*dy);
        ins[i] = keep_inside ? (r[i] <= radius) : (r[i] >= radius);
    }

    // Classify with tolerance — mirrors Python _triangle_relation_to_circle
    bool all_keep, all_discard;
    if (keep_inside) {
        all_keep    = (r[0]<=radius+tol) && (r[1]<=radius+tol) && (r[2]<=radius+tol);
        all_discard = (r[0]>=radius-tol) && (r[1]>=radius-tol) && (r[2]>=radius-tol);
    } else {
        all_keep    = (r[0]>=radius-tol) && (r[1]>=radius-tol) && (r[2]>=radius-tol);
        all_discard = (r[0]<=radius+tol) && (r[1]<=radius+tol) && (r[2]<=radius+tol);
    }
    if (all_keep)    return -1;
    if (all_discard) return  0;

    // Sutherland-Hodgman clip against the circle boundary.
    // "inside" = within the kept region.
    int m = 0;
    for (int i = 0; i < 3; i++) {
        int j = (i + 1) % 3;
        bool s_in = ins[i], p_in = ins[j];
        double isect[2][2];
        int nc = circle_edge_isect(
            tri[i][0], tri[i][1], tri[j][0], tri[j][1], cx, cy, radius, isect
        );
        if (s_in) {
            poly[m][0] = tri[i][0]; poly[m][1] = tri[i][1]; m++;
            if (!p_in && nc > 0) {
                // Exiting kept region: emit the last crossing on s→p
                poly[m][0] = isect[nc-1][0]; poly[m][1] = isect[nc-1][1]; m++;
            }
        } else {
            if (p_in && nc > 0) {
                // Entering kept region: emit first crossing
                poly[m][0] = isect[0][0]; poly[m][1] = isect[0][1]; m++;
            } else if (!p_in && nc == 2) {
                // Edge passes through kept region entirely: emit both crossings
                poly[m][0] = isect[0][0]; poly[m][1] = isect[0][1]; m++;
                poly[m][0] = isect[1][0]; poly[m][1] = isect[1][1]; m++;
            }
        }
    }
    return m;
}

// Clip all triangles against one circle boundary.
//
// tri_in_ptr  : (ntri_in, 3, 2) float64, C-contiguous — input triangle vertices
// ntri_in     : number of input triangles
// cx, cy      : circle centre in pupil metres
// radius      : circle radius in pupil metres
// keep_inside : 1 = keep circle interior; 0 = keep exterior
// tol         : boundary tolerance (Python default 1e-12)
// tri_out_ptr : (ntri_in * 3, 3, 2) float64 pre-allocated output buffer
// n_removed_ptr, n_clipped_ptr : int32* counters (caller zeros before first circle)
//
// Returns: number of output triangles written to tri_out.
int clip_triangles_to_circle(
    size_t tri_in_ptr, int ntri_in,
    double cx, double cy, double radius, int keep_inside, double tol,
    size_t tri_out_ptr,
    size_t n_removed_ptr, size_t n_clipped_ptr
) {
    const double* in  = reinterpret_cast<const double*>(tri_in_ptr);
    double*       out = reinterpret_cast<double*>(tri_out_ptr);
    int* nrem  = reinterpret_cast<int*>(n_removed_ptr);
    int* nclip = reinterpret_cast<int*>(n_clipped_ptr);

    bool ki = (keep_inside != 0);
    int ntri_out = 0;

    for (int k = 0; k < ntri_in; k++) {
        const double* tv = in + k*6;
        double tri[3][2] = {{tv[0],tv[1]},{tv[2],tv[3]},{tv[4],tv[5]}};

        double poly[8][2];
        int nv = clip_one_triangle_to_circle(tri, cx, cy, radius, ki, tol, poly);

        if (nv == -1) {
            // Keep unchanged
            double* dst = out + ntri_out*6;
            dst[0]=tv[0]; dst[1]=tv[1]; dst[2]=tv[2];
            dst[3]=tv[3]; dst[4]=tv[4]; dst[5]=tv[5];
            ntri_out++;
        } else if (nv >= 3) {
            (*nclip)++;
            // Fan-triangulate from poly[0]
            for (int i = 1; i <= nv-2; i++) {
                double ax = poly[i][0]-poly[0][0], ay = poly[i][1]-poly[0][1];
                double bx = poly[i+1][0]-poly[0][0], by = poly[i+1][1]-poly[0][1];
                if (std::abs(ax*by - ay*bx) < 2e-30) continue;
                double* dst = out + ntri_out*6;
                dst[0]=poly[0][0]; dst[1]=poly[0][1];
                dst[2]=poly[i][0]; dst[3]=poly[i][1];
                dst[4]=poly[i+1][0]; dst[5]=poly[i+1][1];
                ntri_out++;
            }
        } else {
            // nv == 0 or degenerate
            (*nrem)++;
        }
    }
    return ntri_out;
}


// ---------------------------------------------------------------------------
// Clip triangles to a strut (band between two parallel lines)
// ---------------------------------------------------------------------------

// Signed distance from point (x,y) to line through (px,py) with direction (cth,sth).
// Positive = left side of the line (looking along the direction).
static inline double line_signed_dist(
    double x, double y,
    double px, double py, double sth, double cth
) {
    return -(x - px)*sth + (y - py)*cth;
}

// Clip a convex polygon against one half-plane defined by a line.
// Keeps vertices where line_signed_dist >= 0 (or <= 0 if flip).
// poly_in: flat (x0,y0, x1,y1, ...) with n vertices.
// poly_out: output buffer.  Returns output vertex count.
static int sh_clip_line(
    const double* poly_in, int n, double* poly_out,
    double px, double py, double sth, double cth, bool keep_positive
) {
    if (n < 3) return 0;
    int m = 0;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        double ix = poly_in[2*i], iy = poly_in[2*i+1];
        double jx = poly_in[2*j], jy = poly_in[2*j+1];
        double di = line_signed_dist(ix, iy, px, py, sth, cth);
        double dj = line_signed_dist(jx, jy, px, py, sth, cth);
        bool i_in = keep_positive ? (di >= 0) : (di <= 0);
        bool j_in = keep_positive ? (dj >= 0) : (dj <= 0);
        if (i_in) {
            poly_out[2*m] = ix; poly_out[2*m+1] = iy; m++;
            if (!j_in) {
                double t = di / (di - dj);
                poly_out[2*m]   = ix + t*(jx - ix);
                poly_out[2*m+1] = iy + t*(jy - iy);
                m++;
            }
        } else if (j_in) {
            double t = di / (di - dj);
            poly_out[2*m]   = ix + t*(jx - ix);
            poly_out[2*m+1] = iy + t*(jy - iy);
            m++;
        }
    }
    return m;
}

// Clip one triangle against a finite strut rectangle, keeping the EXTERIOR.
//
// The strut is the intersection of 4 half-planes:
//   - Between width edges (d1 >= 0 AND d2 >= 0, after orientation)
//   - Between length ends (along-strut distance from center <= half_length)
//
// Strategy: compute the portion of the triangle that is INSIDE all 4 half-planes
// (= inside the strut rectangle), then the kept exterior is the original minus
// that interior. We split iteratively: at each half-plane, the "outside" piece
// is emitted directly, and only the "inside" piece is carried forward to the
// next clip. Whatever remains inside all 4 is discarded.
//
// Returns number of output triangles written to out_buf (flat, 6 doubles each).
// Returns -1 if triangle is entirely outside (keep as-is, nothing written).
// Returns 0 if triangle is entirely inside (discard).
static int clip_one_triangle_to_strut(
    const double tri[3][2],
    double p1x, double p1y, double sth1, double cth1,
    double p2x, double p2y, double sth2, double cth2,
    double cx, double cy, double along_sth, double along_cth, double half_length,
    double tol,
    double* out_buf
) {
    // Compute signed distances to all 4 bounding lines.
    double d1[3], d2[3], dL[3], dR[3];
    for (int i = 0; i < 3; i++) {
        d1[i] = line_signed_dist(tri[i][0], tri[i][1], p1x, p1y, sth1, cth1);
        d2[i] = line_signed_dist(tri[i][0], tri[i][1], p2x, p2y, sth2, cth2);
        // Along-strut distance from center: project onto strut direction.
        // "Inside" the length means |projection| <= half_length.
        // We use two half-planes: one at +half_length (keep < side) and one at -half_length (keep > side).
        // dL = along_dist + half_length (positive = inside the left end)
        // dR = half_length - along_dist (positive = inside the right end)
        double along_dist = (tri[i][0] - cx) * along_cth + (tri[i][1] - cy) * along_sth;
        dL[i] = along_dist + half_length;
        dR[i] = half_length - along_dist;
    }

    // Quick classification: if triangle is entirely outside ANY bounding line, keep as-is.
    if ((d1[0] <= -tol && d1[1] <= -tol && d1[2] <= -tol) ||
        (d2[0] <= -tol && d2[1] <= -tol && d2[2] <= -tol) ||
        (dL[0] <= -tol && dL[1] <= -tol && dL[2] <= -tol) ||
        (dR[0] <= -tol && dR[1] <= -tol && dR[2] <= -tol)) {
        return -1;
    }

    // If entirely inside ALL bounding lines, discard.
    if ((d1[0] >= tol && d1[1] >= tol && d1[2] >= tol) &&
        (d2[0] >= tol && d2[1] >= tol && d2[2] >= tol) &&
        (dL[0] >= tol && dL[1] >= tol && dL[2] >= tol) &&
        (dR[0] >= tol && dR[1] >= tol && dR[2] >= tol)) {
        return 0;
    }

    // Iterative split: 4 half-planes.
    // For each plane, split current "inside" polygon into outside (emit) + inside (carry forward).
    // Planes defined as: point, sth, cth, keep_positive=true means "inside the strut".
    // Length ends: the line at +half_length along the strut has normal = along direction.
    //   Point on line: (cx + half_length*along_cth, cy + half_length*along_sth)
    //   Normal pointing inward: (-along_sth, along_cth)? No — we want "inside" = toward center.
    //   signed_dist for "left end": -(x-ex)*along_sth + (y-ey)*along_cth ... that's perpendicular.
    //   Actually simpler: use the along_dist formulation directly via sh_clip_line with a
    //   synthetic line. The "left end" line passes through (cx - half_length*along_cth, cy - half_length*along_sth)
    //   with perpendicular direction (along_cth, along_sth) as the line direction...
    //   Actually let's just define 4 (px,py,sth,cth) tuples:
    struct HalfPlane { double px, py, sth, cth; };
    // Length end lines: perpendicular to strut direction.
    // The "left end" line at -half_length: point = center - half_length * along_dir.
    //   Its normal pointing "inward" (toward center) is +along direction.
    //   signed_dist = -(x-px)*sth + (y-py)*cth with (sth,cth) = (-along_cth, -along_sth)?
    //   Wait, line_signed_dist(x,y, px,py,sth,cth) = -(x-px)*sth + (y-py)*cth
    //   For the left end: we want "inside" = points where along_dist > -half_length
    //   i.e. (x-cx)*along_cth + (y-cy)*along_sth > -half_length
    //   i.e. (x-(cx-hl*ac))*ac + (y-(cy-hl*as))*as > 0
    //   We need: -(x-px)*sth + (y-py)*cth > 0 to match keep_positive=true
    //   So: sth = -along_cth, cth = along_sth, px = cx - hl*along_cth, py = cy - hl*along_sth
    //   Check: -(x-px)*(-ac) + (y-py)*(as) = (x-px)*ac + (y-py)*as
    //         = (x-cx+hl*ac)*ac + (y-cy+hl*as)*as = (x-cx)*ac + (y-cy)*as + hl*(ac²+as²)
    //         = along_dist + hl. Yes! That's dL. Good.
    // For the right end: "inside" = along_dist < half_length
    //   i.e. hl - along_dist > 0
    //   sth = along_cth, cth = -along_sth, px = cx + hl*along_cth, py = cy + hl*along_sth
    //   Check: -(x-px)*(ac) + (y-py)*(-as) = -(x-cx-hl*ac)*ac - (y-cy-hl*as)*as
    //         = -((x-cx)*ac + (y-cy)*as - hl) = hl - along_dist = dR. Good.
    double lx = cx - half_length*along_cth, ly = cy - half_length*along_sth;
    double rx = cx + half_length*along_cth, ry = cy + half_length*along_sth;
    HalfPlane planes[4] = {
        {p1x, p1y, sth1, cth1},
        {p2x, p2y, sth2, cth2},
        {lx, ly, -along_cth, along_sth},
        {rx, ry, along_cth, -along_sth},
    };

    int ntri_out = 0;
    auto emit_poly = [&](const double* poly, int nv) {
        for (int i = 1; i <= nv - 2; i++) {
            double ax = poly[2*i] - poly[0], ay = poly[2*i+1] - poly[1];
            double bx = poly[2*(i+1)] - poly[0], by = poly[2*(i+1)+1] - poly[1];
            if (std::abs(ax*by - ay*bx) < 2e-30) continue;
            double* dst = out_buf + ntri_out*6;
            dst[0] = poly[0]; dst[1] = poly[1];
            dst[2] = poly[2*i]; dst[3] = poly[2*i+1];
            dst[4] = poly[2*(i+1)]; dst[5] = poly[2*(i+1)+1];
            ntri_out++;
        }
    };

    // We process the triangle through each half-plane. At each step we have a set of
    // "inside" polygons. For each, we split by the next plane: outside pieces are emitted,
    // inside pieces carried forward.
    // Max polygon vertices after N clips: 3 + N = 7. Flat = 14 doubles.
    // We'll maintain a list of "inside" polygons. After 4 planes, max inside polygons = 1
    // (since we're intersecting convex half-planes against a convex polygon = convex result).
    // But we can have multiple outside pieces. Max output triangles ~ 4*3 = 12 (generous).
    double inside_buf[2][16];  // double-buffer for the single inside polygon
    int inside_n = 3;
    double* inside_cur = inside_buf[0];
    double* inside_tmp = inside_buf[1];
    inside_cur[0] = tri[0][0]; inside_cur[1] = tri[0][1];
    inside_cur[2] = tri[1][0]; inside_cur[3] = tri[1][1];
    inside_cur[4] = tri[2][0]; inside_cur[5] = tri[2][1];

    double outside_buf[16];
    for (int p = 0; p < 4; p++) {
        if (inside_n < 3) break;
        // Split inside_cur by planes[p]: outside piece emitted, inside piece kept.
        int n_outside = sh_clip_line(inside_cur, inside_n, outside_buf,
                                     planes[p].px, planes[p].py, planes[p].sth, planes[p].cth, false);
        int n_inside  = sh_clip_line(inside_cur, inside_n, inside_tmp,
                                     planes[p].px, planes[p].py, planes[p].sth, planes[p].cth, true);
        if (n_outside >= 3) emit_poly(outside_buf, n_outside);
        inside_n = n_inside;
        // Swap buffers
        double* swap = inside_cur; inside_cur = inside_tmp; inside_tmp = swap;
    }
    // Whatever remains in inside_cur after all 4 planes is inside the strut → discard.

    return ntri_out > 0 ? ntri_out : 0;
}

// Clip all triangles against one finite strut rectangle, keeping the exterior.
//
// tri_in_ptr  : (ntri_in, 3, 2) float64, C-contiguous
// p1x..cth2  : projected strut edge parameters (from _project_spider_vane)
// cx, cy     : strut center in pupil coords
// length     : strut length (clipping extends ±length/2 from center along strut)
// tol         : boundary tolerance
// tri_out_ptr : pre-allocated output buffer (ntri_in * 6 triangles max)
// n_removed_ptr, n_clipped_ptr : int32* counters
//
// Returns: number of output triangles.
int clip_triangles_to_strut(
    size_t tri_in_ptr, int ntri_in,
    double p1x, double p1y, double sth1, double cth1,
    double p2x, double p2y, double sth2, double cth2,
    double cx, double cy, double length,
    double tol,
    size_t tri_out_ptr,
    size_t n_removed_ptr, size_t n_clipped_ptr
) {
    const double* in  = reinterpret_cast<const double*>(tri_in_ptr);
    double*       out = reinterpret_cast<double*>(tri_out_ptr);
    int* nrem  = reinterpret_cast<int*>(n_removed_ptr);
    int* nclip = reinterpret_cast<int*>(n_clipped_ptr);

    // Orient edges so that the strut interior has positive signed distance
    // to both lines. The midpoint between the two edge centers is inside.
    double mx = 0.5*(p1x + p2x), my = 0.5*(p1y + p2y);
    double d1_mid = line_signed_dist(mx, my, p1x, p1y, sth1, cth1);
    double d2_mid = line_signed_dist(mx, my, p2x, p2y, sth2, cth2);
    double s1 = (d1_mid >= 0) ? 1.0 : -1.0;
    double s2 = (d2_mid >= 0) ? 1.0 : -1.0;
    double eff_sth1 = sth1 * s1, eff_cth1 = cth1 * s1;
    double eff_sth2 = sth2 * s2, eff_cth2 = cth2 * s2;

    // Along-strut direction: average of the two edge directions (they should be parallel).
    double along_cth = 0.5*(cth1 + cth2);
    double along_sth = 0.5*(sth1 + sth2);
    double norm = std::sqrt(along_cth*along_cth + along_sth*along_sth);
    along_cth /= norm;
    along_sth /= norm;
    double half_length = length * 0.5;

    int ntri_out = 0;
    for (int k = 0; k < ntri_in; k++) {
        const double* tv = in + k*6;
        double tri[3][2] = {{tv[0],tv[1]},{tv[2],tv[3]},{tv[4],tv[5]}};

        // Max output per triangle: 6 (split by 4 half-planes)
        double local_buf[6*6];
        int nv = clip_one_triangle_to_strut(
            tri, p1x, p1y, eff_sth1, eff_cth1,
            p2x, p2y, eff_sth2, eff_cth2,
            cx, cy, along_sth, along_cth, half_length,
            tol, local_buf
        );

        if (nv == -1) {
            double* dst = out + ntri_out*6;
            dst[0]=tv[0]; dst[1]=tv[1]; dst[2]=tv[2];
            dst[3]=tv[3]; dst[4]=tv[4]; dst[5]=tv[5];
            ntri_out++;
        } else if (nv > 0) {
            (*nclip)++;
            for (int i = 0; i < nv; i++) {
                double* dst = out + ntri_out*6;
                double* src = local_buf + i*6;
                dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2];
                dst[3]=src[3]; dst[4]=src[4]; dst[5]=src[5];
                ntri_out++;
            }
        } else {
            (*nrem)++;
        }
    }
    return ntri_out;
}


// ---------------------------------------------------------------------------
// Triangle image accumulation
// ---------------------------------------------------------------------------

// One half-plane pass of Sutherland-Hodgman clipping.
// poly/out: flat (x0,y0, x1,y1, ...) doubles.  Returns output vertex count.
static int sh_clip_edge_cpp(
    const double* poly, int n, double* out,
    int axis, double val, int inside_dir
) {
    int m = 0;
    for (int i = 0; i < n; i++) {
        int j = (i == 0) ? n - 1 : i - 1;       // previous vertex index
        double px = poly[2*i],   py = poly[2*i+1];
        double sx = poly[2*j],   sy = poly[2*j+1];
        double pval = (axis == 0) ? px : py;
        double sval = (axis == 0) ? sx : sy;
        bool p_in = (inside_dir > 0) ? (pval >= val) : (pval <= val);
        bool s_in = (inside_dir > 0) ? (sval >= val) : (sval <= val);
        if (p_in) {
            if (!s_in) {
                double t = (val - sval) / (pval - sval);
                out[2*m]   = sx + t*(px - sx);
                out[2*m+1] = sy + t*(py - sy);
                ++m;
            }
            out[2*m]   = px;
            out[2*m+1] = py;
            ++m;
        } else if (s_in) {
            double t = (val - sval) / (pval - sval);
            out[2*m]   = sx + t*(px - sx);
            out[2*m+1] = sy + t*(py - sy);
            ++m;
        }
    }
    return m;
}

// Area of the intersection of a triangle with the unit pixel square [ix±0.5, iy±0.5].
// Clips the triangle against four axis-aligned half-planes using Sutherland-Hodgman.
// A triangle clipped by a rectangle produces at most 7 vertices → 14 doubles; buf size 16 is safe.
static double clip_area_cpp(const double tri[3][2], int ix, int iy) {
    double x0 = ix - 0.5, x1 = ix + 0.5;
    double y0 = iy - 0.5, y1 = iy + 0.5;
    double a[16], b[16];
    a[0]=tri[0][0]; a[1]=tri[0][1];
    a[2]=tri[1][0]; a[3]=tri[1][1];
    a[4]=tri[2][0]; a[5]=tri[2][1];
    int n = 3;
    n = sh_clip_edge_cpp(a, n, b, 0, x0, +1);  if (n < 3) return 0.0;
    n = sh_clip_edge_cpp(b, n, a, 0, x1, -1);  if (n < 3) return 0.0;
    n = sh_clip_edge_cpp(a, n, b, 1, y0, +1);  if (n < 3) return 0.0;
    n = sh_clip_edge_cpp(b, n, a, 1, y1, -1);  if (n < 3) return 0.0;
    // Shoelace on final polygon in `a`
    double area = 0.0;
    for (int i = 0, j = n-1; i < n; j = i++)
        area += a[2*j]*a[2*i+1] - a[2*i]*a[2*j+1];
    return 0.5 * std::abs(area);
}

// Accumulate triangle flux onto a pixel image.
//
// tri_px_ptr   : (ntri, 3, 2) float64, C-contiguous — triangle vertices in
//               centred pixel coords (origin at image centre).
// pupil_areas  : (ntri,) float64 — pupil-space area per triangle (flux proxy).
// proj_areas   : (ntri,) float64 — projected area in pixel² per triangle.
// image_ptr    : (npix*npix,) float64, C-contiguous row-major — written in-place.
// ntri, npix   : array sizes.
//
// For each valid triangle the function:
//   1. Computes a tight pixel bounding box.
//   2. For each pixel in the box, clips the triangle to the pixel square and
//      accumulates flux * overlap_area / proj_area into image.
//
void accumulate_triangles(
    size_t tri_px_ptr,
    size_t pupil_areas_ptr,
    size_t proj_areas_ptr,
    size_t image_ptr,
    int ntri,
    int npix
) {
    const double* tri_px = reinterpret_cast<const double*>(tri_px_ptr);
    const double* pa     = reinterpret_cast<const double*>(pupil_areas_ptr);
    const double* pj     = reinterpret_cast<const double*>(proj_areas_ptr);
    double*       img    = reinterpret_cast<double*>(image_ptr);

    int no2 = (npix - 1) / 2;

    for (int k = 0; k < ntri; k++) {
        double proj = pj[k];
        if (proj <= 0.0) continue;

        // Each triangle is 6 consecutive doubles: v0x,v0y, v1x,v1y, v2x,v2y
        const double* tv = tri_px + k*6;
        double tri[3][2] = { {tv[0],tv[1]}, {tv[2],tv[3]}, {tv[4],tv[5]} };

        // Pre-divide flux by projected area (replicates Python: flux*area/aproj)
        double flux = pa[k] / proj;

        // Bounding box in centred-pixel integer coords.
        // floor(val + 0.5) matches the Python: int(np.floor(val + 0.5))
        int ixmin = (int)std::floor(std::min({tv[0], tv[2], tv[4]}) + 0.5);
        int ixmax = (int)std::floor(std::max({tv[0], tv[2], tv[4]}) + 0.5);
        int iymin = (int)std::floor(std::min({tv[1], tv[3], tv[5]}) + 0.5);
        int iymax = (int)std::floor(std::max({tv[1], tv[3], tv[5]}) + 0.5);

        ixmin = std::max(ixmin, -no2);
        ixmax = std::min(ixmax,  no2);
        iymin = std::max(iymin, -no2);
        iymax = std::min(iymax,  no2);

        for (int iy = iymin; iy <= iymax; iy++) {
            for (int ix = ixmin; ix <= ixmax; ix++) {
                double area = clip_area_cpp(tri, ix, iy);
                if (area > 0.0)
                    img[(iy + no2)*npix + (ix + no2)] += flux * area;
            }
        }
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
    m.def("clip_triangles_to_circle", &clip_triangles_to_circle);
    m.def("clip_triangles_to_strut", &clip_triangles_to_strut);
    m.def("accumulate_triangles", &accumulate_triangles);
}
