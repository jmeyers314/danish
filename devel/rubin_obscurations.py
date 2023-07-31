import batoid
import numpy as np
import matplotlib.pyplot as plt

"""Script to use batoid to model the projection of obscurations along the beam
as a function of the incoming field angle.
"""

telescope = batoid.Optic.fromYaml("LSST_r.yaml")

def model(th):
    """Full trace at field angle th.  Then fit a linear function in x and y
    of surface intersection point vs pupil point.

    Return the coefficients for each surface.
    """
    thx = 0.0
    thy = th
    rays = batoid.RayVector.asPolar(
        telescope,
        theta_x=np.deg2rad(thx),
        theta_y=np.deg2rad(thy),
        wavelength=620e-9,
        nrad=50,
        naz=200,
        inner=2.3
    )
    tf = telescope.traceFull(rays)
    rays = telescope.stopSurface.interact(rays.copy())
    u0, v0 = rays.x, rays.y

    out = {}
    for s in tf.keys():
        if s == 'Detector':
            continue
        u1, v1 = tf[s]['out'].x, tf[s]['out'].y
        rx, resx, _, _, _ = np.polyfit(u0, u1, 1, full=True)
        ry, resy, _, _, _ = np.polyfit(v0, v1, 1, full=True)
        out[s] = rx, ry
    return out

# Determine how surface/pupil coordinate transformations evolve with field
# angle.
scales = {}
centroids = {}
ths = np.linspace(0.0, 2.0, 20)
for th in ths:
    data = model(th)
    for k in data:
        if k not in scales:
            scales[k] = []
            centroids[k] = []
        rx = data[k][0]
        ry = data[k][1]
        scales[k].append(np.mean([rx[0], ry[0]]))  # good enough?
        centroids[k].append(ry[1])

pupil_radii = {}
pupil_motion = {}
for k in scales:
    r, res, _, _, _ = np.polyfit(np.deg2rad(ths), centroids[k], 1, full=True)
    motion = r[0] / np.mean(scales[k])
    obsc = telescope[k].obscuration.original
    if isinstance(obsc, batoid.ObscAnnulus):
        pupil_radii[k+'_outer'] = obsc.outer / np.mean(scales[k])
        pupil_radii[k+'_inner'] = obsc.inner / np.mean(scales[k])
        pupil_motion[k+'_outer'] = np.deg2rad(motion)
        pupil_motion[k+'_inner'] = np.deg2rad(motion)
    elif isinstance(obsc, batoid.ObscCircle):
        pupil_radii[k] = obsc.radius / np.mean(scales[k])
        pupil_motion[k] = np.deg2rad(motion)

print("radii")
for k, v in pupil_radii.items():
    print(f"'{k}': {v}")
print()
print("motion")
for k, v in pupil_motion.items():
    print(f"'{k}': {v}")
