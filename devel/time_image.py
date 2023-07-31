import os
import time
import yaml

import numpy as np
import batoid
import danish
from danish import DonutFactory


def time_image():
    obsc = yaml.safe_load(open(os.path.join(danish.datadir, 'RubinObsc.yaml')))
    factory = DonutFactory(
        obsc_radii=obsc['radii'],
        obsc_centers=obsc['centers'],
        obsc_th_mins=obsc['th_mins'],
    )

    telescope = batoid.Optic.fromYaml("LSST_r.yaml")
    telescope = telescope.withGloballyShiftedOptic("Detector", (0,0,0.0015))
    zref = batoid.zernikeTA(
        telescope, np.deg2rad(1.67), 0.0, 620e-9,
        jmax=66, nrad=20, naz=120, reference='chief', eps=0.61
    )

    N = 200
    np.random.seed(123)
    t0 = time.time()
    for _ in range(N):
        aberrations = np.array(zref)
        aberrations[4] += np.random.uniform(-0.1, 0.1)
        aberrations[5:23] += np.random.uniform(-0.1, 0.1, size=18)
        aberrations *= 620e-9
        img = factory.image(
            aberrations=aberrations, thx=np.deg2rad(1.67), thy=0.0
        )
    t1 = time.time()
    print(f"Time for factory.image(): {(t1-t0)/N*1e3:.2f} ms")


if __name__ == "__main__":
    time_image()
