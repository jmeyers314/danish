import time

import numpy as np
import batoid


def time_image():
    from danish import DonutFactory

    obsc_radii = {
        'M1_inner': 2.5498,
        'M2_inner': 2.3698999752679404,
        'M2_outer': 4.502809953009087,
        'M3_inner': 1.1922312943631603,
        'M3_outer': 5.436574702296011,
        'L1_entrance': 7.697441260764198,
        'L1_exit': 8.106852624652701,
        'L2_entrance': 10.748915941599885,
        'L2_exit': 11.5564127895276,
        'Filter_entrance': 28.082220873785978,
        'Filter_exit': 30.91023954045243,
        'L3_entrance': 54.67312185149621,
        'L3_exit': 114.58705556485711
    }
    obsc_motion = {
        'M1_inner': 0.0,
        'M2_inner': 16.8188788239707,
        'M2_outer': 16.8188788239707,
        'M3_inner': 53.22000661238318,
        'M3_outer': 53.22000661238318,
        'L1_entrance': 131.76650078100135,
        'L1_exit': 137.57031952814913,
        'L2_entrance': 225.6949885074127,
        'L2_exit': 237.01739037674315,
        'Filter_entrance': 802.0137451419788,
        'Filter_exit': 879.8810309773828,
        'L3_entrance': 1597.8959863335774,
        'L3_exit': 3323.60145194633
    }
    factory = DonutFactory(obsc_radii=obsc_radii, obsc_motion=obsc_motion)

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
