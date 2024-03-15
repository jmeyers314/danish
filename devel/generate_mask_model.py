import batoid
from lsst.ts.wep.utils import maskUtils


def main(args):
    optic = batoid.Optic.fromYaml(args.opticFile)
    maskUtils.printMaskModel(
        maskUtils.pruneMaskModel(
            maskUtils.fitMaskModel(
                optic,
                wavelength=args.wavelength,
                deg=args.deg,
                thetaMax=args.thetaMax,
                dTheta=args.dTheta
            )[0]
        )
    )


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("opticFile", type=str, help="Path to the yaml file describing the optical system")
    parser.add_argument("wavelength", type=float, help="Wavelength to use in meters")
    parser.add_argument("--deg", default=3, type=int, help="Degree of polynomial to use")
    parser.add_argument("--thetaMax", default=2.0, type=float, help="Maximum field angle in degrees")
    parser.add_argument("--dTheta", default=0.01, type=float, help="Step size in degrees")
    args = parser.parse_args()

    main(args)
