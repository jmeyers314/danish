import numpy as np
import pytest


def timer(f):
    import functools

    @functools.wraps(f)
    def f2(*args, **kwargs):
        import time
        t0 = time.time()
        result = f(*args, **kwargs)
        t1 = time.time()
        fname = repr(f).split()[1]
        print('time for %s = %.2f' % (fname, t1-t0))
        return result
    return f2


def runtests(filename, parser=None):
    if parser is None:
        from argparse import ArgumentParser
        parser = ArgumentParser()
    parser.add_argument('--profile', action='store_true', help='Profile tests')
    parser.add_argument('--prof_out', default=None, help="Profiler output file")
    args, unknown_args = parser.parse_known_args()
    pytest_args = [filename] + unknown_args + ["--run_slow", "--tb=short", "-s"]

    if args.profile:
        import cProfile, pstats
        pr = cProfile.Profile()
        pr.enable()
    pytest.main(pytest_args)
    if args.profile:
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('tottime')
        ps.print_stats(30)
        if args.prof_out:
            ps.dump_stats(args.prof_out)
