# Danish — Claude Code notes

## Running Python scripts

SIP on macOS strips `PATH`, `PYTHONPATH`, and `LD_LIBRARY_PATH` from child
processes, so a bare `python` command cannot find batoid, danish, or other
LSST-stack packages.  Always source `.env` before running anything:

```bash
(set -a && source .env && python some_script.py)
```

This applies to any command that imports `batoid`, `danish`, or other
LSST-stack packages.
