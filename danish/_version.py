#https://stackoverflow.com/a/7071358/7530778

__version__ = "0.1.2"
tmp = __version__
if "rc" in tmp:
    tmp = tmp[:tmp.find("rc")]
__version_info__ = tuple(map(int, tmp.split('.')))
