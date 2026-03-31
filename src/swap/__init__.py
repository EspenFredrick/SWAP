"""SWAP — Solar Wind Analysis and Propagation toolkit."""
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("swap")
except PackageNotFoundError:
    __version__ = "dev"
