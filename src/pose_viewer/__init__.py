try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
from ._widget import ExampleQWidget
from ._loader import HyperParams
from ._loader import ZebData
from .utils import Animation

__all__ = (
    "ExampleQWidget"
)
