try:
    from ._version import version as __version__
except ImportError:
    __version__ = 'unknown'

from ._sample_data import make_sample_data, make_sample_data_3d
from ._widget import ColocalisationWidget

__all__ = (
    'ColocalisationWidget',
    'make_sample_data',
    'make_sample_data_3d',
)
