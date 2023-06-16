from . import _BasicKits
from . import StanzaKits
from . import GensimKits
from . import resources

# out source pack

# --------------------------------------------------------------------------
# l0 level functions
# --------------------------------------------------------------------------

"""function aliases"""


def alias_delete_whole_dir(*args, **kwargs):
    """delete the whole dir..."""
    _BasicKits._BasicFuncT.delete_whole_dir(*args, **kwargs)


def alias_file_to_list(*args, **kwargs):
    """alias to _BasicT.file_to_list"""
    return _BasicKits.FileT.file_to_list(*args, **kwargs)


def alias_list_to_file(*args, **kwargs):
    """alias to _BasicT.file_to_list"""
    _BasicKits.FileT.list_to_file(*args, **kwargs)


def alias_write_dict_to_csv(*args, **kwargs):
    """alias to _BasicT.write_dict_to_csv"""
    return _BasicKits.FileT.write_dict_to_csv(*args, **kwargs)
