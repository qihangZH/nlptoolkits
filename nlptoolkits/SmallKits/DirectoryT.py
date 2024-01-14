import os
import pathlib
import typing
import warnings


def check_make_directory(check_make_dir, force: bool = False):
    """
    Check and Make directory
    :param check_make_dir: the directory you hope to check-make
    :param force: is force to make the recursive dirs
    """
    folder = os.path.exists(check_make_dir)

    if not folder:
        if force:
            warnings.warn("\033[31myou already set 'force=True' for recursive, however unsafe\033[0m", UserWarning)
            temp_folder_maker = os.makedirs
        else:
            temp_folder_maker = os.mkdir

        temp_folder_maker(check_make_dir)

        print('Make dir:{}'.format(check_make_dir))
    else:
        print('{} already exist.'.format(check_make_dir))


def traverse_dir_items_list(directory, return_type: typing.Literal['all', 'file', 'folder'] = "all", walk=False):
    """
    Traverse through the folder and return the list of item names based on the specified return type.

    Args:
        directory: The directory path to search for items.
        return_type: Return type of items. Options are "all" (default), "file", or "folder".
        walk: Whether to perform a recursive search. Default is False.

    Returns:
        A list of item names.

    """
    if not (return_type in ['all', 'file', 'folder']):
        raise ValueError("Wrong return type, must be ['all', 'file', 'folder']")

    return_item_list = []

    if walk:
        print("\033[31myou already set 'walk=True' for recursive, be aware\033[0m")
        for root, dirs, files in os.walk(directory):
            if (return_type == "all") or (return_type == "folder"):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    return_item_list.append(str(pathlib.Path(dir_path)))
            if (return_type == "all") or (return_type == "file"):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    return_item_list.append(str(pathlib.Path(file_path)))
    else:
        print("\033[31myou already set 'walk=False' for not recursive, be aware\033[0m")
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if (os.path.isdir(item_path)) and (return_type == "all" or return_type == "folder"):
                return_item_list.append(str(pathlib.Path(item_path)))
            elif (os.path.isfile(item_path)) and (return_type == "all" or return_type == "file"):
                return_item_list.append(str(pathlib.Path(item_path)))

    return return_item_list
