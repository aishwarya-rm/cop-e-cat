""" 
I/O utilities
"""

import csv
import pickle
from typing import Any, Dict, List

def print_list(lst: List[Any]) -> None:
    """Print a (potentially heterogeneous) list of items which can individually be converted to string.

    Args:
        lst (List[Any]): List of items to print.
    """    
    for elem in lst:
        print(str(elem))

def load_pickle(fpath: str) -> Any:
    """Load an arbitrary object from a pickled format. Note that no safety checking is performed.

    Args:
        fpath (str): Path to the pickled object to load.

    Returns:
        Any: The object as hydrated from disk.
    """
    with open(fpath, 'rb') as fp:
        return pickle.load(fp)

def save_pickle(obj: Any, out_fpath: str) -> None:
    """Write an arbitrary object to disk using pickle.

    Args:
        obj (Any): Object to write to disk
        out_fpath (str): Path to which object should be written. Any file at this location will be overwritten.
    """
    with open(out_fpath, 'wb') as fp:
        pickle.dump(obj, fp)

def write_list_to_file(fpath: str, lst: List[Any]) -> None:
    """Write to file a (potentially heterogeneous) list of items which can individually be converted to string.

    Args:
        fpath (str): Path to the file to use as a destination. Any existing file will be overwritten.
        lst (List[Any]): List of items to write to file.
    """    
    with open(fpath, 'w') as fp:
        for elem in lst:
            print(str(elem).strip(), file=fp)

def read_list_from_file(fpath: str, skip_header: bool=False) -> List[str]:
    """Parse a file into an array of strings, splitting on newline, and optionally skipping the first row.

    Args:
        fpath (str): File to read.
        skip_header (bool, optional): If True, the first line is skipped as a header. Defaults to False.

    Returns:
        List[str]: Lines of the file, one list entry per line.
    """    
    with open(fpath, 'r') as fp:
        lines = fp.read().splitlines()
        if skip_header:
            return lines[1:]
        else:
            return lines

def dict_from_text_file(fpath: str) -> Dict[str, float]:
    """Parse a two-column CSV file into a dictionary of string keys and float values.
    Will fail an assertion if the second column contains values which are not valid floats.

    Args:
        fpath (str): Path to file to parse.

    Returns:
        Dict[str, float]: Dictionary of string keys and float values.
    """
    out_dict: Dict[str, float] = {}

    with open(fpath, "r") as fp:
        for line in fp:
            comps = list(map(lambda elem: elem.strip(), line.strip().split(",")))
            assert(len(comps) == 2)
            key = comps[0]
            val = comps[1]
            out_dict[key] = float(val)

    return out_dict

def read_dict_iter(filename: str) -> Dict[str, Any]:
    """Parse a tab-separated file (with header row) into a list of dicts,
    one per file line. Values in the header row determine the keys of the
    dicts. Dicts are yielded one at a time.

    Args:
        filename (str): Path to the file to parse.

    Yields:
        Dict[str, Any]: Dict keyed by the fields in the first row.
    """    
    with open(filename) as thefile:
        reader = csv.DictReader(thefile, delimiter='\t')
        for datum in reader:
            yield datum

def dict_to_text_file(fpath: str, in_dict: Dict[Any, Any]) -> None:
    """Write dictionary to disk as two-column csv. Dicts are assumed to be
    keyed by strings and have values that are floats (which will be preserved
    up to 5 significant figures) but this is not formally checked.

    Args:
        fpath (str): File path to use as data destination. Any existing
            file will be overwritten.
        in_dict (Dict[Any, Any]): Dictionary to write to disk.
    """    
    with open(fpath, "w") as fp:
        for key in in_dict:
            print("{},{:.5f}".format(key.strip(), in_dict[key]), file=fp)