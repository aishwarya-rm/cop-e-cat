import argparse
import functools
from typing import TypedDict

class ArgsDict(TypedDict):
    pass

COPECAT_DESC = "TODO: This description should be filled out more."

ARGUMENTS = [
    {
        'flags': ['--verbose', '-v'],
        'named': {
            'action': 'count',
            'default': 0,
            'help': "Set verbosity level. More vs for more verbosity."
        }
    }
]


def init_configuration() -> ArgsDict:
    """Parse the arguments defined in this file.

    Returns:
        ArgsDict: A dictionary of the global command-line arguments in the program.
    """    
    parser = argparse.ArgumentParser(description=COPECAT_DESC)
    for arg in ARGUMENTS:
        parser.add_argument(*arg['flags'], **arg['named'])
    parsed = parser.parse_args()
    print_per_verbose.__dict__['defined_verbosity_level'] = parsed.verbose or 0
    

def local_verbosity(new_level: int):
    """Decorator. Temporarily reset system verbosity level for a given function scope.

    Args:
        new_level (int): The verbosity level that will hold within the defined scope.
    """
    def inner_local_verbosity(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            old_level = print_per_verbose.__dict__['defined_verbosity_level'] or 0
            print_per_verbose.__dict__['defined_verbosity_level'] = new_level
            try:
                result = func(*args, **kwargs)
            finally:
                print_per_verbose.__dict__['defined_verbosity_level'] = old_level
            return result
        return wrapper
    return inner_local_verbosity

def print_per_verbose(requested_verbosity_level: int, msg: str):
    """Print the input string if verbosity is above a defined threshold.

    Args:
        requested_verbosity_level (int): If this level is (strictly) greater than the
            set verbosity level, the output will be printed; otherwise no-op.
        msg (str): Message to print. Note that this value is passed as a string (i.e. any
            evaluations resulting from f-string substitution will happen, regardless of verbosity
            level). Therefore, don't do anything in defining this string that causes side effects
            or is computationally intensive.
    """    
    # defined_verbosity_level is a static value, initialized from command-line argument at setup time in init_configuration().
    # If it hasn't been set (or we're running in a container or something), just do nothing.
    if ('defined_verbosity_level' not in print_per_verbose.__dict__): return
    if (print_per_verbose.defined_verbosity_level < requested_verbosity_level): return
    tabs = max(0, requested_verbosity_level - 1)
    print("\t" * tabs + msg)
    
