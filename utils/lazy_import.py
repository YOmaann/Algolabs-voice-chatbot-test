import importlib.util
import os

def import_local_module(filepath, module_name=None):
    if not isinstance(filepath, str):
        raise TypeError("Expected filepath as a string")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No such file: {filepath}")
    
    if module_name is None:
        # Derive a safe module name from the file name
        module_name = os.path.splitext(os.path.basename(filepath))[0]

    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None:
        raise ImportError(f"Could not load spec from {filepath}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
