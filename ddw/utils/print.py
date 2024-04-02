import inspect
import yaml

def pprint_dict(d):
    """
    Pretty print a dictionary.
    d: dict, dictionary to print
    """
    print(yaml.dump(d, default_flow_style=False))

def print_help_for_function_arguments(func, arg_names=None, print_only_required=False):
    sig = inspect.signature(func)
    if arg_names is None:
        arg_names = sig.parameters.keys()
    if not hasattr(arg_names, '__iter__'):
        arg_names = [arg_names]
    for arg_name in arg_names:
        param = sig.parameters[arg_name]
        default_value = param.default
        required = default_value == inspect.Parameter.empty
        if print_only_required and not required:
            continue
        try:
            default_value = None if required else param.default
            helpstr = param.annotation.__metadata__[0].help
            defaultstr = f"\n\tDEFAULT: {default_value}" if not required else ""
            print(f"ARGUMENT '{arg_name}'\n\tREQUIRED: {required}{defaultstr}\n\tHELP: {helpstr}")
        except:
            print(f"ERROR: Failed to load help for '{arg_name}' from annotation metadata!")