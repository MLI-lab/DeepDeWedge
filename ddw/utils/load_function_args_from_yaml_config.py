import inspect
from typer_config.loaders import yaml_loader


def load_function_args_from_yaml_config(
    function, yaml_config_file, shared_field_name="shared"
):
    """
    This function is used to make the command line interface compatible with the .yaml config files.
    """
    if yaml_config_file is None:
        return {}
    cfg = yaml_loader(yaml_config_file)
    cfg_shared = cfg[shared_field_name] if shared_field_name in cfg.keys() else {}
    if function.__name__ not in cfg.keys():
        raise ValueError(
            f"Field '{function.__name__}' not found in config file '{yaml_config_file}'!"
        )
    cfg = cfg[function.__name__]
    args = {}
    for arg in inspect.getfullargspec(function).args:
        if arg == "config":
            continue
        found_in_common = False
        if arg in cfg_shared.keys():
            args[arg] = cfg_shared[arg]
            found_in_common = True
        if arg in cfg.keys():
            if found_in_common:
                print(
                    f"Overriding value '{args[arg]}' for argument '{arg}' found in field '{shared_field_name}' with value '{cfg[arg]}' from field '{function.__name__}'!"
                )
            args[arg] = cfg[arg]
    return args
