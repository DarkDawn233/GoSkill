import json5 as json
from pathlib import Path


def get_default_args():
    file_path = Path(__file__).parents[2] / 'config' / 'default.json'
    with open(file_path) as f:
        params = json.load(f)
    return params


def get_args(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        file_path = Path(__file__).parents[2] / 'config' / subfolder / f'{config_name}.json'
        with open(file_path) as f:
            args = json.load(f)
        return args


def eval_v(value):
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def update_args(args, params):
    if len(params) == 1:
        return args
    assert params[1] == 'with'
    params = params[2:]
    for _v in params:
        _k, _v = _v.split("=")
        if '.' in _k:
            _k = _k.split('.')
            args[_k[0]][_k[1]] = eval_v(_v)
        else:
            args[_k] = eval_v(_v)
    return args


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
