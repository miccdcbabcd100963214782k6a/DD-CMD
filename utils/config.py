import os
from ast import literal_eval
import yaml

class CfgNode(dict):

    def __init__(self, init_dict=None):
        super().__init__()
        init_dict = {} if init_dict is None else init_dict
        for k, v in init_dict.items():
            self[k] = self._to_node(v)

    def _to_node(self, v):
        if isinstance(v, dict):
            return CfgNode(v)
        return v

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

def load_cfg_from_cfg_file(file: str) -> CfgNode:

    assert os.path.isfile(file) and file.endswith(".yaml"), f"{file} is not a yaml file"
    with open(file, "r") as f:
        cfg_from_file = yaml.safe_load(f)

    flat = {}
    for section in cfg_from_file:
        sec_dict = cfg_from_file[section]
        if isinstance(sec_dict, dict):
            for k, v in sec_dict.items():
                flat[k] = v
        else:
            # allow scalar at top level
            flat[section] = sec_dict

    return CfgNode(flat)

def merge_cfg_from_list(cfg: CfgNode, cfg_list):

    assert len(cfg_list) % 2 == 0, "Override list must be pairs of KEY VALUE"
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        if k not in cfg:
            raise KeyError(f"Unknown config key: {k}")
        # try to keep type
        cur = cfg[k]
        try:
            if isinstance(cur, bool):
                vv = v.lower() in ("true", "1", "yes", "y")
            elif isinstance(cur, int):
                vv = int(v)
            elif isinstance(cur, float):
                vv = float(v)
            elif isinstance(cur, (list, tuple, dict)):
                vv = literal_eval(v)
            else:
                vv = v
        except Exception:
            vv = v
        cfg[k] = vv
    return cfg
