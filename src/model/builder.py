from omegaconf import OmegaConf
import inspect
from typing import Any, Dict

class Registry:
    """A minimal reimplementation of mmcv.Registry"""

    def __init__(self, name: str, build_func=None, parent=None):
        self._name = name
        self._module_dict = {}
        self._parent = parent
        self._build_func = build_func if build_func is not None else build_from_cfg

    def register_module(self, cls=None, name=None):
        """Register a module. Can be used as decorator or function."""
        if cls is None:
            def _register(cls):
                self._register_module(cls, name)
                return cls
            return _register
        self._register_module(cls, name)
        return cls

    def _register_module(self, cls, name=None):
        module_name = name or cls.__name__
        if module_name in self._module_dict:
            raise KeyError(f"{module_name} already registered in {self._name}")
        self._module_dict[module_name] = cls

    def get(self, key: str):
        if key in self._module_dict:
            return self._module_dict[key]
        if self._parent is not None:
            return self._parent.get(key)
        raise KeyError(f"{key} not found in {self._name}")

    def build(self, cfg: Dict[str, Any], *args, **kwargs):
        return self._build_func(cfg, self, *args, **kwargs)

    def __contains__(self, key):
        return key in self._module_dict

    def __repr__(self):
        return f"Registry(name={self._name}, items={list(self._module_dict.keys())})"


def build_from_cfg(cfg: Dict[str, Any], registry: Registry, *args, **kwargs):
    """A minimal version of mmcv.build_from_cfg."""
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict")

    if "type" not in cfg:
        raise KeyError("`cfg` must contain the key 'type'")

    obj_type = cfg["type"]
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError("`type` must be a str or class, but got "
                        f"{type(obj_type)}")

    # remove "type" from kwargs
    args_cfg = {k: v for k, v in cfg.items() if k != "type"}
    return obj_cls(*args, **args_cfg, **kwargs)

MODELS = Registry('model')
def build_model(config):
    model = MODELS.build(OmegaConf.to_container(config, resolve=True))
    return model