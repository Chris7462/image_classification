class ConfigNode:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                v = ConfigNode(v)
            setattr(self, k, v)

class Config:
    def __init__(self, yaml_file):
        import yaml
        with open(yaml_file, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        self.__dict__.update({k: ConfigNode(v) if isinstance(v, dict) else v
                              for k, v in cfg_dict.items()})
