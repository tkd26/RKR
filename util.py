import yaml

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)