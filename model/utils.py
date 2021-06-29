import torch

# Don't remove this file and don't change the imports of load_state_dict_from_url
# from other files. We need this so that we can swap load_state_dict_from_url with
# a custom internal version in fbcode.
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

def load_init_model_state(from_model, to_model):
    from_state_keys = [k for k, v in from_model.state_dict().items()]
    model_dict = to_model.state_dict()
    for k, v in to_model.state_dict().items():
        if k in from_state_keys:
            model_dict[k] = from_model.state_dict()[k]
    to_model.load_state_dict(model_dict)
    return to_model
