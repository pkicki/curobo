import torch
_original = torch.cuda.get_device_properties
class _Properties:
    def __init__(self, value):
        self.value = value
    def __getattr__(self, name):
        if name == 'shared_memory_per_block':
            return 49152
        return getattr(self.value, name)
def _properties(device):
    return _Properties(_original(device))
torch.cuda.get_device_properties = _properties
