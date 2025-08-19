from fla.models import MambaConfig

class EnhancedMambaConfig(MambaConfig):
    def __init__(self, sparse_arch="none", data_max_length=2048, sparse_keys=64, num_hash=64, **kwargs):
        self.sparse_arch = sparse_arch
        self.data_max_length = data_max_length
        self.sparse_keys = sparse_keys
        self.num_hash = num_hash
        super().__init__(**kwargs)