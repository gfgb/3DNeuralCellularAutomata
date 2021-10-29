import numpy as np
import torch

class SamplePool:
    def __init__(self, x, _parent=None, _parent_idx=None):
        self.x = x
        self._parent = _parent
        self._parent_idx = _parent_idx
        self._size = x.shape[0]

    def sample(self, n):
        idx = torch.from_numpy(np.random.choice(self._size, n, False).astype(np.int64))
        batch = self.x[idx, ...]
        batch = SamplePool(x=batch, _parent=self, _parent_idx=idx)
        return batch

    def commit(self):
        self._parent.x[self._parent_idx] = self.x