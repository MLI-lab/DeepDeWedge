import torch
from torch.utils.data import BatchSampler

class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """
    The MultiEpochsDataLoader is a PyTorch dataloader that re-uses worker processes rather than re-initializing them every epoch.
    For DeepDeWedge, the MultiEpochsDataLoader significantly reduces the fitting time when epochs are short.
    This is due to computationally expensive spatial rotations applied whenever a new sub-tomogram pair is sampled from the training set.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        # Replace batch_sampler with an instance of _RepeatSampler
        kwargs = {
            "sampler": self.batch_sampler.sampler,
            "batch_size": self.batch_sampler.batch_size,
            "drop_last": self.batch_sampler.drop_last,
        }
        self.batch_sampler = _RepeatSampler(**kwargs)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(BatchSampler):
    """Sampler that repeats batches indefinitely.

    Args:
        batch_sampler (BatchSampler): The original batch sampler to repeat.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        while True:
            yield from super().__iter__()