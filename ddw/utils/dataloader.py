import torch
from torch.utils.data import BatchSampler

class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """
    The MultiEpochsDataLoader is a PyTorch dataloader that re-uses worker processes rather than re-initializing the every epoch (see https://github.com/pytorch/pytorch/issues/15849#issuecomment-518126031).
    For DeepDeWedge, we found that the MultiEpochsDataLoader significantly reduces the fitting time compared to the standard dataloder when epochs are short, i.e., consist of few batches.
    This is likely due to the computationally expensive spatial rotations that are applied whenever a new sub-tomogram pair is sampled from the training set.
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
        self.batch_size = self.batch_sampler.batch_size  # ensure batch_size is set because it was not found when updating the subtomo missing wedges in the multi gpu setup
        self.batch_sampler = _RepeatSampler(**kwargs)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(BatchSampler):
    """
    Wrapper of BatchSampler that repeats batches indefinitely.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        while True:
            yield from super().__iter__()