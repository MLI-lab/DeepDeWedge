import torch


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """
    The MultiEpochsDataLoader is a PyTorch dataloader that re-uses worker processes rather than re-initializing the every epoch(see https://github.com/pytorch/pytorch/issues/15849#issuecomment-518126031).
    For DeepDeWedge, we found that the MultiEpochsDataLoader significantly reduces the fitting time compared to the standard dataloder when epochs are short, i.e., consist of few batches.
    This is likely due to the computationally expensive spatial rotations that are applied whenever a new sub-tomogram pair is sampled from the training set.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
