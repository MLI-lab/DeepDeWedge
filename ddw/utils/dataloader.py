import torch
from torch.utils.data.distributed import DistributedSampler

MultiEpochsDataLoader = torch.utils.data.DataLoader

# class MultiEpochsDataLoader(torch.utils.data.DataLoader):
#     def __init__(self, dataset, batch_size, num_workers=0, ddp_enabled=True, **kwargs):
#         # If DDP is enabled, use DistributedSampler; otherwise, use the default sampler
#         if ddp_enabled:
#             sampler = DistributedSampler(dataset, shuffle=kwargs.get('shuffle', True))
#         else:
#             sampler = kwargs.get('sampler', None)
        
#         batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=False)

#         # Use _RepeatSampler to repeat the sampler in both cases
#         self.batch_sampler = _RepeatSampler(batch_sampler)
        
#         super().__init__(dataset, batch_sampler=self.batch_sampler, num_workers=num_workers, **kwargs)
#         self._DataLoader__initialized = True
#         self.iterator = super().__iter__()

#     def __len__(self):
#         # Return the correct length for distributed training
#         return len(self.batch_sampler.sampler)

#     def __iter__(self):
#         # Ensure we maintain proper iteration
#         for i in range(len(self)):
#             yield next(self.iterator)


# class _RepeatSampler:
#     """Sampler that repeats batches forever."""
    
#     def __init__(self, sampler):
#         self.sampler = sampler

#     def __iter__(self):
#         while True:
#             yield from iter(self.sampler)

