import torch
import tqdm


def get_avg_model_input_mean_and_var(dataloader, batches=None, verbose=False):
    if batches is None:
        batches = 3 * len(dataloader)
    chunks = []
    bar = (
        tqdm.tqdm(range(batches), desc="Getting average model input mean and variance")
        if verbose
        else range(batches)
    )
    iter_loader = iter(dataloader)
    for _ in bar:
        try:
            item = next(iter_loader)
        except StopIteration:
            iter_loader = iter(dataloader)
            item = next(iter_loader)
        chunks.append(item["model_input"])
    chunks = torch.concat(chunks, 0)
    means = chunks.mean(dim=(-1, -2, -3))
    vars = chunks.var(dim=(-1, -2, -3))
    if verbose:
        print(
            f"Average model input mean: {means.mean()} (Variance over inputs: {means.var()})"
        )
        print(
            f"Average model input varariance: {vars.mean()} (Variance over inputs: {vars.var()})"
        )
    return means.mean().cpu().item(), vars.mean().cpu().item()
