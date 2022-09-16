import torch
from torch.utils import data
import tqdm


class BasicDataset(data.Dataset):
    def __init__(self, documents, transform=None):
        super().__init__()
        self.documents = documents
        self.transform = transform

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, item):
        if self.transform is None:
            return self.documents[item]
        else:
            r = self.transform(self.documents[item])
            return r


def create_batch(args):
    """
    Create a singleton batch in a manner similar to the PyTorch dataloader

    :param args: single data point for batching
    """
    if isinstance(args, (tuple, list)):
        return tuple([create_batch(x) for x in args])
    if isinstance(args, dict):
        return {k: create_batch(args[k]) for k in args}
    if isinstance(args, torch.Tensor):
        return args.unsqueeze(0)
    if isinstance(args, (float, int)):
        return torch.tensor([args])
    raise TypeError('only tensors and tuples of tensors recursively supported...')


def unpack_batch(args):
    """
    Unpack a batch into lines of tensor output.

    :param args: a batch of model outputs

    >>> unpack_batch(torch.randn(1, 10))[0].shape
    torch.Size([10])
    >>> out = unpack_batch([torch.randn(2, 10), torch.randn(2, 3, 5)])
    >>> type(out)
    <class 'list'>
    >>> len(out)
    2
    >>> out = unpack_batch({'a': torch.randn(2, 10), 'b': torch.randn(2, 3, 5)})
    >>> [type(x) for x in out]
    [<class 'dict'>, <class 'dict'>]
    >>> out[0]['a'].shape
    torch.Size([10])
    >>> out[0]['b'].shape
    torch.Size([3, 5])
    >>> out = unpack_batch({'a': {'b': torch.randn(2, 10)}})
    >>> out[0]['a']['b'].shape
    torch.Size([10])
    >>> out[1]['a']['b'].shape
    torch.Size([10])
    """

    if isinstance(args, torch.Tensor):
        return [args[i] for i in range(args.shape[0])]
    else:
        if isinstance(args, list) or isinstance(args, tuple):
            tmp = [unpack_batch(x) for x in args]
            batch_size = len(tmp[0])
            return [[x[i] for x in tmp] for i in range(batch_size)]
        elif isinstance(args, dict):
            tmp = {k: unpack_batch(v) for k, v in args.items()}
            batch_size = len(next(iter(tmp.values())))
            return [{k: v[i] for k, v in tmp.items()} for i in range(batch_size)]
        else:
            raise NotImplementedError


def apply_model(model, args, single=True, verbose=False, **kwargs):
    """
    Apply model to args including pre-processing, forward pass and post-processing.

    :param model: model object including methods *preprocess*, *forward* and *postprocess*
    :param args: single or multiple data points over which to evaluate model
    :param single: toggle to apply model to single or multiple (batched) datapoints.
    :param verbose: display progress bar
    :param kwargs: key, value pairs to be passed to dataloader
    """
    if single:
        prepared = model.preprocess(args)
        singleton_batch = create_batch(prepared)
        output = model.forward(singleton_batch)
        output = unpack_batch(output)[0]
        if hasattr(model, 'postprocess'):
            return model.postprocess(output)
        return output
    else:
        inputs = BasicDataset(args, model.preprocess)
        loader = data.DataLoader(inputs, **kwargs)
        out = []
        if verbose:
            progress = tqdm.tqdm(total=len(args))
        for batch in loader:
            tmp = model.forward(batch)
            tmp = unpack_batch(tmp)
            if hasattr(model, 'postprocess'):
                tmp = list(map(model.postprocess, tmp))
            out.extend(tmp)
            if verbose:
                progress.update(len(tmp))
        return out