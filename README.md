# torchapply
Apply a torch model to some datapoints. 

Here's an example:

```python
import torch
from torch import tensor


class Main(torch.nn.Module):
    def __init__(self, model_0, model_1):
        super().__init__()
        self.model_0 = model_0
        self.model_1 = model_1
        self.dictionary = {'apple': 0, 'orange': 1, 'pear': 2}

    def preprocess(self, arg):
        return [
            {
                'a': {'b': self.dictionary[arg[0]['a']['b']]},
                'c': self.dictionary[arg[0]['c']]
            },
            torch.tensor([self.dictionary[x] for x in arg[1]])
        ]

    def forward(self, args):
        return self.model_0(args[0]), self.model_1(args[1])
      
    def postprocess(self, arg):
        total = [arg[0]['a']['b'].sum(), arg[0]['c'].sum(), arg[1].sum()]
        return {'score': sum(total), 'decision': sum(total) > 0}
        

class ModelA(torch.nn.Module):
    def forward(self, args):
        return {'b': torch.randn(args['b'].shape[0], 10)}


class ModelC(torch.nn.Module):
    def forward(self, args):
        return torch.randn(args.shape[0], 10)


class Model1(torch.nn.Module):
    def forward(self, args):
        return torch.randn(args.shape[0], 10)


class Model0(torch.nn.Module):
    def __init__(self, model_a, model_c):
        super().__init__()
        self.model_a = model_a
        self.model_c = model_c

    def forward(self, args):
        return {'a': self.model_a(args['a']), 'c': self.model_c(args['c'])}


model = Main(
    model_0=Model0(
        model_a=ModelA(),
        model_c=ModelC()
    ),
    model_1=Model1()
)
```

Apply to a single datapoint:

```python
from torchapply import apply_model

apply_model(
   model, 
   ({'a': {'b': 'orange'}, 'c': 'pear'}, ('apple', 'apple')),
   single=True
)
```

Apply to multiple datapoints:

```python
from torchapply import apply_model

apply_model(
    model,
    [({'a': {'b': 'orange'}, 'c': 'pear'}, ('apple', 'apple')) for _ in range(10)],
    single=False
)
```

