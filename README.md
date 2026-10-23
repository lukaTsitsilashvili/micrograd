# micrograd-project

A tiny neural network library in pure Python with automatic differentiation, based on Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

## Structure

- `micrograd/engine.py`: the `Value` class, which tracks a computation graph and runs backpropagation
- `micrograd/network.py`: `Neuron`, `Layer` and `MLP` classes
- `main.py`: gradient check and a small training demo

## Run

No dependencies needed (Python 3.6+):

```bash
python3 main.py
```
