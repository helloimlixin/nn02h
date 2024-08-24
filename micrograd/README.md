The `micrograd` is a minimalistic autograd engine that implements *backpropagation*, which allows you to efficiently compute
gradients of some type of loss function with respect to the weights of a neural network. By backpropagation we can then iteratively
tune the parameters of the neural network to minimize the loss function, improving the neural network's accuracy. Backpropagation
is the mathematical core of many modern deep neural network engines.

The basic functionality of the `micrograd` is to allow you to build mathematical expressions:
```python
from micrograd.engine import Value

a = Value(2.0)
b = Value(1.0)
c = a + b # build a computation graph with a and b the child nodes of c
d = a * b + b ** 4
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e ** 2
g = f / 2.0
g += 10.0 / f
```
One more thing to note this micrograd is supported for scalar only operations.