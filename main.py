import random

from micrograd.engine import Value
from micrograd.layers import MLP

random.seed(1) 

print("=== 1. Autograd ===")

a = Value(2.0)
b = Value(-3.0)
c = a * b + b ** 2      
c.backward()

print("c      =", c.data)   
print("dc/da  =", a.grad)   
print("dc/db  =", b.grad)   


print("\n=== 2. Gradient check ===")


def f(x):
    return (x * x * x + x.tanh() * 2 + 1 / (x + 3)).data


x = Value(0.5)
y = x * x * x + x.tanh() * 2 + 1 / (x + 3)
y.backward()

h = 1e-6
numeric = (f(Value(0.5 + h)) - f(Value(0.5 - h))) / (2 * h)
print("autograd gradient :", x.grad)
print("numeric gradient  :", numeric)   



print("\n=== 3. The network ===")

model = MLP(2, [4, 4, 1])    
print(model)
print("number of parameters:", len(model.parameters()))   
print("output for [1.0, -2.0]:", model([1.0, -2.0]))


print("\n=== 4. Training ===")

xs = [
    [2.0, 3.0],
    [3.0, -1.0],
    [0.5, 1.0],
    [1.0, 1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  

learning_rate = 0.02

for step in range(1000):
    preds = [model(x) for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))

    model.zero_grad()
    loss.backward()

    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if step % 100 == 0 or step == 999:
        print(f"step {step:3d}   loss {loss.data:.6f}")

print("\nTargets    :", ys)
print("Predictions:", [round(model(x).data, 3) for x in xs])