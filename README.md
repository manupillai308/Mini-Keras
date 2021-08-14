# Mini-Keras
Keras like implementation of Deep Learning architectures from scratch using numpy.


## How to contribute?

The project contains implementations for various activation functions, layers, loss functions, model structures and optimizers in files
```activation.py, layer.py, loss.py, model.py``` and ```optimizer.py``` respectively.

Given below is list of available implementations (which may or may not require any improvements).


|Activation Functions| Status|
|---|---|
|Sigmoid| Available|
|ReLU| Required|
|Softmax| Required|

|Layer| Status|
|---|---|
|Dense| Available|
|Conv2D| Available|
|MaxPool2D| Available|
|Flatten| Available|
|BasicRNN| Required|

|Loss Function| Status|
|---|---|
|BinaryCrossEntropy| Available|
|CategoricalCrossEntropy| Required|

|Model Structure| Status|
|---|---|
|Sequential| Available|

|Optimizer| Status|
|---|---|
|GradientDescentOptimizer| Available|
|AdamOptimizer| Required|
|AdaGradOptimizer| Required|
|GradientDescentOptimizer (with Nesterov)| Required|

Each of the implementations are class-based and follows a keras like structure. A typical model training with Mini-Keras looks like this,
```python
from model import Sequential
from layer import Dense, Conv2D, MaxPool2D, Flatten
from loss import BinaryCrossEntropy
from activation import Sigmoid
from optimizer import GradientDescentOptimizer

model = Sequential()
model.add(Conv2D, ksize=3, stride=1, activation=Sigmoid(), input_size=(8,8,1), filters=1, padding=0)
model.add(MaxPool2D, ksize=2, stride=1, padding=0)
model.add(Conv2D, ksize=2, stride=1, activation=Sigmoid(), filters=1, padding=0)
model.add(Flatten)
model.add(Dense, units=1, activation=Sigmoid())
model.summary()

model.compile(BinaryCrossEntropy())

print("Initial Loss", model.evaluate(X, y)[0])
model.fit(X, y, n_epochs=100, batch_size=300, learning_rate=0.003, optimizer=GradientDescentOptimizer(), verbose=1)
print("Final Loss", model.evaluate(X, y)[0])
```

As you might have noticed, its very similar to how one will do it in **Keras**.

### Testing new functionalities

The ```run.py``` consists of a small code snippet that can be used to test if your new implementation is working properly or not.

### Implementation Details

All the implementations have a forward propagation and a backward propagation equivalent available as a method in the corresponding class. Below are the details for implementing all the functionalities under different categories.
