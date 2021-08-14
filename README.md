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

#### 1. Activation Function

Every new implementation of Activation function should be a class containing 3 methods,

- ```__call__``` : returns ```eval```
- ```eval``` : Method that returns the output of the activation function for an input X of shape (features, batch_size).
- ```grad_input``` : Method that returns the Jacobian of the activation function with respect to the input of the function. 
I.e for an activation function, this method returns 

<div align="center"><img src="http://www.sciweavers.org/tex2img.php?eq=%5Cbegin%7Bbmatrix%7D%0A%20%20%20%20%5Cfrac%7B%5Cpartial%20f_1%7D%7B%5Cpartial%20x_1%7D%20%26%20%5Cfrac%7B%5Cpartial%20f_1%7D%7B%5Cpartial%20x_2%7D%20%26%20%5Cldots%20%26%20%5Cfrac%7B%5Cpartial%20f_1%7D%7B%5Cpartial%20x_n%7D%20%5C%5C%0A%20%20%20%20%5Cvdots%20%26%20%26%20%5Cddots%20%26%20%5Cvdots%20%5C%5C%0A%20%20%20%20%5Cfrac%7B%5Cpartial%20f_n%7D%7B%5Cpartial%20x_1%7D%20%26%20%5Cfrac%7B%5Cpartial%20f_n%7D%7B%5Cpartial%20x_2%7D%20%26%20...%20%26%20%5Cfrac%7B%5Cpartial%20f_n%7D%7B%5Cpartial%20x_n%7D%20%5C%5C%0A%20%20%5Cend%7Bbmatrix%7D&bc=White&fc=Black&im=jpg&fs=36&ff=arev&edit=0" align="center" border="0" alt="\begin{bmatrix}    \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \ldots & \frac{\partial f_1}{\partial x_n} \\    \vdots & & \ddots & \vdots \\    \frac{\partial f_n}{\partial x_1} & \frac{\partial f_n}{\partial x_2} & ... & \frac{\partial f_n}{\partial x_n} \\  \end{bmatrix}" width="250" height="125" /></div>


It should be noted that all the implementations must be using numpy and loops are required to be avoided at most of the places. 
For an example, check ```activation.Sigmoid``` class in the repository.

<!-- #### 2. Layers -->

