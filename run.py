from model import Sequential
from layer import Dense, Conv2D, MaxPool2D, Flatten
from loss import BinaryCrossEntropy
from activation import Sigmoid
from optimizer import GradientDescentOptimizer

if __name__ == "__main__":
    from sklearn.datasets import load_digits
    data = load_digits(n_class=2)
    X, y = data['data'].reshape(-1,8,8,1)/16, data['target'].reshape(-1,1)

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