import numpy as np

class Sequential:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.outputs = []

    def add(self, Layer, *args, **kwargs):
        if not kwargs.get("input_size"):
            if len(self.layers) > 0:
                kwargs["input_size"] = self.layers[-1].get_output_size()
            else:
                raise ValueError("input_size is required for first layer in Sequential model")
        self.layers.append(Layer(*args, **kwargs))
        return self
    
    def summary(self):
        from tabulate import tabulate

        headers = ["Layer Type", "Output Shape", "No. of parameters"]
        summary_ = []
        params = 0
        for layer in self.layers:
            p = layer.get_no_of_params()
            params += p
            summary_.append([layer.__class__.__name__, layer.get_output_size(), p])
        
        print(tabulate(summary_, headers=headers))
        print("Total No. of parameters:", params)
    
    def get_batch(self, X, y, batch_size):
        ixs = np.arange(X.shape[0])
        np.random.shuffle(ixs)
        for i in range(0, X.shape[0], batch_size):
            x_batch, y_batch = X[ixs[i:i+batch_size]], y[ixs[i:i+batch_size]]
            if len(x_batch):
                yield (x_batch, y_batch)
        return 
    
    def fit(self, X, y, n_epochs, learning_rate, optimizer, batch_size=1, verbose=1):
        if len(y.shape) < 2:
            raise ValueError(f"Incompatible shape of y {y.shape}, try reshaping y using y.reshape(-1,1)")
        
        self.optimizer = optimizer.set_lr(learning_rate)
        for i in range(n_epochs):
            if verbose == 1:
                print(f"Epoch: {i+1}/{n_epochs}")
            
            progress_bar = self.__progress_bar(50, int(50*batch_size/X.shape[0]))
            for (X_batch, y_batch) in self.get_batch(X, y, batch_size):
                _, outputs, _gradients_ = self.forward_propagation(X_batch)
                grads = self.backward_propagation(outputs, _gradients_, y_batch.reshape(-1,1))
                self._update_params(grads)
                if verbose == 1:
                    try:
                        _loss = self._eval_loss(X_batch, y_batch)
                        print("\r" + next(progress_bar), f"Loss: {np.round(_loss, 4)}", end="")
                    except StopIteration:
                        pass
            if verbose == 1:
                _loss = self._eval_loss(X, y)
                bar =  "|" + "-"*50 + ">" + " "*0 + "|"
                print("\r" + bar, f"Loss: {np.round(_loss, 4)}")
        if verbose == 0:
            print(f"\rEpoch: {i+1} Loss:{self._eval_loss(X, y)}", end="")
            
        print("")
            
    def forward_propagation(self, X, eval=False):
        output = X.T
        outputs = [output]
        gradients = []
        for layer in self.layers:
            if not eval:
                grad_ = layer.gradient_dict(output)
                gradients.append(grad_)
            output = layer.eval(output)
            outputs.append(output)

        return output.T, outputs, gradients
    
    def backward_propagation(self, outputs, gradients, y):
        grad_loss = self.loss.grad_input(outputs[-1], y)
        outputs = outputs[:-1]
        grads = []
        for grad, output, layer in list(zip(gradients, outputs, self.layers))[::-1]:
            grad_w, grad_b, grad_loss = layer.backprop_grad(grad_loss, grad)
            grads.append((grad_w, grad_b))
        
        return grads
    
    def _update_params(self, grads):
        for ((grad_w, grad_b), layer) in zip(grads, self.layers[::-1]):
            layer.update(grad_w, grad_b, self.optimizer)
    
    def predict(self, X):
        return self._eval(X)
    
    def predict_classes(self, X, threshold=0.5):
        return (self.predict(X) > threshold).astype("int")
    
    def evaluate(self, X, y):
        if len(y.shape) < 2:
            raise ValueError(f"Incompatible shape of y {y.shape}, try reshaping y using y.reshape(-1,1)")
        return self._eval_loss(X, y), (y == self.predict_classes(X)).astype('int')
    
    def _eval(self, X):
        return self.forward_propagation(X, eval=True)[0]
    
    def compile(self, loss):
        self.loss = loss
    
    def __progress_bar(self, size, inc):
        step = 0
        inc += 1
        while step <= size:

            bar = "|" + "-"*step + ">" + " "*(size-step) + "|"
            yield bar
            step += inc
        
        return
    
    def _eval_loss(self, X, y_true):
        if len(y_true.shape) < 2:
            raise ValueError(f"Incompatible shape of y {y_true.shape}, try reshaping y using y.reshape(-1,1)")
            
        if self.loss is None:
            raise RuntimeError("Model not compiled")
            
        return self.loss(self._eval(X), y_true)