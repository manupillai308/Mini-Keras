import numpy as np

class Dot:
    def __init__(self, input_size, units):
        self.W = np.random.randn(input_size, units)
        self.b = np.random.randn(units, 1)

    def __call__(self, X):
        return self.W.T.dot(X) + self.b

    def grad_w(self, X):
        I = np.identity(self.b.shape[0])
        m1 = np.stack([I]*self.W.shape[0], axis=1)
        grad = np.einsum('ijk,jm->mijk', m1, X)
        return grad
    
    def grad_b(self, X):
        return np.stack([np.identity(self.b.shape[0])]*X.shape[1], axis=0)

    def grad_input(self, X):
        return np.stack([self.W.T]*X.shape[1], axis=0)
    
    def get_output_size(self):
        return self.b.shape
    
    def get_no_of_params(self):
        return np.prod(self.W.shape) + np.prod(self.b.shape)
    
    def update(self, gradW, gradb, optimizer, method):
        if method == "minimize":
            self.W = optimizer.minimize(self.W, gradW)
            self.b = optimizer.minimize(self.b, gradb)
        elif method == "maximize":
            self.W = optimizer.maximize(self.W, gradW)
            self.b = optimizer.maximize(self.b, gradb)

class Dense:
    
    def __init__(self, units, activation, input_size):
        if isinstance(input_size, tuple):
            if len(input_size) <= 2:
                input_size = input_size[0]
            else:
                raise RuntimeError(f"Incompatible input shape, got {input_size}")
        self.units = units
        self.dot = Dot(input_size, units)
        self.activation = activation
        self.input_size = input_size

    def get_output_size(self):
        return self.dot.get_output_size()

    def get_no_of_params(self):
        return self.dot.get_no_of_params()

    def eval(self, X):
        return self.activation(self.dot(X))

    def grad_parameters(self, X):
        da_dI = self.activation.grad_input(self.dot(X))
        dI_dw = self.dot.grad_w(X)
        da_dw = np.einsum('mij,mjkl->mikl', da_dI, dI_dw)
        
        dI_db = self.dot.grad_b(X)
        da_db = np.einsum('mij,mjk->mik', da_dI, dI_db)
        return (da_dw, da_db)
    
    def gradient_dict(self, output):
        grad_ = {}
        grad_["input"] = self.grad_input(output)
        grad_["w"], grad_["b"] = self.grad_parameters(output)

        return grad_


    def grad_input(self, X):
        g1 = self.activation.grad_input(self.dot(X))

        g2 = self.dot.grad_input(X)

        return np.einsum('mij,mjk->mik', g1, g2)
    
    @staticmethod
    def backprop_grad(grad_loss, grad):
        grad_w = np.einsum('mij,mjkl->mikl', grad_loss, grad["w"]).sum(axis=0)[0]
        grad_b = np.einsum('mij,mjk->mik', grad_loss, grad["b"]).sum(axis=0).T
        grad_loss = np.einsum('mij,mjk->mik', grad_loss, grad["input"])

        return grad_w, grad_b, grad_loss

    def update(self, grad_w, grad_b, optimizer, method="minimize"):
        self.dot.update(grad_w, grad_b, optimizer, method)
        
class Conv2D:
    
    def __init__(self, ksize, stride, padding, activation, filters, input_size):
        if input_size[0] <= 0 or input_size[1] <= 0:
            raise ValueError(f"Input image size is invalid, got {input_size}")
        self.kernels = []
        self.stride = stride
        self.padding = padding
        self.input_size = input_size
        self.ksize = ksize
        self.filters = filters
        
        self.bias = np.random.randn(filters).reshape(1, -1)
        for i in range(filters):
            self.kernels.append(np.random.randn(ksize, ksize, input_size[-1]))
        self.activation = activation
        
    @staticmethod
    def _rotate(inp):
        assert len(inp.shape) == 4, f"Shape mismatch, input map should have 4 dim, got {len(inp.shape)}"

        return np.flip(inp, axis=(1,2))

    @staticmethod
    def _inside_pad(inp, pad_width):
        assert len(inp.shape) == 4, f"Shape mismatch, input map should have 4 dim, got {len(inp.shape)}"
        
        if pad_width == 0:
            return inp
        ix = np.repeat(np.arange(1, inp.shape[1]), pad_width)

        inp = np.insert(inp, ix, 0, axis=1)
        return  np.insert(inp, ix, 0, axis=2)


    @staticmethod
    def _pad(inp, pad_width):
        assert len(inp.shape) == 4, f"Shape mismatch, input map should have 4 dim, got {len(inp.shape)}"
        if pad_width == 0:
            return inp

        return np.pad(inp, [(0, 0), (pad_width, pad_width), (pad_width, pad_width), (0,0)])

    @staticmethod
    def _convolution_op_w_kernel(inp, kernel, stride=1):
        
        assert len(inp.shape) == 4, f"Shape mismatch, input map should have 4 dim, got {len(inp.shape)}"
        assert len(kernel.shape) == 4, f"Shape mismatch, kernel should have 4 dim, got {len(kernel.shape)}"
        assert inp.shape[-1] == kernel.shape[-1], f"Shape mismatch, input map should have same channel as kernel, got {inp.shape} & {kernel.shape}"
        assert kernel.shape[1] == kernel.shape[2], "Non square kernels are not supported"
        assert (inp.shape[1] >= kernel.shape[0]) or (inp.shape[2] >= kernel.shape[1]), f"Input map shape less than kernel, got {inp.shape[1:3]} for kernel {kernel.shape[:-1]}"

        kernel = Conv2D._rotate(kernel)

        start_rloc = 0
        end_rloc = kernel.shape[1]

        oup = []

        while end_rloc <= inp.shape[1]:

            start_cloc = 0
            end_cloc = kernel.shape[2]
            output = []

            while end_cloc <= inp.shape[2]:
                conv = inp[:, start_rloc:end_rloc, start_cloc:end_cloc]*kernel
                output.append(conv.sum(axis=(1,2,3)))

                start_cloc += stride
                end_cloc += stride

            oup.append(output)

            start_rloc += stride
            end_rloc += stride        
        
        oup = np.expand_dims(oup, 0)
        oup = np.transpose(oup, [3,1,2,0])
        assert len(oup.shape) == 4, f"Shape mismatch at convolution op, got {oup.shape}"
        
        return oup
    
    def _convolution_op(self, inp, stride): 
        
        feature_maps = []
        for kernel in self.kernels:
            oup = self._convolution_op_w_kernel(inp, np.expand_dims(kernel, 0), stride)
            feature_maps.append(oup[...,0])
        
        return np.stack(feature_maps, axis=-1)
    
    def get_output_size(self):
        m, n, k, p, s = self.input_size[0], self.input_size[1], self.ksize, self.padding, self.stride
        return (m-k+2*p)//s + 1, (n-k+2*p)//s + 1, self.filters

    def get_no_of_params(self):
        return (self.ksize*self.ksize*self.input_size[-1]*self.filters) + self.filters

    def eval(self, X):
        out = self._convolution_op(X.T, self.stride) + self.bias
        b, h, w, c = out.shape
        a_out = self.activation(out.reshape(b, h*w*c).T)
        return a_out.T.reshape(b, h, w, c).T

    def grad_activation(self, X):
        out = self._convolution_op(X.T, self.stride) + self.bias
        b, h, w, c = out.shape
        
        da_dI = self.activation.grad_input(out.reshape(b, h*w*c).T)
        da_dI = np.diagonal(da_dI, axis1=1, axis2=2)
        
        return da_dI.T.reshape(b, h, w, c)
    
        
    def gradient_dict(self, output):
        grad_ = {}
        grad_["input"] = self.get_input(output)
        grad_["activation"] = self.grad_activation(output)

        return grad_
    
    def get_input(self, X):
        out_h, out_w, _ = self.get_output_size()
        h = (out_h-1)*self.stride-2*self.padding+self.ksize
        w = (out_w-1)*self.stride-2*self.padding+self.ksize
        return X.T[:, :h, :w, :]
    
    def backprop_grad(self, abcd, grad): # abcd -> grad_loss
        pqrs = grad["activation"]
        
        b, h, w, c = abcd.shape
        
        kernels = pqrs[:, :h, :w, :] * abcd
        kernels = self._inside_pad(kernels, self.stride-1)
        inps = grad["input"]
        grad_ws = []
        
        #### GRAD W---------------------------
        for i in range(kernels.shape[-1]):
            kernel = kernels[..., i]
            grad_w = []
            for j in range(inps.shape[-1]):
                inp = inps[..., j]
                oup = self._convolution_op_w_kernel(np.expand_dims(inp,-1), np.expand_dims(kernel, -1))
                oup = self._rotate(oup).sum(axis=0)
                grad_w.append(oup[...,0])
            
            grad_w = np.array(grad_w)
            grad_ws.append(np.transpose(grad_w, [1,2,0]))
        ### -----------------------------------
        
        #### GRAD I---------------------------
        inp = self._pad(kernels, self.ksize-1)
        kernels = self.kernels
        
        grad_I = np.empty_like(grad["input"], dtype="float32")
        
        for i in range(self.input_size[-1]):
            kernel = np.dstack([kernels[j][...,i] for j in range(len(kernels))])
            oup = self._convolution_op_w_kernel(inp, np.expand_dims(kernel, 0))
            grad_I[..., i] = oup[...,0]
            
        ### -----------------------------------
        
        #### GRAD b---------------------------
        grad_bs = np.sum(pqrs * abcd, axis=(1,2,0))
        
        return grad_ws, grad_bs.reshape(1,-1), self._pad_grad_I(grad_I)

    def _pad_grad_I(self, grad_I):
        return np.pad(grad_I, [(0, 0), (0, self.input_size[0] - grad_I.shape[1]), (0, self.input_size[1] - grad_I.shape[2]), (0,0)])
        
    def update(self, grad_w, grad_b, optimizer, method="minimize"):
        if method=="minimize":
            self.bias = optimizer.minimize(self.bias, grad_b)
            for i in range(len(self.kernels)):
                self.kernels[i] = optimizer.minimize(self.kernels[i], grad_w[i])
        else:
            self.bias = optimizer.maximize(self.bias, grad_b)
            for i in range(len(self.kernels)):
                self.kernels[i] = optimizer.maximize(self.kernels[i], grad_w[i])
        

class Flatten:
    def __init__(self, input_size):
        self.h, self.w, self.c = input_size

    def get_output_size(self):
        return (self.h*self.w*self.c, 1)

    def get_no_of_params(self):
        return 0

    def eval(self, X):
        return X.T.reshape(-1, self.h*self.w*self.c).T

    def grad_parameters(self, X):
        pass
    
    def gradient_dict(self, output):
        grad_ = {}
        return grad_

    def grad_input(self, X):
        pass
    
    def backprop_grad(self, grad_loss, grad):
        # m x 1 x self.h*self.w*self.c
        return None, None, grad_loss[:, 0, :].reshape(-1, self.h, self.w, self.c)

    def update(self, grad_w, grad_b, optimizer, method="minimize"):
        pass
        

class MaxPool2D:
    
    def __init__(self, ksize, stride, padding, input_size):
        if input_size[0] <= 0 or input_size[1] <= 0:
            raise ValueError(f"Input image size is invalid, got {input_size}")
        self.stride = stride
        self.padding = padding
        self.input_size = input_size
        self.ksize = ksize

    @staticmethod
    def _pad(inp, pad_width):
        assert len(inp.shape) == 4, f"Shape mismatch, input map should have 4 dim, got {len(inp.shape)}"
        if pad_width == 0:
            return inp

        return np.pad(inp, [(0, 0), (pad_width, pad_width), (pad_width, pad_width), (0,0)])

    @staticmethod
    def _convolution_op_w_kernel(inp, ksize, stride=1):

        assert len(inp.shape) == 4, f"Shape mismatch, input map should have 4 dim, got {len(inp.shape)}"

        start_rloc = 0
        end_rloc = ksize

        oup = []
        ix = []
        inp_ixs = np.dstack(np.meshgrid(np.arange(inp.shape[1]), np.arange(inp.shape[2]))[::-1])
        inp_ixs = np.repeat(np.expand_dims(inp_ixs, 0), inp.shape[0], axis=0)

        while end_rloc <= inp.shape[1]:

            start_cloc = 0
            end_cloc = ksize
            output = []
            indexes = []

            while end_cloc <= inp.shape[2]:
                conv = inp[:, start_rloc:end_rloc, start_cloc:end_cloc]
                ixs = inp_ixs[:, start_rloc:end_rloc, start_cloc:end_cloc]

                b, h, w, c = conv.shape
                conv = conv.reshape(b, h*w, c)
                output.append(conv.max(axis=1))
                indexes.append(ixs.reshape(b, h*w, 2)[np.arange(b).reshape(-1,1), conv.argmax(axis=1)])

                start_cloc += stride
                end_cloc += stride

            oup.append(output)
            ix.append(indexes)

            start_rloc += stride
            end_rloc += stride        

        oup = np.transpose(oup, [2,0,1,3])
        ix = np.transpose(ix, [2,0,1,3, 4])
        assert len(oup.shape) == 4, f"Shape mismatch at convolution op, got {oup.shape}"
        assert len(ix.shape) == 5, f"Shape mismatch at convolution op, got {ix.shape}"

        return oup, ix
    
    def _convolution_op(self, inp, stride): 

        oup, ix = self._convolution_op_w_kernel(inp, self.ksize, stride)
        
        return oup, ix
    
    def get_output_size(self):
        m, n, k, p, s = self.input_size[0], self.input_size[1], self.ksize, self.padding, self.stride
        return (m-k+2*p)//s + 1, (n-k+2*p)//s + 1, self.input_size[-1]

    def get_no_of_params(self):
        return 0

    def eval(self, X, eval=True):
        out, ix = self._convolution_op(X.T, self.stride)
        return out.T if eval else (out.T, ix)

    def grad_activation(self, X):
        pass
        
    def gradient_dict(self, X):
        grad_ = {}
        _, grad_["max_indexes"] = self.eval(X, False)
        grad_["input"] = self.get_input(X)

        return grad_
    
    def get_input(self, X):
        out_h, out_w, _ = self.get_output_size()
        h = (out_h-1)*self.stride-2*self.padding+self.ksize
        w = (out_w-1)*self.stride-2*self.padding+self.ksize
        return np.zeros_like(X.T[:, :h, :w, :])
    
    def backprop_grad(self, abcd, grad): # abcd -> grad_loss
        grad_I = grad["input"]
        max_indexes = grad["max_indexes"]
        b, h, w, c = grad_I.shape
        
        #### GRAD I---------------------------
        for i in range(abcd.shape[1]):
            for j in range(abcd.shape[2]):
                grad_I[np.arange(b).reshape(-1,1), max_indexes[:, i, j, :, 0], max_indexes[:, i, j, :, 1], np.arange(c).reshape(1,-1)] += abcd[:, i, j, :]
        ### -----------------------------------

        return None, None, self._pad_grad_I(grad_I)

    def _pad_grad_I(self, grad_I):
        return np.pad(grad_I, [(0, 0), (0, self.input_size[0] - grad_I.shape[1]), (0, self.input_size[1] - grad_I.shape[2]), (0,0)])
    
    def update(self, grad_w, grad_b, optimizer, method="minimize"):
        pass
        
class BasicRNN:
    
    def __init__(self, output_units, hidden_units, activation, input_size):
        if isinstance(input_size, tuple):
            if len(input_size) != 2: # input_size => (timestep, input_features)
                raise RuntimeError(f"Incompatible input shape, got {input_size}")
                
        self.output_activation = activation
        self.timestep = input_size[0]
        self.input_units = input_size[1]
        self.output_units = output_units
        self.hidden_units = hidden_units
        
        self.hidden_layer = Dense(units=hidden_units, activation=ReLU(), input_size=self.input_units+self.hidden_units)
        self.output_layer = Dense(units=output_units, activation=self.output_activation, input_size=self.hidden_units)

    def get_output_size(self):
        return (self.timestep, self.output_units)

    def get_no_of_params(self):
        return self.hidden_layer.get_no_of_params() + self.output_layer.get_no_of_params()

    def eval(self, X, start_sequence=None):
        h_t = np.zeros((self.hidden_units, X.shape[-1]))
        timestep = X.shape[1]
        if start_sequence is not None:
            assert h_t.shape == start_sequence, f"Sequence start hidden state received incompatible shape, got {start_sequence.shape}, expected {h_t.shape}"
            h_t = start_sequence
        
        y = np.empty((self.output_units, timestep, X.shape[-1]))
        for i in range(timestep):
            x_t = X[:, i, :]
            x_t_stacked = np.vstack([x_t, h_t])
            h_t = self.hidden_layer.eval(x_t_stacked)
            y_t = self.output_layer.eval(h_t)
            y[:, i, :] = y_t
        
        return y

    def grad_parameters_T(self, x_t, h_t_1):
        
        x_t_stacked = np.vstack([x_t, h_t_1])
        h_t = self.hidden_layer.eval(x_t_stacked)
        
        dyt_param_output = self.output_layer.grad_parameters(h_t) # (dw, db)
        dyt_ht = self.output_layer.grad_input(h_t)
        
        dht_param = self.hidden_layer.grad_parameters(x_t_stacked)
        
        dyt_param_hidden = (np.einsum('mij,mjkl->mikl', dyt_ht, dht_param[0]), np.einsum('mij,mjk->mik', dyt_ht, dht_param[1]))
        
        return dyt_param_output, dyt_param_hidden, dht_param, h_t
    
    def grad_input_T(self, x_t, h_t_1):
        x_t_stacked = np.vstack([x_t, h_t_1])
        h_t = self.hidden_layer.eval(x_t_stacked)
        
        dyt_ht = self.output_layer.grad_input(h_t)
        
        dht_x_t_stacked = self.hidden_layer.grad_input(x_t_stacked)
        
        dht_x_t = dht_x_t_stacked[:, :, :self.input_units]
        dht_h_t_1 = dht_x_t_stacked[:, :, self.input_units:]
        

        dyt_x_t_stacked = np.einsum('mij,mjk->mik', dyt_ht, dht_x_t_stacked)

        dyt_x_t = dyt_x_t_stacked[:, :, :self.input_units]
        dyt_h_t_1 = dyt_x_t_stacked[:, :, self.input_units:]
        
        assert dyt_x_t.shape[-1] == self.input_units, f"Shape mistmatch in input gradient for step t, {dyt_x_t.shape[-1]} != {self.input_units}"
        assert dyt_h_t_1.shape[-1] == self.hidden_units, f"Shape mistmatch in input gradient for step t, {dyt_h_t_1.shape[-1]} != {self.hidden_units}"
        
        return dyt_x_t, dyt_h_t_1, dht_x_t, dht_h_t_1, h_t
    
    
    def grad_input(self, X, start_sequence=None):
        h_t = np.zeros((self.hidden_units, X.shape[-1]))
        timestep = X.shape[1]
        
        if start_sequence is not None:
            assert h_t.shape == start_sequence, f"Sequence start hidden state received incompatible shape, got {start_sequence.shape}, expected {h_t.shape}"
            h_t = start_sequence
        
        dy_dx = np.zeros((X.shape[-1], timestep, timestep, self.output_units, self.input_units))
        dy_dh_1 = np.zeros((X.shape[-1], timestep, timestep, self.output_units, self.hidden_units))
        
        gradient_across_time = {}
        
        for i in range(timestep):
            x_t = X[:, i, :]
            dyt_x_t, dyt_h_t_1, dht_x_t, dht_h_t_1, h_t = self.grad_input_T(x_t, h_t)
            dy_dx[:, i, i] = dyt_x_t
            dy_dh_1[:, i, i] = dyt_h_t_1
            
            gradient_across_time[i] = {}
            gradient_across_time[i]["dht_x_t"] = dht_x_t
            gradient_across_time[i]["dht_h_t_1"] = dht_h_t_1
            
            for j in range(i-1, -1):
                dy_dx[:, i, j] = np.einsum('mij,mjk->mik', dyt_h_t_1, gradient_across_time[j]["dht_x_t"])
                dyt_h_t_1 = np.einsum('mij,mjk->mik', dyt_h_t_1, gradient_across_time[j]["dht_h_t_1"])
                dy_dh_1[:, i, j] = dyt_h_t_1
            
            
        return dy_dx, dy_dh_1
        
        
    def grad_parameters(self, X, dy_dh_1, start_sequence=None):
        h_t = np.zeros((self.hidden_units, X.shape[-1]))
        timestep = X.shape[1]
        
        if start_sequence is not None:
            assert h_t.shape == start_sequence, f"Sequence start hidden state received incompatible shape, got {start_sequence.shape}, expected {h_t.shape}"
            h_t = start_sequence
        
        dy_dx = np.zeros((X.shape[-1], timestep, timestep, self.output_units, self.input_units))
        
        dy_dw_output = np.zeros((X.shape[-1], timestep, timestep, self.hidden_units, self.output_units, self.hidden_units))
        dy_dw_hidden = np.zeros((X.shape[-1], timestep, timestep, self.input_units, self.hidden_units, self.input_units))
        dy_db_hidden = np.zeros((X.shape[-1], timestep, timestep, self.hidden_units, self.hidden_units))
        dy_db_output = np.zeros((X.shape[-1], timestep, timestep, self.output_units, self.output_units))
        
        for i in range(timestep):
            x_t = X[:, i, :]
            (dyt_dw_output, dyt_db_output), (dyt_dw_hidden, dyt_db_hidden), (dht_dw_hidden, dht_db_hidden), h_t = self.grad_parameters_T(x_t, h_t)
            dy_dw_output[:, i, i] = dyt_dw_output
            dy_db_output[:, i, i] = dyt_db_output
            
            dy_db_hidden[:, i, i] = dyt_db_hidden
            dy_dw_hidden[:, i, i] = dyt_dw_hidden
            
            for j in range(i+1, timestep):
                dy_dw_hidden[:, j, i] = np.einsum('mij,mjkl->mikl', dy_dh_1[:, j, i], dht_dw_hidden)
                dy_db_hidden[:, j, i] = np.einsum('mij,mjk->mik', dy_dh_1[:, j, i], dht_db_hidden)
            
        return (dy_dw_output, dy_db_output), (dy_dw_hidden, dy_db_hidden)
    
    
    def gradient_dict(self, output):
        grad_ = {}
        grad_["input"], dy_dh_1 = self.grad_input(output)
        grad_["output"], grad_["hidden"] = self.grad_parameters(output, dy_dh_1)

        return grad_
    
    @staticmethod
    def backprop_grad(grad_loss, grad):
        raise NotImplementedError("backprop_grad not implemented")
        
        grad_w = np.einsum('mij,mjkl->mikl', grad_loss, grad["w"]).sum(axis=0)[0]
        grad_b = np.einsum('mij,mjk->mik', grad_loss, grad["b"]).sum(axis=0).T
        grad_loss = np.einsum('mij,mjk->mik', grad_loss, grad["input"])

        return grad_w, grad_b, grad_loss

    def update(self, grad_w, grad_b, optimizer, method="minimize"):
        self.dot.update(grad_w, grad_b, optimizer, method)
