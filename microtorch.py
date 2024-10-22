import torch

class Embedding():
    def __repr__(self):
        return f"Embedding:{tuple(self.embedding.shape)}"
    def __init__(self, dim1,dim2):
        self.embedding = torch.randn((dim1,dim2),requires_grad=True)
    def __call__(self,x):
        return self.embedding[x]
    def parameters(self):
        return [self.embedding]
    
class Linear():
    def __init__(self, dim1,dim2,scale = 0.1):
        self.w = (torch.randn((dim1, dim2)) * scale).requires_grad_(True)
        self.b = (torch.randn(dim2) * scale*0.1).requires_grad_(True)
    def __call__(self,x):
        out = x @ self.w + self.b
        return out
    def parameters(self):
        return [self.w] + ([] if self.b is None else [self.b])
    def __repr__(self):
        return f"Linear:{tuple(self.w.shape)}"

class Flatten():
    def __init__(self):
        pass
    def __call__(self, x, axis=None):
        if x.dim() == 2:
            if axis == 0:
                return x.view(-1, 1)
            elif axis == 1:
                return x.view(1, -1)
            elif axis is None:
                return x.view(-1)
            else:
                raise ValueError("Axis invalid")
        if x.dim() ==3:
            if axis == 0:
                return x.view(x.shape[0], -1)
            elif axis ==1:
                return x.view(-1, x.shape[2])
            elif axis is None:
                return x.view(x.shape[0], -1)
            else:
                return ValueError("Axis invalid")
    def parameters(self):
        return []

class Sequential():
    def __init__(self,layers):
        self.layers = layers
    def __call__(self,x):
        out = x
        for i in self.layers:
            out = i(out)
        return out
    def parameters(self):
        params = []
        for i in self.layers:
            params.extend(i.parameters())  
        return params

class Tanh():
    def __init__(self):
        pass
    def __call__(self,x):
        return torch.tanh(x)
    def parameters(self):
        return []