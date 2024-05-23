import torch


class VGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.param_hist = []

    def step(self, parameters):
        with torch.no_grad():
            for param in parameters:
                print(param.grad)
                param.data -= self.lr * param.grad

    def zero_grad(self, parameters):
        with torch.no_grad():
            for param in parameters:
                param.grad = None


