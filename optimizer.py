import numpy as np

class Optimizer:
    def __init__(self, params, name, lr):
        self.params = params
        self.name = name
        self.lr = lr

    def update(self, global_idx):
        pass


class SGD(Optimizer):
    def __init__(self,params,name,lr):
        Optimizer.__init__(self,params,name,lr)

    def update(self):
        self.params[self.name]=self.params[self.name]- lr * self.params['d' + self.name]


class RmsProp(Optimizer):
    def __init__(self,params,name,lr,):
        Optimizer.__init__(self,params,name,lr)

    def update(self):

class rprop(Optimizer):
    def __init__(self,params,name,lr,gradsq = 0):
        Optimizer.__init__(self,params,name,lr)

    def update(self):
        self.gradsq += self.params['d' + self.name]**2


class Momentum(Optimizer):
    def __init__(self, params,name,alpha = 0.9):
        Optimizer.__init__(self,params,name)
        self.alpha = alpha
        self.v = np.zeros_like(self.params[self.name])

    def update(self):
        self.v = self.alpha * self.v + ((1 - self.alpha) * self.params['d' + self.name]
        self.params[self.name] -= self.alpha * self.v


class Nesterov(Optimizer):
    def __init__(self, params,name,alpha = 0.9):
        Optimizer.__init__(self,params,name)
        self.alpha = alpha
        self.v = np.zeros_like(self.params[self.name])

    def update(self):


class Adam(Optimizer):
    def __init__(self,params,name,beta1 = 0.9,beta2 = 0.995,alpha = 0.1, e = 0.0001):
        Optimizer.__init__(self,params,name)
        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = alpha
        self.e = e
        self.v = np.zeros_like(self.params[self.name])
        self.r = np.zeros_like(self.params[self.name])


    def update(self):
        self.v = self.beta1 * self.v + ((1-beta1) * self.params['d' + self.name]) / (1 - self.beta1**t)
        self.r = self.beta2 * self.r + ((1-beta2) * self.params['d' + self.name]**2) / (1 - self.beta2**t)
        self.params[self.name] -= (self.alpha * self.v) / (np.sqrt(self.r) + self.e)





class Upper:
    def __init__(self,params,optimizer_name,lr,**kwargs):
        self.params = params
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.layer_params = []
        for param in self.params:
            param_names = param.keys()
            for param_name in param_names:
                self.layer_params.append(param_name)
    def update(self):
        self.optimizer_name()
