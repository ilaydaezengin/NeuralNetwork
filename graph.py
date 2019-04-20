import layers
import numpy as np

class Graph:
    def __init__(self):
        self.layer = []

    def add(self, layer):
        self.layer.append(layer)
        return self

    def get_parameters(layer):
        attr_list = []
        for layer in layers:
            if hasattr(layer,'params'):
                attr_list.append(layer.params)
        return attr_list

    def forward(self,x):
        y = x
        self.xlist = [y]
        for layer in self.layer:
            y = layer.forward(y)
            self.xlist.append(y)
        return y



    def backward(self,x,dy):
        i = len(self.xlist)-2
        for layer in reversed(self.layer):
            dy = layer.backward(self.xlist[i],dy)
            i -= 1
        return dy


data, labels = layers.shape_training_data()
label_names_file = layers.unpickle("cifar-10-batches-py/batches.meta")
C = len(label_names_file[b'label_names'])



alpha = 0.1
batch_size = 64

l = 0.0001
E = np.transpose(np.sum(data, axis=0))/data.shape[0]
v = np.sum(np.square(data-E), axis=0)/data.shape[0]
data -=E
data /=v
beta1 = 0.9
beta2 = 0.995
e = 0.0001
perm = np.random.permutation(data.shape[0])
idx = 0


databatch,labelbatch = layers.get_next_batch(perm,data,labels,idx,batch_size)
N,D = databatch.shape
shape1 = (D,D)
shape2 = (D,C)



def f():
    model = Graph()
    model.add(layers.conv(layers.xaxier_initilizer,layers.zero_initilizer,1,1,32,4,3,3))
    model.add(layers.Relu())
    model.add(layers.conv(layers.xaxier_initilizer,layers.zero_initilizer,1,2,16,32,3,3))
    model.add(layers.Relu())
    model.add(layers.max_pooling(layers.xaxier_initilizer,layers.zero_initilizer,0,2,2,2))
    model.add(layers.flatten())
    model.add(layers.FullConn(layers.xaxier_initilizer,layers.zero_initilizer,(1024,10)))
    model.add(layers.Relu())
    crit = layers.softmax_with_loss()
    def f(x):
        logits = model.forward(x)
        prob = crit.forward(logits)
        dy,loss = crit.backward(prob,[1,2,3])
        dx = model.backward(x,dy)
        return loss,dx
    return f
