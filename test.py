import layers
import numpy as np
from gradcheck import gradcheck
import graph

def fc_x_forward():
    N,D,C = 4,5,6
    fc = layers.FullConn(layers.xaxier_initilizer,layers.zero_initilizer,(D,C))
    dm = layers.dummy()
    def f(x):
        y = fc.forward(x)
        yout = dm.forward(y)
        dy = dm.backward(y,1)
        dx = fc.backward(x,dy)
        return yout,dx
    return f

x = np.random.rand(4,5)
f = fc_x_forward()
gradcheck(f,x)

def fc_w_forward():
    N,D,C = 4,5,6
    x0 = np.random.rand(N,D)
    fc = layers.FullConn(layers.xaxier_initilizer,layers.zero_initilizer,(D,C))
    dm = layers.dummy()
    def f(x):
        fc.params['w'] = x
        y = fc.forward(x0)
        yout = dm.forward(y)
        dy = dm.backward(y,1)
        dx = fc.backward(x0,dy)
        return yout,fc.params['dw']
    return f

#w = np.random.rand(5,6)
#f = fc_w_forward()
#gradcheck(f,w)



def fc_b_forward():
    N,D,C = 4,5,6
    x0 = np.random.rand(N,D)
    fc = layers.FullConn(layers.xaxier_initilizer,layers.zero_initilizer,(D,C))
    dm = layers.dummy()
    def f(x):
        fc.params['b'] = x
        y = fc.forward(x0)
        yout = dm.forward(y)
        dy = dm.backward(y,1)
        dx = fc.backward(x0,dy)
        return yout,fc.params['db']
    return f

#b = np.zeros((1,6))
#f = fc_b_forward()
#gradcheck(f,b)

def pool_x_forward():
    N,c,h,w = 5,10,5,5
    pool = layers.max_pooling(layers.xaxier_initilizer,layers.zero_initilizer,0,2,2,2)
    dm = layers.dummy()
    def f(x):
        y = pool.forward(x)
        yout = dm.forward(y)
        dy = dm.backward(y,1)
        dx = pool.backward(x,dy)
        return yout,dx
    return f

x = np.random.rand(5,10,5,5)
f = pool_x_forward()
gradcheck(f,x)




def conv_x_forward():
    N,c,h,w = 5,1,10,10
    conv = layers.conv(layers.xaxier_initilizer,layers.zero_initilizer,1,1,20,1,3,3)
    dm = layers.dummy()
    def f(x):
        y = conv.forward(x)
        yout = dm.forward(y)
        dy = dm.backward(y,1)
        dx = conv.backward(x,dy)
        return yout,dx
    return f

#x = np.random.rand(5,1,10,10)
#f = conv_x_forward()
#gradcheck(f,x)


def conv_w_forward():
    x0 = np.random.rand(5,1,10,10)
    N,c,h,w = x0.shape
    conv = layers.conv(layers.xaxier_initilizer,layers.zero_initilizer,1,1,20,1,3,3)
    dm = layers.dummy()
    def f(x):
        conv.params['w'] = x
        y = conv.forward(x0)
        yout = dm.forward(y)
        dy = dm.backward(y,1)
        dx = conv.backward(x0,dy)
        return yout,conv.params['dw']
    return f

#w = np.random.rand(20,1,3,3)
#f = conv_w_forward()
#gradcheck(f,w)


def relu_x_forward():
    N,D,C = 4,5,6
    relu = layers.Relu()
    dm = layers.dummy()
    def f(x):
        y = relu.forward(x)
        yout = dm.forward(y)
        dy = dm.backward(y,1)
        dx = relu.backward(x,dy)
        return yout,dx
    return f

#x = np.random.rand(10,3,15,15)
#f = relu_x_forward()
#gradcheck(f,x)

def graph_forward():
    model = graph.Graph()
    model.add(layers.conv(layers.xaxier_initilizer,layers.zero_initilizer,1,1,32,4,3,3))
    #model.add(layers.Relu())
    model.add(layers.conv(layers.xaxier_initilizer,layers.zero_initilizer,1,2,16,32,3,3))
    #model.add(layers.Relu())
    model.add(layers.max_pooling(layers.xaxier_initilizer,layers.zero_initilizer,0,2,2,2))
    model.add(layers.flatten())
    model.add(layers.FullConn(layers.xaxier_initilizer,layers.zero_initilizer,(1024,10)))
    #model.add(layers.Relu())
    crit = layers.softmax_with_loss()
    y = np.array([1,2,3])
    def foo(x):
        logits = model.forward(x)
        prob = crit.forward(logits)
        dy,loss = crit.backward(logits,y)
        dx = model.backward(x,dy)
        return loss,dx
    return foo


x =np.random.rand(3,4,32,32)
f = graph_forward()
gradcheck(f,x)


def softmax_x_forward():
    labels = np.asarray([2,3,1,4])
    softmax = layers.softmax_with_loss()
    def f(x):
        p = softmax.forward(x)
        dy,loss = softmax.backward(x,labels)
        return loss,dy
    return f

#x =np.random.rand(4,5)
#f = softmax_x_forward()
#gradcheck(f,x)
