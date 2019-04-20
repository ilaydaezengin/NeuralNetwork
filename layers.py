from glob import glob
import numpy as np
import pickle
import im2col

class Layer:
    def __init__(self):
        self.params = {}

    def forward(self,x):
        pass
    def backward(self,dy):
        pass


class FullConn(Layer):
    def __init__(self,kernel_initilizer,bias_initilizer,shape):
        Layer.__init__(self)
        self.params['w'] = kernel_initilizer(shape)
        self.params['b'] = bias_initilizer(shape[1])

    def forward(self,x):
        W = self.params['w']
        b = self.params['b']
        return np.dot(x,W) + b

    def backward(self,x,dy):
        W = self.params['w']
        db = np.sum(dy, axis=0)
        dw = np.dot(x.T,dy)
        dx = np.dot(dy,W.T)
        self.params['dw'] = dw
        self.params['db'] = db
        return dx

class softmax_with_loss(Layer):
    def forward(self,x):
        y = x - np.max(x, axis=1, keepdims = True)
        p = np.exp(y)
        p = p / np.sum(p, axis=1, keepdims=True)
        return p

    def backward(self,x,labels):
        y = x - np.max(x, axis=1, keepdims = True)
        p = np.exp(y)
        p = p / np.sum(p, axis=1, keepdims=True)
        loss = -np.sum(np.log(p[np.arange(labels.shape[0]),labels])) / (labels.shape[0])
        dy = np.copy(p)
        dy[np.arange(labels.shape[0]),labels] -= 1
        dy /= labels.shape[0]
        return dy,loss


class conv(Layer):
    def __init__(self,kernel_initilizer,bias_initilizer,pad,stride,kernel_number,depth,kernel_h,kernel_w):
        Layer.__init__(self)
        self.params['w'] = kernel_initilizer((kernel_number,depth,kernel_h,kernel_w))
        self.params['b'] = bias_initilizer((kernel_number,1))
        self.pad = pad
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.kernel_number = kernel_number
        self.stride = stride

    def forward(self,x):
        N,c,h,w = x.shape
        y_height = ((h + 2 * self.pad - self.kernel_h) // self.stride) + 1
        y_width = ((w + 2 * self.pad - self.kernel_w) // self.stride) + 1
        xcol = im2col.im2col_indices(x,self.kernel_h,self.kernel_w,self.pad,self.stride)
        W = self.params['w'].reshape(self.kernel_number,-1)
        b = self.params['b']
        y = np.dot(W,xcol) + b
        y = y.reshape(self.kernel_number,y_height,y_width,N).transpose(3,0,1,2)
        return y

    def backward(self,x,dy):
        xcol = im2col.im2col_indices(x,self.kernel_h,self.kernel_w,self.pad,self.stride)
        dy = dy.transpose(1,2,3,0).reshape(self.kernel_number,-1)
        db = np.sum(dy, axis=1)
        dw = np.dot(dy,xcol.T).reshape(self.kernel_number,self.kernel_h,self.kernel_w,-1).transpose(0,3,1,2)
        self.params['dw'] = dw
        self.params['db'] = db
        W_shaped = self.params['w'].reshape(self.kernel_number,-1)
        dx = np.dot(dy.T,W_shaped).T
        dx_im = im2col.col2im_indices(dx,x.shape,self.kernel_h,self.kernel_w,self.pad,self.stride)
        return dx_im


class max_pooling(Layer):
    def __init__(self,kernel_initilizer,bias_initilizer,pad,kernel_h,kernel_w,stride):
        Layer.__init__(self)
        self.params['w'] = kernel_initilizer((kernel_h,kernel_w))
        self.params['b'] = bias_initilizer((kernel_h,kernel_w))
        self.pad = pad
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride = stride

    def forward(self,x):
        N,c,h,w = x.shape
        y_height = ((h + 2 * self.pad - self.kernel_h) // self.stride) + 1
        y_width = ((w + 2 * self.pad - self.kernel_w) // self.stride) + 1
        self.params['y_height'] = y_height
        self.params['y_width'] = y_width
        xshaped = x.reshape(N * c, 1, h, w)
        xcol = im2col.im2col_indices(xshaped,self.kernel_h,self.kernel_w,self.pad,self.stride)
        max_x = np.argmax(xcol, axis=0)
        self.params['max_x'] = max_x
        y = xcol[max_x,range(max_x.size)]
        y = y.reshape(y_height,y_width,N,c).transpose(2,3,0,1)
        return y

    def backward(self,x,dy):
        N,c,h,w = x.shape
        xshaped = x.reshape(N * c, 1, h, w)
        xcol = im2col.im2col_indices(xshaped,self.kernel_h,self.kernel_w,self.pad,self.stride)
        dxcol = np.zeros_like(xcol)
        dy = dy.transpose(2,3,0,1).ravel()
        dxcol[self.params['max_x'],range(self.params['max_x'].size)] = dy
        dx = im2col.col2im_indices(dxcol,(N * c, 1, h, w),self.kernel_h,self.kernel_w,self.pad, self.stride)
        dx = dx.reshape(x.shape)
        return dx

class dummy(Layer):
    def forward(self,x):
        return np.sum(x)
    def backward(self,x,dy):
        dy = np.ones_like(x)
        return dy


class flatten(Layer):
    def forward(self,x):
        y = x.reshape(x.shape[0],-1)
        return y
    def backward(self,x,dy):
        return dy.reshape(*x.shape)

class dropout(Layer):

    def __init__(self, prob):
        self.prob = prob
        self.params['prob'] = prob

    def forward(self,x):
        self.droplayer = np.random.binomial(1,prob,size=x.shape)
        y = x * self.droplayer
        return y

    def backward(self,dy):
        dx = dy * self.droplayer
        return dx


class batch_normalization(Layer):
    def forward(self,x,w,b):
        E = np.mean(x, axis=0)
        v = np.var(x, axis=0)
        xhat = (x - E)/ np.sqrt(v**2 + E)
        y = w.xhat + b
        return y,xhat,w,b

    def backward(self,dy):
        dw = np.sum(dy * xhat,axis=0)
        db = np.sum(dy, axis=0)
        dxhat = dy * w
        dE = np.sum(dxhat, (-1 / np.sqrt(v**2 + E)), axis=0)
        dv = dxhat * (-0.5 * np.sum((x-E) * (v**2 + E)**(-1.5), axis=0))
        dx = 1 / (N * np.sqrt(v**2 +E)) * (N * dxhat - np.sum(dxhat, axis=0) - (xhat * np.sum(dxhat*x, axis=0)))
        return dx



class Relu(Layer):
    def forward(self,x):
        self.x = x
        return np.maximum(x,0)

    def backward(self,x,dy):
        dx = np.copy(dy)
        dx[x <= 0] = 0
        return dx


def zero_initilizer(shape):
    b = np.zeros(shape)
    return b

def xaxier_initilizer(shape):
    W = np.random.rand(*shape) * np.sqrt(2.0 / shape[0])
    return W




def unpickle(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary

def shape_training_data():
    filename = 'cifar-10-batches-py/data_batch_'
    data_list = []
    label_list = []
    for i in range(1,6):
        datafile = unpickle(filename+str(i))
        data = datafile[b'data'].astype(np.float32)
        labels = datafile[b'labels']
        data_list.append(data)
        label_list.append(labels)
    N = len(data_list)*(data_list[0].shape[0])
    D = data_list[0].shape[1]
    data_array = np.asarray(data_list).reshape(N,D)
    label_array = np.asarray(label_list).reshape(N)
    return data_array,label_array




def get_next_batch(perm, data,labels,idx, batch_size):
    if data.shape[0]-idx < batch_size:
        index_array = perm[idx:]
    else:
        index_array = perm[idx:idx+batch_size]
    databatch = data[index_array]
    labelbatch = labels[index_array]
    return np.asarray(databatch), np.asarray(labelbatch)



def adam(beta1,beta2,alpha,var_list):
    data = []
    for i in range(len(var_list)):
        v = np.zeros(var_list[i].shape)
        r = np.zeros(var_list[i].shape)
        data.append((v,r))
    def update(t,vlist,d_vlist):
        for i in range(len(d_vlist)):
            v = (beta1 * data[i][0] + ((1 - beta1) * d_vlist[i])) / (1 - beta1**t)
            r = (beta2 * data[i][1] + ((1 - beta2) * d_vlist[i]**2)) / (1 - beta2**t)
            vlist[i] = vlist[i] - alpha * data[i][0] / (np.sqrt(data[i][1]) + e)
            data[i] = (v,r)
        return vlist
    return update
