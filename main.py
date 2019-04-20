from glob import glob
import numpy as np
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


def shape_training_data():
    a = unpickle("cifar-10-batches-py/data_batch_")
    b1 = a[b'data'].astype(np.float32)
    b2 = a[b'labels']
    filenames = glob("cifar-10-batches-py/data_batch_")
    filenames.remove("cifar-10-batches-py/data_batch_1")
    for fname in filenames:
        d = unpickle(fname)
        x = d[b'data'].astype(np.float32)
        x2 = d[b'labels']
        b1 = np.concatenate([b1,x])
        b2 = np.concatenate([b2,x2])
    return (b1, b2)


def shape_test_data(a1):
    d = unpickle(a1)
    x = d[b'data']
    y = d[b'labels']
    return (x,y)

d = np.random.permutation(50000)
data, labels = shape_training_data()
def get_next_batch(idx, batch_size):
    idxArr = d[idx:idx+batch_size]
    databatch = data[idxArr]
    labelbatch = labels[idxArr]
    return np.asarray(databatch), np.asarray(labelbatch)

alpha = 0.0001
batch_size = 32
w = np.random.rand(3072,10)
b = np.zeros([1,10])
idx = 0
while True:
    data, labels = get_next_batch(idx,batch_size)
    y = np.dot(data,w) + b
    y = y - np.max(y, axis=1, keepdims=True)
    exp_of_output=np.exp(y)
    p = np.exp(y)/np.sum(np.exp(y), axis=1,keepdims=True)
    loss = -np.sum(np.log(p[np.arange(batch_size),labels]))/data.shape[0]
    print(loss)
    p[np.arange(batch_size),labels]-=1
    db = np.sum(p, axis=0)
    dw = np.dot(np.transpose(data), p)
    dx = np.dot(w, np.transpose(p))
    idx = idx + batch_size
    w = w - alpha * dw
    b = b - alpha * db
