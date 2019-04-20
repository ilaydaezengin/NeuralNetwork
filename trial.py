from glob import glob
import numpy as np
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary

def shape_training_data():
    filename='cifar-10-batches-py/data_batch_'
    data_list=[]
    label_list=[]
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
    if N-idx < batch_size:
        index_array = perm[idx:]
    else:
        index_array = perm[idx:idx+batch_size]
    databatch = data[index_array]
    labelbatch = labels[index_array]
    return np.asarray(databatch), np.asarray(labelbatch)



def d_relu(x):
    return 1. * (x > 0)




data, labels = shape_training_data()
label_names_file = unpickle("cifar-10-batches-py/batches.meta")
C = len(label_names_file[b'label_names'])

N = data.shape[0]
D = data.shape[1]
alpha = 0.1
batch_size = 64
L2 = D
L1 = D
W_1 = np.random.rand(D,L1) * np.sqrt(2.0/D)
b_1 = np.zeros([1,L1])
W_2 = np.random.rand(L1,L2)* np.sqrt(2.0/L1)
b_2 = np.zeros([1,L2])
W_3 = np.random.rand(L2,C) * np.sqrt(2.0/L2)
b_3 = np.zeros([1,C])
l = 0.0001
E = np.transpose(np.sum(data, axis=0))/N
v = np.sum(np.square(data-E), axis=0)/N
data -=E
data /=v

for i in range(4):
    perm = np.random.permutation(N)
    idx = 0
    while idx<N:
        databatch,labelbatch = get_next_batch(perm,data,labels,idx,batch_size)

        x_1 = databatch
        z_1 = np.dot(x_1,W_1) + b_1
        x_2 = np.maximum(z_1,0)

        z_2 = np.dot(x_2,W_2) + b_2
        x_3 = np.maximum(z_2,0)

        y = np.dot(x_3, W_3) +b_3
        y = y - np.max(y, axis=1, keepdims=True)
        yexp = np.exp(y)
        p = yexp / np.sum(yexp, axis=1, keepdims=True)
        dataloss = -np.sum(np.log(p[np.arange(x_3.shape[0]),labelbatch]))/(x_3.shape[0])
        loss = dataloss + (0.5*l*np.sum(W_1*W_1)) + (0.5*l*np.sum(W_2*W_2)) + (0.5*l*np.sum(W_3*W_3))

        dy = np.copy(p)
        dy[np.arange(x_3.shape[0]),labelbatch] -= 1
        dy /= x_3.shape[0]

        db_3 = np.sum(dy, axis=0)
        dw_3 = np.dot(np.transpose(x_3), dy)
        dx_3 = (np.dot(W_3, np.transpose(dy))).T
        dz_2 = dx_3 * d_relu(z_2)


        db_2 = np.sum(dz_2, axis=0)
        dw_2 = np.dot(np.transpose(x_2), dz_2)
        dx_2 = (np.dot(W_2, np.transpose(dz_2))).T
        dz_1 = dx_2 * d_relu(z_1)

        db_1 = np.sum(dz_1, axis=0)
        dw_1 = np.dot(np.transpose(x_1), dz_1)
        dx_1 = np.dot(W_1, np.transpose(dz_1))

        dw_1 += l * W_1
        dw_2 += l * W_2
        dw_3 += l * W_3

        W_1 = W_1 - alpha * dw_1
        b_1 = b_1 - alpha * db_1
        W_2 = W_2 - alpha * dw_2
        b_2 = b_2 - alpha * db_2
        W_3 = W_3 - alpha * dw_3
        b_3 = b_3 - alpha * db_3
        idx = idx + batch_size
        print("epoch: {0}, loss: {1}".format(i,loss))

    test_file = unpickle("cifar-10-batches-py/test_batch")
    testdata = test_file[b'data'].astype(np.float32)
    testdata -=E
    testdata /=v
    testlabels = test_file[b'labels']
    t_x1 = testdata
    t_z1 = np.dot(t_x1,W_1) + b_1
    t_x2 = np.maximum(t_z1,0)
    t_z2 = np.dot(t_x2,W_2) + b_2
    t_x3 = np.maximum(t_z2,0)
    t_y = np.dot(t_x3, W_3) +b_3
    t_y = t_y - np.max(t_y, axis=1, keepdims=True)
    t_yexp = np.exp(t_y)
    prob = t_yexp / np.sum(t_yexp, axis=1, keepdims=True)
    test_array = np.argmax(prob, axis=1).T
    print(np.count_nonzero(test_array==testlabels)*100/testdata.shape[0])
