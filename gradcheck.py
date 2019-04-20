
import numpy as np


def gradcheck(f, x):
    """ Gradient check for a function f.

    Arguments:
    f -- a function that takes a single argument and outputs the
         cost and its gradients
    x -- the point (numpy array) to check the gradient at
    """

    fx, grad = f(x)
    h = 1e-4

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        temp = x[idx]
        x[idx] = temp + h
        f1,f1_grad = f(x)
        x[idx] = temp - h
        f2,f2_grad = f(x)
        x[idx] = temp
        numgrad = (f1 - f2) / (2 * h)
        # Compare gradients
        reldiff = abs(numgrad - grad[idx]) / max(1, abs(numgrad), abs(grad[idx]))
        if reldiff > 1e-5:
            print('Gradient check failed.')
            print('First gradient error found at index {}'.format(str(idx)))
            print('Your gradient: {} , Numerical gradient: {}'.format(grad[idx], numgrad))
            return

        it.iternext()

    print('Gradient check passed!')




def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print('Running sanity checks...')
    gradcheck(quad, np.array(123.456))      # scalar test
    gradcheck(quad, np.random.randn(3,))    # 1-D test
    gradcheck(quad, np.random.randn(4,5))   # 2-D test

if __name__ == "__main__":
    sanity_check()
