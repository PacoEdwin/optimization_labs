import lab1
import math
import random
import numpy as np
import numdifftools as nd

##########################################################################################
def conjugate_gradient_method(A, b, eps, x = None):
    if x is None:
        x = np.random.normal(size=[len(b)])
    
    r = b - np.dot(A, x)
    p = r
    r_dot = np.dot(np.transpose(r), r)

    for _ in b:
        Ap = np.dot(A, p)
        alpha = r_dot / np.dot(np.transpose(p), Ap)

        x = x + np.dot(alpha, p)
        r = r - np.dot(alpha, Ap)

        if np.linalg.norm(r) < eps:
            break

        r_new_dot = np.dot(np.transpose(r), r)
        p = r + (r_new_dot/r_dot)*p
        r_dot = r_new_dot

    return x

def test_conjugate_gradient_method():
    print('conjugate gradient method')
    
    eps = 1e-7
    n = 2

    A = np.array([[1,0], [0,1]])
    b = np.ones(n)

    res = conjugate_gradient_method(A, b, eps)
    print('custom functiom test', res)

    A = np.array([[202,-200],[-200, 200]])
    b = np.array([2,0])

    res = conjugate_gradient_method(A, b, eps)
    print('quadratic functiom test', res)

    print('')

##########################################################################################


##########################################################################################
def newton_method(func, eps, x):
    Gradient = nd.Gradient(func)
    Hessian = nd.Hessian(func)
    
    while True:
        Hx = Hessian(x)
        Gx = Gradient(x)
        delta_x = np.linalg.solve(Hx, -Gx)
        if np.array([math.isnan(val) for val in delta_x]).any():
            raise Exception("NaN")

        if np.linalg.norm(delta_x) < eps:
            return x

        x += delta_x

    return x

def test_newton_method():
    print('newton method test')

    eps = 1e-7
    
    x = np.random.normal(size=[2])
    func = lambda x: 100*((x[1] - x[0]) ** 2) + ((1 - x[0]) ** 2)
    x = newton_method(func, eps, x)
    print('quadratic function test:', x)
    
    x = np.random.normal(size=[2])
    func = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    x = newton_method(func, eps, x)
    print('Rosenbrock functiom test', x)

    # 2 / exp(((x - 1) / 2)^2 + (y- 1)^2) +  3 / exp(((x - 2) / 3)^2 + ((y - 3) / 2)^2)
    # 2 * exp(-((x - 1) / 2)^2 - (y- 1)^2) +  3 * exp(-((x - 2) / 3)^2 - ((y - 3) / 2)^2)
    def func(x):
        return (2 * math.exp(-((x[0] - 1) / 2) ** 2 - (x[1] - 1) ** 2) + \
            3 * math.exp(-((x[0] - 2) / 3) ** 2 - ((x[1] - 3) / 2) ** 2))

    eps = 1e-8
    i = 0
    while i < 5:
        x = lab1.get_random_point([[1.1, 2.4], [1, 3]])
        try:
            x = newton_method(func, eps, x.copy())
            i += 1
        except:
            print('Bad starting point', x)

        print('Test func:', x)
        print('f(x) = ', func(x))

    print('')

##########################################################################################

if __name__ == "__main__":
    test_conjugate_gradient_method()
    test_newton_method()