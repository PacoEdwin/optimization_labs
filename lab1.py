import random
import numpy as np

##########################################################################################
def dichotomy(a, b, eps, delta, func):
    assert(a < b)
    if b - a < eps:
        func_a = func(a)
        func_b = func(b)
        if func_a < func_b:
            return a, func_a
        else:
            return b, func_b

    print('segment: [{}, {}]'.format(a, b))
    mid = (a + b)/2
    
    x1 = mid - delta
    assert(x1 > a)

    x2 = mid + delta
    assert(x2 < b)

    func_x1 = func(x1)
    func_x2 = func(x2)

    if func_x1 == func_x2:
        return dichotomy(x1, x2, eps, delta, func)
    if func_x1 < func_x2:
        return dichotomy(a, x2, eps, delta, func)
    else:
        return dichotomy(x1, b, eps, delta, func)

def test_dichotomy():
    print('fibonacci dichotomy')

    func = lambda x: (x ** 3) / 3 - (x ** 2) / 2 - x - 1
    a = 1
    b = 2
    eps = 0.000001
    delta = random.uniform(0, eps/2)

    if delta == eps/2:
        delta -= np.finfo(float).eps

    res = dichotomy(a, b, eps, delta , func)
    print('dichotomy result: ', res)
##########################################################################################


##########################################################################################
def l_golden_section(a, b, x2, func_x2, eps, func):
    assert(a < b)
    if b - a < eps:
        return x2, func_x2

    print('segment: [{}, {}]'.format(a, b))

    golden = (3 - 5 ** 0.5) / 2 

    x1 = a + golden*(b - a) 
    func_x1 = func(x1)

    if func_x1 < func_x2:
        return l_golden_section(a, x2, x1, func_x1, eps, func)
    else:
        return r_golden_section(x1, b, x2, func_x2, eps, func)

def r_golden_section(a, b, x1, func_x1, eps, func):
    assert(a < b)
    if b - a < eps:
        return x1, func_x1

    print('segment: [{}, {}]'.format(a, b))

    golden = (5 ** 0.5 - 1) / 2

    x2 = a + golden*(b - a)
    func_x2 = func(x2)

    if func_x1 < func_x2:
        return l_golden_section(a, x2, x1, func_x1, eps, func)
    else:
        return r_golden_section(x1, b, x2, func_x2, eps, func)

def golden_section(a, b, eps, func):
    golden = (3 - 5 ** 0.5) / 2

    x1 = a + golden * (b - a)
    x2 = x1 + golden * (b - x1)
    
    func_x1 = func(x1)
    func_x2 = func(x2)

    if func_x1 < func_x2:
        return l_golden_section(a, x2, x1, func_x1, eps, func)
    else:
        return r_golden_section(x1, b, x2, func_x2, eps, func)

def test_golden_section():
    print('golden section test')

    func = lambda x: (x ** 3) / 3 - (x ** 2) / 2 - x - 1
    a = 1
    b = 2
    eps = 0.000001

    res = golden_section(a, b, eps, func)
    print('golden section result', res)
##########################################################################################


##########################################################################################
def get_fibonacci_sequence(val):
    fibs = [1, 1]
    while fibs[-1] < val:
        fibs.append(fibs[-1] + fibs[-2])
    
    return fibs

def fibonacci_method_inner(a, b, n, fibs, eps, func):
    x1 = a + fibs[n - 2] / fibs[n] * (b - a)
    x2 = a + fibs[n - 1] / fibs[n] * (b - a)
    
    func_x1 = func(x1)
    func_x2 = func(x2)

    while n != 2:
        n -= 1
        if func_x1 < func_x2:
            b = x2
            x2 = x1
            func_x2 = func_x1
            x1 = a + fibs[n - 2] / fibs[n] * (b - a)
            func_x1 = func(x1)
        else:
            a = x1
            x1 = x2
            func_x1 = func_x2
            x2 = a + fibs[n - 1] / fibs[n] * (b - a)
            func_x2 = func(x2)
        
        print('segment: [{}, {}]'.format(a, b))


    return (a + b) / 2,func((a + b) / 2)


def fibonacci_method(a, b, eps, func):
    assert(a < b)

    fibs = get_fibonacci_sequence((b - a) / eps)
    n = len(fibs) - 1

    return fibonacci_method_inner(a, b, n, fibs, eps, func)

def test_fibonacci_method():
    print('fibonacci method test')

    func = lambda x: (x ** 3) / 3 - (x ** 2) / 2 - x - 1
    a = 1
    b = 2
    eps = 0.000001
    res = fibonacci_method(a, b, eps, func)
    print('fibonacci method result:', res)

##########################################################################################


##########################################################################################
def get_random_point(domain):
    return np.array([random.uniform(a, b) for a, b in domain])

def gradient_descent(domain, eps, func, gradient_func, x = None):
    def check_condition(x, x_history, func, gradient_func, eps):
        return abs(func(*x) - func(*x_history[-1])) > eps 
            # and abs(x - x_history[-1]) > eps \
            # and \ # abs(gradient_func(*x)) > eps \

    if x is None:
        x = get_random_point(domain)

    # check dimensions
    assert(len(domain) == len(x))

    x_history = []
    condition = True
    while condition:
        x_history.append(x)

        calculate_new_point = lambda h : x_history[-1] - h * gradient_func(*x_history[-1])
        h, _ = fibonacci_method(0, 10, eps, lambda point: func(*calculate_new_point(point)))
        x = calculate_new_point(h)
        
        condition = check_condition(x, x_history, func, gradient_func, eps).all()
    
    return x, x_history

def test_gradient_descent():
    eps = 1e-6
    domain = [[-10, 10], [-10, 10]]

    #func = lambda x, y: x ** 2 + y ** 2
    #gradient_func =  lambda x, y: np.array([2*x, 2*y])    
    func = lambda x, y: 100*((x - y) ** 2) + ((1 - y) ** 2)
    gradient_func =  lambda x, y: np.array([200*(x - y), -200*x + 202*y - 2])

    res, x_history = gradient_descent(domain, eps, func, gradient_func)
    print(res)
##########################################################################################

if __name__ == "__main__":
    test_gradient_descent()
    # test_dichotomy()
    # test_golden_section()
    # test_fibonacci_method()