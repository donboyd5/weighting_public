# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:31:58 2020

@author: donbo
"""


# import autograd.numpy as np
from autograd import grad
from autograd import hessian_vector_product as hvp
from autograd import elementwise_grad as egrad
from autograd import make_hvp as make_hvp
from autograd import jacobian
from autograd import hessian
from scipy.optimize import minimize

import inspect
from inspect import signature

f_hvp = hvp(f)
def f_hvp_wrap(x, p, xmat, targets, objscale, diff_weights):
    return f_hvp(x, p, xmat=xmat, targets=targets, objscale=objscale, diff_weights=diff_weights)

p = mtp.Problem(h=6, s=3, k=2)
p = mtp.Problem(h=10, s=3, k=2)
p = mtp.Problem(h=100, s=5, k=3)
p = mtp.Problem(h=1000, s=10, k=6)
p = mtp.Problem(h=10000, s=30, k=10)
p = mtp.Problem(h=20000, s=50, k=30)
p = mtp.Problem(h=30000, s=50, k=30)

xmat = p.xmat
wh = p.wh
targets = p.targets
h = xmat.shape[0]
s = targets.shape[0]
k = targets.shape[1]

diff_weights = get_diff_weights(targets)

# A = lil_matrix((h, h * s))
# for i in range(0, h):
#     A[i, range(i*s, i*s + s)] = 1
# A
# b = A.todense()  # ok to look at dense version if small

# fast way to fill A
# get i and j indexes of nonzero values, and data
inz = np.arange(0, h).repeat(s)
jnz = np.arange(0, s * h)
A = sp.sparse.coo_matrix((np.ones(h*s), (inz, jnz)), shape=(h, h * s))
# A2.todense() - A.todense()

A = A.tocsr()  # csr format is faster for our calculations
# A = A.tocsc()
lincon = sp.optimize.LinearConstraint(A, wh, wh)

wsmean = np.mean(wh) / targets.shape[0]
wsmin = np.min(wh) / targets.shape[0]
wsmax = np.max(wh)  # no state can get more than all of the national weight

objscale = 1

bnds = sp.optimize.Bounds(wsmin / 10, wsmax)

# starting values (initial weights), as an array
# x0 = np.full(h * s, 1)
# x0 = np.full(p.h * p.s, wsmean)
# initial weights that satisfy constraints
x0 = np.ones((h, s)) / s
x0 = np.multiply(x0, wh.reshape(x0.shape[0], 1)).flatten()

# verify that starting values satisfy adding-up constraint
np.square(np.round(x0.reshape((h, s)).sum(axis=1) - wh, 2)).sum()

# pv = x0 * 2
# pvr = pv[::-1]
# vec = f_hvp_wrap(x0, pvr, xmat, targets, objscale, diff_weights)

resapprox2 = minimize(f, x0,
               method='trust-constr',
               bounds=bnds,
               constraints=lincon,  # lincon lincon_feas
               jac=gfun,
               hess='2-point',
               # hessp=f_hvp_wrap,
               args=(xmat, targets, objscale, diff_weights),
               options={'maxiter': 100, 'verbose': 2,
                        'gtol': 1e-4, 'xtol': 1e-4,
                        'initial_tr_radius': 1,  # default 1
                        'factorization_method': 'AugmentedSystem'})

reshvp = minimize(f, x0,
               method='trust-constr',
               bounds=bnds,
               constraints=lincon,  # lincon lincon_feas
               jac=gfun,
               # hess='2-point',
               hessp=f_hvp_wrap,  # f_hvp_wrap wrap2
               args=(xmat, targets, objscale, diff_weights),
               options={'maxiter': 100, 'verbose': 2,
                        'gtol': 1e-4, 'xtol': 1e-4,
                        'initial_tr_radius': 1,  # default 1
                        'factorization_method': 'AugmentedSystem'})


# %% older below here



def rosen(x):
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

minimize(
    rosen, x0, method='Newton-CG',
    jac=grad(rosen), hessp=hvp(rosen),
    options={'xtol': 1e-8, 'disp': True})


def rosen2(x, a, b):
    z = a * b
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

hvpr = hvp(rosen2)
hvpr(x0, tns, a=a1, b=b1) # this works
# wrap it in the regular calling arrangement
def hvprw(x, p, a, b):
    return hvpr(x, p, a=a, b=b)
hvprw(x0, tns, a1, b1)

minimize(
    rosen2, x0, method='Newton-CG',
    jac=grad(rosen2), hessp=hvprw,
    args=(a1, b1),
    options={'xtol': 1e-8, 'disp': True})


def rosen3(x, *, a, b):
    z = a * b
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosen4(x, **a, **b):
    z = a * b
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)


x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

minimize(
    rosen, x0, method='Newton-CG',
    jac=jacobian(rosen), hessp=hvp(rosen),
    options={'xtol': 1e-8, 'disp': True})

minimize(
    rosen, x0, method='Newton-CG',
    jac=jacobian(rosen2), hessp=hvp(rosen2),
    args=(a=5, b=7),
    options={'xtol': 1e-8, 'disp': True})

a1 = 12
b1 = 7

fn = make_hvp(rosen)
fn = hvp(rosen)
fn(x0)
x2 = x0 * 2
fn(x0, x2)

fn = hvp(rosen2)
sig = signature(fn)
str(sig)
inspect.signature(rosen2)
inspect.getfullargspec(rosen2)

inspect.signature(fn)
inspect.getfullargspec(fn)

# the returned function has arguments (*args, tensor, **kwargs)
tns = x0
x2 = x0 * x0
fn(x0, tns, a=a1, b=b1) # this works
fn(x2, tns, a=a1, b=b1) # it is not constant

def fx(x, t):
    diffs = x - t
    diffs = diffs * 37.0
    sqdiffs = np.square(diffs)
    obj = np.sum(sqdiffs)
    return obj

# the vector product is constant
x1 = np.array([1, 2, 3, 4], dtype='float64')
x2 = np.array([12, 27, 23, 41], dtype='float64')
t1 = np.array([7, 9, 13, 14], dtype='float64')
tnx = np.array([2, 4, 6, 7], dtype='float64')
tnx2 = np.array([-10, 4, 36, 27], dtype='float64')
fx(x1, t1)
fx(x2, t1)

# [19166., 24642., 35594., 38332.]
h = np.array([[1, 0, 2],
              [0, 20, 0],
              [0, 0, 10]])
ph = np.array([1.0, 2.0, 3.0])
h.dot(ph)


hx2 = hessian(fx)
hm = hx2(x1, t1)
np.dot(hm, tnx)
np.dot(hm, tnx2)
np.dot(tnx, hm)

hx = hvp(fx)
hx(x1, tnx, t1)
hx(x2, tnx, t1)
hx(x2, tnx2, t1)
hx(x2, tnx2, t1*2)

x1[-1]
x1[:-2]

fn = hvp(rosen3)
sig = signature(fn)
str(sig)

fn(x0)
x2 = x0 * 2
fn(x0, x2, a1, b1)
fn(x0, x2, a=a1, b=b1)

def foo(a, b, c=4, *arglist, **keywords): pass
inspect.getfullargspec(foo)
inspect.signature(foo)

def foo(a, b, c=4, *arglistxx, **keywordsyy): pass
inspect.getfullargspec(foo)
inspect.signature(foo)

tuple('abc')
('a', 'b', 'c')
tuple( ('a', 'b', 'c'))
tuple( [1, 2, 3] )
('a', 1, 'c')


fn = hvp(rosen2)(x, a, b)
# rosen2(x, a, b):
minimize(
    rosen2, x0, method='Newton-CG',
    jac=grad(rosen2), hessp=fn,
    args=(a=a1, b=b1),
    options={'xtol': 1e-8, 'disp': True})

def hessian_tensor_product(fun, argnum=0):
    """Builds a function that returns the exact Hessian-tensor product.
    The returned function has arguments (*args, tensor, **kwargs), and for
    vectors takes roughly 4x as long to evaluate as the original function."""
    fun_grad = grad(fun, argnum)
    def vector_dot_grad(*args, **kwargs):
        args, vector = args[:-1], args[-1]
        return np.tensordot(fun_grad(*args, **kwargs), vector, np.ndim(vector))
    return grad(vector_dot_grad, argnum)
hessian_vector_product = hessian_tensor_product



def foo(a, *, b:int, **kwargs):
    pass

def foo(a, b, c, d):
    pass
foo2 = foo
sig = signature(foo2)
str(sig)


str(sig.parameters['b'])


sig.parameters['b'].annotation







minimize(
    rosen2, x0, method='Newton-CG',
    jac=jacobian(rosen2), hessp=hvp(rosen2),
    options={'xtol': 1e-8, 'disp': True})

hessian_tensor_product
minimize(
    rosen2, x0, method='Newton-CG',
    jac=jacobian(rosen2), hessp=hvp(rosen2),
    args=(a, b),
    options={'xtol': 1e-8, 'disp': True})

minimize(
    rosen2, x0, method='Newton-CG',
    jac=egrad(rosen2),
    # hess='2-point',
    args=(a, b),
    options={'xtol': 1e-8, 'disp': True})
