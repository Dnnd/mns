
# coding: utf-8

# In[109]:


import numpy as np


def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def derivative(y, x, w, i):
    ret = -y * x[i] * sigmoid(-y * np.dot(x, w))
    return ret


class GradDescenter:

    def __init__(self, x, y, w, dimx):
        self.x = np.array(x)
        self.y = np.array(y)
        self.w = np.array(w)
        self.n = 0
        self.order = len(y)
        self.dimx = dimx

    def step(self, n):
        next_w = self.w - 1/np.sqrt(n + 1) * self.grad()
        eps = np.linalg.norm(next_w - self.w)
        self.w = next_w
        return eps

    def grad(self):
        return np.array([self.part_deriv(i) for i in range(0, self.dimx)])

    def part_deriv(self, i):
        terms = [derivative(self.y[k], self.x[k], self.w, i)
                 for k in range(0, self.order)]
        res = np.sum(terms)
        return res

    def train(self, eps):
        err = np.inf
        n = 0
        while err > eps:
            err = self.step(n)
            n = n + 1
        return n


class LdaClassifier:

    def __init__(self, T, train_set):
        self.train_set = train_set
        self.T = T
        self._cov = None
        self._m1 = None
        self._m2 = None

    def train(self):

        m0, m1 = self.get_mean()
        cov = self.get_cov()

        cov_inv = np.linalg.inv(cov)
        self.w = np.linalg.inv(cov)*(m0 - m1)
        self.c = 0.5 * (self.T - np.transpose(m0) * cov_inv *
                        m0 + np.transpose(m1) * cov_inv * m1)

    def get_cov(self):
        if self._cov is not None:
            return self._cov

        n1 = len(train_set[0]['y'])
        n2 = len(train_set[1]['y'])

        x1 = [np.transpose(np.matrix(x)) for x in train_set[0]['x']]
        x2 = [np.transpose(np.matrix(x)) for x in train_set[1]['x']]

        m1, m2 = self.get_mean()

        s1 = 1/n1 * np.sum([(x - m1) * np.transpose(x - m1)
                            for x in x1], axis=0)
        s2 = 1/n2 * np.sum([(x - m2) * np.transpose(x - m2)
                            for x in x2], axis=0)

        self._cov = n1/(n1 + n2) * s1 + n2/(n1 + n2) * s2
        return self._cov

    def get_mean(self):
        if self._m1 is not None and self._m2 is not None:
            return self._m1, self._m2

        x1 = train_set[0]['x']
        x2 = train_set[1]['x']

        n1 = len(x1)
        n2 = len(x2)

        m1 = 1/n1 * np.sum(x1, axis=0)
        m2 = 1/n2 * np.sum(x2, axis=0)

        self._m1 = np.transpose(np.matrix(m1))
        self._m2 = np.transpose(np.matrix(m2))

        return self._m1, self._m2


# In[156]:


from numpy import random as rnd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D


def genDividable2d(begin, end, bias, size):

    base = (begin + end) / 2

    mean1 = [base, base*(bias)]
    mean2 = [base, base*(-bias)]

    cov = [[1, 0], [0, 0.1]]
    negones = -1*np.ones(size)

    data1 = np.random.multivariate_normal(mean1, cov, size)
    data2 = np.random.multivariate_normal(mean2, cov, size)

    return np.stack((negones, data1[:, 0], data1[:, 1]), axis=-1), np.stack((negones, data2[:, 0], data2[:, 1]), axis=-1)


def genUndividable2d(begin, end, size):
    base = (begin + end) / 2
    k = rnd.uniform(0, 1)

    mean = [base, base*k]

    cov = [[1, 0], [0, 0.1]]
    negones = -1*np.ones(size)

    data1 = np.random.multivariate_normal(mean, cov, size)
    data2 = np.random.multivariate_normal(mean, cov, size)
    return np.stack((negones, data1[:, 0], data1[:, 1]), axis=-1), np.stack((negones, data2[:, 0], data2[:, 1]), axis=-1)


def genDividable3d(begin, end, bias, size):

    base = (begin + end) / 2

    mean1 = [base, base, base*(bias)]
    mean2 = [base, base, base*(-bias)]

    cov = [[1, 0, 0], [0, 1, 0], [0, 0, 0.1]]
    negones = -1*np.ones(size)

    data1 = np.random.multivariate_normal(mean1, cov, size)
    data2 = np.random.multivariate_normal(mean2, cov, size)

    return (np.stack((negones, data1[:, 0], data1[:, 1], data1[:, 2]), axis=-1),
            np.stack((negones, data2[:, 0], data2[:, 1], data2[:, 2]), axis=-1))


def linearize(w):
    def lin(x):
        return -x*w[1]/w[2] + w[0]/w[2]
    return lin


# In[154]:


size = 50
q = 5
eps = 0.1

data = []
data1 = []
data2 = []

y1 = -1 * np.ones(size)
y2 = np.ones(size)
y = np.concatenate((y1, y2))

y1t = -1 * np.ones(size * (q - 1))
y2t = np.ones(size * (q - 1))
yt = np.concatenate((y1t, y2t))

for i in range(0, q):
    d1, d2 = genDividable2d(0, 1, 1.2, size)
    data1.append(d1)
    data2.append(d2)


data1 = np.array(data1)
data2 = np.array(data2)

beg_w = [0, 1, 1]
err_1 = 0
err_2 = 0

for i in range(0, q):
    d1 = data1[i]
    d2 = data2[i]
    plt.figure(1)
    plt.plot(d1[:, 1], d1[:, 2], 'o', color=(i * 1/q, i * 1/q, i * 1/q))
    plt.plot(d2[:, 1], d2[:, 2], 'o', color=(i * 1/q, i * 1/q, i * 1/q))

    x = np.concatenate((d1, d2), axis=0)

    data1_t = np.delete(data1, i, 0)
    data2_t = np.delete(data2, i, 0)
    x1_t = []
    x2_t = []
    xt = []
    for row in data1_t:
        for j in row:
            x1_t.append(j)

    for row in data2_t:
        for j in row:
            x2_t.append(j)

    for j in x1_t:
        xt.append(j)

    for j in x2_t:
        xt.append(j)

    x1_t = np.array(x1_t)
    x2_t = np.array(x2_t)

    xt = np.array(xt)

    gd = GradDescenter(xt, yt, beg_w, 3)
    gd.train(0.1)

    lin = linearize(gd.w)

    for j in range(0, len(y)):
        if (lin(x[j][1]) < x[j][2]) and (y[j] == 1):
            err_1 += 1
        elif (lin(x[j][1]) > x[j][2]) and (y[j] == -1):
            err_1 += 1

    train_set = np.array([{'x': x1_t[:, 1:], 'y': y1t},
                          {'x': x2_t[:, 1:], 'y': y2t}])
    lda = LdaClassifier(0.1, train_set)
    lda.train()
    w = np.transpose(lda.w)[0]
    c = lda.c[0, 0]
    for j in range(0, len(y)):
        t = x[j][1]
        val = -t*w[0, 0]/w[0, 1] + c/w[0, 1]
        if (val < x[j][2]) and (y[j] == 1):
            err_2 += 1
        elif (val > x[j][2]) and (y[j] == -1):
            err_2 += 1

err_1 = 1/q * err_1
err_2 = 1/q * err_2
print('Error from grad descent: {}'.format(err_1))
print('Error from LDA: {}'.format(err_2))


# In[29]:


size = 50
data1, data2 = genDividable2d(0, 1, 1.5, size)

y1 = -1 * np.ones(size)
y2 = np.ones(size)

x1 = data1[:, 1:]
x2 = data2[:, 1:]

train_set = np.array([{'x': x1, 'y': y1}, {'x': x2, 'y': y2}])
lda = LdaClassifier(0.1, train_set)
lda.train()

w = np.transpose(lda.w)[0]
c = lda.c[0, 0]

xs = np.arange(-5, 5, 0.01)
ys = [-x*w[0, 0]/w[0, 1] + c/w[0, 1] for x in xs]
plt.figure(2)
plt.plot(xs, ys, 'g')

plt.plot(data1[:, 1], data1[:, 2], 'ro')
plt.plot(data2[:, 1], data2[:, 2], 'bo')
plt.show()
print()


# In[157]:


size = 25
data1, data2 = genDividable3d(0, 1, 1, size)

y1 = -1 * np.ones(size)
y2 = np.ones(size)

y = np.concatenate((y1, y2))
x = np.concatenate((data1, data2))
xs = np.arange(-5, 10, 5)

w = [0, 1, 1, 0]
gd = GradDescenter(x, y, w, 4)
gd.train(0.1)

x1 = data1[:, 1:]
x2 = data2[:, 1:]

train_set = np.array([{'x': x1, 'y': y1}, {'x': x2, 'y': y2}])
lda = LdaClassifier(0.1, train_set)
lda.train()
lw = np.transpose(lda.w)

xv, yv = np.meshgrid(xs, xs)
zvg = -1/gd.w[3] * (-gd.w[1]*xv - gd.w[2]*yv + gd.w[0])
zvl = -1/lw[0, 2] * (-lw[0, 0]*xv - lw[0, 1]*yv + lda.c[0, 0])

fig = plt.figure(3)

ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xv, yv, zvg, color='g')
ax.plot_surface(xv, yv, zvl, color='y')
ax.scatter(data1[:, 1], data1[:, 2], data1[:, 3], color='r')
ax.scatter(data2[:, 1], data2[:, 2], data2[:, 3], color='b')


# In[14]:


size = 100
data1, data2 = genDividable2d(0, 1, 1, size)

y1 = -1 * np.ones(size)
y2 = np.ones(size)

y = np.concatenate((y1, y2))
x = np.concatenate((data1, data2))


x1 = data1[:, 1:]
x2 = data2[:, 1:]

train_set = np.array([{'x': x1, 'y': y1}, {'x': x2, 'y': y2}])
lda = LdaClassifier(0.1, train_set)
lda.train()
lw = np.transpose(lda.w)

w = [0, 1, 1]
gd = GradDescenter(x, y, w, 3)
gd.train(0.1)
lin = linearize(gd.w)

xs = np.arange(-5, 5, 0.01)
ysg = [lin(x) for x in xs]

lw = np.transpose(lda.w)
ysl = [-x*lw[0, 0]/lw[0, 1] + lda.c[0, 0]/lw[0, 1] for x in xs]
plt.figure(4)
plt.plot(xs, ysg, 'g')
plt.plot(xs, ysl, 'y')
plt.plot(data1[:, 1], data1[:, 2], 'ro')
plt.plot(data2[:, 1], data2[:, 2], 'bo')
plt.show()
