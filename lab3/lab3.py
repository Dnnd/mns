
# coding: utf-8

# In[4]:


N = 10
data_1 = [
    726044,
    615432,
    508077,
    807863,
    755223,
    848953,
    384558,
    666686,
    515201,
    483331,     
]

data_2 = [
    529082,
    729957,
    650570,
    445834,
    343280,
    959903,
    730049,
    730640,
    973224,
    258006,   
]

var_1 =  sorted(data_1)
var_2 = sorted(data_2)


import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
median_ranks = []
for i in range(0, len(data_1)):
    median_ranks.append( ((i + 1) - 0.3 ) / (len(data_1) + 0.4))
    


# In[5]:


print('mean_1: {}, mean_2: {}'.format(np.mean(data_1), np.mean(data_2)))
t, pval = stats.ttest_ind(var_1,var_2)
print('same with pval: {}'.format(pval))


# In[10]:


#y = a + bx
def regression(x, y):    
    num = np.sum( (x - np.mean(x))*(y - np.mean(y)))
    den = np.sum( (x - np.mean(x))**2)
    b = num / den
    a = np.mean(y) - b * np.mean(x)
    return a, b

def lin(a, b):
    def f(x):
        return a + b*x
    return np.vectorize(f)

x_1 = np.log(np.array(var_1))
x_2 = np.log(np.array(var_2))

y = np.array(median_ranks)
y = np.log(np.log(1/(1 - y)))


c_1, beta_1 = regression(x_1, y)
c_2, beta_2 = regression(x_2, y)

nu_1 = np.exp(-c_1/beta_1)
nu_2 = np.exp(-c_2/beta_2)

print('beta_1: {}, nu_1: {}'.format(beta_1, nu_1))
print('beta_2: {}, nu_2: {}'.format(beta_2, nu_2))
lin_1 = lin(c_1, beta_1)
lin_2 = lin(c_2, beta_2)

plt.figure(0)
plt.plot(x_1, y, 'ro')
plt.plot(x_2, y, 'bo')

plt.plot(x_1,lin_1(x_1), color='r')
plt.plot(x_2, lin_2(x_2), color='b')
plt.grid()
plt.show()


# In[8]:


def getR(beta, nu):
    def R(t):
        return np.exp(-((t/nu)**beta))
    return np.vectorize(R)

R_1 = getR(beta_1, nu_1)
R_2 = getR(beta_2, nu_2)

points = np.linspace(0, 2000000, num=2000000)
plt.plot(points, R_1(points))
plt.plot(points, R_2(points))

plt.grid()
plt.legend(['Выборка 1', 'Bыборка 2'])
plt.ticklabel_format(axis='x', style='sci')
plt.show()


# In[9]:


part = 0.99
t_1 = nu_1 * np.power(-np.log(part), 1/beta_1) / 100
t_2 = nu_2 * np.power(-np.log(part), 1/beta_2) / 100
print('t1: {}, t2: {}'.format(t_1/360, t_2/360))

