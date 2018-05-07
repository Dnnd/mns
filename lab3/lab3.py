
# coding: utf-8

# In[19]:


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


# In[116]:


import numpy as np
import matplotlib.pyplot as plt
import math
mean_1 = np.sum(data_1)/len(data_1) 
mean_2 = np.sum(data_1)/len(data_2)


# In[50]:


median_ranks = []
for i in range(0, len(data_1)):
    median_ranks.append( ((i + 1) - 0.3 ) / (len(data_1) + 0.4))


# In[108]:


from scipy import stats

#y = a + bx
def regression(x, y, N):    
    num = np.sum( (x - np.mean(x))*(y - np.mean(y)))
    den = np.sum( (x - np.mean(x))**2)
    b = num / den
    a = np.mean(y) - b * np.mean(x)
    return a, b

x_1 = np.log(np.array(var_1))
x_2 = np.log(np.array(var_2))

y = np.array(median_ranks)
y = np.log(np.log(1/(1 - y)))


c_1, beta_1 = regression(x_1, y, N)
c_2, beta_2 = regression(x_2, y, N)

nu_1 = np.exp(-c_1/beta_1)
nu_2 = np.exp(-c_2/beta_2)

print('beta_1: {}, nu_1: {}'.format(beta_1, nu_1))
print('beta_2: {}, nu_2: {}'.format(beta_2, nu_2))


# In[122]:


def getR(beta, nu):
    def R(t):
        return np.exp(-((t/nu)**beta))
    return np.vectorize(R)

R_1 = getR(beta_1, nu_1)
R_2 = getR(beta_2, nu_2)

points = np.linspace(0, 2000000, num=10000000)
plt.plot(points, R_1(points))
plt.plot(points, R_2(points))
plt.grid()
plt.legend(['Выборка 1', 'B'])
plt.axis(style='sci')
plt.show()


# In[121]:


part = 0.1
t_1 = nu_1 * math.log(-np.log(part), beta) / 100
t_2 = nu_2 * math.log(-np.log(part), beta)/100
print(t_1)
print(t_2)

