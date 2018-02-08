import numpy as np
a= np.arange(10,100,5)
b= np.ones([4,3,2],dtype=float)
s1=a.reshape(3,2,3)
print(s1)