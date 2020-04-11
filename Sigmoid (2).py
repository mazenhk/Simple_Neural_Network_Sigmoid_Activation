import numpy as np
from math import ceil,floor

def sigmoid(x):
	return 1/(1+np.exp(-x))
	
def sig_derv(x):
	return x* (1-x)
	
x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1],[1,0,1],[1,1,1],[0,0,1],[0,1,1]])
					
z = np.array([[0],[1],[1],[1],[1],[1],[0],[1]])

np.random.seed(1)

syn0=2*np.random.random((3,8)) - 1
#print(syn0)
syn1=2*np.random.random((8,3)) - 1
syn2=2*np.random.random((3,8)) - 1
syn3=2*np.random.random((8,1)) - 1

for j in range(100000):
	l0 = x
	l1 = sigmoid(np.dot(l0,syn0))
	l2 = sigmoid(np.dot(l1,syn1))
	
	l3 = sigmoid(np.dot(l2,syn2))
	l4 = sigmoid(np.dot(l3,syn3))
	
	l4_err = z - l4
	
	if(j % 1000 )== 0:
		print("Error ", np.mean(np.abs(l4_err)))

	adj1 = l4_err * sig_derv(l4)
	
	syn3 += np.dot(l3.T,adj1)

l5 = []
for i in l4:
    if(i >= 0.5):
        i = ceil(i)
    elif(i < 0.5):
        i = floor(i)
        
    l5.append(i)
    
print("Output after training")
print(l5)

x = np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,1,1]])

l0 = x
l1 = sigmoid(np.dot(l0,syn0))
l2 = sigmoid(np.dot(l1,syn1))
	
l3 = sigmoid(np.dot(l2,syn2))
l4 = sigmoid(np.dot(l3,syn3))
	
l4_err = z - l4
	
if(j % 1000 )== 0:
	print("Error ", np.mean(np.abs(l4_err)))

adj1 = l4_err * sig_derv(l4)
	
syn3 += np.dot(l3.T,adj1)

print(l4)
print(syn3)

l5 = []
for i in l4:
    if(i >= 0.5):
        i = ceil(i)
    elif(i < 0.5):
        i = floor(i)
        
    l5.append(i)
    
print("Output after training")
print(l5)
