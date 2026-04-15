# 4/14/2026
# i need to figure out exactly how numpy works so i can use it properly


import numpy as np

#array=np.array([2,3,4])

#print(array)
#print(np.roll(array,0))
#print(np.roll(array,1))
#print(np.roll(array,2))
#print(np.roll(array,3,axis=0))
#
#
#rng=np.random.default_rng()
#rand=rng.random((2,3))
#print(rand)
#
#
#list0=np.arange(1,61)
#print(list0)
#
#matrix0=list0.reshape(6,10)
#print(matrix0)
#
#print(np.roll(matrix0,3))
#print(np.roll(matrix0,3,axis=0))
#print(np.roll(matrix0,3,axis=1))
#
#list1=np.arange(1,21)
#
#print(np.repeat(list1,3))
#print(np.tile(list1,3))
#
#print(np.repeat(list1,3).reshape(20,3)) #cool
#print(np.tile(list1,3).reshape(3,20)) #cool
#
#
#list=np.arange(1,9)
#
#print(list)
#print(np.repeat(list,3))
#print(np.tile(list,3))
#print(np.repeat(list,8).reshape(8,8)) #cool
#print(np.tile(list,8).reshape(8,8)) #cool

#array=np.tile(np.arange(1,9),8).reshape(8,8)
#print(array)
#print(array[:,0:5])
#print(array[:,3:8])
#
#array[:,3:8]=array[:,0:5]
##cool! it works! :D
#print(array)

# reset array
array=np.tile(np.arange(1,9),8).reshape(8,8)
zeros=int(np.zeros((8,3)))
int_arr=zeros.floor()
print(int_arr)
array=np.concat([array[:,0:5],zeros],axis=1)
print(array)



#matrix[:,0:8]=matrix[:,2:10]
#print(matrix)