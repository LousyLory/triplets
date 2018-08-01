import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

f = open('op4.txt')
A = f.readlines()
f.close()

A = [l.strip('\n') for l in A]

loss = []
tr_a = []
val_a = []

Q = []
R = []
for i in range(len(A)):
	if 'epoch' in A[i]:
		Q.append(A[i])
	else:
		R.append(A[i])

for i in range(len(R)):
	#print R[i]
	_,sep,Vals = R[i].rpartition('loss ')
	vals,_,tr_val = Vals.partition(' ')
	tr,_,val = tr_val.rpartition(' ')
	#print vals
	loss.append(float(vals))
	#tr_a.append(float(tr))
	#val_a.append(float(val))

tr_a = []
val_a = []
for i in range(len(Q)):
	_,sep,t_a = Q[i].rpartition('training: ')
	t_a,_,_ = t_a.partition(' ')
	_,sep,v_a = Q[i].rpartition('validation: ')
	tr_a.append(float(t_a))
	val_a.append(float(v_a))


sns.set_style("darkgrid")
#sns.distplot(loss, color='g')
plt.plot(loss, color='g', label='loss')
#plt.plot(tr_a, color='r', label='training accuracy')
#plt.plot(val_a, color='b', label='validation accuracy')
plt.legend(loc='upper right')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('loss vs iteration')
plt.show()

sns.set_style("darkgrid")
plt.plot(tr_a, color='r', label='training accuracy')
plt.plot(val_a, color='b', label='validation accuracy')
plt.legend(loc='upper right')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.title('accuracy vs iteration')
plt.show()