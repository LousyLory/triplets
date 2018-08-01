import cv2

f = open('./val_txt_files/word_files.txt')
A = f.readlines()
f.close()

f = open('./val_txt_files/y_labels.txt')
B = f.readlines()
f.close()

A = [l.strip('\n\r') for l in A]

count = 0
for i in A:
	Q = cv2.imread(i+'.png')
	if Q is None:
		print i 
		print B[count]
	count = count+1
