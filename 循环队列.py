a=list(None for i in range(10))
print (a)
h=r=0

def in_value(h,r,lis,value):
	if  r!=h :
		lis[r]=value
		r=(r+1)%len(lis)
		print (lis)
		return (r)
		
	elif lis[r]==None:
		lis[r]=value
		r=(r+1)%len(lis)
		print (lis)
		return (r)
	else:
		print("full")
		return (r)

def out_value(h,r,lis):
	
	if 	h!=r:
		lis[h]=None
		h=(h+1)%len(lis)
		print(lis)
		return h
	elif lis[h]!=None:
		lis[h]=None
		h=(h+1)%len(lis)
		print(lis)
		return h
		
	else:
		print("empty")
		return h
for i in range(4):
	r=in_value(h,r,a,i)

for i in range(2):
	h=out_value(h,r,a)
for i in range(13):
	r=in_value(h,r,a,i)
	
for i in range(12):
	h=out_value(h,r,a)
