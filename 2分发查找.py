lis=[i for i in range(1,101,2)]
h=len(lis)-1
l=0
value=35
print(lis)
def search(h,l,lis,value):
	mid=int((h+l)/2)
	if l==mid or h==mid:
		return("not find")
	elif value==lis[l]:
		return l
	elif value==lis[h]:
		return h
	elif value==lis[mid]:
		return mid
	elif value>lis[mid]:
		return search(h,mid+1,lis,value)
	else:
		return search(mid-1,l,lis,value)
index=search(h,l,lis,value)
print(index) 
		
