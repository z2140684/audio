lis=[3,6,4,98,0,4,58,3,5,8,956,73,5,7,423579,567,2,6,468,34,7,678,2,3,78,26]

def swap(lis,low,high):
	temp=lis[low]
	lis[low]=lis[high]
	lis[high]=temp
	
def point(lis,low,high):
	'''找到指定点，并把小的放在指定点左边，大的放右边'''
	p=lis[low]#每次定第一个点为指定点
	while(low<high):
		while(p<=lis[high] and low<high):
			high-=1	
		swap(lis,low,high)
		while(p>=lis[low] and low<high):
			low+=1
		swap(lis,low,high)
	return low
		
def Qsort(lis,low,high):
	if (low<high):
		p=point(lis,low,high)
		Qsort(lis,low,p-1)
		Qsort(lis,p+1,high)

Qsort(lis,0,len(lis)-1)
print(lis)		
