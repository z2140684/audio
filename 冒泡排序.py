lis=[3,6,4,98,0,4,58,3,5,8,956,73,5,7,9,567,2]
lenth=len(lis)
i=0
while(lenth!=1):#将最大的放在开头，在把第二大的放在他后面
	current=0
	for i in range(lenth-1):#lenth-1使其不用和最大的多一次比较
		if lis[i]>lis[i+1]:
			current=lis[i]
			lis[i]=lis[i+1]
			lis[i+1]=current
	lenth-=1
print(lis)
	
