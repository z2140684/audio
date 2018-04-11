lis=[3,6,4,98,0,4,58,3,5,8,956,73,5,7,423579,567,2,6,468,34,7,678,2,3,78,26]
for i in range(1,len(lis)):
	if lis[i]<lis[i-1]:
		temp=lis[i]
		for j in range(i,0,-1):
			if lis[j-1]<temp:#当lis[j-1]<temp,将值插入到lis[j]中
				lis[j]=temp
				break
			lis[j]=lis[j-1]
			if j-1==0:#若到头还不break，说明该值目前最小应该放入lis[0]
				lis[0]=temp
				 
print(lis)
				
