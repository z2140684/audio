lis=[3,6,4,98,0,4,58,3,5,8,956,73,5,7,9,567,2,6,468,34,7,678,2,3,78,26,42357]
lenth=len(lis)
rear=lenth
head=0
while(head!=rear-1):
'''����С�ķ��ڿ�ͷ����������ȥ'''
	min=head
	for i in range(head,rear):
		if lis[i]<lis[min]:
			min=i#min��¼��С���±�
	current=lis[head]
	lis[head]=lis[min]
	lis[min]=current
	head+=1
print(lis)
