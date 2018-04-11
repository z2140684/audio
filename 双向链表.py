class node():#实用类来做节点
	def __init__(self,value,next=None,front=None):
		self.value=value
		self.next=next
		self.front=front
def creat_list(n):
	if n<=0:
		return (False)
	else:
		root=node(0)
		tmp=root
		for i in range(n-1):
			tmp.next=node(i+1)
			tmp.next.front=tmp
			tmp=tmp.next
		tmp.next=root
		root.front=tmp
	return root
def insert(root,n,t):
	p=root
	for i in range(1,n-1):
		p=p.next
	tem=node(t)
	tem.next=p.next
	p.next=tem#对类操作是全局的
	
def delet(root,n):
	p=root
	for i in range(1,n-1):
		p=p.next
	p.next=p.next.next
	
def search(root,value):
	p=root
	i=0
	while(1):
		if p.value==value:
			return i
		i+=1
		if p.next==root:
			break
		p=p.next
		
	
root=creat_list(5)
p=root
for i in range(5):
	print(p.value)
	p=p.next
	
insert(root,4,77)
p=root
for i in range(8):
	print(p.value)
	p=p.next
delet(root,4)
p=root
for i in range(8):
	print(p.value)
	p=p.next
print ('qweqwe'+str(search(root,3)))
