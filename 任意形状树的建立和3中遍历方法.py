class bitnode():
	'''二叉树的节点'''
	def __init__(self):
		self.value=None
		self.lchild=None
		self.rchild=None
		self.level=None

def creat_tree_front(node,i):
	'''按前序遍历建立树,i记录层数'''
	v=input("input value and 空格为不赋值\n")
	if v!=' ':
		node.value=v
		node.level=i
		node.lchild=bitnode()
		creat_tree_front(node.lchild,i+1)
		node.rchild=bitnode()
		creat_tree_front(node.rchild,i+1)

def search_front(head):
	'''前序遍历'''
	if head.value !=None:
		print(head.value+'  '+'in the '+str(head.level)+' level')
		search_front(head.lchild)
		search_front(head.rchild)
	
def search_mid(head):
	'''中序遍历'''
	if head.value !=None:
		search_mid(head.lchild)
		print (head.value+'  '+'in the '+str(head.level)+' level')
		search_mid(head.rchild)
	
def search_back(head):
	'''后许遍历'''
	if head.value !=None:
		search_back(head.lchild)
		search_back(head.rchild)
		print(head.value+'  '+'in the '+str(head.level)+' level')	
		
def search_level(head):
	'''按层遍历'''
	if head==None:
		return
	q=[]
	q.append(head)
	while(1):
		if len(q)==0:
			break
		current=q.pop(0)
		print(current.value+'  '+'in the '+str(current.level)+' level')
		if current.lchild.value != None:#此处注意current.lchild可能建立了，但其值为None
			q.append(current.lchild)
		if current.rchild.value !=None:
			q.append(current.rchild)
		

	
	
head=bitnode()#建立类要加括号！！！
level_count=1
creat_tree_front(head,level_count)
print('front')
search_front(head)
print('mid:')
search_mid(head)
print('back')
search_back(head)
print('level')
search_level(head)
