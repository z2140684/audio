'''建立节点类'''
class node():
	def __init__(self,value=None,char=None):
		self.lchild=None
		self.rchild=None
		self.value=value
		self.char=char
		
'''按中序遍历树函数'''	
def search_mid(root):
	if root.lchild!=None:
		search_mid(root.lchild)
	if root.value!=None and root.char!=None:
		print(root.char)
	if root.rchild!=None:
		search_mid(root.rchild)



def build_tree(string):
	s=string
	
	'''建立字典，统计string中的字母出现次数'''
	dit={}
	for i in range(len(s)):
		if s[i] not in dit.keys():
			dit[s[i]]=1
		else:
			dit[s[i]]+=1

	'''创建一个list，放入出现字母，并从大到小排序'''
	pri=list(dit.keys())
	print(dit)
	print("before sort list:"+str(pri))
	lenth=len(pri)
	for i in range(lenth-1):
		current=pri[i]
		for j in range(i+1,lenth):
			if dit[pri[j]]>dit[pri[i]]:
				pri[i]=pri[j]
				pri[j]=current
				current=pri[i]
	print("after sort:"+str(pri))			
	

			
	'''建立新链表，每个字母建立一个节点，按字母频率从大到小放入链表'''
	pri_node=[]
	while(pri):
		k=pri.pop(0)
		pri_node.append(node(dit[k],k))

	
	'''按value大小，给类链表排序的函数'''
	def resort(que):
		for i in range(len(que)-1):
			current=que[i]
			for j in range(i+1,len(que)):
				if que[j].value>que[i].value:
					que[i]=que[j]
					que[j]=current
					current=que[i]
		return que

	'''建立赫夫曼树'''
	while(len(pri_node)>1):
		a=pri_node.pop()
		b=pri_node.pop()
		up=node(value=a.value+b.value)
		up.lchild=a
		up.rchild=b
		pri_node.append(up)
		
		pri_node=resort(pri_node)
		
	root=pri_node.pop()
	print("view the tree as mid sequence:")
	search_mid(root)
	return root 
	
'''获得字母编码'''
def get_leaves_code(root,code,L_code):
	if root.lchild!=None:
		code1=code+'0'
		get_leaves_code(root.lchild,code1,L_code)
	if  root.char!=None:
		L_code[root.char]=code
	if root.rchild!=None:
		code2=code+'1'
		get_leaves_code(root.rchild,code2,L_code)
		

'''编码string'''
def encode(string,L_code):
	codeline=''
	for i in range(len(string)):
		codeline=codeline+str(L_code[string[i]])
	return codeline
	
'''解码'''
def decode(codeline,L_code):
	check=''
	i=0
	message=''
	while(i!=len(codeline)):
		check=check+codeline[i]
		i+=1
		for key,value in L_code.items():
			if check==value:
				message=message+key
				check=''
				break
	return message
			
			
		
		
	
	
string="kssssaatttaaabb"
print("message input:"+string)
L_code={}
code=''
root=build_tree(string)
get_leaves_code(root,code,L_code)
print(L_code)
codeline=encode(string,L_code)
print("message is encoded as:"+codeline)

message=decode(codeline,L_code)

print("message is decoded as:"+message)
	


	
