lis=[[0,6,1,5,99,99],[6,0,5,99,3,99],[1,5,0,7,5,4],[5,99,7,0,99,2],[99,3,5,99,0,6],[99,99,4,2,6,0]]
path=[[0,0,0] for i in range(10)]

def minpath(lis,path):
	'''��ͼ�����б߰�Ȩֵ��С������path3ά�б��з���'''
	for m in range(10):#10����
		start=0
		end=0
		weigth=0
		min=99
		count=0
		for i in range(6):
			for j in range(count,6):
				if lis[i][j]!=0 and lis[i][j]<min :
					min=lis[i][j]
					start=i
					end=j
			count+=1
		lis[start][end]=99
		path[m]=[start,end,min]
	return path

def parent(f,if_loop):
	'''�ж��Ƿ��·��һ���б�����'''
	while(if_loop[f]!=0):
		f=if_loop[f]
	return f

def min_tree_bilid(path):
	'''ÿ��ѡȡ��С�ߣ��ж��Ƿ��·'''
	if_loop=[0  for i in range(6)]
	
	for p in path:
		n=parent(p[0],if_loop)
		m=parent(p[1],if_loop)
		'''m!=n���ʾ����һ�����ϣ�û��·'''
		if m!=n:
			if_loop[n]=m
			print("we chooise start: "+str(p[0])+" end: "+str(p[1]))
				  
				  
path=minpath(lis,path)	
print(path)
min_tree_bilid(path)
