
def is_Topologistic(Graph_list��node_num):
	collect=[]
	counter=0
	while(1):
		for graph in Graph_list:
			'''�ռ����Ϊ0�Ľڵ�'''
			if graph.in==0:
				collect.append(graph)
				
		if collect==[]:δ����
			break
			
		while(collect):
			g=collect.pop(0)
			counter+=1#��������pop�����ڵ㣬��counter>���ڽڵ�����ͼ�л��������˽ṹ
			while(g.next!=None):
				g=g.next
				g.in-=1
				if g.in==0:
					collect.append(g)
		
		if counter>node_num:
			return None
		else:
			return True
		
