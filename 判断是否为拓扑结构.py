
def is_Topologistic(Graph_list，node_num):
	collect=[]
	counter=0
	while(1):
		for graph in Graph_list:
			'''收集入度为0的节点'''
			if graph.in==0:
				collect.append(graph)
				
		if collect==[]:未命名
			break
			
		while(collect):
			g=collect.pop(0)
			counter+=1#用来计数pop几个节点，若counter>大于节点数则图有环，非拓扑结构
			while(g.next!=None):
				g=g.next
				g.in-=1
				if g.in==0:
					collect.append(g)
		
		if counter>node_num:
			return None
		else:
			return True
		
