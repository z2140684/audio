import numpy as np
map=[[0,1,0,1,1],[1,0,1,1,0],[0,1,0,0,1],[1,1,0,0,0],[1,0,1,0,0]]
'''map 对应节点连接关系'''
def travel(start,map):
	for i in range(len(map[start])):#清理该节点所有连接线，表示已经遍历（清理列）
		map[start][i]=0
	print("we arrived "+str(start))
	for i in range(len(map)):#顺行便利
		if map[i][start]!=0:
			travel(i,map)
			
start=0
travel(start,map)
