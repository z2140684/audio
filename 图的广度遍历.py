map=[[0,1,0,0,0],[1,0,1,1,0],[0,1,0,0,1],[0,1,0,0,1],[0,0,1,1,0]]

def travel(map,que):
	start=que.pop(0)
	print("we arrived "+str(start))
	for i in range(len(map[start])):
		map[start][i]=0
	for i in range(len(map)):
		if map[i][start]==1:
			que.append(i)
			for j in range(len(map[i])):
				map[i][j]=0
	if que:
		travel(map,que)
			
start=4
que=[]
que.append(start)
travel(map,que)
