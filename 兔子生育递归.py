def fib(i):#i 代表第几个月
	if i<2:
		return 1 if i==1 else 0
	return fib(i-1)+fib(i-2)

print(fib(8))
