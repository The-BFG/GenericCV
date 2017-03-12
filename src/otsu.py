def media(count, start, end):
	somma = 0
	for i in range(start, end):
		somma += count[i]*(i+1)
	return somma

def otsu(count):
	sigma = []
	for t in range(len(count)):
		w1 = sum(count[:t])
		w2 = sum(count[t:])
		#print(w2)
		if(w1 > 0):
			m1 = media(count, 0, t)/w1
		else:
			m1 = 0
		if(w2 > 0):
			m2 = media(count, t, len(count))/w2
		else:
			m2 = 0
		sigma.append(w1*w2*((m1-m2)**2))
	print(max(sigma), sigma)
	return max(sigma) 
	
otsu([0.4, 0.3, 0.2, 0.1])
