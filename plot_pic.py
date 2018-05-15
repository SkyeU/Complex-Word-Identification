from matplotlib import pyplot as plt  
import re
print plt.style.available
plt.style.use(u'seaborn-poster')
x_eng = []
y_eng = []
file_name ='logspa'
with open(file_name+'.txt', 'r') as f:
	for line in f:
		line = re.findall(r"\d+\.?\d*",line)
		print(line)
		x_eng.append(int(line[0]))
		y_eng.append(float(line[7]))

		# re.findall(r"\d+\.?\d*",string)
plt.plot(x_eng, y_eng,'r')
plt.plot(x_eng, y_eng,'r')
plt.xlabel("Training epoch")
plt.ylabel("F1 score")
plt.title("Training Curve of Spanish")
plt.savefig(file_name+".pdf",dpi=300)
plt.show()