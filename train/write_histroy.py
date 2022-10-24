import os

arr = os.listdir()
for a in arr:
	tmp = a.split('-')
	if len(tmp) == 3 and tmp[0] == "gogui" and tmp[1] == "twogtp":
		target_dir = a

fp = open(target_dir + '/P1B_P2W.log', 'r')
lines = fp.readlines()
fp.close()

m = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'J':8}

f_h = open('history.txt', 'a')
f_w = open('win_or_loss.txt', 'a')
for line in lines:
	li = line.strip('\n').split()
	if len(li) > 2 and li[2] == "resign":
		f_h.write('\n')
		if li[0][0] == "B":
			f_w.write("0\n")
		else:
			f_w.write("1\n")

	if len(li) > 1 and li[1] == "play":
		pos = int(m[li[3][0]]) * 9 + int(li[3][1]) - 1
		if li[2] == "B":
			f_h.write(str(pos) + ' ')
		else:
			f_h.write(str(pos + 81) + ' ')


f_h.close()
f_w.close()



