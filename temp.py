import sys

f = open(sys.argv[1]).read().split('\n')
f = [i.strip() for i in f if len(i.strip()) > 0]

out = open(sys.argv[1], 'w')

for i in f:
    i = i.split(' ')
    f = float(i[2])
    j = "%s %.2f" % (i[0], f)
    out.write(j + '\n')

out.close()
