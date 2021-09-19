import sys
import os
f=open(sys.argv[1],'r')
g=open(sys.argv[2],'w')
lines = f.readlines()
count = 0
for line in lines:
    src,label = line.strip().split(' ')
    g.write(str(count)+'\t'+str(label)+'\t'+src+'\n')
    count = count + 1

    
