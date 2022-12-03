import os
import operator

d = {100:100,1:2,2:3,3:4}
s = sorted(d.items(),key=operator.itemgetter(0),reverse=True)
print(s[0][0])