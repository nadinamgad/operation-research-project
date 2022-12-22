import numpy as np
basic = [
    ['s1', '0'],
     ['x', '4']
]
cons = [
    [1,2,1,0],
    [6,2,0,1]
]

row = 1
col = 0

# new value = old - (corr key col * corr row value)


# a = basic[0][1] * cons[0][0]
# a += basic[1][1] * cons[1][0]
# a += basic[2][1] * cons[2][0]
#
# a = basic[0][1] * cons[0][1]
# a += basic[1][1] * cons[1][1]
# a += basic[2][1] * cons[2][1]
#
# a = basic[0][1] * cons[0][2]
# a += basic[1][1] * cons[1][2]
# a += basic[2][1] * cons[2][2]
zj=[]
a = 0
b = 0
print('et2y alah ', basic[a][1])
while(b < len(cons[0])):
    ans = 0
    print('ff ', b)
    while(a < len(basic)):
        ans += int(basic[a][1]) * cons[a][b]
        print('ans ', ans)
        a+= 1
    zj.insert(b, ans)
    b += 1
    a = 0
print('zz ', zj)

cj_zj = []

new_cons = np.copy(cons)
print(new_cons)
new_cons.tolist()
for i in new_cons:
    print(i[0])