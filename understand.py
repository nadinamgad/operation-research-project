import numpy as np
from tabulate import tabulate
c = [1, 1, 0, 0, 0]
A = [
    [-1, 1, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [0, 1, 0, 0, 1]
]
b = [2, 4, 4]


def to_tableau(c, A, b):
    xb = [eq + [x] for eq, x in zip(A, b)]
    z = c + [0]
    return xb + [z]


def can_be_improved(tableau):
    z = tableau[-1]
    return any(x > 0 for x in z[:-1])


import math


def get_pivot_position(tableau):
    z = tableau[-1]
    column = next(i for i, x in enumerate(z[:-1]) if x > 0)

    restrictions = []
    for eq in tableau[:-1]:
        el = eq[column]
        restrictions.append(math.inf if el <= 0 else eq[-1] / el)

    row = restrictions.index(min(restrictions))
    return row, column


def pivot_step(tableau, pivot_position):
    new_tableau = [[] for eq in tableau]

    i, j = pivot_position
    pivot_value = tableau[i][j]
    new_tableau[i] = np.array(tableau[i]) / pivot_value

    for eq_i, eq in enumerate(tableau):
        if eq_i != i:
            multiplier = np.array(new_tableau[i]) * tableau[eq_i][j]
            new_tableau[eq_i] = np.array(tableau[eq_i]) - multiplier
    #print('new tablaeu: ', new_tableau)
    return new_tableau


def is_basic(column):
    return sum(column) == 1 and len([c for c in column if c == 0]) == len(column) - 1


def get_solution(tableau):
    columns = np.array(tableau).T
    solutions = []
    for column in columns:
        solution = 0
        if is_basic(column):
            one_index = column.tolist().index(1)
            solution = columns[-1][one_index]
        solutions.append(solution)

    return solutions


def simplex(c, A, b):
    tableau = to_tableau(c, A, b)

    while can_be_improved(tableau):
        pivot_position = get_pivot_position(tableau)
        tableau = pivot_step(tableau, pivot_position)

    return get_solution(tableau)


solution = simplex(c, A, b)
print('solution: ', solution)

data = [["Mavs", 99],
        ["Suns", 91],
        ["Spurs", 94],
        ["Nets", 88]]

# define header names
col_names = ["Team", "Points"]

# display table
#print(tabulate(data, headers=col_names, tablefmt="fancy_grid"))
a1 = [1,2]
a2 = a1.copy()
a2.append(3)
print(a1, a2)

def identity(n):
    m=[[0 for x in range(n)] for y in range(n)]
    for i in range(0,n):
        m[i][i] = 1
    return m

idn = identity(2)
print(idn)
basic = [
    ['s1', '0']
     ['s2', '0']
]
cons = [
    [1,2,1,0]
    [6,-2,0,1]
]

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
qq = 0
ff = 0
print('et2y alah ', basic[qq][1])
while(b < len(cons)):
    ans = 0
    while(qq < len(basic)):
        ans += int(basic[qq][1]) * cons[qq][ff]

        qq+= 1
    zj.insert(ff, ans)
    ff += 1
print('zz ', zj)