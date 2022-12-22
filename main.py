import pandas as pd
import tkinter as tk
from tkinter import ttk
import re
import copy
# from tkinter import *
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
from tabulate import tabulate

# coordinates are calculated wrong and graph only plots one graph
# canvas = tk.Tk()
# canvas.title("problems solver")
# canvas.minsize(1000,800)
obj_fun = None
decision_var = None
cons = None
goal = None
data = []
temp = []
temp3 = []
variables = []
constraints = []
cons_matrix = []
string_cons_matrix = []
final_table_data = []
table_data = []
RHS = []
ratio = []
iden = []
basic = []
obj_fun_matrix = []
temp_obj_fun_matrix = []
cons_matrix_with_letters = []
numb_variables = None
count_letters = 0
index_of_missed_letter = []
decision_var1 = None
coordinates = []
constants = []
operators = []
x_list = []
y_list = []
zj = []
cj_zj = []
ct = 0
row = 0
found = True
all_slacks = True
surplus = False
artificial_variable = False
surplus_index = []
av_index = []
slack_index = []
columns = []
optimal = True
max_bool = False
min_bool = False
min_value = None
max_value = None
pivot_column_index = None
pivot_row_index = None


# new functions ------------------------------------------------------------------------------------------------------------------------------------------

def Iterate():
    global ratio
    global zj
    global pivot_column_index
    global pivot_row_index
    global cj_zj
    # print(max_bool, min_bool)
    getpivotcolumn()
    ratio.clear()
    calculateRatio()
    print('ratio: ', ratio)
    getpivotrow()
    # print("Pivot",pivot_row_index, pivot_column_index)
    editbasic()
    print('basic \n', basic)
    pivotColNewValues()
    # print(cons_matrix)
    print('cons matrix: \n', cons_matrix)
    consMatrixNewValues()
    zj.clear()
    cj_zj.clear()
    calculateZj()
    calculate_cj_zj()
    drawTableau()
    # print('RHS', RHS)
    # print('cj - zj = ', cj_zj)
    checkoptimality()
    # zj.clear()
    # cj_zj.clear()
    # ratio.clear()
    print('*********************************************************************')


def consMatrixNewValues():
    global new_cons
    global cons_matrix
    # new value = old value - ( corr key col * corr row value )
    a = 0
    j = 0
    new_cons = np.copy(cons_matrix)
    # print(new_cons)
    while (a < len(cons_matrix)):
        # print(cons_matrix[a])
        if a == pivot_row_index:
            a += 1
            if a == len(cons_matrix):
                break
        while (j < len(cons_matrix[a])):
            # print('j ', j)
            # print('a ', a)
            # print(cons_matrix[a][j])
            # print(cons_matrix[a][pivot_column_index])
            # print(cons_matrix[pivot_row_index][j])
            new_cons[a][j] = cons_matrix[a][j] - (cons_matrix[a][pivot_column_index] * cons_matrix[pivot_row_index][j])
            # print(RHS[a], cons_matrix[a][pivot_column_index], RHS[pivot_row_index])
            j += 1
        RHS[a] = float(RHS[a]) - (cons_matrix[a][pivot_column_index] * float(RHS[pivot_row_index]))
        # print(RHS)
        a += 1
        j = 0
        new_cons.tolist()
        cons_matrix = new_cons.copy()
        # extractDigits(cons_matrix)
    # print(cons_matrix)
    # print('RHS', RHS)


def pivotColNewValues():
    global cons_matrix
    global RHS
    # print(cons_matrix)
    # print(cons_matrix[pivot_row_index][pivot_column_index])
    divisor = cons_matrix[pivot_row_index][pivot_column_index]
    a = 0
    while (a < len(cons_matrix[pivot_row_index])):
        cons_matrix[pivot_row_index][a] /= divisor
        a += 1

    RHS[pivot_row_index] = float(RHS[pivot_row_index]) / divisor

    # print(cons_matrix)
    # print(RHS)


def editbasic():
    global basic
    print(columns)
    print(temp_obj_fun_matrix)
    basic[pivot_row_index][0] = columns[pivot_column_index]
    basic[pivot_row_index][1] = temp_obj_fun_matrix[pivot_column_index]


def getpivotrow():
    global pivot_row_index
    # global min_value

    # min non zero value
    min_value = float('inf')
    for i in range(len(ratio)):
        if ratio[i] >= 0:
            if ratio[i] < min_value:
                min_value = ratio[i]

    pivot_row_index = ratio.index((min_value))
    # print('row ', pivot_row_index)


def getpivotcolumn():
    global min_value
    global max_value
    global pivot_column_index
    if max_bool:
        # print('hah')
        max_value = max(cj_zj)
        print('max val: ', max_value)
        pivot_column_index = cj_zj.index(max_value)
        print('index: ', pivot_column_index)
    if min_bool:
        min_value = min(cj_zj)
        pivot_column_index = cj_zj.index(min_value)
        print('index: ', pivot_column_index)


def calculateRatio():
    global ratio
    global pivot_column_index
    tm = []
    # print('cons matrix : \n', cons_matrix)
    for i in cons_matrix:
        # print(i[pivot_column_index])
        tm.append(i[pivot_column_index])
    # print('tm ', tm)
    a = 0
    while (a < len(tm)):
        ans = 0
        print('rhs ', a, ' = ', RHS[a])
        ans = float(RHS[a]) / (float(tm[a]))
        ratio.insert(a, ans)
        a += 1
    # print('ratio ', ratio)


def checkoptimality():
    global optimal
    global max_bool
    global min_bool
    # print('goal: ', goal)
    a = 0
    ct = 0
    if goal == 'max':
        max_bool = True
        while (a < len(cj_zj)):
            if cj_zj[a] > 0:
                # print('7sl')
                optimal = False
            else:
                # print('m7slsh')
                ct += 1
            a += 1
        if ct == len(cj_zj):
            # print(ct, 'ebl3')
            optimal = True
            print('optimal at : \n', basic)
    else:
        min_bool = True
        while (a < len(cj_zj)):
            if cj_zj[a] < 0:
                optimal = False
            else:
                ct += 1
            a += 1
        if ct == len(cj_zj):
            optimal = True
    print('optimality status: ', optimal)


def calculate_cj_zj():
    global cj_zj
    # print('hi ', temp_obj_fun_matrix)
    a = 0
    while (a < len(temp_obj_fun_matrix)):
        ans = 0
        ans = temp_obj_fun_matrix[a] - zj[a]
        cj_zj.insert(a, ans)
        a += 1
    # print('cj - zj = ', cj_zj)


def calculateZj():
    global zj
    a = 0
    b = 0
    # print('et2y alah ', basic[a][1])
    while (b < len(cons_matrix[0])):
        ans = 0
        # print('ff ', b)
        while (a < len(basic)):
            ans += int(basic[a][1]) * cons_matrix[a][b]
            # print('ans ', ans)
            a += 1
        zj.insert(b, ans)
        b += 1
        a = 0
    a = 0
    ans = 0
    while (a < len(basic)):
        ans += float(basic[a][1]) * float(RHS[a])
        a += 1
    zj.append(ans)
    print('zj:  ', zj)


def identity(n):
    m = [[0 for x in range(n)] for y in range(n)]
    for i in range(0, n):
        m[i][i] = 1
    return m


def extractDigits(lst):
    return [[el] for el in lst]


def addslacks():
    temporary = []
    temporary.extend(slack_index)
    temporary.extend(surplus_index)
    temporary.extend(av_index)
    print(temporary)


def createfirst2rows():
    global columns
    global temp_obj_fun_matrix
    if all_slacks:
        i = 0
        while (i < len(obj_fun_matrix)):
            string = 's' + str(i + 1)
            columns.append(string)
            i += 1
        # print('columns: ',columns)

    temp_obj_fun_matrix.extend(obj_fun_matrix)
    J = 0
    a = len(obj_fun_matrix)
    while (J < a):
        temp_obj_fun_matrix.append(0)
        J += 1
    # print('hi ', obj_fun_matrix)
    # print('heeey', temp_obj_fun_matrix)


def drawTableau():
    global pivot_column_index
    global pivot_row_index
    final_table_data2 = []

    # createfirst2rows()
    # string_obj_fun_matrix = [str(i) for i in temp_obj_fun_matrix]
    # **************************************************************** el hagat el sabta **************************************************************************************
    table_data = string_obj_fun_matrix.copy()
    table_data.insert(0, 'Basic')
    table_data.insert(1, 'CB')
    table_data.insert(len(table_data), 'R.H.S')
    table_data.append('Ratio')
    # columns.append(' ')
    # columns.append(' ')
    final_table_data2.append(table_data)
    print('kk', columns)
    print('gg', final_table_data2)
    # ************************************************************ el hagat elly bbtghyr **************************************************************************************
    a = 0
    print("string_cons_matrix before", string_cons_matrix)
    while (a < len(cons_matrix)):
        string_cons_matrix[a] = [str(i) for i in cons_matrix[a]]
        a += 1
    print('st cons ', string_cons_matrix)
    a = 0
    be = []
    while (a < len(cons_matrix)):
        if a < len(basic):
            be = basic[a]
            string_cons_matrix[a].insert(0, be[0])
            string_cons_matrix[a].insert(1, be[1])
        string_cons_matrix[a].append(RHS[a])
        a += 1
    # for i in range(len(ratio)):
    #     final_table_data2[i+1][-1].append(ratio[i])

    final_table_data2.extend(string_cons_matrix)

    zj_final = zj.copy()
    zj_final.insert(0, '')
    zj_final.insert(1, 'zj')
    cj_zj_final = cj_zj.copy()
    cj_zj_final.insert(0, '')
    cj_zj_final.insert(1, 'cj - zj')
    final_table_data2.insert(len(final_table_data2), zj_final)
    final_table_data2.insert(len(final_table_data2), cj_zj_final)

    # for i in range(len(ratio)):
    #     final_table_data2[i+1].append(ratio[i])

    print("final_table: ", final_table_data2)
    print(tabulate(final_table_data2, headers=columns, tablefmt="fancy_grid"))
    final_table_data2.clear()


def InitializeTableau():
    global RHS
    global basic
    global string_obj_fun_matrix
    createfirst2rows()
    string_obj_fun_matrix = [str(i) for i in temp_obj_fun_matrix]
    table_data = string_obj_fun_matrix.copy()
    table_data.insert(0, 'Basic')
    table_data.insert(1, 'CB')
    table_data.insert(len(table_data), 'R.H.S')
    table_data.append('Ratio')
    # print('dataaa', table_data)
    columns.append(' ')
    columns.append(' ')
    # string_obj_fun_matrix.append('R.H.S')
    # string_obj_fun_matrix.append('Ratio')
    final_table_data.append(table_data)
    a = 0
    while (a < len(cons_matrix)):
        string_cons_matrix.insert(a, [str(i) for i in cons_matrix[a]])
        a += 1
    print('gee', string_cons_matrix)
    a = 0
    while (a < len(string_cons_matrix)):
        RHS.insert(a, string_cons_matrix[a][-1])
        string_cons_matrix[a].remove(string_cons_matrix[a][-1])
        cons_matrix[a].remove(cons_matrix[a][-1])
        a += 1

    # print(len(string_obj_fun_matrix), string_obj_fun_matrix)
    # print(len(string_cons_matrix[0]), string_cons_matrix[0])
    diff = len(string_obj_fun_matrix) - len(string_cons_matrix[0])
    # print(len(cons_matrix))
    iden = identity(len(cons_matrix))
    # print('identity', iden)
    ct = 0
    for i in columns:
        if i != ' ':
            ct += 1
    mid = ct / 2
    # print(mid)
    a = int(mid)
    tm = []
    while (a < mid * 2):
        tm.append(columns[a])
        a += 1
    # print(tm)
    basic = extractDigits(tm)
    for i in basic:
        i.append('0')
    a = 0
    be = []
    while (a < len(cons_matrix)):
        string_cons_matrix[a].extend(iden[a])
        cons_matrix[a].extend(iden[a])
        if a < len(basic):
            be = basic[a]
            string_cons_matrix[a].insert(0, be[0])
            string_cons_matrix[a].insert(1, be[1])
        string_cons_matrix[a].append(RHS[a])
        a += 1
    # print('Basic matrix', basic)
    # print('string cons matrix after merge: ', string_cons_matrix)
    # print('cons matrix: ', cons_matrix)
    # print('columns: ', columns)
    # print('RHS: ', RHS)
    calculateZj()
    calculate_cj_zj()
    checkoptimality()
    final_table_data.extend(string_cons_matrix)
    zj_final = zj.copy()
    zj_final.insert(0, '')
    zj_final.insert(1, 'zj')
    cj_zj_final = cj_zj.copy()
    cj_zj_final.insert(0, '')
    cj_zj_final.insert(1, 'cj - zj')
    final_table_data.insert(len(final_table_data), zj_final)
    final_table_data.insert(len(final_table_data), cj_zj_final)

    # print(RHS)
    print('ftt', final_table_data)
    print(tabulate(final_table_data, headers=columns, tablefmt="fancy_grid"))
    # zj.clear()
    # cj_zj.clear()
    print('*********************************************************************')


def MakeObjFunMatrix(data):
    global obj_fun_matrix
    obj_fun_matrix.extend(data[0])
    # remove null in last 2 cells
    i = 0
    while (i < 2):
        obj_fun_matrix.remove(obj_fun_matrix[len(obj_fun_matrix) - 1])
        i += 1


def checkoperators():
    global all_slacks
    global artificial_variable
    global surplus_index
    global av_index
    global slack_index
    greater = '>'
    greater_equal = '>='
    equal = '='
    smaller_equal = '<='
    if greater in operators:
        surplus = True
        all_slacks = False
        print("subtract s: ", surplus)
        # surplus_index.append(operators.index(greater))
    if greater_equal in operators:
        surplus = True
        all_slacks = False
        print("subtract s: ", surplus)
        # surplus_index.append(operators.index(greater_equal))
        getIndices(greater_equal, surplus_index)
    if equal in operators:
        artificial_variable = True
        print("av: ", artificial_variable)
        # av_index.append(operators.index(equal))
        getIndices(equal, av_index)
    if smaller_equal in operators:
        # all_slacks = True
        # print("slacks: ", all_slacks)
        # slack_index.append(operators.index(smaller_equal))
        getIndices(smaller_equal, slack_index)


def getIndices(operator, operator_list):
    i = 0
    while (i < len(operators)):
        if operators[i] == operator:
            operator_list.append(i)
        i += 1


def removenulls(data):
    # global data
    i = 0
    while (i < len(data)):
        data[i].remove(data[i][len(data[i]) - 1])
        i += 1
    # print(data)


def removeoperators(data):
    i = 0
    while (i < len(data)):
        data[i].remove(data[i][len(data[i]) - 1])
        i += 1


def MakeConstraintsMatrix(data):
    global cons_matrix
    i = 1
    while (i < len(data)):
        # print('hi')
        # store.clear()
        cons_matrix.append(data[i])
        i += 1

    # print('constraints: ', cons_matrix)


def Coordinates():
    global x_list
    global y_list
    global constants
    for i in range(len(cons_matrix)):
        for j in range(len(cons_matrix[i]) - 1):
            # print('check values: ', cons_matrix[i][j + 1], '|' , cons_matrix[i][j - 1])
            if 0 in cons_matrix[i]:
                index = cons_matrix[i].index(0)
                if index == 0:
                    constants.append((i, 1))
                else:
                    constants.append((i, 0))
                break
            if j % 2 == 0:
                x = 0
                y = float(cons_matrix[i][len(cons_matrix[i]) - 1]) / float(cons_matrix[i][j + 1])
            else:
                x = float(cons_matrix[i][len(cons_matrix[i]) - 1]) / float(cons_matrix[i][j - 1])
                y = 0
            coordinates.append((x, y))
    # print('coordinates: ', coordinates)
    # print('indexes: ', constants)

    for i in coordinates:
        # print(i[0])
        x_list.append(i[0])
        y_list.append(i[1])
    # print('x: ', x_list)
    # print('y: ', y_list)


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def calculateOptimalSolGraphical():
    global obj_fun_matrix
    global goal
    lhs = []
    rhs_ = []
    obj_fun_matrix2 = []
    if goal == 'max':
        print('convert !')
        obj_fun_matrix2 = copy.deepcopy(obj_fun_matrix)
        i =0
        while(i < len(obj_fun_matrix2)):
            obj_fun_matrix2[i] *= -1
            i += 1
    lhs = copy.deepcopy(cons_matrix)
    i=0
    while(i<len(lhs)):
        lhs[i].remove(lhs[i][-1])
        i += 1

    a = 0
    while(a<len(cons_matrix)):
        rhs_.insert(a, int(cons_matrix[a][-1]))
        a += 1

    a=0
    bnd = []
    while(a < len(obj_fun_matrix)):
        bnd.append((0, float("inf")))
        a += 1
    x1_bounds = (0, None)
    x2_bounds = (0, None)
    opt = linprog(c=obj_fun_matrix2, A_ub=lhs, b_ub=rhs_, method='highs-ds')
    if opt.status == 0: print(f'The solution is optimal.')
    print(f'Objective value: z = {opt.fun * -1}')
    print(f'Solution: x1 = {opt.x[0]}, x2 = {opt.x[1]}')
    print(obj_fun_matrix)
    print(obj_fun_matrix2)
    print(cons_matrix)
    print(lhs)
    print(rhs_)
    #print(opt)
def PlotGraph():
    x_plot = []
    y_plot = []
    i = 0
    a = 0
    cmap = get_cmap(len(x_list))
    while (i < len(x_list)):
        x_plot.append(x_list[i])
        x_plot.append(x_list[i + 1])
        y_plot.append(y_list[i])
        y_plot.append(y_list[i + 1])
        plt.plot(x_plot, y_plot, color=cmap(i), linewidth=3,
                 marker='o', markerfacecolor='black', markersize=12)
        plt.fill_between(x_plot, 0, y_plot, color='pink', alpha=.2)
        x_plot.clear()
        y_plot.clear()
        i += 2
        a += 1
        # check intersection
        # if a == 2:
        #     a = 0
        #     [xi, yi] = polyxpoly(x, y, xbox, ybox);
        #     mapshow(xi, yi, 'DisplayType', 'point', 'Marker', 'o')

    plt.scatter(x_plot, y_plot, color='pink')

    for i in range(len(constants)):
        val = cons_matrix[constants[i][0]][len(cons_matrix[constants[i][0]]) - 1] / cons_matrix[constants[i][0]][
            constants[i][1]]
        if constants[i][1] == 1:
            # print('val: ', val)
            plt.axhline(y=val, color='y', linestyle='-')
        else:
            plt.axvline(x=val, color='b', label='axvline - full height')
    # setting x and y axis range
    # plt.ylim(1, 100)
    # plt.xlim(1, 100)

    # naming the x axis
    plt.xlabel('x - axis')
    # naming the y axis
    plt.ylabel('y - axis')

    # giving a title to my graph
    plt.title('graph visualization')
    # plt.legend()
    # function to show the plot
    plt.show()


def collectoperators(data):
    global operators
    i = 1
    while (i < len(data)):
        operators.append(data[i][len(data[i]) - 2])
        i += 1


def finalgraphicalmethod():
    Coordinates()
    calculateOptimalSolGraphical()
    PlotGraph()


def simplexMethod():
    checkoperators()
    InitializeTableau()
    while (optimal == False):
        print('yarabbb')
        Iterate()
    zj.clear()
    # Iterate()


# old functions ------------------------------------------------------------------------------------------------------------------------------------------

def mainf():
    transform_con_text_to_matrix()
    countDecisionVariables()
    # print(cons_matrix)
    CheckMissingVariables()
    print(cons_matrix)
    # print('index_of_missed_letter: ', index_of_missed_letter)
    Coordinates()
    PlotGraph()


def countDecisionVariables():
    global numb_variables
    global decision_var1
    if obj_fun != None:
        obj_fun1 = obj_fun.get()
        decision_var1 = decision_var.get()
        decision_var1 = re.sub(r"\s+", "", decision_var1)
        # print('decision_var1',decision_var1)
        numb_variables = len(decision_var1)
        # print(numb_variables)


def CheckMissingVariables():
    # print('hi',len(cons_matrix[0]) - 1)
    index = []  # index in 2d array
    global numb_variables
    global cons_matrix_with_letters
    global index_of_missed_letter
    global cons_matrix
    for i in range(len(cons_matrix)):
        if len(cons_matrix[i]) - 1 < numb_variables:
            print('missin variable')
            index.append(i)
            print(index)
        else:
            print('all well')
    # print('index: ', index)
    # Add zeros in index of missed variables in cons matrix
    for i in index:
        # print('i: ', i)
        # for i in range(len(cons_matrix_with_letters)):
        # for j in cons_matrix_with_letters[i]:
        # print('j: ', j)
        for r in range(len(decision_var1)):
            # print('r: ', r)
            # print('text; ' ,cons_matrix_with_letters[i][0])
            if decision_var1[r] not in cons_matrix_with_letters[i][0]:
                index_of_missed_letter.append((i, r))  # index in decision variables string
                print(index_of_missed_letter, ', ', i)
    print('index of missed variable: ', index_of_missed_letter)

    for i in index_of_missed_letter:
        cons_matrix[i[0]].insert(i[1], '0')
    # print(cons_matrix)


def transform_con_text_to_matrix():
    temp2 = []
    temp3 = []
    global cons_matrix_with_letters
    global cons_matrix
    for i in constraints:
        temp = '' + str(i.get())
        temp2.append(temp)
        cons_matrix_with_letters.append(temp2)
        for i in temp2:
            temp3 = (re.findall(r'[\d\.\-\+]+', i))
        quantifier = '+'

        if quantifier in temp3:
            temp3.remove(quantifier)

        cons_matrix.append(temp3)
    # print(cons_matrix)
    # print(cons_matrix_with_letters)


def drawgraph():
    # y = None
    a = 0
    for i in range(len(coordinates)):
        for j in range(len(coordinates[i])):
            print('check: ', coordinates[i][j])
            print('j: ', j)
            store = coordinates[i][j]
            if j % 2 == 0:
                x = [store[a], store[a + 1]]
            else:
                # print('y: ', store[a], store[a + 1])
                y = [store[a], store[a + 1]]
                print('x: ', x, 'y: ', y)
                plt.plot(x, y)

    plt.show()


def coordinates_func():
    global coordinates
    temp4 = []
    for i in range(len(cons_matrix)):
        print('last cell: ', cons_matrix[i][len(cons_matrix)])
        print('first cell: ', cons_matrix[i][0])
        point = cons_matrix[i][len(cons_matrix)] / cons_matrix[i][0]
        if i % 2 == 0:
            temp4.append((0, point))
            coordinates.append(temp4)
        else:
            temp4.append((point, 0))
            coordinates.append(temp4)
    print('coordinates: ', coordinates)


def countLetter(text):
    global count_letters
    for i in text:
        if (i.isalpha()):
            count_letters = count_letters + 1


def check_missing_variable(text):
    global index_of_missed_letter
    global found
    for i in decision_var1:
        print('i: ', i)
        print('text: ', text)
        if i in text:
            print('found')
        else:
            found = False
            missed = i
            index_of_missed_letter.append(decision_var1.find(missed))
            # print('index_of_missed_letter: ',index_of_missed_letter)


def graphicalMethod():
    # print(ct)
    global found
    found = True
    for i in constraints:
        temp = '' + str(i.get())
        temp2 = [int(j) for j in temp.split() if j.isdigit()]
        countLetter(temp)
        if count_letters != numb_variables:
            print('count_letters: ', count_letters)
            print('numb_variables: ', numb_variables)
            print('there is variable of value 0')
            if numb_variables != 0:
                print('a7eh')
                # print(decision_var1)
                check_missing_variable(temp)
            else:
                print('please enter decision variables first')
                messagebox.showinfo("Information", "please enter decision variables first")
            if found == False:
                temp2.insert(index_of_missed_letter, 0)

        print('count: ', count_letters)
        cons_matrix.append(temp2)

    print('cons matrix', cons_matrix)
    print('cons matrix size:', len(cons_matrix))
    coordinates_func()
    drawgraph()


def formulateprob():  # put cooeficients in array and detect variables in equation
    global numb_variables
    global decision_var1
    if obj_fun != None:
        obj_fun1 = obj_fun.get()
        decision_var1 = decision_var.get()
        decision_var1 = re.sub(r"\s+", "", decision_var1)
        # print('decision_var1',decision_var1)
        numb_variables = len(decision_var1)
        if numb_variables <= 2:
            print('graphical method')
            graphicalMethod()
        else:
            print('simplex')
        # print(type(obj_fun1))
        temp = [int(i) for i in obj_fun1.split() if i.isdigit()]
        variables = " ".join(re.split("[^a-zA-Z]*", obj_fun1))
        variables = re.sub(r"\s+", "", variables)
        # variables = [int(i) for i in res1.split()]
        # variables.append(res1)
        # print('numerical values: ',temp)
        # print('vaariables: ', variables)


def Addconstraint():
    global ct
    global row
    global cons
    global constraints
    ct = ct + 1
    label = ttk.Label(canvas, text="please enter constraint", font=('Times', 10))
    label.grid(column=0, row=row)
    con = tk.StringVar()
    cons = ttk.Entry(canvas, width=20, textvariable=con)
    cons.grid(column=1, row=row)
    row = row + 1
    constraints.append(cons)


# graphical method ------------------------------------------------------------------------------------------------------------------------------------------

dataframe1 = pd.read_excel('readfile.xlsx')
# print(dataframe1.loc[:,"sign"])
operators = dataframe1["operator"].tolist()
goal = dataframe1.columns[3]
# print('goal: ', goal)
# operator in obj function is nan
operators.remove(operators[0])
# print(operators)
data = dataframe1.values.tolist()
# Readfromfile()
removenulls(data)
# print('data: ', data)
MakeConstraintsMatrix(data)
removenulls(cons_matrix)
MakeObjFunMatrix(data)
columns = dataframe1.columns.tolist()
i = 0
while (i < 3):
    columns.remove(columns[len(columns) - 1])
    i += 1
# print('data: ', data)
# print('constraints: ', cons_matrix)
# print('obj func: ', obj_fun_matrix)
# print('columns: ', columns)
if len(obj_fun_matrix) <= 2:
    finalgraphicalmethod()
    simplexMethod()
else:
    simplexMethod()

# simplex method ------------------------------------------------------------------------------------------------------------------------------------------

# dataframe2 = pd.read_excel('simplex.xlsx')
# #print(dataframe2.columns)
# columns = dataframe2.columns.tolist()
# i=0
# while(i<3):
#     del columns[len(columns) - 1]
#     i += 1
#
# #print('colss: ', columns)
# data2 = dataframe2.values.tolist()
# collectoperators(data2)
# # print('operators: ', operators)
# removenulls(data2)
# # print('simplex matrix: ', data2)
# MakeConstraintsMatrix(data2)
# removeoperators(cons_matrix)
# print(cons_matrix)
# checkoperators()
# # print('surplus: ', surplus_index)
# # print('slacks: ', slack_index)
# # print('av: ', av_index)
# addslacks()
# print(tabulate(data, headers=col_names, tablefmt="grid", showindex="always"))
# print(data)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------

# label = ttk.Label(canvas, text="please enter decision variables and pay attention to the order", font=('Times', 24))
# label.grid(column=1, row= row)
# row = row + 1
# dv = tk.StringVar()
# decision_var = ttk.Entry(canvas, width=20, textvariable= dv)
# decision_var.grid(column= 1, row= row, padx=5, pady=20)
# row = row + 1
# label = ttk.Label(canvas, text="enter objective function", font=('Times', 24))
# label.grid(column= 1, row= ro Fw, padx=5, pady=20)
# row = row + 1
# label = ttk.Label(canvas, text="please leave space between numbers and variables. Also, if a variable have value '1' please write it.", font=('Times', 15))
# label.grid(column= 1, row= row, padx=5, pady=20)
# row = row + 1
#
# obj = tk.StringVar()
# obj_fun = ttk.Entry(canvas, width= 20, textvariable= obj)
# obj_fun.grid(column= 1, row= row)
# row = row + 1
#
#
# button = ttk.Button(canvas, text= "Add Constraint", command= Addconstraint)
# button.grid(column=1, row = row, padx=5, pady=20)
# row = row + 1
#
# button = ttk.Button(canvas, text= "Submit" ,command= mainf)
# button.grid(column= 1, row= row, padx=5, pady=20)
# row = row + 1
#
# canvas.mainloop()
# canvas.pack()