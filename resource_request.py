# This is the code used to generate the data for Figure 4(a) in the paper.

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas
# from scipy.special import softmaxscipy

import matplotlib.pyplot as plt
import seaborn as sns

import datetime

def printGraph(X, Y, name):
    plt.figure('Draw')
    plt.plot(X, Y)
    # plt.scatter(X, Y, color = 'r', marker='.')
    plt.xlabel('p_defender_align')
    plt.ylabel('defender resource')
    plt.draw()
    plt.savefig(name + ".pdf")
    plt.close()
    print("print figure finish: " + name)

def lp_calc(pw, pa, ra, pd, rd, uc, uu, p, r, t):
    ilp_model = gp.Model('utility_optimization')

    x = ilp_model.addVars(len(pa), len(pa) + 2, name='x')

    m = np.zeros((len(pa), len(p)))

    for i in range(len(p)):
        for j in range(len(pa)):
            if uc[j, i] < uu[j, i]:
                m[j, i] = len(pa) + 1
            else:
                m[j, i] = j + 1

    ilp_model.setObjective(gp.quicksum(
        (p[i] * ((1 - pw) * (x[t, 0] * rd[t] + (1 - x[t, 0]) * pd[t]) + pw * (
                x[t, m[t, i]] * rd[t] + (1 - x[t, m[t, i]]) * pd[t])))
        for i in range(len(p))), GRB.MAXIMIZE)

    ilp_model.addConstrs(gp.quicksum(
        (p[i] * ((1 - pw) * (x[t, 0] * pa[t] + (1 - x[t, 0]) * ra[t]) + pw * (
                x[t, m[t, i]] * pa[t] + (1 - x[t, m[t, i]]) * ra[t])))
        for i in range(len(p)))
                         >= gp.quicksum(
        (p[i] * ((1 - pw) * (x[j, 0] * pa[j] + (1 - x[j, 0]) * ra[j]) + pw * (
                x[j, m[j, i]] * pa[j] + (1 - x[j, m[j, i]]) * ra[j])))
        for i in range(len(p))) for j in range(len(pa)))

    ilp_model.addConstrs(x[t1, i] >= x[t1, len(pa) + 1] for i in range(len(pa) + 2) for t1 in range(len(pa)))
    ilp_model.addConstrs(x[t1, i] <= x[t1, t1 + 1] for i in range(len(pa) + 2) for t1 in range(len(pa)))

    ilp_model.addConstrs(x[t1, i] >= 0 for t1 in range(len(pa)) for i in range(len(pa) + 2))
    ilp_model.addConstrs(x[t1, i] <= 1 for t1 in range(len(pa)) for i in range(len(pa) + 2))
    ilp_model.addConstrs(gp.quicksum(x[t, i] for t in range(len(pa))) <= r for i in range(len(pa) + 2))

    ilp_model.optimize()

    try:
        print(ilp_model.objVal)
    except Exception as e:
        print(e)
        return -np.inf, -np.inf
    x1 = ilp_model.getAttr('X', x)
    attacker_val = sum((p[i] * ((1 - pw) * (x1[t, 0] * pa[t] + (1 - x1[t, 0]) * ra[t]) + pw * (
                x1[t, m[t, i]] * pa[t] + (1 - x1[t, m[t, i]]) * ra[t])))
                       for i in range(len(p)))
    print(attacker_val)

    return ilp_model.objVal, attacker_val



def softmax(x):
    return np.exp(x) / sum(np.exp(x))

import pickle

def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':



    type_number = 3
    target_numbers = [5, 10, 20]
    target_number = 10
    alpha = 0.3
    eps = 1e-8

    res = dict()
    for target_number in target_numbers:
        res[target_number] = []

    min_p0 = 0




    for num_instances in range(5):
        r = 1
        pw = 0.3
        maxv = -np.inf

        list_a = []

        for target_number in target_numbers:
            r = 1
            list_result = []
            pa = -np.random.rand(target_number)
            ra = np.random.rand(target_number)
            uc = np.random.rand(target_number, type_number)

            pd = -np.random.rand(target_number)
            rd = np.random.rand(target_number)

            uu = np.zeros_like(uc)
            uu[:, 0] = uc[:, 0] - 0.1
            uu[:, 1] = uc[:, 1] + 0.1
            uu[:, 2] = np.random.rand(target_number)

            p = np.array([1.0, 0, 0.0])

            target_u = -np.inf

            for i in range(0, target_number):
                cur_utility, _ = lp_calc(pw, pa, ra, pd, rd, uc, uu, p, r, i)
                target_u = max(target_u, cur_utility)


            for p0 in np.arange(1.0, min_p0, -0.01):
                p = np.array([p0, 1 - p0, 0])
                while True:
                    maxv = -np.inf
                    for i in range(target_number):
                        def_v, att_v = lp_calc(pw, pa, ra, pd, rd, uc, uu, p, r, i)
                        print('current instance:')
                        print(num_instances, target_number)
                        if def_v > maxv:
                            maxv = def_v
                            attv = att_v
                    if target_u - maxv > eps:
                        r = r + 0.1
                        if r > 20:
                            min_p0 = p0
                            break
                    else:
                        break
                list_result.append(r)
            res[target_number].append(list_result)
    save(res, 'tmp/resource_request_n.pickle')
    for target_number in target_numbers:
        print_x = np.arange(1.0, min_p0, 0.01)
        for i in range(len(res[target_number])):
            res[target_number][i] = res[target_number][i][:len(print_x)]
        cur_res = np.array(res[target_number])


        save(cur_res, 'tmp/resource_result_atk_n_{}.pickle'.format(str(target_number)))





