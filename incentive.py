# This is the code used to generate the data for Figure 4(b) in the paper.

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas
# from scipy.special import softmaxscipy

import matplotlib.pyplot as plt
import seaborn as sns

import datetime

def printGraph(X, Y, Y1, name):
    plt.figure('Draw')
    plt.plot(X, Y)
    plt.plot(X, Y1)

    legend = []

    legend.append('defense utility gain')
    legend.append('expected additional payoff')

    plt.legend(legend, loc='lower right')
    # plt.scatter(X, Y, color = 'r', marker='.')
    plt.xlabel('addtional reward')
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
        (p[i] * ((1 - pw) * (x[t, 0] * rd[t] + (1 - x[t, 0]) * pd[t]) + pw * (x[t, m[t, i]] * rd[t] + (1 - x[t, m[t, i]]) * pd[t])))
        for i in range(len(p))), GRB.MAXIMIZE)

    ilp_model.addConstrs(gp.quicksum(
        (p[i] * ((1 - pw) * (x[t, 0] * pa[t] + (1 - x[t, 0]) * ra[t]) + pw * (x[t, m[t, i]] * pa[t] + (1 - x[t, m[t, i]]) * ra[t])))
        for i in range(len(p)))
                         >= gp.quicksum(
        (p[i] * ((1 - pw) * (x[j, 0] * pa[j] + (1 - x[j, 0]) * ra[j]) + pw * (x[j, m[j, i]] * pa[j] + (1 - x[j, m[j, i]]) * ra[j])))
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
        return -np.inf, -np.inf, 0

    x1 = ilp_model.getAttr('X', x)
    attacker_val = sum((p[i] * ((1 - pw) * (x1[t, 0] * pa[t] + (1 - x1[t, 0]) * ra[t]) + pw * (
                x1[t, m[t, i]] * pa[t] + (1 - x1[t, m[t, i]]) * ra[t])))
                       for i in range(len(p)))
    print(attacker_val)

    return ilp_model.objVal, attacker_val, sum(p[i] * x1[t, m[t, i]] * pw for i in range(len(p)))



def softmax(x):
    return np.exp(x) / sum(np.exp(x))

import pickle

def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':



    type_number = 2
    target_numbers = [10, 30, 100, 200, 300, 400, 500, 600]
    target_number = 10
    alpha = 0.3
    eps = 1e-8

    res = []
    payoff_res = []

    target_us = []



    for num_instances in range(50):
        # print(p.shape)
        r = 1
        pw = 0.3
        maxv = -np.inf

        list_a = []
        list_result = []
        list_pay = []

        scale = 0.1

        pa = -np.random.rand(target_number)
        ra = np.random.rand(target_number)
        uc = np.random.rand(target_number, type_number)


        pd = -np.random.rand(target_number)
        rd = np.random.rand(target_number)

        uu = np.random.rand(target_number, type_number)

        uc *= scale
        uu *= scale

        p = np.random.rand(type_number)
        p = p / np.sum(p)
        r = 1

        target_u = -np.inf

        for i in range(0, target_number):
            cur_utility, _, _ = lp_calc(pw, pa, ra, pd, rd, uc, uu, p, r, i)
            target_u = max(target_u, cur_utility)
        target_us.append(target_u)


        for up in np.arange(0, 0.1, 0.001):
            uc = uc + up
            maxv = -np.inf
            payoff = 0
            for i in range(0, target_number):
                cur_utility, _ , xt= lp_calc(pw, pa, ra, pd, rd, uc, uu, p, r, i)
                if cur_utility > maxv:
                    maxv = cur_utility
                    payoff = xt * up
            list_result.append(maxv)
            list_pay.append(payoff)
        res.append(list_result)
        payoff_res.append(list_pay)
    print(res)
    print_x = np.arange(0, 0.1, 0.001)
    res = np.array(res)
    payoff_res = np.array(payoff_res)

    print(res)

    save(res, 'tmp/incentive_n.pickle')
    save(payoff_res, 'tmp/incentive_payoff.pickle')

    
    # The following part is to draw a draft with the data, not exactly painting the figure in the paper

    mean_res = np.mean(res, axis=0)
    print(mean_res.shape)
    mean_res = mean_res.reshape(-1)

    mean_payoff = np.mean(payoff_res, axis=0)
    mean_payoff = mean_payoff.reshape(-1)

    printGraph(print_x, mean_res, mean_payoff, 'incentive')




