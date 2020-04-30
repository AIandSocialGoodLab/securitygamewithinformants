# This is the code used to generate the data for Figure 3 in the paper.

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
	# plt.plot(X, Y)
	plt.scatter(X, Y, color = 'r', marker='.')
	plt.draw()
	plt.savefig(name + ".svg")
	plt.close()
	print("print figure finish: " + name)

def lp_calc(pw, pa, ra, pd, rd, uc, uu, p, r, t):
    ilp_model = gp.Model('utility_optimization')

    x = ilp_model.addVars(len(pa), 2 * len(pa) + 1, name='x')
    # t = ilp_model.addVar(vtype='I', name='x')

    m = np.zeros((len(pa), len(p)))

    for i in range(len(p)):
        for j in range(len(pa)):
            if uc[j, i] < uu[j, i]:
                m[j, i] = 2 * j + 1
            else:
                m[j, i] = 2 * j + 2

    ilp_model.setObjective(gp.quicksum(
        (p[i] * ((1 - pw) * (x[t, 0] * rd[t] + (1 - x[t, 0]) * pd[t]) + pw * (x[t, m[t, i]] * rd[t] + (1 - x[t, m[t, i]]) * pd[t])))
        for i in range(len(p))), GRB.MAXIMIZE)

    ilp_model.addConstrs(gp.quicksum(
        (p[i] * ((1 - pw) * (x[t, 0] * pa[t] + (1 - x[t, 0]) * ra[t]) + pw * (x[t, m[t, i]] * pa[t] + (1 - x[t, m[t, i]]) * ra[t])))
        for i in range(len(p)))
                         >= gp.quicksum(
        (p[i] * ((1 - pw) * (x[j, 0] * pa[j] + (1 - x[j, 0]) * ra[j]) + pw * (x[j, m[j, i]] * pa[j] + (1 - x[j, m[j, i]]) * ra[j])))
        for i in range(len(p))) for j in range(len(pa)))

    ilp_model.addConstrs(x[t1, i] >= x[t1, 2 * t1 + 1] for i in range(2 * len(pa) + 1) for t1 in range(len(pa)))
    ilp_model.addConstrs(x[t1, i] <= x[t1, 2 * t1 + 2] for i in range(2 * len(pa) + 1) for t1 in range(len(pa)))

    ilp_model.addConstrs(x[t1, i] >= 0 for t1 in range(len(pa)) for i in range(2 * len(pa) + 1))
    ilp_model.addConstrs(x[t1, i] <= 1 for t1 in range(len(pa)) for i in range(2 * len(pa) + 1))
    ilp_model.addConstrs(gp.quicksum(x[t, i] for t in range(len(pa))) <= r for i in range(2 * len(pa) + 1))

    ilp_model.optimize()

    try:
        print(ilp_model.objVal)
    except Exception as e:
        print(e)
        return -np.inf

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
    target_number = 10
    alpha = 0.3

    # data need for the linear program

    r = 5
    pw = 0.3
    maxv = -np.inf

    list_a = []
    list_result = []

    pa = -np.random.rand(target_number)
    ra = np.random.rand(target_number)
    uc = np.random.rand(target_number, type_number)

    pd = -ra + alpha * np.random.normal(size=target_number)
    rd = -pa + alpha * np.random.normal(size=target_number)

    uu = np.zeros_like(uc)
    uu[:, 0] = uc[:, 0] - 0.1
    uu[:, 1] = uc[:, 1] + 0.1
    uu[:, 2] = np.random.rand(target_number)

    heat_results = np.empty((50, 50))
    heat_results[:, :] = np.nan

    attv_res = np.copy(heat_results)

    for beta1 in np.arange(0, 1, 0.02):
        for beta2 in np.arange(0, 1 - beta1, 0.02):
            p = np.array([beta1, beta2, 1 - beta1 - beta2])
            mean_maxv = 0
            mean_attv = 0

            for _ in range(50):
                maxv = -np.inf
                attv = 0
                tmp_random = np.random.rand()

                for i in range(target_number):
                    def_v, att_v = lp_calc(pw, pa, ra, pd, rd, uc, uu, p, r, i)
                    if def_v > maxv:
                        maxv = def_v
                        attv = att_v
                mean_maxv += maxv
                mean_attv += attv

            mean_maxv /= 10
            mean_attv /= 10

            heat_results[int(beta1 / 0.02), int(beta2 / 0.02)] = mean_maxv
            attv_res[int(beta1 / 0.02), int(beta2 / 0.02)] = mean_attv

    # Save the model for painting convenience

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save(heat_results, 'tmp/defender_v_{}.pickle'.format(current_time))
    save(attv_res, 'tmp/attacker_v_{}.pickle'.format(current_time))

    # The following part is to draw a draft with the data, not exactly painting the figure in the paper


    minv = np.nanmin(heat_results)
    maxv = np.nanmax(heat_results)

    print('minv, maxv:')

    print(minv)
    print(maxv)

    heat_results = (heat_results - minv) / (maxv - minv)

    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.imshow(heat_results)

    cbar = ax.figure.colorbar(im, ax=ax)

    plt.xticks(np.arange(0, 50, 10), np.arange(0, 1, 0.2))
    plt.yticks(np.arange(0, 50, 10), np.arange(0, 1, 0.2))

    plt.xlabel('p(attacker_align)')
    plt.ylabel('p(defender_align)')


    plt.savefig('fig/heat_def_utility.pdf')
    plt.close()

    minv = np.nanmin(attv_res)
    maxv = np.nanmax(attv_res)

    print('minv, maxv:')

    print(minv)
    print(maxv)

    heat_results = (attv_res - minv) / (maxv - minv)

    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.imshow(attv_res)

    cbar = ax.figure.colorbar(im, ax=ax)

    plt.xticks(np.arange(0, 50, 10), np.arange(0, 1, 0.2))
    plt.yticks(np.arange(0, 50, 10), np.arange(0, 1, 0.2))

    plt.xlabel('p(attacker_align)')
    plt.ylabel('p(defender_align)')


    plt.savefig('fig/heat_atk_utility.pdf')
    plt.close()




