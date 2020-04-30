# This is the code used to generate the data for Figure 5 in the paper.

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas
# from scipy.special import softmaxscipy

import matplotlib.pyplot as plt
import seaborn as sns

import datetime

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

    print('current target_number:')
    print(len(pa))

    try:
        print(ilp_model.objVal)
    except Exception as e:
        print(e)
        return -np.inf, -np.inf, None
    x1 = ilp_model.getAttr('X', x)
    attacker_val = sum((p[i] * ((1 - pw) * (x1[t, 0] * pa[t] + (1 - x1[t, 0]) * ra[t]) + pw * (
                x1[t, m[t, i]] * pa[t] + (1 - x1[t, m[t, i]]) * ra[t])))
                       for i in range(len(p)))
    print(attacker_val)

    return ilp_model.objVal, attacker_val, x1



def softmax(x):
    return np.exp(x) / sum(np.exp(x))

import pickle

def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':



    type_number = 4
    target_numbers = range(10, 80, 2)
    alpha = 0.3

    runtimes_list = []
    result_d = []
    result_a = []
    result_s = []
    result_s_a = []
    result_s_s = []


    for target_number in target_numbers:
        r = 5
        pw = 0.3


        result_d_i = []
        result_a_i = []
        result_s_i = []
        result_s_a_i = []
        result_s_s_i = []

        for instance_number in range(10):
            pa = -np.random.rand(target_number)
            ra = np.random.rand(target_number)
            uc = np.random.rand(target_number, type_number)

            pd = -np.random.rand(target_number)
            rd = np.random.rand(target_number)

            uu = np.zeros_like(uc)
            uu[:, 0] = uc[:, 0] - 0.1
            uu[:, 1] = uc[:, 1] + 0.1
            uu[:, 2] = np.random.rand(target_number)
            uu[:, 3] = np.random.rand(target_number)
            maxv = -np.inf

            # The following part is to use different way to generate uc so that we get different type of scenario
        

            p = np.array([1, 0, 0, 0])
            opt_x = None


            for i in range(target_number):
                def_v, att_v, cur_x = lp_calc(pw, pa, ra, pd, rd, uc, uu, p, r, i)
                if def_v > maxv:
                    maxv = def_v
                    attv = att_v
                    opt_x = cur_x

            result_d_i.append(maxv)

            maxv = -np.inf

            p = np.array([0, 1, 0, 0])

            for i in range(target_number):
                def_v, att_v,_ = lp_calc(pw, pa, ra, pd, rd, uc, uu, p, r, i)
                if def_v > maxv:
                    maxv = def_v
                    attv = att_v

            result_a_i.append(maxv)

            m = np.zeros((len(pa), len(p)))

            for i in range(len(p)):
                for j in range(len(pa)):
                    if uc[j, i] < uu[j, i]:
                        m[j, i] = len(pa) + 1
                    else:
                        m[j, i] = j + 1

            bst_t = 0
            bst_v = -np.inf

            for cur_t in range(target_number):
                tmp_v = sum(
                    (p[i] * ((1 - pw) * (opt_x[cur_t, 0] * pa[cur_t] + (1 - opt_x[cur_t, 0]) * ra[cur_t]) + pw * (
                            opt_x[cur_t, m[cur_t, i]] * pa[cur_t] + (1 - opt_x[cur_t, m[cur_t, i]]) * ra[cur_t])))
                    for i in range(len(p)))
                if tmp_v > bst_v:
                    bst_v = tmp_v
                    bst_t = cur_t

            stupid_v = sum(
                (p[i] * ((1 - pw) * (opt_x[bst_t, 0] * rd[bst_t] + (1 - opt_x[bst_t, 0]) * pd[bst_t]) + pw * (
                        opt_x[bst_t, m[bst_t, i]] * rd[bst_t] + (1 - opt_x[bst_t, m[bst_t, i]]) * pd[bst_t])))
                for i in range(len(p)))

            result_s_a_i.append(stupid_v)


            maxv = -np.inf

            p = np.array([0, 0, 0.5, 0.5])

            for i in range(target_number):
                def_v, att_v, _ = lp_calc(pw, pa, ra, pd, rd, uc, uu, p, r, i)
                if def_v > maxv:
                    maxv = def_v
                    attv = att_v

            result_s_i.append(maxv)

            m = np.zeros((len(pa), len(p)))

            for i in range(len(p)):
                for j in range(len(pa)):
                    if uc[j, i] < uu[j, i]:
                        m[j, i] = len(pa) + 1
                    else:
                        m[j, i] = j + 1

            bst_t = 0
            bst_v = -np.inf

            for cur_t in range(target_number):
                tmp_v = sum(
                    (p[i] * ((1 - pw) * (opt_x[cur_t, 0] * pa[cur_t] + (1 - opt_x[cur_t, 0]) * ra[cur_t]) + pw * (
                            opt_x[cur_t, m[cur_t, i]] * pa[cur_t] + (1 - opt_x[cur_t, m[cur_t, i]]) * ra[cur_t])))
                    for i in range(len(p)))
                if tmp_v > bst_v:
                    bst_v = tmp_v
                    bst_t = cur_t

            stupid_v = sum(
                (p[i] * ((1 - pw) * (opt_x[bst_t, 0] * rd[bst_t] + (1 - opt_x[bst_t, 0]) * pd[bst_t]) + pw * (
                        opt_x[bst_t, m[bst_t, i]] * rd[bst_t] + (1 - opt_x[bst_t, m[bst_t, i]]) * pd[bst_t])))
                for i in range(len(p)))

            result_s_s_i.append(stupid_v)
        result_s.append(np.mean(np.array(result_s_i)))
        result_d.append(np.mean(np.array(result_d_i)))
        result_a.append(np.mean(np.array(result_a_i)))
        result_s_a.append(np.mean(np.array(result_s_a_i)))
        result_s_s.append(np.mean(np.array(result_s_s_i)))

    result_s = np.array(result_s)
    result_a = np.array(result_a)
    result_d = np.array(result_d)
    result_s_s = np.array(result_s_s)
    result_s_a = np.array(result_s_a)

    # Save the result

    save(result_d, 'tmp/target_defender_n1.pickle')
    save(result_a, 'tmp/target_attacker_n1.pickle')
    save(result_s, 'tmp/target_strategic_n1.pickle')
    save(result_s_a, 'tmp/target_stupid_attacker_n1.pickle')
    save(result_s_s, 'tmp/target_stupid_strategic_n1.pickle')

    



