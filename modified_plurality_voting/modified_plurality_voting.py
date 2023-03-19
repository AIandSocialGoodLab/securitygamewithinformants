import numpy as np
import os
import shutil
from pathlib import Path
import config
import gurobipy as gp
from gurobipy import GRB

from game import Game

def recursive(arr, count, s_hat):
    if count == 0:
        return arr
    test_list = [[i] for i in range(s_hat+1)]
    arr = [a + b for a in arr for b in test_list]
    return recursive(arr, count - 1, s_hat)

def feasible_message(message):
    first, second, third = 0, 0, 0
    for m in message:
        if m >= first:
            third = second
            second = first
            first = m
        elif m >= second:
            third = second
            second = m
        else:
            third = m
    return second <= 1 and third == 0

def generate_message_vars(s_hat, t):
    arr = recursive([[]], t, s_hat)
    res = []
    for a in arr:
        if feasible_message(a):
            res.append(a)
    target_list = [[i] for i in range(t)]
    final_messages = [a + b for a in res for b in target_list]
    final_message_sets = [tuple(f) for f in final_messages]
    final_message_sets.sort()
    return final_message_sets

def calc_defender_utility(target_idx, game, x, s_hat):
    res = 0.0
    for s in range(s_hat + 1):
        message_var = [0]*game.n_targets + [target_idx]
        message_var[target_idx] = s 
        message_var = tuple(message_var)
        res += game.p_plus[(s, game.n_informants, target_idx)] * (
            game.P_d[target_idx] + \
            (game.R_d[target_idx] - game.P_d[target_idx]) * \
            x[message_var]
        )
    for s in range(s_hat + 1, game.n_informants + 1):
        res += game.p_plus[(s, game.n_informants, target_idx)] * (
            game.P_d[target_idx] + \
            (game.R_d[target_idx] - game.P_d[target_idx]) * \
            game.s_infty[target_idx]
        )
    return res


def calc_attacker_utility(target_idx, game, x, s_hat):
    res = 0.0
    for s in range(s_hat + 1):
        message_var = [0]*game.n_targets + [target_idx]
        message_var[target_idx] = s 
        message_var = tuple(message_var)
        res += game.p_plus[(s, game.n_informants, target_idx)] * (
            game.R_a[target_idx] + \
            (game.P_a[target_idx] - game.R_a[target_idx]) * \
            x[message_var]
        )
    for s in range(s_hat + 1, game.n_informants + 1):
        res += game.p_plus[(s, game.n_informants, target_idx)] * (
            game.R_a[target_idx] + \
            (game.P_a[target_idx] - game.R_a[target_idx]) * \
            game.s_infty[target_idx]
        )
    return res

def calc_interim_converage_prob(target_idx, reported_message, game, x, s_hat):
    res = 0.0
    if reported_message.endswith('-') or reported_message.endswith('e'):
        for s in range(min(s_hat, game.n_informants-1)+1):
            message_var = [0]*game.n_targets + [target_idx]
            message_var[target_idx] = s 
            message_var = tuple(message_var)
            res += game.p_plus[(s, game.n_informants-1, target_idx)] * x[message_var]
    else:
        reported_target = int(reported_message.split("_")[1])
        if reported_target == target_idx:
            for s in range(s_hat):
                message_var = [0]*game.n_targets + [target_idx]
                message_var[target_idx] = s + 1
                message_var = tuple(message_var)
                res += game.p_plus[(s, game.n_informants-1, target_idx)] * x[message_var]
            if s_hat < game.n_informants:
                res += game.p_plus[(s_hat, game.n_informants-1, target_idx)] * game.s_infty[target_idx]
        else:
            for s in range(min(s_hat, game.n_informants-1)+1):
                message_var = [0]*game.n_targets + [target_idx]
                message_var[target_idx] = s
                message_var[reported_target] = 1
                message_var = tuple(message_var)
                res += game.p_plus[(s, game.n_informants-1, target_idx)] * x[message_var]
    return res


def save_solutions_to_file(solutions, objs, statuses, folder):
    optimal_solution_idx = np.argmax(objs)
    for idx in range(len(objs)):
        if idx == optimal_solution_idx:
            file_name = '{}/target_{}_solution_opt.txt'.format(folder, idx)
        else:
            file_name = '{}/target_{}_solution.txt'.format(folder, idx)
        output_content = ['Status: {}'.format(statuses[idx])]
        if statuses[idx] == 'OPTIMAL':
            output_content.append('')
            for k, v in solutions[idx].items():
                output_content.append('{}: {}'.format(k, v))
        with open(file_name, 'w') as f:
            f.writelines(f"{l}\n" for l in output_content)


def solve_for_target(target_idx, game, write_lp, lp_dir, s_hat):
    model = gp.Model("target_{}_program".format(target_idx))
    model.setParam("OutputFlag", 0)

    # variable_set target+1 dimensions, where the first t dimension represents the number of informants who report that target and the last dimension represent the defender probability given the message.
    variable_set = generate_message_vars(s_hat=s_hat, t=game.n_targets)

    x = model.addVars(variable_set, name="message")
    #####################################################
    # add feasiblity constraints for the model variables#
    #####################################################
    for va in variable_set:
        model.addConstr(x[va] <= game.s_infty[va[-1]])
    
    arr = recursive([[]], game.n_targets, s_hat)
    res = []
    for a in arr:
        if feasible_message(a):
            res.append(a)
    for r in res:
        cur = 0.0
        for t in range(game.n_targets):
            cur += x[tuple(r+[t])]
        model.addConstr(cur <= game.n_resources) # feasibility constraint, each message's corresponding allocation proability can't exceed the numebr of resources
    
    model.setObjective(calc_defender_utility(target_idx=target_idx, game=game, x=x, s_hat=s_hat), GRB.MAXIMIZE)

    current_target_attacker_utility = calc_attacker_utility(target_idx=target_idx, game=game, x=x, s_hat=s_hat)

    for t in range(game.n_targets):
        if t != target_idx:
            model.addConstr(
                current_target_attacker_utility >= calc_attacker_utility(target_idx=t, game=game, x=x, s_hat=s_hat),
                name='IC_attacker_for_target_{}'.format(t)
            )
    for t in range(game.n_targets):
        for informant_type in range(game.n_types):
            truthful_message = 'target_{}_'.format(t)
            for message in game.truth_message_per_type[t]:
                if int(message.split("_")[1]) == informant_type:
                    truthful_message += message[-1]
                    break
            truthful_message_coverage_prob = calc_interim_converage_prob(target_idx=t, reported_message=truthful_message, game=game, x=x, s_hat=s_hat)
            for message in game.all_messages:
                if message == truthful_message: continue
                misreport_message_coverage_prob = calc_interim_converage_prob(target_idx=t, reported_message=message, game=game, x=x, s_hat=s_hat)
                if truthful_message.endswith('+'):
                    model.addConstr(
                        truthful_message_coverage_prob >= misreport_message_coverage_prob,
                        name='IC_{}_targetRelated_type_{}_message_{}'.format(t, informant_type, message)
                    )
                else:
                    model.addConstr(
                        truthful_message_coverage_prob <= misreport_message_coverage_prob,
                        name='IC_{}_targetRelated_type_{}_message_{}'.format(t, informant_type, message)
                    )
    
    if write_lp:
        file_name = '{}/target_{}.lp'.format(lp_dir, target_idx)
        model.write(file_name)
    
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        obj, solution_dict, attacker_u = None, None, None
    else:
        obj = model.objVal
        solution_dict = {}
        for va in variable_set:
            solution_dict[va] = x[va].x

        attacker_u = calc_attacker_utility(target_idx=target_idx, game=game, x=x, s_hat=s_hat).getValue()

    return model.Status, obj, solution_dict, attacker_u

def main(argv=None):
    config_file = config.cfg
    n_informants = None
    s_hat = config_file.s_hat
    type_dist = None
    cur_round = 1
    n_resources = config_file.n_resources
    if argv == None: 
        n_informants = config_file.n_informants
        type_dist_file = config_file.type_dist_file + "_" + str(cur_round) + "_10targets.txt"
        type_dist = np.loadtxt(type_dist_file)
    else: 
        cur_round = argv[2]
        n_informants = argv[1]
        type_dist = np.array([float(argv[0])/10.0, 1.0 - float(argv[0])/10.0]) 

    p_w = config_file.prob_observe
    
    informant_covered_payoff_file = config_file.direct_informant_covered_payoff_file
    informant_uncovered_payoff_file = config_file.direct_informant_uncovered_payoff_file
    attacker_payoff_file = config_file.attacker_payoff_file + "_" + str(cur_round) + "_10targets.txt"
    defender_payoff_file = config_file.defender_payoff_file + "_" + str(cur_round) + "_10targets.txt"
    
    write_lp = config_file.write_lp
    save_solutions = config_file.save_solutions
    output_dir = config_file.output_dir

    informant_covered_payoff = np.loadtxt(informant_covered_payoff_file)
    informant_uncovered_payoff = np.loadtxt(informant_uncovered_payoff_file)
    R_a, P_a = np.loadtxt(attacker_payoff_file)
    R_d, P_d = np.loadtxt(defender_payoff_file)

    n_types, n_targets = informant_covered_payoff.shape
    assert len(type_dist) == n_types
    assert informant_uncovered_payoff.shape == informant_covered_payoff.shape
    assert len(R_a) == n_targets
    assert len(P_a) == n_targets
    assert len(R_d) == n_targets
    assert len(P_d) == n_targets
    assert np.all(R_a >= P_a)
    assert np.all(R_d >= P_d)

    if write_lp and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    game = Game(
        n_informants,
        n_resources,
        p_w,
        n_types,
        n_targets,
        type_dist,
        informant_covered_payoff,
        informant_uncovered_payoff,
        R_a,
        P_a,
        R_d,
        P_d,
    )

    status = {
        1: 'LOADED',
        2: 'OPTIMAL',
        3: 'INFEASIBLE',
        4: 'INF_OR_UNBD',
        5: 'UNBOUNDED',
        6: 'CUTOFF',
        7: 'ITERATION_LIMIT',
        8: 'NODE_LIMIT',
        9: 'TIME_LIMIT',
        10: 'SOLUTION_LIMIT',
        11: 'INTERRUPTED',
        12: 'NUMERIC',
        13: 'SUBOPTIMAL',
        14: 'INPROGRESS',
        15: 'USER_OBJ_LIMIT'
    }

    objectives = [np.NINF] * n_targets
    attacker_obj = [np.NINF] * n_targets
    solutions = [0] * n_targets
    statuses = [0] * n_targets

    for t in range(n_targets):
        status_code, obj, sol, attacker_u = solve_for_target(t, game, write_lp, output_dir, s_hat)

        statuses[t] = status[status_code]
        if obj is not None:
            objectives[t] = obj
            solutions[t] = sol
            attacker_obj[t] = attacker_u
            
    idx = np.argmax(objectives)
    if save_solutions:
        save_solutions_to_file(solutions, objectives, statuses, output_dir)
    return objectives[idx], attacker_obj[idx]

if __name__ == '__main__':
    main()