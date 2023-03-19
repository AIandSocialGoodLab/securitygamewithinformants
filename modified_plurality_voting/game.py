import numpy as np
from scipy.special import comb
import gurobipy as gp
from gurobipy import GRB

class Game:
    def __init__(
            self,
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
    ):
        self.n_informants = n_informants
        self.n_resources = n_resources
        self.p_w = p_w
        self.n_types = n_types
        self.n_targets = n_targets
        self.type_dist = type_dist
        self.informant_covered_payoff = informant_covered_payoff
        self.informant_uncovered_payoff = informant_uncovered_payoff
        self.R_a = R_a
        self.P_a = P_a
        self.R_d = R_d
        self.P_d = P_d

        self.p_plus = {}
        for s in range(n_informants + 1):
            for t in range(n_targets):
                self.p_plus[(n_informants,n_informants-1,t)] = 0.0
                p_plus_t = 0.0
                for informant_type in range(n_types):
                    if informant_covered_payoff[informant_type][t] > informant_uncovered_payoff[informant_type][t]:
                        p_plus_t += type_dist[informant_type]
                self.p_plus[(s, n_informants, t)] = comb(n_informants, s) * ( ((p_plus_t * p_w) ** s) * ((1-p_plus_t*p_w) ** (n_informants-s)) )
                if s <= n_informants - 1:
                    self.p_plus[(s, n_informants-1, t)] = comb(n_informants-1, s) * (((p_plus_t * p_w) ** s) * ((1-p_plus_t*p_w) ** (n_informants-1-s)))
        self.s_infty = self.no_informant_unlimited_resource_strategy()
        self.all_messages = set(
            ['target_{}_{}'.format(t, a) for t in range(n_targets) for a in ['+', '-']]
        )
        self.all_messages.add('no_message')
        self.truth_message_per_type = [set() for _ in range(n_targets)]
        for target in range(n_targets):
            for informant_type in range(n_types):
                if informant_covered_payoff[informant_type][target] > informant_uncovered_payoff[informant_type][target]:
                    self.truth_message_per_type[target].add('type_{}_+'.format(informant_type))
                else:
                    self.truth_message_per_type[target].add('type_{}_-'.format(informant_type))

    def no_informant_unlimited_resource_strategy(self):
        objectives = [np.NINF] * self.n_targets
        solutions = {i:[0.0]*self.n_targets for i in range(self.n_targets)}
        
        for t in range(self.n_targets):
            
            model = gp.Model("target_{}_program".format(t))
            
            x = []
            for i in range(self.n_targets):
                x += [model.addVar(lb=0.0, ub=1.0, name="x_{}".format(i))]
            
            obj = x[t] * self.R_d[t] + (1-x[t]) * self.P_d[t]
            current_target_attacker_utility = x[t] * self.P_a[t] + (1-x[t]) * self.R_a[t]
            model.setObjective(obj, GRB.MAXIMIZE)
            
            for i in range(self.n_targets):
                if i == t: continue
                misreport_target_attacker_utility = x[i] * self.P_a[i] + (1-x[i]) * self.R_a[i]
                model.addConstr(current_target_attacker_utility >= misreport_target_attacker_utility, name="target_{}".format(t))

            model.optimize()

            if model.Status == GRB.OPTIMAL:
                objectives[t] = model.objVal
                for i in range(self.n_targets):
                    solutions[t][i] = x[i].x
        strategies = [1.0] * self.n_targets
        t_infty = np.argmax(objectives)
        strategies[t_infty] = solutions[t_infty][t_infty]
        return strategies