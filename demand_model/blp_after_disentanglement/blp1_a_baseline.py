##############################################################################
# (a) Model: structured char only, instruments from structured char,
#            random coefficients on [price + structured char only].
##############################################################################

import pyblp
import numpy as np
import pandas as pd
import random
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
pyblp.options.seed = SEED
pyblp.options.digits = 2
pyblp.options.verbose = True
pyblp.__version__

uk_product_data = pd.read_csv("uk_blp_products_1.csv")

# Demand instruments: only structured char
demand_instruments = pyblp.build_blp_instruments(pyblp.Formulation('1 + hpwt + mpd + space'), uk_product_data)

for i, column in enumerate(demand_instruments.T):
    uk_product_data[f'demand_instruments{i}'] = column

logit_formulation = pyblp.Formulation('1 + prices + hpwt + mpd + space')
logit_instruments = demand_instruments
logit_problem = pyblp.Problem(logit_formulation, uk_product_data)
logit_results = logit_problem.solve()

logit_results

supply_instruments = np.c_[
    pyblp.build_blp_instruments(pyblp.Formulation('1 + log(hpwt) + log(mpg) + log(space)'), uk_product_data),
    pyblp.build_blp_instruments(pyblp.Formulation('0 + trend'), uk_product_data)[:, 0],
    uk_product_data['mpd'],
]

for i, column in enumerate(supply_instruments.T):
    uk_product_data[f'supply_instruments{i}'] = column

uk_product_data.head()

uk_agent_data = pd.read_csv("uk_blp_agents_h4.csv")

uk_agent_data.head()

# Model Formulation with No Visual Char in Demand
uk_product_formulations_with_supply = (
   pyblp.Formulation('1 + hpwt + mpd + space'),
   pyblp.Formulation('1 + prices + hpwt + mpd + space'),
   pyblp.Formulation('1 + log(hpwt) + log(mpg) + log(space) + trend')
)

uk_product_formulations_with_supply

agent_formulation = pyblp.Formulation('0 + I(1 / income)')
agent_formulation

uk_problem_with_supply = pyblp.Problem(uk_product_formulations_with_supply, uk_product_data, agent_formulation, uk_agent_data, costs_type='log')

uk_problem_with_supply

# random coefficients (constant, price, hpwt, mpd, space)
uk_initial_sigma = np.diag([1.5, 0, 8.5, 8.5, 8.5])
uk_initial_pi = np.c_[[0, -14, 0, 0, 0]]

uk_sigma_bounds = (
        np.zeros_like(uk_initial_sigma),
        np.diag([100, 0, 100, 100, 100])
)
uk_pi_bounds = (
        np.c_[[0, -100, 0, 0, 0]],
        np.c_[[0, -0.1, 0, 0, 0]]
)

uk_results_with_supply = uk_problem_with_supply.solve(
        uk_initial_sigma,
        uk_initial_pi,
        sigma_bounds=uk_sigma_bounds,
        pi_bounds=uk_pi_bounds,
        W_type='clustered',
        se_type='clustered',
        initial_update=True,
        optimization=pyblp.Optimization('l-bfgs-b'),
        costs_bounds=(0.001, None),
        iteration=pyblp.Iteration('squarem', {'atol': 1e-14})
)
uk_results_with_supply

instrument_results = uk_results_with_supply.compute_optimal_instruments(method='approximate')
updated_problem = instrument_results.to_problem()

updated_results = updated_problem.solve(
    uk_results_with_supply.sigma,
    uk_results_with_supply.pi,
    optimization=pyblp.Optimization('bfgs', {'gtol': 1e-5}),
    method='1s'
)
updated_results

uk_elasticities = updated_results.compute_elasticities()
uk_mean_elasticities = pd.DataFrame(updated_results.extract_diagonal_means(uk_elasticities))
print(uk_mean_elasticities.mean())
print(uk_mean_elasticities.std())

uk_markups = pd.DataFrame(updated_results.compute_markups())
print(uk_markups.mean())
print(uk_markups.std())

counterfactual_simulation = pyblp.Simulation(
        product_formulations = (pyblp.Formulation('1 + hpwt + mpd + space'),
        pyblp.Formulation('1 + prices + hpwt + mpd + space'),
        pyblp.Formulation('1 + log(hpwt) + log(mpg) + log(space) + trend')),
        product_data = uk_product_data,
        beta = updated_results.beta,
        sigma = updated_results.sigma,
        gamma = updated_results.gamma,
        pi = updated_results.pi,
        agent_formulation = pyblp.Formulation('0 + I(1 / income)'),
        costs_type = 'log',
        agent_data = uk_agent_data,
        xi = updated_results.xi,
        omega = updated_results.omega
)

counterfactual_simulation_endogenous = counterfactual_simulation.replace_endogenous()
counterfactual_shares = pd.DataFrame(pyblp.data_to_dict(counterfactual_simulation_endogenous.product_data))

counterfactual_shares.to_csv('counterfactual_shares1_a.csv',index=False)

uk_product_data.to_csv('exp_uk_product_data_a_1.csv',index=False)
# pd.DataFrame(uk_results_with_supply.xi_fe, columns = ['xi_fe']).to_csv('exp_xi_fe_a_1.csv',index=False)
pd.DataFrame(updated_results.xi_fe, columns = ['xi_fe']).to_csv('exp_xi_fe_a_1.csv',index=False) 

'''

bmw_wo_viz_sim = pyblp.Simulation(
        product_formulations = (pyblp.Formulation('1 + gas + engine + space', absorb='C(clustering_ids)'),
        pyblp.Formulation('1 + prices + gas + engine + space'),
        pyblp.Formulation('1 + log(gas) + log(mpg) + log(space) + trend')),
        product_data = uk_product_data,
        beta = uk_results_with_supply.beta,
        sigma = uk_results_with_supply.sigma,
        gamma = uk_results_with_supply.gamma,
        pi = uk_results_with_supply.pi,
        agent_formulation = pyblp.Formulation('0 + I(1 / income)'),
        costs_type = 'log',
        agent_data = uk_agent_data,
        xi = uk_results_with_supply.xi,
        omega = uk_results_with_supply.omega
)

bmw_rep_end_wo_viz = bmw_wo_viz_sim.replace_endogenous()
bmw_shares_wo_viz = pd.DataFrame(pyblp.data_to_dict(bmw_rep_end_wo_viz.product_data))

uk_logit_het1 = (
   pyblp.Formulation('1 + prices + gas + engine + space + Viz1 + Viz4 + Viz5')
)
uk_logit_problem_het1 = pyblp.Problem(uk_logit_het1, uk_product_data)
uk_logit_results_het1 = uk_logit_problem_het1.solve()
uk_logit_results_het1

logit_elasticities_het1 = uk_logit_results_het1.compute_elasticities()
logit_mean_elasticities_het1 = pd.DataFrame(uk_logit_results_het1.extract_diagonal_means(logit_elasticities_het1))
print(logit_mean_elasticities_het1.mean())
print(logit_mean_elasticities_het1.std())

logit_markups_het1 = pd.DataFrame(uk_logit_results_het1.compute_markups())
print(logit_markups_het1.mean())
print(logit_markups_het1.std())

uk_logit_het2 = (
   pyblp.Formulation('1 + prices + gas + engine + space + Viz1*C(Segment) + Viz4*C(Segment) + Viz5*C(Segment)')
)
uk_logit_problem_het2 = pyblp.Problem(uk_logit_het2, uk_product_data)
uk_logit_results_het2 = uk_logit_problem_het2.solve()
uk_logit_results_het2

logit_elasticities_het2 = uk_logit_results_het2.compute_elasticities()
logit_mean_elasticities_het2 = pd.DataFrame(uk_logit_results_het2.extract_diagonal_means(logit_elasticities_het2))
print(logit_mean_elasticities_het2.mean())
print(logit_mean_elasticities_het2.std())

logit_markups_het2 = pd.DataFrame(uk_logit_results_het2.compute_markups())
print(logit_markups_het2.mean())
print(logit_markups_het2.std())

uk_agent_data_het = pd.read_csv("uk_blp_agents_h7.csv")

uk_product_formulations_with_supply_het = (
   pyblp.Formulation('1 + gas + engine + space + Viz1 + Viz4 + Viz5', absorb='C(clustering_ids)'),
   pyblp.Formulation('1 + prices + gas + engine + space  + Viz1 + Viz4 + Viz5'),
   pyblp.Formulation('1 + log(gas) + log(mpg) + log(space) + trend')
)

uk_problem_with_supply = pyblp.Problem(uk_product_formulations_with_supply_het, uk_product_data, agent_formulation, uk_agent_data_het, costs_type='log')

uk_initial_sigma_het = np.diag([2, 0, 1, 6, 7, 0.1, 0.3, 0.5])
uk_initial_pi_het = np.c_[[0, -10, 0, 0, 0, 0, 0, 0]]

uk_sigma_bounds_het = (
        np.zeros_like(uk_initial_sigma_het),
        np.diag([100, 0, 100, 100, 100, 100, 100, 100])
)
uk_pi_bounds_het = (
        np.c_[[0, -100, 0, 0, 0, 0, 0, 0]],
        np.c_[[0, -0.1, 0, 0, 0, 0, 0, 0]]
)

uk_results_with_supply_het = uk_problem_with_supply.solve(
        uk_initial_sigma_het,
        uk_initial_pi_het,
        sigma_bounds=uk_sigma_bounds_het,
        pi_bounds=uk_pi_bounds_het,
        initial_update=True,
        optimization=pyblp.Optimization('l-bfgs-b'),
        costs_bounds=(0.001, None),
        iteration=pyblp.Iteration('squarem', {'atol': 1e-14})
)
uk_results_with_supply_het

cl_elasticities = uk_results_with_supply_het.compute_elasticities()
cl_mean_elasticities = pd.DataFrame(uk_results_with_supply_het.extract_diagonal_means(cl_elasticities))
print(cl_mean_elasticities.mean())
print(cl_mean_elasticities.std())

cl_markups = pd.DataFrame(uk_results_with_supply_het.compute_markups())
print(cl_markups.mean())
print(cl_markups.std())

bmw_w_viz_wo_change_sim = pyblp.Simulation(
        product_formulations = (pyblp.Formulation('1 + gas + engine + space + Viz1 + Viz4 + Viz5', absorb='C(clustering_ids)'),
        pyblp.Formulation('1 + prices + gas + engine + space + Viz1 + Viz4 + Viz5'),
        pyblp.Formulation('1 + log(gas) + log(mpg) + log(space) + trend')),
        product_data = uk_product_data,
        beta = uk_results_with_supply_het.beta,
        sigma = uk_results_with_supply_het.sigma,
        gamma = uk_results_with_supply_het.gamma,
        pi = uk_results_with_supply_het.pi,
        agent_formulation = pyblp.Formulation('0 + I(1 / income)'),
        costs_type = 'log',
        agent_data = uk_agent_data_het,
        xi = uk_results_with_supply_het.xi,
        omega = uk_results_with_supply_het.omega
)

bmw_rep_end_wo_change_sim = bmw_w_viz_wo_change_sim.replace_endogenous()
bmw_shares_wo_change_sim = pd.DataFrame(pyblp.data_to_dict(bmw_rep_end_wo_change_sim.product_data))

test_data = pd.read_csv("test_data.csv")

bmw_w_viz_w_change_sim = pyblp.Simulation(
        product_formulations = (pyblp.Formulation('1 + gas + engine + space + Viz1 + Viz4 + Viz5', absorb='C(clustering_ids)'),
        pyblp.Formulation('1 + prices + gas + engine + space + Viz1 + Viz4 + Viz5'),
        pyblp.Formulation('1 + log(gas) + log(mpg) + log(space) + trend')),
        product_data = test_data,
        beta = uk_results_with_supply_het.beta,
        sigma = uk_results_with_supply_het.sigma,
        gamma = uk_results_with_supply_het.gamma,
        pi = uk_results_with_supply_het.pi,
        agent_formulation = pyblp.Formulation('0 + I(1 / income)'),
        costs_type = 'log',
        agent_data = uk_agent_data_het,
        xi = uk_results_with_supply_het.xi,
        omega = uk_results_with_supply_het.omega
)

bmw_rep_end_w_change_sim = bmw_w_viz_w_change_sim.replace_endogenous()
bmw_shares_w_change_sim = pd.DataFrame(pyblp.data_to_dict(bmw_rep_end_w_change_sim.product_data))

bmw_shares_wo_viz.to_csv('v2_bmw_shares_wo_viz.csv',index=False)
bmw_shares_wo_change_sim.to_csv('v2_bmw_shares_wo_change_sim.csv',index=False)
bmw_shares_w_change_sim.to_csv('v2_bmw_shares_w_change_sim.csv',index=False)

'''

