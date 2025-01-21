import pyblp
import numpy as np
import pandas as pd
pyblp.options.digits = 2
pyblp.options.verbose = True
pyblp.__version__

uk_product_data = pd.read_csv("uk_blp_products.csv")

demand_instruments = pyblp.build_blp_instruments(pyblp.Formulation('1 + hpwt + mpd + space'), uk_product_data)

for i, column in enumerate(demand_instruments.T):
    uk_product_data[f'demand_instruments{i}'] = column

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

uk_product_formulations_with_supply = (
   pyblp.Formulation('1 + hpwt + mpd + space', absorb='C(clustering_ids)'),
   pyblp.Formulation('1 + prices + hpwt + mpd + space'),
   pyblp.Formulation('1 + log(hpwt) + log(mpg) + log(space) + trend')
)

uk_product_formulations_with_supply

agent_formulation = pyblp.Formulation('0 + I(1 / income)')
agent_formulation

uk_problem_with_supply = pyblp.Problem(uk_product_formulations_with_supply, uk_product_data, agent_formulation, uk_agent_data, costs_type='log')

uk_problem_with_supply

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
#        W_type='clustered',
#        se_type='clustered',
        initial_update=True,
        optimization=pyblp.Optimization('l-bfgs-b'),
        costs_bounds=(0.001, None),
        iteration=pyblp.Iteration('squarem', {'atol': 1e-14})
)
uk_results_with_supply

uk_product_data.to_csv('exp_uk_product_data.csv',index=False)
pd.DataFrame(uk_results_with_supply.xi_fe, columns = ['xi_fe']).to_csv('exp_xi_fe.csv',index=False)


