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

baseline_product_data = pd.read_csv("uk_blp_products_1.csv")

# Demand instruments: only structured char
baseline_demand_instruments = pyblp.build_blp_instruments(pyblp.Formulation('1 + hpwt + mpd + space'), baseline_product_data)

for i, column in enumerate(baseline_demand_instruments.T):
    baseline_product_data[f'demand_instruments{i}'] = column

baseline_logit_formulation = pyblp.Formulation('1 + prices + hpwt + mpd + space')
baseline_logit_problem = pyblp.Problem(baseline_logit_formulation, baseline_product_data)
baseline_logit_results = baseline_logit_problem.solve()
baseline_logit_results
baseline_elasticities = baseline_logit_results.compute_elasticities()
baseline_mean_elasticities = pd.DataFrame(baseline_logit_results,extract_diagonal_means(baseline_elasticities))
print(baseline_mean_elasticities.mean())
print(baseline_mean_elasticities.std())
baseline_markups = pd.DataFrame(baseline_logit_results.compute_markups())
print(baseline_markups.mean())
print(baseline_markups.std())



visual_logit_formulation = pyblp.Formulation('1 + prices + hpwt + mpd + space + bodyshape + boxiness + grille_height + grille_width')
visual_logit_problem = pyblp.Problem(visual_logit_formulation, baseline_product_data)
visual_logit_results = visual_logit_problem.solve()
visual_logit_results
visual_elasticities = visual_logit_results.compute_elasticities()
visual_mean_elasticities = pd.DataFrame(visual_logit_results,extract_diagonal_means(visual_elasticities))
print(visual_mean_elasticities.mean())
print(visual_mean_elasticities.std())
visual_markups = pd.DataFrame(visual_logit_results.compute_markups())
print(visual_markups.mean())
print(visual_markups.std())



fullvisual_product_data = pd.read_csv("uk_blp_products_1.csv")
fullvisual_demand_instruments = pyblp.build_blp_instruments(pyblp.Formulation('1 + hpwt + mpd + space + boxiness + grille_height + grille_width'),fullvisual_product_data)

for i, column in enumerate(fullvisual_demand_instruments.T):
    fullvisual_product_data[f'demand_instruments{i}'] = column

fullvisual_logit_formulation = pyblp.Formulation('1 + prices + hpwt + mpd + space + bodyshape + boxiness + grille_height + grille_width')
fullvisual_logit_problem = pyblp.Problem(fullvisual_logit_formulation,fullvisual_product_data)
fullvisual_logit_results = fullvisual_logit_problem.solve()
fullvisual_logit_results
fullvisual_elasticities = fullvisual_logit_results.compute_elasticities()
fullvisual_mean_elasticities = pd.DataFrame(fullvisual_elasticities,extract_diagonal_means(fullvisual_elasticities))
print(fullvisual_mean_elasticities.mean())
print(fullvisual_mean_elasticities.std())
fullvisual_markups = pd.DataFrame(fullvisual_logit_results.compute_markups())
print(fullvisual_markups.mean())
print(fullvisual_markups.std())

