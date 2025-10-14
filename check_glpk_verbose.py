from pathlib import Path
import sys
ROOT = Path('OptiChat')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'Feas'))
from scripts.run_water_network_coupled import COUPLED_NETWORK_CONFIG
from water_network_generic import build_water_network_model
from pyomo.environ import SolverFactory
model = build_water_network_model(COUPLED_NETWORK_CONFIG)
solver = SolverFactory('glpk')
results = solver.solve(model, tee=True)
print(results.solver.termination_condition)
