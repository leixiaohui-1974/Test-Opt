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
print('available:', solver.available(exception_flag=False))
if solver.executable() is None:
    solver.set_executable('glpsol')
print('exec:', solver.executable())
results = solver.solve(model)
print('termination:', results.solver.termination_condition)
