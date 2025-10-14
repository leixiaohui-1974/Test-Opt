import sys
from pathlib import Path
ROOT = Path('OptiChat')
sys.path.insert(0, str(ROOT/'Feas'))
sys.path.insert(0, str(ROOT/'scripts'))
from run_water_network_coupled import COUPLED_NETWORK_CONFIG
from water_network_generic import build_water_network_model
from pyomo.environ import SolverFactory

model = build_water_network_model(COUPLED_NETWORK_CONFIG)
solver = SolverFactory('appsi_highs')
print('available:', solver.available(exception_flag=False))
try:
    result = solver.solve(model, load_solutions=False)
    print('termination:', result.solver.termination_condition)
    print('best_feasible:', result.solver.best_feasible_objective)
except Exception as exc:
    import traceback
    traceback.print_exc()
