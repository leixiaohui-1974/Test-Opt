import sys
from pathlib import Path
ROOT = Path('OptiChat/scripts').resolve().parents[0]
sys.path.insert(0, str(ROOT / 'Feas'))
sys.path.insert(0, str(ROOT / 'scripts'))
from run_water_network_gate_chain import build_gate_chain_config
from water_network_generic import build_water_network_model
from pyomo.environ import SolverFactory

config = build_gate_chain_config()
for edge in config['edges']:
    attrs = edge.get('attributes', {}) or {}
    attrs.pop('gate_formula', None)
model = build_water_network_model(config)
opt = SolverFactory('appsi_highs')
res = opt.solve(model, tee=True)
print(res.solver.status, res.solver.termination_condition)
