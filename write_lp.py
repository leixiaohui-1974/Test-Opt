from pathlib import Path
import sys
ROOT = Path('OptiChat')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'Feas'))
from scripts.run_water_network_coupled import COUPLED_NETWORK_CONFIG
from water_network_generic import build_water_network_model
model = build_water_network_model(COUPLED_NETWORK_CONFIG)
model.write('debug.lp', format='lp', io_options={'symbolic_solver_labels': True})
print('written debug.lp')
