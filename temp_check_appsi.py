from pyomo.environ import SolverFactory
import json
solver = SolverFactory('appsi_highs')
available = solver.available(exception_flag=False)
print(json.dumps({'available': available}))
