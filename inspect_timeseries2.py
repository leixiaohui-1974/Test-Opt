import pandas as pd
from pathlib import Path
path = Path('OptiChat/reports/water_network/coupled_tailwater_timeseries.csv')
df = pd.read_csv(path)
print('storage_end min/max:', df['storage_end'].min(), df['storage_end'].max())
print(df[['period','storage_end','gate_flow','pump_flow']])
