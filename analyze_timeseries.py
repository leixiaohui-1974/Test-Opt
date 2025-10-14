import pandas as pd
from pathlib import Path
path = Path('OptiChat/reports/water_network/coupled_tailwater_timeseries.csv')
df = pd.read_csv(path)
print(df)
print('storage min/max', df['storage_end'].min(), df['storage_end'].max())
print('tailwater storage min/max', df['tailwater_storage'].min(), df['tailwater_storage'].max())
