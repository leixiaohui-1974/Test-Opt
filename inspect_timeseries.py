import pandas as pd
from pathlib import Path
path = Path('OptiChat/reports/water_network/coupled_tailwater_timeseries.csv')
df = pd.read_csv(path)
print(df.head())
print(df[['pump_flow','demand','shortage']].describe())
