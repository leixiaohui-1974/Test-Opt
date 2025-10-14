from pathlib import Path
text=Path('OptiChat/Feas/water_network_generic.py').read_text('utf-8')
print(text.count('Piecewise('))
