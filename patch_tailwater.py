from pathlib import Path
path = Path('OptiChat/scripts/run_water_network_coupled.py')
text = path.read_text('utf-8')
old = '            "states": {\n                "stage": {\n'
new = '            "states": {\n                "storage": {\n                    "initial": 0.0,\n                    "bounds": (-200.0, 500.0),\n                    "role": "storage",\n                },\n                "stage": {\n'
if old not in text:
    raise SystemExit('pattern not found')
path.write_text(text.replace(old, new, 1), encoding='utf-8')
