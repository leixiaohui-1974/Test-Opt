from pathlib import Path
path = Path('OptiChat/scripts/demo_mpc_tailwater.py')
text = path.read_text('utf-8')
old = "    linear_data = export_linearization(config)\n    couplings = linear_data[\"state_couplings\"]\n    if not couplings:\n        raise ValueError(\"No coupling information found; cannot build MPC model.\")\n\n    slope = float(couplings[0][\"slope\"])\n    intercept = float(couplings[0][\"intercept\"])\n"
new = "    linear_data = export_linearization(config)\n    couplings = linear_data[\"state_couplings\"]\n    if couplings:\n        slope = float(couplings[0][\"slope\"])\n        intercept = float(couplings[0][\"intercept\"])\n    else:\n        piecewise_entry = next((entry for entry in linear_data[\"state_piecewise\"]\n                                  if entry[\"target\"][0] == \"tailwater\"), None)\n        if piecewise_entry and len(piecewise_entry[\"breakpoints\"]) >= 2:\n            x0, x1 = piecewise_entry[\"breakpoints\"][0:2]\n            y0, y1 = piecewise_entry[\"values\"][0:2]\n            slope = float((y1 - y0) / (x1 - x0))\n            intercept = float(y0 - slope * x0)\n        else:\n            raise ValueError(\"No coupling data available for tailwater stage.\")\n"
if old not in text:
    raise SystemExit('pattern not found in demo script')
path.write_text(text.replace(old, new, 1), encoding='utf-8')
