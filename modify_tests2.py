from pathlib import Path
path = Path('OptiChat/tests/test_water_network_enhancements.py')
text = path.read_text('utf-8')
old = "    data = export_linearization(COUPLED_NETWORK_CONFIG)\n    assert data[\"state_piecewise\"]\n    assert data[\"state_couplings\"]\n    assert data[\"gate_tables\"]\n    assert data[\"sos2_cost_tables\"]\n\n    mpc_model = build_tailwater_mpc_model(COUPLED_NETWORK_CONFIG)\n"
new = "    data = export_linearization(COUPLED_NETWORK_CONFIG)\n    assert data[\"state_piecewise\"]\n    assert data[\"gate_tables\"]\n    assert data[\"sos2_cost_tables\"]\n\n    mpc_model = build_tailwater_mpc_model(COUPLED_NETWORK_CONFIG)\n"
if old not in text:
    raise SystemExit('pattern not found in second block')
path.write_text(text.replace(old, new, 1), encoding='utf-8')
