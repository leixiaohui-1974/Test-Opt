\* Source Pyomo model name=GenericWaterNetwork *\

min 
obj:
+1 sos2_cost(tailwater_pump_t01)
+1 sos2_cost(tailwater_pump_t02)
+1 sos2_cost(tailwater_pump_t03)
+1 sos2_cost(tailwater_pump_t04)
+1 sos2_cost(tailwater_pump_t05)
+1 sos2_cost(tailwater_pump_t06)
+1 sos2_cost(tailwater_pump_t07)
+1 sos2_cost(tailwater_pump_t08)
+1 sos2_cost(tailwater_pump_t09)
+1 sos2_cost(tailwater_pump_t10)
+1 sos2_cost(tailwater_pump_t11)
+1 sos2_cost(tailwater_pump_t12)
+10000.0 shortage(demand_t01)
+10000.0 shortage(demand_t02)
+10000.0 shortage(demand_t03)
+10000.0 shortage(demand_t04)
+10000.0 shortage(demand_t05)
+10000.0 shortage(demand_t06)
+10000.0 shortage(demand_t07)
+10000.0 shortage(demand_t08)
+10000.0 shortage(demand_t09)
+10000.0 shortage(demand_t10)
+10000.0 shortage(demand_t11)
+10000.0 shortage(demand_t12)

s.t.

c_u_gate_flow_limit(reservoir_gate_t01)_:
+1 flow(reservoir_gate_t01)
-1 gate_capacity(reservoir_gate_t01)
<= 0

c_u_gate_flow_limit(reservoir_gate_t02)_:
+1 flow(reservoir_gate_t02)
-1 gate_capacity(reservoir_gate_t02)
<= 0

c_u_gate_flow_limit(reservoir_gate_t03)_:
+1 flow(reservoir_gate_t03)
-1 gate_capacity(reservoir_gate_t03)
<= 0

c_u_gate_flow_limit(reservoir_gate_t04)_:
+1 flow(reservoir_gate_t04)
-1 gate_capacity(reservoir_gate_t04)
<= 0

c_u_gate_flow_limit(reservoir_gate_t05)_:
+1 flow(reservoir_gate_t05)
-1 gate_capacity(reservoir_gate_t05)
<= 0

c_u_gate_flow_limit(reservoir_gate_t06)_:
+1 flow(reservoir_gate_t06)
-1 gate_capacity(reservoir_gate_t06)
<= 0

c_u_gate_flow_limit(reservoir_gate_t07)_:
+1 flow(reservoir_gate_t07)
-1 gate_capacity(reservoir_gate_t07)
<= 0

c_u_gate_flow_limit(reservoir_gate_t08)_:
+1 flow(reservoir_gate_t08)
-1 gate_capacity(reservoir_gate_t08)
<= 0

c_u_gate_flow_limit(reservoir_gate_t09)_:
+1 flow(reservoir_gate_t09)
-1 gate_capacity(reservoir_gate_t09)
<= 0

c_u_gate_flow_limit(reservoir_gate_t10)_:
+1 flow(reservoir_gate_t10)
-1 gate_capacity(reservoir_gate_t10)
<= 0

c_u_gate_flow_limit(reservoir_gate_t11)_:
+1 flow(reservoir_gate_t11)
-1 gate_capacity(reservoir_gate_t11)
<= 0

c_u_gate_flow_limit(reservoir_gate_t12)_:
+1 flow(reservoir_gate_t12)
-1 gate_capacity(reservoir_gate_t12)
<= 0

c_e_mass_balance(reservoir_t01)_:
+1 flow(reservoir_gate_t01)
+1 state(reservoir_storage_t01)
= 480.0

c_e_mass_balance(reservoir_t02)_:
+1 flow(reservoir_gate_t02)
-1 state(reservoir_storage_t01)
+1 state(reservoir_storage_t02)
= 78.0

c_e_mass_balance(reservoir_t03)_:
+1 flow(reservoir_gate_t03)
-1 state(reservoir_storage_t02)
+1 state(reservoir_storage_t03)
= 76.0

c_e_mass_balance(reservoir_t04)_:
+1 flow(reservoir_gate_t04)
-1 state(reservoir_storage_t03)
+1 state(reservoir_storage_t04)
= 75.0

c_e_mass_balance(reservoir_t05)_:
+1 flow(reservoir_gate_t05)
-1 state(reservoir_storage_t04)
+1 state(reservoir_storage_t05)
= 74.0

c_e_mass_balance(reservoir_t06)_:
+1 flow(reservoir_gate_t06)
-1 state(reservoir_storage_t05)
+1 state(reservoir_storage_t06)
= 70.0

c_e_mass_balance(reservoir_t07)_:
+1 flow(reservoir_gate_t07)
-1 state(reservoir_storage_t06)
+1 state(reservoir_storage_t07)
= 68.0

c_e_mass_balance(reservoir_t08)_:
+1 flow(reservoir_gate_t08)
-1 state(reservoir_storage_t07)
+1 state(reservoir_storage_t08)
= 66.0

c_e_mass_balance(reservoir_t09)_:
+1 flow(reservoir_gate_t09)
-1 state(reservoir_storage_t08)
+1 state(reservoir_storage_t09)
= 64.0

c_e_mass_balance(reservoir_t10)_:
+1 flow(reservoir_gate_t10)
-1 state(reservoir_storage_t09)
+1 state(reservoir_storage_t10)
= 60.0

c_e_mass_balance(reservoir_t11)_:
+1 flow(reservoir_gate_t11)
-1 state(reservoir_storage_t10)
+1 state(reservoir_storage_t11)
= 58.0

c_e_mass_balance(reservoir_t12)_:
+1 flow(reservoir_gate_t12)
-1 state(reservoir_storage_t11)
+1 state(reservoir_storage_t12)
= 55.0

c_e_mass_balance(tailwater_t01)_:
-1 flow(reservoir_gate_t01)
+1 flow(tailwater_pump_t01)
+1 state(tailwater_storage_t01)
= 0

c_e_mass_balance(tailwater_t02)_:
-1 flow(reservoir_gate_t02)
+1 flow(tailwater_pump_t02)
-1 state(tailwater_storage_t01)
+1 state(tailwater_storage_t02)
= 0

c_e_mass_balance(tailwater_t03)_:
-1 flow(reservoir_gate_t03)
+1 flow(tailwater_pump_t03)
-1 state(tailwater_storage_t02)
+1 state(tailwater_storage_t03)
= 0

c_e_mass_balance(tailwater_t04)_:
-1 flow(reservoir_gate_t04)
+1 flow(tailwater_pump_t04)
-1 state(tailwater_storage_t03)
+1 state(tailwater_storage_t04)
= 0

c_e_mass_balance(tailwater_t05)_:
-1 flow(reservoir_gate_t05)
+1 flow(tailwater_pump_t05)
-1 state(tailwater_storage_t04)
+1 state(tailwater_storage_t05)
= 0

c_e_mass_balance(tailwater_t06)_:
-1 flow(reservoir_gate_t06)
+1 flow(tailwater_pump_t06)
-1 state(tailwater_storage_t05)
+1 state(tailwater_storage_t06)
= 0

c_e_mass_balance(tailwater_t07)_:
-1 flow(reservoir_gate_t07)
+1 flow(tailwater_pump_t07)
-1 state(tailwater_storage_t06)
+1 state(tailwater_storage_t07)
= 0

c_e_mass_balance(tailwater_t08)_:
-1 flow(reservoir_gate_t08)
+1 flow(tailwater_pump_t08)
-1 state(tailwater_storage_t07)
+1 state(tailwater_storage_t08)
= 0

c_e_mass_balance(tailwater_t09)_:
-1 flow(reservoir_gate_t09)
+1 flow(tailwater_pump_t09)
-1 state(tailwater_storage_t08)
+1 state(tailwater_storage_t09)
= 0

c_e_mass_balance(tailwater_t10)_:
-1 flow(reservoir_gate_t10)
+1 flow(tailwater_pump_t10)
-1 state(tailwater_storage_t09)
+1 state(tailwater_storage_t10)
= 0

c_e_mass_balance(tailwater_t11)_:
-1 flow(reservoir_gate_t11)
+1 flow(tailwater_pump_t11)
-1 state(tailwater_storage_t10)
+1 state(tailwater_storage_t11)
= 0

c_e_mass_balance(tailwater_t12)_:
-1 flow(reservoir_gate_t12)
+1 flow(tailwater_pump_t12)
-1 state(tailwater_storage_t11)
+1 state(tailwater_storage_t12)
= 0

c_e_mass_balance(demand_t01)_:
+1 shortage(demand_t01)
+1 flow(tailwater_pump_t01)
= 60.0

c_e_mass_balance(demand_t02)_:
+1 shortage(demand_t02)
+1 flow(tailwater_pump_t02)
= 62.0

c_e_mass_balance(demand_t03)_:
+1 shortage(demand_t03)
+1 flow(tailwater_pump_t03)
= 64.0

c_e_mass_balance(demand_t04)_:
+1 shortage(demand_t04)
+1 flow(tailwater_pump_t04)
= 66.0

c_e_mass_balance(demand_t05)_:
+1 shortage(demand_t05)
+1 flow(tailwater_pump_t05)
= 68.0

c_e_mass_balance(demand_t06)_:
+1 shortage(demand_t06)
+1 flow(tailwater_pump_t06)
= 70.0

c_e_mass_balance(demand_t07)_:
+1 shortage(demand_t07)
+1 flow(tailwater_pump_t07)
= 74.0

c_e_mass_balance(demand_t08)_:
+1 shortage(demand_t08)
+1 flow(tailwater_pump_t08)
= 78.0

c_e_mass_balance(demand_t09)_:
+1 shortage(demand_t09)
+1 flow(tailwater_pump_t09)
= 80.0

c_e_mass_balance(demand_t10)_:
+1 shortage(demand_t10)
+1 flow(tailwater_pump_t10)
= 78.0

c_e_mass_balance(demand_t11)_:
+1 shortage(demand_t11)
+1 flow(tailwater_pump_t11)
= 76.0

c_e_mass_balance(demand_t12)_:
+1 shortage(demand_t12)
+1 flow(tailwater_pump_t12)
= 72.0

c_u_flow_capacity(reservoir_gate_t01)_:
+1 flow(reservoir_gate_t01)
<= 120.0

c_u_flow_capacity(reservoir_gate_t02)_:
+1 flow(reservoir_gate_t02)
<= 120.0

c_u_flow_capacity(reservoir_gate_t03)_:
+1 flow(reservoir_gate_t03)
<= 120.0

c_u_flow_capacity(reservoir_gate_t04)_:
+1 flow(reservoir_gate_t04)
<= 120.0

c_u_flow_capacity(reservoir_gate_t05)_:
+1 flow(reservoir_gate_t05)
<= 120.0

c_u_flow_capacity(reservoir_gate_t06)_:
+1 flow(reservoir_gate_t06)
<= 120.0

c_u_flow_capacity(reservoir_gate_t07)_:
+1 flow(reservoir_gate_t07)
<= 120.0

c_u_flow_capacity(reservoir_gate_t08)_:
+1 flow(reservoir_gate_t08)
<= 120.0

c_u_flow_capacity(reservoir_gate_t09)_:
+1 flow(reservoir_gate_t09)
<= 120.0

c_u_flow_capacity(reservoir_gate_t10)_:
+1 flow(reservoir_gate_t10)
<= 120.0

c_u_flow_capacity(reservoir_gate_t11)_:
+1 flow(reservoir_gate_t11)
<= 120.0

c_u_flow_capacity(reservoir_gate_t12)_:
+1 flow(reservoir_gate_t12)
<= 120.0

c_u_flow_capacity(tailwater_pump_t01)_:
+1 flow(tailwater_pump_t01)
<= 90.0

c_u_flow_capacity(tailwater_pump_t02)_:
+1 flow(tailwater_pump_t02)
<= 90.0

c_u_flow_capacity(tailwater_pump_t03)_:
+1 flow(tailwater_pump_t03)
<= 90.0

c_u_flow_capacity(tailwater_pump_t04)_:
+1 flow(tailwater_pump_t04)
<= 90.0

c_u_flow_capacity(tailwater_pump_t05)_:
+1 flow(tailwater_pump_t05)
<= 90.0

c_u_flow_capacity(tailwater_pump_t06)_:
+1 flow(tailwater_pump_t06)
<= 90.0

c_u_flow_capacity(tailwater_pump_t07)_:
+1 flow(tailwater_pump_t07)
<= 90.0

c_u_flow_capacity(tailwater_pump_t08)_:
+1 flow(tailwater_pump_t08)
<= 90.0

c_u_flow_capacity(tailwater_pump_t09)_:
+1 flow(tailwater_pump_t09)
<= 90.0

c_u_flow_capacity(tailwater_pump_t10)_:
+1 flow(tailwater_pump_t10)
<= 90.0

c_u_flow_capacity(tailwater_pump_t11)_:
+1 flow(tailwater_pump_t11)
<= 90.0

c_u_flow_capacity(tailwater_pump_t12)_:
+1 flow(tailwater_pump_t12)
<= 90.0

c_e_state_piecewise(0)_pw_t01_CC_constraint1_:
+1 state(reservoir_storage_t01)
-250.0 state_piecewise(0)_pw_t01_CC_lambda(1)
-350.0 state_piecewise(0)_pw_t01_CC_lambda(2)
-450.0 state_piecewise(0)_pw_t01_CC_lambda(3)
-600.0 state_piecewise(0)_pw_t01_CC_lambda(4)
= 0

c_e_state_piecewise(0)_pw_t01_CC_constraint2_:
+1 state(reservoir_level_t01)
-202.0 state_piecewise(0)_pw_t01_CC_lambda(1)
-208.0 state_piecewise(0)_pw_t01_CC_lambda(2)
-215.0 state_piecewise(0)_pw_t01_CC_lambda(3)
-225.0 state_piecewise(0)_pw_t01_CC_lambda(4)
= 0

c_e_state_piecewise(0)_pw_t01_CC_constraint3_:
+1 state_piecewise(0)_pw_t01_CC_lambda(1)
+1 state_piecewise(0)_pw_t01_CC_lambda(2)
+1 state_piecewise(0)_pw_t01_CC_lambda(3)
+1 state_piecewise(0)_pw_t01_CC_lambda(4)
= 1

c_u_state_piecewise(0)_pw_t01_CC_constraint4(1)_:
+1 state_piecewise(0)_pw_t01_CC_lambda(1)
-1 state_piecewise(0)_pw_t01_CC_bin_y(1)
<= 0

c_u_state_piecewise(0)_pw_t01_CC_constraint4(2)_:
+1 state_piecewise(0)_pw_t01_CC_lambda(2)
-1 state_piecewise(0)_pw_t01_CC_bin_y(1)
-1 state_piecewise(0)_pw_t01_CC_bin_y(2)
<= 0

c_u_state_piecewise(0)_pw_t01_CC_constraint4(3)_:
+1 state_piecewise(0)_pw_t01_CC_lambda(3)
-1 state_piecewise(0)_pw_t01_CC_bin_y(2)
-1 state_piecewise(0)_pw_t01_CC_bin_y(3)
<= 0

c_u_state_piecewise(0)_pw_t01_CC_constraint4(4)_:
+1 state_piecewise(0)_pw_t01_CC_lambda(4)
-1 state_piecewise(0)_pw_t01_CC_bin_y(3)
<= 0

c_e_state_piecewise(0)_pw_t01_CC_constraint5_:
+1 state_piecewise(0)_pw_t01_CC_bin_y(1)
+1 state_piecewise(0)_pw_t01_CC_bin_y(2)
+1 state_piecewise(0)_pw_t01_CC_bin_y(3)
= 1

c_e_state_piecewise(0)_pw_t02_CC_constraint1_:
+1 state(reservoir_storage_t02)
-250.0 state_piecewise(0)_pw_t02_CC_lambda(1)
-350.0 state_piecewise(0)_pw_t02_CC_lambda(2)
-450.0 state_piecewise(0)_pw_t02_CC_lambda(3)
-600.0 state_piecewise(0)_pw_t02_CC_lambda(4)
= 0

c_e_state_piecewise(0)_pw_t02_CC_constraint2_:
+1 state(reservoir_level_t02)
-202.0 state_piecewise(0)_pw_t02_CC_lambda(1)
-208.0 state_piecewise(0)_pw_t02_CC_lambda(2)
-215.0 state_piecewise(0)_pw_t02_CC_lambda(3)
-225.0 state_piecewise(0)_pw_t02_CC_lambda(4)
= 0

c_e_state_piecewise(0)_pw_t02_CC_constraint3_:
+1 state_piecewise(0)_pw_t02_CC_lambda(1)
+1 state_piecewise(0)_pw_t02_CC_lambda(2)
+1 state_piecewise(0)_pw_t02_CC_lambda(3)
+1 state_piecewise(0)_pw_t02_CC_lambda(4)
= 1

c_u_state_piecewise(0)_pw_t02_CC_constraint4(1)_:
+1 state_piecewise(0)_pw_t02_CC_lambda(1)
-1 state_piecewise(0)_pw_t02_CC_bin_y(1)
<= 0

c_u_state_piecewise(0)_pw_t02_CC_constraint4(2)_:
+1 state_piecewise(0)_pw_t02_CC_lambda(2)
-1 state_piecewise(0)_pw_t02_CC_bin_y(1)
-1 state_piecewise(0)_pw_t02_CC_bin_y(2)
<= 0

c_u_state_piecewise(0)_pw_t02_CC_constraint4(3)_:
+1 state_piecewise(0)_pw_t02_CC_lambda(3)
-1 state_piecewise(0)_pw_t02_CC_bin_y(2)
-1 state_piecewise(0)_pw_t02_CC_bin_y(3)
<= 0

c_u_state_piecewise(0)_pw_t02_CC_constraint4(4)_:
+1 state_piecewise(0)_pw_t02_CC_lambda(4)
-1 state_piecewise(0)_pw_t02_CC_bin_y(3)
<= 0

c_e_state_piecewise(0)_pw_t02_CC_constraint5_:
+1 state_piecewise(0)_pw_t02_CC_bin_y(1)
+1 state_piecewise(0)_pw_t02_CC_bin_y(2)
+1 state_piecewise(0)_pw_t02_CC_bin_y(3)
= 1

c_e_state_piecewise(0)_pw_t03_CC_constraint1_:
+1 state(reservoir_storage_t03)
-250.0 state_piecewise(0)_pw_t03_CC_lambda(1)
-350.0 state_piecewise(0)_pw_t03_CC_lambda(2)
-450.0 state_piecewise(0)_pw_t03_CC_lambda(3)
-600.0 state_piecewise(0)_pw_t03_CC_lambda(4)
= 0

c_e_state_piecewise(0)_pw_t03_CC_constraint2_:
+1 state(reservoir_level_t03)
-202.0 state_piecewise(0)_pw_t03_CC_lambda(1)
-208.0 state_piecewise(0)_pw_t03_CC_lambda(2)
-215.0 state_piecewise(0)_pw_t03_CC_lambda(3)
-225.0 state_piecewise(0)_pw_t03_CC_lambda(4)
= 0

c_e_state_piecewise(0)_pw_t03_CC_constraint3_:
+1 state_piecewise(0)_pw_t03_CC_lambda(1)
+1 state_piecewise(0)_pw_t03_CC_lambda(2)
+1 state_piecewise(0)_pw_t03_CC_lambda(3)
+1 state_piecewise(0)_pw_t03_CC_lambda(4)
= 1

c_u_state_piecewise(0)_pw_t03_CC_constraint4(1)_:
+1 state_piecewise(0)_pw_t03_CC_lambda(1)
-1 state_piecewise(0)_pw_t03_CC_bin_y(1)
<= 0

c_u_state_piecewise(0)_pw_t03_CC_constraint4(2)_:
+1 state_piecewise(0)_pw_t03_CC_lambda(2)
-1 state_piecewise(0)_pw_t03_CC_bin_y(1)
-1 state_piecewise(0)_pw_t03_CC_bin_y(2)
<= 0

c_u_state_piecewise(0)_pw_t03_CC_constraint4(3)_:
+1 state_piecewise(0)_pw_t03_CC_lambda(3)
-1 state_piecewise(0)_pw_t03_CC_bin_y(2)
-1 state_piecewise(0)_pw_t03_CC_bin_y(3)
<= 0

c_u_state_piecewise(0)_pw_t03_CC_constraint4(4)_:
+1 state_piecewise(0)_pw_t03_CC_lambda(4)
-1 state_piecewise(0)_pw_t03_CC_bin_y(3)
<= 0

c_e_state_piecewise(0)_pw_t03_CC_constraint5_:
+1 state_piecewise(0)_pw_t03_CC_bin_y(1)
+1 state_piecewise(0)_pw_t03_CC_bin_y(2)
+1 state_piecewise(0)_pw_t03_CC_bin_y(3)
= 1

c_e_state_piecewise(0)_pw_t04_CC_constraint1_:
+1 state(reservoir_storage_t04)
-250.0 state_piecewise(0)_pw_t04_CC_lambda(1)
-350.0 state_piecewise(0)_pw_t04_CC_lambda(2)
-450.0 state_piecewise(0)_pw_t04_CC_lambda(3)
-600.0 state_piecewise(0)_pw_t04_CC_lambda(4)
= 0

c_e_state_piecewise(0)_pw_t04_CC_constraint2_:
+1 state(reservoir_level_t04)
-202.0 state_piecewise(0)_pw_t04_CC_lambda(1)
-208.0 state_piecewise(0)_pw_t04_CC_lambda(2)
-215.0 state_piecewise(0)_pw_t04_CC_lambda(3)
-225.0 state_piecewise(0)_pw_t04_CC_lambda(4)
= 0

c_e_state_piecewise(0)_pw_t04_CC_constraint3_:
+1 state_piecewise(0)_pw_t04_CC_lambda(1)
+1 state_piecewise(0)_pw_t04_CC_lambda(2)
+1 state_piecewise(0)_pw_t04_CC_lambda(3)
+1 state_piecewise(0)_pw_t04_CC_lambda(4)
= 1

c_u_state_piecewise(0)_pw_t04_CC_constraint4(1)_:
+1 state_piecewise(0)_pw_t04_CC_lambda(1)
-1 state_piecewise(0)_pw_t04_CC_bin_y(1)
<= 0

c_u_state_piecewise(0)_pw_t04_CC_constraint4(2)_:
+1 state_piecewise(0)_pw_t04_CC_lambda(2)
-1 state_piecewise(0)_pw_t04_CC_bin_y(1)
-1 state_piecewise(0)_pw_t04_CC_bin_y(2)
<= 0

c_u_state_piecewise(0)_pw_t04_CC_constraint4(3)_:
+1 state_piecewise(0)_pw_t04_CC_lambda(3)
-1 state_piecewise(0)_pw_t04_CC_bin_y(2)
-1 state_piecewise(0)_pw_t04_CC_bin_y(3)
<= 0

c_u_state_piecewise(0)_pw_t04_CC_constraint4(4)_:
+1 state_piecewise(0)_pw_t04_CC_lambda(4)
-1 state_piecewise(0)_pw_t04_CC_bin_y(3)
<= 0

c_e_state_piecewise(0)_pw_t04_CC_constraint5_:
+1 state_piecewise(0)_pw_t04_CC_bin_y(1)
+1 state_piecewise(0)_pw_t04_CC_bin_y(2)
+1 state_piecewise(0)_pw_t04_CC_bin_y(3)
= 1

c_e_state_piecewise(0)_pw_t05_CC_constraint1_:
+1 state(reservoir_storage_t05)
-250.0 state_piecewise(0)_pw_t05_CC_lambda(1)
-350.0 state_piecewise(0)_pw_t05_CC_lambda(2)
-450.0 state_piecewise(0)_pw_t05_CC_lambda(3)
-600.0 state_piecewise(0)_pw_t05_CC_lambda(4)
= 0

c_e_state_piecewise(0)_pw_t05_CC_constraint2_:
+1 state(reservoir_level_t05)
-202.0 state_piecewise(0)_pw_t05_CC_lambda(1)
-208.0 state_piecewise(0)_pw_t05_CC_lambda(2)
-215.0 state_piecewise(0)_pw_t05_CC_lambda(3)
-225.0 state_piecewise(0)_pw_t05_CC_lambda(4)
= 0

c_e_state_piecewise(0)_pw_t05_CC_constraint3_:
+1 state_piecewise(0)_pw_t05_CC_lambda(1)
+1 state_piecewise(0)_pw_t05_CC_lambda(2)
+1 state_piecewise(0)_pw_t05_CC_lambda(3)
+1 state_piecewise(0)_pw_t05_CC_lambda(4)
= 1

c_u_state_piecewise(0)_pw_t05_CC_constraint4(1)_:
+1 state_piecewise(0)_pw_t05_CC_lambda(1)
-1 state_piecewise(0)_pw_t05_CC_bin_y(1)
<= 0

c_u_state_piecewise(0)_pw_t05_CC_constraint4(2)_:
+1 state_piecewise(0)_pw_t05_CC_lambda(2)
-1 state_piecewise(0)_pw_t05_CC_bin_y(1)
-1 state_piecewise(0)_pw_t05_CC_bin_y(2)
<= 0

c_u_state_piecewise(0)_pw_t05_CC_constraint4(3)_:
+1 state_piecewise(0)_pw_t05_CC_lambda(3)
-1 state_piecewise(0)_pw_t05_CC_bin_y(2)
-1 state_piecewise(0)_pw_t05_CC_bin_y(3)
<= 0

c_u_state_piecewise(0)_pw_t05_CC_constraint4(4)_:
+1 state_piecewise(0)_pw_t05_CC_lambda(4)
-1 state_piecewise(0)_pw_t05_CC_bin_y(3)
<= 0

c_e_state_piecewise(0)_pw_t05_CC_constraint5_:
+1 state_piecewise(0)_pw_t05_CC_bin_y(1)
+1 state_piecewise(0)_pw_t05_CC_bin_y(2)
+1 state_piecewise(0)_pw_t05_CC_bin_y(3)
= 1

c_e_state_piecewise(0)_pw_t06_CC_constraint1_:
+1 state(reservoir_storage_t06)
-250.0 state_piecewise(0)_pw_t06_CC_lambda(1)
-350.0 state_piecewise(0)_pw_t06_CC_lambda(2)
-450.0 state_piecewise(0)_pw_t06_CC_lambda(3)
-600.0 state_piecewise(0)_pw_t06_CC_lambda(4)
= 0

c_e_state_piecewise(0)_pw_t06_CC_constraint2_:
+1 state(reservoir_level_t06)
-202.0 state_piecewise(0)_pw_t06_CC_lambda(1)
-208.0 state_piecewise(0)_pw_t06_CC_lambda(2)
-215.0 state_piecewise(0)_pw_t06_CC_lambda(3)
-225.0 state_piecewise(0)_pw_t06_CC_lambda(4)
= 0

c_e_state_piecewise(0)_pw_t06_CC_constraint3_:
+1 state_piecewise(0)_pw_t06_CC_lambda(1)
+1 state_piecewise(0)_pw_t06_CC_lambda(2)
+1 state_piecewise(0)_pw_t06_CC_lambda(3)
+1 state_piecewise(0)_pw_t06_CC_lambda(4)
= 1

c_u_state_piecewise(0)_pw_t06_CC_constraint4(1)_:
+1 state_piecewise(0)_pw_t06_CC_lambda(1)
-1 state_piecewise(0)_pw_t06_CC_bin_y(1)
<= 0

c_u_state_piecewise(0)_pw_t06_CC_constraint4(2)_:
+1 state_piecewise(0)_pw_t06_CC_lambda(2)
-1 state_piecewise(0)_pw_t06_CC_bin_y(1)
-1 state_piecewise(0)_pw_t06_CC_bin_y(2)
<= 0

c_u_state_piecewise(0)_pw_t06_CC_constraint4(3)_:
+1 state_piecewise(0)_pw_t06_CC_lambda(3)
-1 state_piecewise(0)_pw_t06_CC_bin_y(2)
-1 state_piecewise(0)_pw_t06_CC_bin_y(3)
<= 0

c_u_state_piecewise(0)_pw_t06_CC_constraint4(4)_:
+1 state_piecewise(0)_pw_t06_CC_lambda(4)
-1 state_piecewise(0)_pw_t06_CC_bin_y(3)
<= 0

c_e_state_piecewise(0)_pw_t06_CC_constraint5_:
+1 state_piecewise(0)_pw_t06_CC_bin_y(1)
+1 state_piecewise(0)_pw_t06_CC_bin_y(2)
+1 state_piecewise(0)_pw_t06_CC_bin_y(3)
= 1

c_e_state_piecewise(0)_pw_t07_CC_constraint1_:
+1 state(reservoir_storage_t07)
-250.0 state_piecewise(0)_pw_t07_CC_lambda(1)
-350.0 state_piecewise(0)_pw_t07_CC_lambda(2)
-450.0 state_piecewise(0)_pw_t07_CC_lambda(3)
-600.0 state_piecewise(0)_pw_t07_CC_lambda(4)
= 0

c_e_state_piecewise(0)_pw_t07_CC_constraint2_:
+1 state(reservoir_level_t07)
-202.0 state_piecewise(0)_pw_t07_CC_lambda(1)
-208.0 state_piecewise(0)_pw_t07_CC_lambda(2)
-215.0 state_piecewise(0)_pw_t07_CC_lambda(3)
-225.0 state_piecewise(0)_pw_t07_CC_lambda(4)
= 0

c_e_state_piecewise(0)_pw_t07_CC_constraint3_:
+1 state_piecewise(0)_pw_t07_CC_lambda(1)
+1 state_piecewise(0)_pw_t07_CC_lambda(2)
+1 state_piecewise(0)_pw_t07_CC_lambda(3)
+1 state_piecewise(0)_pw_t07_CC_lambda(4)
= 1

c_u_state_piecewise(0)_pw_t07_CC_constraint4(1)_:
+1 state_piecewise(0)_pw_t07_CC_lambda(1)
-1 state_piecewise(0)_pw_t07_CC_bin_y(1)
<= 0

c_u_state_piecewise(0)_pw_t07_CC_constraint4(2)_:
+1 state_piecewise(0)_pw_t07_CC_lambda(2)
-1 state_piecewise(0)_pw_t07_CC_bin_y(1)
-1 state_piecewise(0)_pw_t07_CC_bin_y(2)
<= 0

c_u_state_piecewise(0)_pw_t07_CC_constraint4(3)_:
+1 state_piecewise(0)_pw_t07_CC_lambda(3)
-1 state_piecewise(0)_pw_t07_CC_bin_y(2)
-1 state_piecewise(0)_pw_t07_CC_bin_y(3)
<= 0

c_u_state_piecewise(0)_pw_t07_CC_constraint4(4)_:
+1 state_piecewise(0)_pw_t07_CC_lambda(4)
-1 state_piecewise(0)_pw_t07_CC_bin_y(3)
<= 0

c_e_state_piecewise(0)_pw_t07_CC_constraint5_:
+1 state_piecewise(0)_pw_t07_CC_bin_y(1)
+1 state_piecewise(0)_pw_t07_CC_bin_y(2)
+1 state_piecewise(0)_pw_t07_CC_bin_y(3)
= 1

c_e_state_piecewise(0)_pw_t08_CC_constraint1_:
+1 state(reservoir_storage_t08)
-250.0 state_piecewise(0)_pw_t08_CC_lambda(1)
-350.0 state_piecewise(0)_pw_t08_CC_lambda(2)
-450.0 state_piecewise(0)_pw_t08_CC_lambda(3)
-600.0 state_piecewise(0)_pw_t08_CC_lambda(4)
= 0

c_e_state_piecewise(0)_pw_t08_CC_constraint2_:
+1 state(reservoir_level_t08)
-202.0 state_piecewise(0)_pw_t08_CC_lambda(1)
-208.0 state_piecewise(0)_pw_t08_CC_lambda(2)
-215.0 state_piecewise(0)_pw_t08_CC_lambda(3)
-225.0 state_piecewise(0)_pw_t08_CC_lambda(4)
= 0

c_e_state_piecewise(0)_pw_t08_CC_constraint3_:
+1 state_piecewise(0)_pw_t08_CC_lambda(1)
+1 state_piecewise(0)_pw_t08_CC_lambda(2)
+1 state_piecewise(0)_pw_t08_CC_lambda(3)
+1 state_piecewise(0)_pw_t08_CC_lambda(4)
= 1

c_u_state_piecewise(0)_pw_t08_CC_constraint4(1)_:
+1 state_piecewise(0)_pw_t08_CC_lambda(1)
-1 state_piecewise(0)_pw_t08_CC_bin_y(1)
<= 0

c_u_state_piecewise(0)_pw_t08_CC_constraint4(2)_:
+1 state_piecewise(0)_pw_t08_CC_lambda(2)
-1 state_piecewise(0)_pw_t08_CC_bin_y(1)
-1 state_piecewise(0)_pw_t08_CC_bin_y(2)
<= 0

c_u_state_piecewise(0)_pw_t08_CC_constraint4(3)_:
+1 state_piecewise(0)_pw_t08_CC_lambda(3)
-1 state_piecewise(0)_pw_t08_CC_bin_y(2)
-1 state_piecewise(0)_pw_t08_CC_bin_y(3)
<= 0

c_u_state_piecewise(0)_pw_t08_CC_constraint4(4)_:
+1 state_piecewise(0)_pw_t08_CC_lambda(4)
-1 state_piecewise(0)_pw_t08_CC_bin_y(3)
<= 0

c_e_state_piecewise(0)_pw_t08_CC_constraint5_:
+1 state_piecewise(0)_pw_t08_CC_bin_y(1)
+1 state_piecewise(0)_pw_t08_CC_bin_y(2)
+1 state_piecewise(0)_pw_t08_CC_bin_y(3)
= 1

c_e_state_piecewise(0)_pw_t09_CC_constraint1_:
+1 state(reservoir_storage_t09)
-250.0 state_piecewise(0)_pw_t09_CC_lambda(1)
-350.0 state_piecewise(0)_pw_t09_CC_lambda(2)
-450.0 state_piecewise(0)_pw_t09_CC_lambda(3)
-600.0 state_piecewise(0)_pw_t09_CC_lambda(4)
= 0

c_e_state_piecewise(0)_pw_t09_CC_constraint2_:
+1 state(reservoir_level_t09)
-202.0 state_piecewise(0)_pw_t09_CC_lambda(1)
-208.0 state_piecewise(0)_pw_t09_CC_lambda(2)
-215.0 state_piecewise(0)_pw_t09_CC_lambda(3)
-225.0 state_piecewise(0)_pw_t09_CC_lambda(4)
= 0

c_e_state_piecewise(0)_pw_t09_CC_constraint3_:
+1 state_piecewise(0)_pw_t09_CC_lambda(1)
+1 state_piecewise(0)_pw_t09_CC_lambda(2)
+1 state_piecewise(0)_pw_t09_CC_lambda(3)
+1 state_piecewise(0)_pw_t09_CC_lambda(4)
= 1

c_u_state_piecewise(0)_pw_t09_CC_constraint4(1)_:
+1 state_piecewise(0)_pw_t09_CC_lambda(1)
-1 state_piecewise(0)_pw_t09_CC_bin_y(1)
<= 0

c_u_state_piecewise(0)_pw_t09_CC_constraint4(2)_:
+1 state_piecewise(0)_pw_t09_CC_lambda(2)
-1 state_piecewise(0)_pw_t09_CC_bin_y(1)
-1 state_piecewise(0)_pw_t09_CC_bin_y(2)
<= 0

c_u_state_piecewise(0)_pw_t09_CC_constraint4(3)_:
+1 state_piecewise(0)_pw_t09_CC_lambda(3)
-1 state_piecewise(0)_pw_t09_CC_bin_y(2)
-1 state_piecewise(0)_pw_t09_CC_bin_y(3)
<= 0

c_u_state_piecewise(0)_pw_t09_CC_constraint4(4)_:
+1 state_piecewise(0)_pw_t09_CC_lambda(4)
-1 state_piecewise(0)_pw_t09_CC_bin_y(3)
<= 0

c_e_state_piecewise(0)_pw_t09_CC_constraint5_:
+1 state_piecewise(0)_pw_t09_CC_bin_y(1)
+1 state_piecewise(0)_pw_t09_CC_bin_y(2)
+1 state_piecewise(0)_pw_t09_CC_bin_y(3)
= 1

c_e_state_piecewise(0)_pw_t10_CC_constraint1_:
+1 state(reservoir_storage_t10)
-250.0 state_piecewise(0)_pw_t10_CC_lambda(1)
-350.0 state_piecewise(0)_pw_t10_CC_lambda(2)
-450.0 state_piecewise(0)_pw_t10_CC_lambda(3)
-600.0 state_piecewise(0)_pw_t10_CC_lambda(4)
= 0

c_e_state_piecewise(0)_pw_t10_CC_constraint2_:
+1 state(reservoir_level_t10)
-202.0 state_piecewise(0)_pw_t10_CC_lambda(1)
-208.0 state_piecewise(0)_pw_t10_CC_lambda(2)
-215.0 state_piecewise(0)_pw_t10_CC_lambda(3)
-225.0 state_piecewise(0)_pw_t10_CC_lambda(4)
= 0

c_e_state_piecewise(0)_pw_t10_CC_constraint3_:
+1 state_piecewise(0)_pw_t10_CC_lambda(1)
+1 state_piecewise(0)_pw_t10_CC_lambda(2)
+1 state_piecewise(0)_pw_t10_CC_lambda(3)
+1 state_piecewise(0)_pw_t10_CC_lambda(4)
= 1

c_u_state_piecewise(0)_pw_t10_CC_constraint4(1)_:
+1 state_piecewise(0)_pw_t10_CC_lambda(1)
-1 state_piecewise(0)_pw_t10_CC_bin_y(1)
<= 0

c_u_state_piecewise(0)_pw_t10_CC_constraint4(2)_:
+1 state_piecewise(0)_pw_t10_CC_lambda(2)
-1 state_piecewise(0)_pw_t10_CC_bin_y(1)
-1 state_piecewise(0)_pw_t10_CC_bin_y(2)
<= 0

c_u_state_piecewise(0)_pw_t10_CC_constraint4(3)_:
+1 state_piecewise(0)_pw_t10_CC_lambda(3)
-1 state_piecewise(0)_pw_t10_CC_bin_y(2)
-1 state_piecewise(0)_pw_t10_CC_bin_y(3)
<= 0

c_u_state_piecewise(0)_pw_t10_CC_constraint4(4)_:
+1 state_piecewise(0)_pw_t10_CC_lambda(4)
-1 state_piecewise(0)_pw_t10_CC_bin_y(3)
<= 0

c_e_state_piecewise(0)_pw_t10_CC_constraint5_:
+1 state_piecewise(0)_pw_t10_CC_bin_y(1)
+1 state_piecewise(0)_pw_t10_CC_bin_y(2)
+1 state_piecewise(0)_pw_t10_CC_bin_y(3)
= 1

c_e_state_piecewise(0)_pw_t11_CC_constraint1_:
+1 state(reservoir_storage_t11)
-250.0 state_piecewise(0)_pw_t11_CC_lambda(1)
-350.0 state_piecewise(0)_pw_t11_CC_lambda(2)
-450.0 state_piecewise(0)_pw_t11_CC_lambda(3)
-600.0 state_piecewise(0)_pw_t11_CC_lambda(4)
= 0

c_e_state_piecewise(0)_pw_t11_CC_constraint2_:
+1 state(reservoir_level_t11)
-202.0 state_piecewise(0)_pw_t11_CC_lambda(1)
-208.0 state_piecewise(0)_pw_t11_CC_lambda(2)
-215.0 state_piecewise(0)_pw_t11_CC_lambda(3)
-225.0 state_piecewise(0)_pw_t11_CC_lambda(4)
= 0

c_e_state_piecewise(0)_pw_t11_CC_constraint3_:
+1 state_piecewise(0)_pw_t11_CC_lambda(1)
+1 state_piecewise(0)_pw_t11_CC_lambda(2)
+1 state_piecewise(0)_pw_t11_CC_lambda(3)
+1 state_piecewise(0)_pw_t11_CC_lambda(4)
= 1

c_u_state_piecewise(0)_pw_t11_CC_constraint4(1)_:
+1 state_piecewise(0)_pw_t11_CC_lambda(1)
-1 state_piecewise(0)_pw_t11_CC_bin_y(1)
<= 0

c_u_state_piecewise(0)_pw_t11_CC_constraint4(2)_:
+1 state_piecewise(0)_pw_t11_CC_lambda(2)
-1 state_piecewise(0)_pw_t11_CC_bin_y(1)
-1 state_piecewise(0)_pw_t11_CC_bin_y(2)
<= 0

c_u_state_piecewise(0)_pw_t11_CC_constraint4(3)_:
+1 state_piecewise(0)_pw_t11_CC_lambda(3)
-1 state_piecewise(0)_pw_t11_CC_bin_y(2)
-1 state_piecewise(0)_pw_t11_CC_bin_y(3)
<= 0

c_u_state_piecewise(0)_pw_t11_CC_constraint4(4)_:
+1 state_piecewise(0)_pw_t11_CC_lambda(4)
-1 state_piecewise(0)_pw_t11_CC_bin_y(3)
<= 0

c_e_state_piecewise(0)_pw_t11_CC_constraint5_:
+1 state_piecewise(0)_pw_t11_CC_bin_y(1)
+1 state_piecewise(0)_pw_t11_CC_bin_y(2)
+1 state_piecewise(0)_pw_t11_CC_bin_y(3)
= 1

c_e_state_piecewise(0)_pw_t12_CC_constraint1_:
+1 state(reservoir_storage_t12)
-250.0 state_piecewise(0)_pw_t12_CC_lambda(1)
-350.0 state_piecewise(0)_pw_t12_CC_lambda(2)
-450.0 state_piecewise(0)_pw_t12_CC_lambda(3)
-600.0 state_piecewise(0)_pw_t12_CC_lambda(4)
= 0

c_e_state_piecewise(0)_pw_t12_CC_constraint2_:
+1 state(reservoir_level_t12)
-202.0 state_piecewise(0)_pw_t12_CC_lambda(1)
-208.0 state_piecewise(0)_pw_t12_CC_lambda(2)
-215.0 state_piecewise(0)_pw_t12_CC_lambda(3)
-225.0 state_piecewise(0)_pw_t12_CC_lambda(4)
= 0

c_e_state_piecewise(0)_pw_t12_CC_constraint3_:
+1 state_piecewise(0)_pw_t12_CC_lambda(1)
+1 state_piecewise(0)_pw_t12_CC_lambda(2)
+1 state_piecewise(0)_pw_t12_CC_lambda(3)
+1 state_piecewise(0)_pw_t12_CC_lambda(4)
= 1

c_u_state_piecewise(0)_pw_t12_CC_constraint4(1)_:
+1 state_piecewise(0)_pw_t12_CC_lambda(1)
-1 state_piecewise(0)_pw_t12_CC_bin_y(1)
<= 0

c_u_state_piecewise(0)_pw_t12_CC_constraint4(2)_:
+1 state_piecewise(0)_pw_t12_CC_lambda(2)
-1 state_piecewise(0)_pw_t12_CC_bin_y(1)
-1 state_piecewise(0)_pw_t12_CC_bin_y(2)
<= 0

c_u_state_piecewise(0)_pw_t12_CC_constraint4(3)_:
+1 state_piecewise(0)_pw_t12_CC_lambda(3)
-1 state_piecewise(0)_pw_t12_CC_bin_y(2)
-1 state_piecewise(0)_pw_t12_CC_bin_y(3)
<= 0

c_u_state_piecewise(0)_pw_t12_CC_constraint4(4)_:
+1 state_piecewise(0)_pw_t12_CC_lambda(4)
-1 state_piecewise(0)_pw_t12_CC_bin_y(3)
<= 0

c_e_state_piecewise(0)_pw_t12_CC_constraint5_:
+1 state_piecewise(0)_pw_t12_CC_bin_y(1)
+1 state_piecewise(0)_pw_t12_CC_bin_y(2)
+1 state_piecewise(0)_pw_t12_CC_bin_y(3)
= 1

c_e_state_piecewise(1)_pw_t01_CC_constraint1_:
+1 state(reservoir_storage_t01)
-250.0 state_piecewise(1)_pw_t01_CC_lambda(1)
-400.0 state_piecewise(1)_pw_t01_CC_lambda(2)
-550.0 state_piecewise(1)_pw_t01_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t01_CC_constraint2_:
+1 state(tailwater_stage_t01)
-112.0 state_piecewise(1)_pw_t01_CC_lambda(1)
-118.0 state_piecewise(1)_pw_t01_CC_lambda(2)
-130.0 state_piecewise(1)_pw_t01_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t01_CC_constraint3_:
+1 state_piecewise(1)_pw_t01_CC_lambda(1)
+1 state_piecewise(1)_pw_t01_CC_lambda(2)
+1 state_piecewise(1)_pw_t01_CC_lambda(3)
= 1

c_u_state_piecewise(1)_pw_t01_CC_constraint4(1)_:
+1 state_piecewise(1)_pw_t01_CC_lambda(1)
-1 state_piecewise(1)_pw_t01_CC_bin_y(1)
<= 0

c_u_state_piecewise(1)_pw_t01_CC_constraint4(2)_:
+1 state_piecewise(1)_pw_t01_CC_lambda(2)
-1 state_piecewise(1)_pw_t01_CC_bin_y(1)
-1 state_piecewise(1)_pw_t01_CC_bin_y(2)
<= 0

c_u_state_piecewise(1)_pw_t01_CC_constraint4(3)_:
+1 state_piecewise(1)_pw_t01_CC_lambda(3)
-1 state_piecewise(1)_pw_t01_CC_bin_y(2)
<= 0

c_e_state_piecewise(1)_pw_t01_CC_constraint5_:
+1 state_piecewise(1)_pw_t01_CC_bin_y(1)
+1 state_piecewise(1)_pw_t01_CC_bin_y(2)
= 1

c_e_state_piecewise(1)_pw_t02_CC_constraint1_:
+1 state(reservoir_storage_t02)
-250.0 state_piecewise(1)_pw_t02_CC_lambda(1)
-400.0 state_piecewise(1)_pw_t02_CC_lambda(2)
-550.0 state_piecewise(1)_pw_t02_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t02_CC_constraint2_:
+1 state(tailwater_stage_t02)
-112.0 state_piecewise(1)_pw_t02_CC_lambda(1)
-118.0 state_piecewise(1)_pw_t02_CC_lambda(2)
-130.0 state_piecewise(1)_pw_t02_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t02_CC_constraint3_:
+1 state_piecewise(1)_pw_t02_CC_lambda(1)
+1 state_piecewise(1)_pw_t02_CC_lambda(2)
+1 state_piecewise(1)_pw_t02_CC_lambda(3)
= 1

c_u_state_piecewise(1)_pw_t02_CC_constraint4(1)_:
+1 state_piecewise(1)_pw_t02_CC_lambda(1)
-1 state_piecewise(1)_pw_t02_CC_bin_y(1)
<= 0

c_u_state_piecewise(1)_pw_t02_CC_constraint4(2)_:
+1 state_piecewise(1)_pw_t02_CC_lambda(2)
-1 state_piecewise(1)_pw_t02_CC_bin_y(1)
-1 state_piecewise(1)_pw_t02_CC_bin_y(2)
<= 0

c_u_state_piecewise(1)_pw_t02_CC_constraint4(3)_:
+1 state_piecewise(1)_pw_t02_CC_lambda(3)
-1 state_piecewise(1)_pw_t02_CC_bin_y(2)
<= 0

c_e_state_piecewise(1)_pw_t02_CC_constraint5_:
+1 state_piecewise(1)_pw_t02_CC_bin_y(1)
+1 state_piecewise(1)_pw_t02_CC_bin_y(2)
= 1

c_e_state_piecewise(1)_pw_t03_CC_constraint1_:
+1 state(reservoir_storage_t03)
-250.0 state_piecewise(1)_pw_t03_CC_lambda(1)
-400.0 state_piecewise(1)_pw_t03_CC_lambda(2)
-550.0 state_piecewise(1)_pw_t03_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t03_CC_constraint2_:
+1 state(tailwater_stage_t03)
-112.0 state_piecewise(1)_pw_t03_CC_lambda(1)
-118.0 state_piecewise(1)_pw_t03_CC_lambda(2)
-130.0 state_piecewise(1)_pw_t03_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t03_CC_constraint3_:
+1 state_piecewise(1)_pw_t03_CC_lambda(1)
+1 state_piecewise(1)_pw_t03_CC_lambda(2)
+1 state_piecewise(1)_pw_t03_CC_lambda(3)
= 1

c_u_state_piecewise(1)_pw_t03_CC_constraint4(1)_:
+1 state_piecewise(1)_pw_t03_CC_lambda(1)
-1 state_piecewise(1)_pw_t03_CC_bin_y(1)
<= 0

c_u_state_piecewise(1)_pw_t03_CC_constraint4(2)_:
+1 state_piecewise(1)_pw_t03_CC_lambda(2)
-1 state_piecewise(1)_pw_t03_CC_bin_y(1)
-1 state_piecewise(1)_pw_t03_CC_bin_y(2)
<= 0

c_u_state_piecewise(1)_pw_t03_CC_constraint4(3)_:
+1 state_piecewise(1)_pw_t03_CC_lambda(3)
-1 state_piecewise(1)_pw_t03_CC_bin_y(2)
<= 0

c_e_state_piecewise(1)_pw_t03_CC_constraint5_:
+1 state_piecewise(1)_pw_t03_CC_bin_y(1)
+1 state_piecewise(1)_pw_t03_CC_bin_y(2)
= 1

c_e_state_piecewise(1)_pw_t04_CC_constraint1_:
+1 state(reservoir_storage_t04)
-250.0 state_piecewise(1)_pw_t04_CC_lambda(1)
-400.0 state_piecewise(1)_pw_t04_CC_lambda(2)
-550.0 state_piecewise(1)_pw_t04_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t04_CC_constraint2_:
+1 state(tailwater_stage_t04)
-112.0 state_piecewise(1)_pw_t04_CC_lambda(1)
-118.0 state_piecewise(1)_pw_t04_CC_lambda(2)
-130.0 state_piecewise(1)_pw_t04_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t04_CC_constraint3_:
+1 state_piecewise(1)_pw_t04_CC_lambda(1)
+1 state_piecewise(1)_pw_t04_CC_lambda(2)
+1 state_piecewise(1)_pw_t04_CC_lambda(3)
= 1

c_u_state_piecewise(1)_pw_t04_CC_constraint4(1)_:
+1 state_piecewise(1)_pw_t04_CC_lambda(1)
-1 state_piecewise(1)_pw_t04_CC_bin_y(1)
<= 0

c_u_state_piecewise(1)_pw_t04_CC_constraint4(2)_:
+1 state_piecewise(1)_pw_t04_CC_lambda(2)
-1 state_piecewise(1)_pw_t04_CC_bin_y(1)
-1 state_piecewise(1)_pw_t04_CC_bin_y(2)
<= 0

c_u_state_piecewise(1)_pw_t04_CC_constraint4(3)_:
+1 state_piecewise(1)_pw_t04_CC_lambda(3)
-1 state_piecewise(1)_pw_t04_CC_bin_y(2)
<= 0

c_e_state_piecewise(1)_pw_t04_CC_constraint5_:
+1 state_piecewise(1)_pw_t04_CC_bin_y(1)
+1 state_piecewise(1)_pw_t04_CC_bin_y(2)
= 1

c_e_state_piecewise(1)_pw_t05_CC_constraint1_:
+1 state(reservoir_storage_t05)
-250.0 state_piecewise(1)_pw_t05_CC_lambda(1)
-400.0 state_piecewise(1)_pw_t05_CC_lambda(2)
-550.0 state_piecewise(1)_pw_t05_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t05_CC_constraint2_:
+1 state(tailwater_stage_t05)
-112.0 state_piecewise(1)_pw_t05_CC_lambda(1)
-118.0 state_piecewise(1)_pw_t05_CC_lambda(2)
-130.0 state_piecewise(1)_pw_t05_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t05_CC_constraint3_:
+1 state_piecewise(1)_pw_t05_CC_lambda(1)
+1 state_piecewise(1)_pw_t05_CC_lambda(2)
+1 state_piecewise(1)_pw_t05_CC_lambda(3)
= 1

c_u_state_piecewise(1)_pw_t05_CC_constraint4(1)_:
+1 state_piecewise(1)_pw_t05_CC_lambda(1)
-1 state_piecewise(1)_pw_t05_CC_bin_y(1)
<= 0

c_u_state_piecewise(1)_pw_t05_CC_constraint4(2)_:
+1 state_piecewise(1)_pw_t05_CC_lambda(2)
-1 state_piecewise(1)_pw_t05_CC_bin_y(1)
-1 state_piecewise(1)_pw_t05_CC_bin_y(2)
<= 0

c_u_state_piecewise(1)_pw_t05_CC_constraint4(3)_:
+1 state_piecewise(1)_pw_t05_CC_lambda(3)
-1 state_piecewise(1)_pw_t05_CC_bin_y(2)
<= 0

c_e_state_piecewise(1)_pw_t05_CC_constraint5_:
+1 state_piecewise(1)_pw_t05_CC_bin_y(1)
+1 state_piecewise(1)_pw_t05_CC_bin_y(2)
= 1

c_e_state_piecewise(1)_pw_t06_CC_constraint1_:
+1 state(reservoir_storage_t06)
-250.0 state_piecewise(1)_pw_t06_CC_lambda(1)
-400.0 state_piecewise(1)_pw_t06_CC_lambda(2)
-550.0 state_piecewise(1)_pw_t06_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t06_CC_constraint2_:
+1 state(tailwater_stage_t06)
-112.0 state_piecewise(1)_pw_t06_CC_lambda(1)
-118.0 state_piecewise(1)_pw_t06_CC_lambda(2)
-130.0 state_piecewise(1)_pw_t06_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t06_CC_constraint3_:
+1 state_piecewise(1)_pw_t06_CC_lambda(1)
+1 state_piecewise(1)_pw_t06_CC_lambda(2)
+1 state_piecewise(1)_pw_t06_CC_lambda(3)
= 1

c_u_state_piecewise(1)_pw_t06_CC_constraint4(1)_:
+1 state_piecewise(1)_pw_t06_CC_lambda(1)
-1 state_piecewise(1)_pw_t06_CC_bin_y(1)
<= 0

c_u_state_piecewise(1)_pw_t06_CC_constraint4(2)_:
+1 state_piecewise(1)_pw_t06_CC_lambda(2)
-1 state_piecewise(1)_pw_t06_CC_bin_y(1)
-1 state_piecewise(1)_pw_t06_CC_bin_y(2)
<= 0

c_u_state_piecewise(1)_pw_t06_CC_constraint4(3)_:
+1 state_piecewise(1)_pw_t06_CC_lambda(3)
-1 state_piecewise(1)_pw_t06_CC_bin_y(2)
<= 0

c_e_state_piecewise(1)_pw_t06_CC_constraint5_:
+1 state_piecewise(1)_pw_t06_CC_bin_y(1)
+1 state_piecewise(1)_pw_t06_CC_bin_y(2)
= 1

c_e_state_piecewise(1)_pw_t07_CC_constraint1_:
+1 state(reservoir_storage_t07)
-250.0 state_piecewise(1)_pw_t07_CC_lambda(1)
-400.0 state_piecewise(1)_pw_t07_CC_lambda(2)
-550.0 state_piecewise(1)_pw_t07_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t07_CC_constraint2_:
+1 state(tailwater_stage_t07)
-112.0 state_piecewise(1)_pw_t07_CC_lambda(1)
-118.0 state_piecewise(1)_pw_t07_CC_lambda(2)
-130.0 state_piecewise(1)_pw_t07_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t07_CC_constraint3_:
+1 state_piecewise(1)_pw_t07_CC_lambda(1)
+1 state_piecewise(1)_pw_t07_CC_lambda(2)
+1 state_piecewise(1)_pw_t07_CC_lambda(3)
= 1

c_u_state_piecewise(1)_pw_t07_CC_constraint4(1)_:
+1 state_piecewise(1)_pw_t07_CC_lambda(1)
-1 state_piecewise(1)_pw_t07_CC_bin_y(1)
<= 0

c_u_state_piecewise(1)_pw_t07_CC_constraint4(2)_:
+1 state_piecewise(1)_pw_t07_CC_lambda(2)
-1 state_piecewise(1)_pw_t07_CC_bin_y(1)
-1 state_piecewise(1)_pw_t07_CC_bin_y(2)
<= 0

c_u_state_piecewise(1)_pw_t07_CC_constraint4(3)_:
+1 state_piecewise(1)_pw_t07_CC_lambda(3)
-1 state_piecewise(1)_pw_t07_CC_bin_y(2)
<= 0

c_e_state_piecewise(1)_pw_t07_CC_constraint5_:
+1 state_piecewise(1)_pw_t07_CC_bin_y(1)
+1 state_piecewise(1)_pw_t07_CC_bin_y(2)
= 1

c_e_state_piecewise(1)_pw_t08_CC_constraint1_:
+1 state(reservoir_storage_t08)
-250.0 state_piecewise(1)_pw_t08_CC_lambda(1)
-400.0 state_piecewise(1)_pw_t08_CC_lambda(2)
-550.0 state_piecewise(1)_pw_t08_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t08_CC_constraint2_:
+1 state(tailwater_stage_t08)
-112.0 state_piecewise(1)_pw_t08_CC_lambda(1)
-118.0 state_piecewise(1)_pw_t08_CC_lambda(2)
-130.0 state_piecewise(1)_pw_t08_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t08_CC_constraint3_:
+1 state_piecewise(1)_pw_t08_CC_lambda(1)
+1 state_piecewise(1)_pw_t08_CC_lambda(2)
+1 state_piecewise(1)_pw_t08_CC_lambda(3)
= 1

c_u_state_piecewise(1)_pw_t08_CC_constraint4(1)_:
+1 state_piecewise(1)_pw_t08_CC_lambda(1)
-1 state_piecewise(1)_pw_t08_CC_bin_y(1)
<= 0

c_u_state_piecewise(1)_pw_t08_CC_constraint4(2)_:
+1 state_piecewise(1)_pw_t08_CC_lambda(2)
-1 state_piecewise(1)_pw_t08_CC_bin_y(1)
-1 state_piecewise(1)_pw_t08_CC_bin_y(2)
<= 0

c_u_state_piecewise(1)_pw_t08_CC_constraint4(3)_:
+1 state_piecewise(1)_pw_t08_CC_lambda(3)
-1 state_piecewise(1)_pw_t08_CC_bin_y(2)
<= 0

c_e_state_piecewise(1)_pw_t08_CC_constraint5_:
+1 state_piecewise(1)_pw_t08_CC_bin_y(1)
+1 state_piecewise(1)_pw_t08_CC_bin_y(2)
= 1

c_e_state_piecewise(1)_pw_t09_CC_constraint1_:
+1 state(reservoir_storage_t09)
-250.0 state_piecewise(1)_pw_t09_CC_lambda(1)
-400.0 state_piecewise(1)_pw_t09_CC_lambda(2)
-550.0 state_piecewise(1)_pw_t09_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t09_CC_constraint2_:
+1 state(tailwater_stage_t09)
-112.0 state_piecewise(1)_pw_t09_CC_lambda(1)
-118.0 state_piecewise(1)_pw_t09_CC_lambda(2)
-130.0 state_piecewise(1)_pw_t09_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t09_CC_constraint3_:
+1 state_piecewise(1)_pw_t09_CC_lambda(1)
+1 state_piecewise(1)_pw_t09_CC_lambda(2)
+1 state_piecewise(1)_pw_t09_CC_lambda(3)
= 1

c_u_state_piecewise(1)_pw_t09_CC_constraint4(1)_:
+1 state_piecewise(1)_pw_t09_CC_lambda(1)
-1 state_piecewise(1)_pw_t09_CC_bin_y(1)
<= 0

c_u_state_piecewise(1)_pw_t09_CC_constraint4(2)_:
+1 state_piecewise(1)_pw_t09_CC_lambda(2)
-1 state_piecewise(1)_pw_t09_CC_bin_y(1)
-1 state_piecewise(1)_pw_t09_CC_bin_y(2)
<= 0

c_u_state_piecewise(1)_pw_t09_CC_constraint4(3)_:
+1 state_piecewise(1)_pw_t09_CC_lambda(3)
-1 state_piecewise(1)_pw_t09_CC_bin_y(2)
<= 0

c_e_state_piecewise(1)_pw_t09_CC_constraint5_:
+1 state_piecewise(1)_pw_t09_CC_bin_y(1)
+1 state_piecewise(1)_pw_t09_CC_bin_y(2)
= 1

c_e_state_piecewise(1)_pw_t10_CC_constraint1_:
+1 state(reservoir_storage_t10)
-250.0 state_piecewise(1)_pw_t10_CC_lambda(1)
-400.0 state_piecewise(1)_pw_t10_CC_lambda(2)
-550.0 state_piecewise(1)_pw_t10_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t10_CC_constraint2_:
+1 state(tailwater_stage_t10)
-112.0 state_piecewise(1)_pw_t10_CC_lambda(1)
-118.0 state_piecewise(1)_pw_t10_CC_lambda(2)
-130.0 state_piecewise(1)_pw_t10_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t10_CC_constraint3_:
+1 state_piecewise(1)_pw_t10_CC_lambda(1)
+1 state_piecewise(1)_pw_t10_CC_lambda(2)
+1 state_piecewise(1)_pw_t10_CC_lambda(3)
= 1

c_u_state_piecewise(1)_pw_t10_CC_constraint4(1)_:
+1 state_piecewise(1)_pw_t10_CC_lambda(1)
-1 state_piecewise(1)_pw_t10_CC_bin_y(1)
<= 0

c_u_state_piecewise(1)_pw_t10_CC_constraint4(2)_:
+1 state_piecewise(1)_pw_t10_CC_lambda(2)
-1 state_piecewise(1)_pw_t10_CC_bin_y(1)
-1 state_piecewise(1)_pw_t10_CC_bin_y(2)
<= 0

c_u_state_piecewise(1)_pw_t10_CC_constraint4(3)_:
+1 state_piecewise(1)_pw_t10_CC_lambda(3)
-1 state_piecewise(1)_pw_t10_CC_bin_y(2)
<= 0

c_e_state_piecewise(1)_pw_t10_CC_constraint5_:
+1 state_piecewise(1)_pw_t10_CC_bin_y(1)
+1 state_piecewise(1)_pw_t10_CC_bin_y(2)
= 1

c_e_state_piecewise(1)_pw_t11_CC_constraint1_:
+1 state(reservoir_storage_t11)
-250.0 state_piecewise(1)_pw_t11_CC_lambda(1)
-400.0 state_piecewise(1)_pw_t11_CC_lambda(2)
-550.0 state_piecewise(1)_pw_t11_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t11_CC_constraint2_:
+1 state(tailwater_stage_t11)
-112.0 state_piecewise(1)_pw_t11_CC_lambda(1)
-118.0 state_piecewise(1)_pw_t11_CC_lambda(2)
-130.0 state_piecewise(1)_pw_t11_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t11_CC_constraint3_:
+1 state_piecewise(1)_pw_t11_CC_lambda(1)
+1 state_piecewise(1)_pw_t11_CC_lambda(2)
+1 state_piecewise(1)_pw_t11_CC_lambda(3)
= 1

c_u_state_piecewise(1)_pw_t11_CC_constraint4(1)_:
+1 state_piecewise(1)_pw_t11_CC_lambda(1)
-1 state_piecewise(1)_pw_t11_CC_bin_y(1)
<= 0

c_u_state_piecewise(1)_pw_t11_CC_constraint4(2)_:
+1 state_piecewise(1)_pw_t11_CC_lambda(2)
-1 state_piecewise(1)_pw_t11_CC_bin_y(1)
-1 state_piecewise(1)_pw_t11_CC_bin_y(2)
<= 0

c_u_state_piecewise(1)_pw_t11_CC_constraint4(3)_:
+1 state_piecewise(1)_pw_t11_CC_lambda(3)
-1 state_piecewise(1)_pw_t11_CC_bin_y(2)
<= 0

c_e_state_piecewise(1)_pw_t11_CC_constraint5_:
+1 state_piecewise(1)_pw_t11_CC_bin_y(1)
+1 state_piecewise(1)_pw_t11_CC_bin_y(2)
= 1

c_e_state_piecewise(1)_pw_t12_CC_constraint1_:
+1 state(reservoir_storage_t12)
-250.0 state_piecewise(1)_pw_t12_CC_lambda(1)
-400.0 state_piecewise(1)_pw_t12_CC_lambda(2)
-550.0 state_piecewise(1)_pw_t12_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t12_CC_constraint2_:
+1 state(tailwater_stage_t12)
-112.0 state_piecewise(1)_pw_t12_CC_lambda(1)
-118.0 state_piecewise(1)_pw_t12_CC_lambda(2)
-130.0 state_piecewise(1)_pw_t12_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t12_CC_constraint3_:
+1 state_piecewise(1)_pw_t12_CC_lambda(1)
+1 state_piecewise(1)_pw_t12_CC_lambda(2)
+1 state_piecewise(1)_pw_t12_CC_lambda(3)
= 1

c_u_state_piecewise(1)_pw_t12_CC_constraint4(1)_:
+1 state_piecewise(1)_pw_t12_CC_lambda(1)
-1 state_piecewise(1)_pw_t12_CC_bin_y(1)
<= 0

c_u_state_piecewise(1)_pw_t12_CC_constraint4(2)_:
+1 state_piecewise(1)_pw_t12_CC_lambda(2)
-1 state_piecewise(1)_pw_t12_CC_bin_y(1)
-1 state_piecewise(1)_pw_t12_CC_bin_y(2)
<= 0

c_u_state_piecewise(1)_pw_t12_CC_constraint4(3)_:
+1 state_piecewise(1)_pw_t12_CC_lambda(3)
-1 state_piecewise(1)_pw_t12_CC_bin_y(2)
<= 0

c_e_state_piecewise(1)_pw_t12_CC_constraint5_:
+1 state_piecewise(1)_pw_t12_CC_bin_y(1)
+1 state_piecewise(1)_pw_t12_CC_bin_y(2)
= 1

c_e_gate_piecewise(reservoir_gate_t01)_CC_constraint1_:
+1 gate_opening(reservoir_gate_t01)
-0.4 gate_piecewise(reservoir_gate_t01)_CC_lambda(2)
-0.7 gate_piecewise(reservoir_gate_t01)_CC_lambda(3)
-1 gate_piecewise(reservoir_gate_t01)_CC_lambda(4)
= 0

c_e_gate_piecewise(reservoir_gate_t01)_CC_constraint2_:
+1 gate_capacity(reservoir_gate_t01)
-45.0 gate_piecewise(reservoir_gate_t01)_CC_lambda(2)
-80.0 gate_piecewise(reservoir_gate_t01)_CC_lambda(3)
-120.0 gate_piecewise(reservoir_gate_t01)_CC_lambda(4)
= 0

c_e_gate_piecewise(reservoir_gate_t01)_CC_constraint3_:
+1 gate_piecewise(reservoir_gate_t01)_CC_lambda(1)
+1 gate_piecewise(reservoir_gate_t01)_CC_lambda(2)
+1 gate_piecewise(reservoir_gate_t01)_CC_lambda(3)
+1 gate_piecewise(reservoir_gate_t01)_CC_lambda(4)
= 1

c_u_gate_piecewise(reservoir_gate_t01)_CC_constraint4(1)_:
+1 gate_piecewise(reservoir_gate_t01)_CC_lambda(1)
-1 gate_piecewise(reservoir_gate_t01)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(reservoir_gate_t01)_CC_constraint4(2)_:
+1 gate_piecewise(reservoir_gate_t01)_CC_lambda(2)
-1 gate_piecewise(reservoir_gate_t01)_CC_bin_y(1)
-1 gate_piecewise(reservoir_gate_t01)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(reservoir_gate_t01)_CC_constraint4(3)_:
+1 gate_piecewise(reservoir_gate_t01)_CC_lambda(3)
-1 gate_piecewise(reservoir_gate_t01)_CC_bin_y(2)
-1 gate_piecewise(reservoir_gate_t01)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(reservoir_gate_t01)_CC_constraint4(4)_:
+1 gate_piecewise(reservoir_gate_t01)_CC_lambda(4)
-1 gate_piecewise(reservoir_gate_t01)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(reservoir_gate_t01)_CC_constraint5_:
+1 gate_piecewise(reservoir_gate_t01)_CC_bin_y(1)
+1 gate_piecewise(reservoir_gate_t01)_CC_bin_y(2)
+1 gate_piecewise(reservoir_gate_t01)_CC_bin_y(3)
= 1

c_e_gate_piecewise(reservoir_gate_t02)_CC_constraint1_:
+1 gate_opening(reservoir_gate_t02)
-0.4 gate_piecewise(reservoir_gate_t02)_CC_lambda(2)
-0.7 gate_piecewise(reservoir_gate_t02)_CC_lambda(3)
-1 gate_piecewise(reservoir_gate_t02)_CC_lambda(4)
= 0

c_e_gate_piecewise(reservoir_gate_t02)_CC_constraint2_:
+1 gate_capacity(reservoir_gate_t02)
-45.0 gate_piecewise(reservoir_gate_t02)_CC_lambda(2)
-80.0 gate_piecewise(reservoir_gate_t02)_CC_lambda(3)
-120.0 gate_piecewise(reservoir_gate_t02)_CC_lambda(4)
= 0

c_e_gate_piecewise(reservoir_gate_t02)_CC_constraint3_:
+1 gate_piecewise(reservoir_gate_t02)_CC_lambda(1)
+1 gate_piecewise(reservoir_gate_t02)_CC_lambda(2)
+1 gate_piecewise(reservoir_gate_t02)_CC_lambda(3)
+1 gate_piecewise(reservoir_gate_t02)_CC_lambda(4)
= 1

c_u_gate_piecewise(reservoir_gate_t02)_CC_constraint4(1)_:
+1 gate_piecewise(reservoir_gate_t02)_CC_lambda(1)
-1 gate_piecewise(reservoir_gate_t02)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(reservoir_gate_t02)_CC_constraint4(2)_:
+1 gate_piecewise(reservoir_gate_t02)_CC_lambda(2)
-1 gate_piecewise(reservoir_gate_t02)_CC_bin_y(1)
-1 gate_piecewise(reservoir_gate_t02)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(reservoir_gate_t02)_CC_constraint4(3)_:
+1 gate_piecewise(reservoir_gate_t02)_CC_lambda(3)
-1 gate_piecewise(reservoir_gate_t02)_CC_bin_y(2)
-1 gate_piecewise(reservoir_gate_t02)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(reservoir_gate_t02)_CC_constraint4(4)_:
+1 gate_piecewise(reservoir_gate_t02)_CC_lambda(4)
-1 gate_piecewise(reservoir_gate_t02)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(reservoir_gate_t02)_CC_constraint5_:
+1 gate_piecewise(reservoir_gate_t02)_CC_bin_y(1)
+1 gate_piecewise(reservoir_gate_t02)_CC_bin_y(2)
+1 gate_piecewise(reservoir_gate_t02)_CC_bin_y(3)
= 1

c_e_gate_piecewise(reservoir_gate_t03)_CC_constraint1_:
+1 gate_opening(reservoir_gate_t03)
-0.4 gate_piecewise(reservoir_gate_t03)_CC_lambda(2)
-0.7 gate_piecewise(reservoir_gate_t03)_CC_lambda(3)
-1 gate_piecewise(reservoir_gate_t03)_CC_lambda(4)
= 0

c_e_gate_piecewise(reservoir_gate_t03)_CC_constraint2_:
+1 gate_capacity(reservoir_gate_t03)
-45.0 gate_piecewise(reservoir_gate_t03)_CC_lambda(2)
-80.0 gate_piecewise(reservoir_gate_t03)_CC_lambda(3)
-120.0 gate_piecewise(reservoir_gate_t03)_CC_lambda(4)
= 0

c_e_gate_piecewise(reservoir_gate_t03)_CC_constraint3_:
+1 gate_piecewise(reservoir_gate_t03)_CC_lambda(1)
+1 gate_piecewise(reservoir_gate_t03)_CC_lambda(2)
+1 gate_piecewise(reservoir_gate_t03)_CC_lambda(3)
+1 gate_piecewise(reservoir_gate_t03)_CC_lambda(4)
= 1

c_u_gate_piecewise(reservoir_gate_t03)_CC_constraint4(1)_:
+1 gate_piecewise(reservoir_gate_t03)_CC_lambda(1)
-1 gate_piecewise(reservoir_gate_t03)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(reservoir_gate_t03)_CC_constraint4(2)_:
+1 gate_piecewise(reservoir_gate_t03)_CC_lambda(2)
-1 gate_piecewise(reservoir_gate_t03)_CC_bin_y(1)
-1 gate_piecewise(reservoir_gate_t03)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(reservoir_gate_t03)_CC_constraint4(3)_:
+1 gate_piecewise(reservoir_gate_t03)_CC_lambda(3)
-1 gate_piecewise(reservoir_gate_t03)_CC_bin_y(2)
-1 gate_piecewise(reservoir_gate_t03)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(reservoir_gate_t03)_CC_constraint4(4)_:
+1 gate_piecewise(reservoir_gate_t03)_CC_lambda(4)
-1 gate_piecewise(reservoir_gate_t03)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(reservoir_gate_t03)_CC_constraint5_:
+1 gate_piecewise(reservoir_gate_t03)_CC_bin_y(1)
+1 gate_piecewise(reservoir_gate_t03)_CC_bin_y(2)
+1 gate_piecewise(reservoir_gate_t03)_CC_bin_y(3)
= 1

c_e_gate_piecewise(reservoir_gate_t04)_CC_constraint1_:
+1 gate_opening(reservoir_gate_t04)
-0.4 gate_piecewise(reservoir_gate_t04)_CC_lambda(2)
-0.7 gate_piecewise(reservoir_gate_t04)_CC_lambda(3)
-1 gate_piecewise(reservoir_gate_t04)_CC_lambda(4)
= 0

c_e_gate_piecewise(reservoir_gate_t04)_CC_constraint2_:
+1 gate_capacity(reservoir_gate_t04)
-45.0 gate_piecewise(reservoir_gate_t04)_CC_lambda(2)
-80.0 gate_piecewise(reservoir_gate_t04)_CC_lambda(3)
-120.0 gate_piecewise(reservoir_gate_t04)_CC_lambda(4)
= 0

c_e_gate_piecewise(reservoir_gate_t04)_CC_constraint3_:
+1 gate_piecewise(reservoir_gate_t04)_CC_lambda(1)
+1 gate_piecewise(reservoir_gate_t04)_CC_lambda(2)
+1 gate_piecewise(reservoir_gate_t04)_CC_lambda(3)
+1 gate_piecewise(reservoir_gate_t04)_CC_lambda(4)
= 1

c_u_gate_piecewise(reservoir_gate_t04)_CC_constraint4(1)_:
+1 gate_piecewise(reservoir_gate_t04)_CC_lambda(1)
-1 gate_piecewise(reservoir_gate_t04)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(reservoir_gate_t04)_CC_constraint4(2)_:
+1 gate_piecewise(reservoir_gate_t04)_CC_lambda(2)
-1 gate_piecewise(reservoir_gate_t04)_CC_bin_y(1)
-1 gate_piecewise(reservoir_gate_t04)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(reservoir_gate_t04)_CC_constraint4(3)_:
+1 gate_piecewise(reservoir_gate_t04)_CC_lambda(3)
-1 gate_piecewise(reservoir_gate_t04)_CC_bin_y(2)
-1 gate_piecewise(reservoir_gate_t04)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(reservoir_gate_t04)_CC_constraint4(4)_:
+1 gate_piecewise(reservoir_gate_t04)_CC_lambda(4)
-1 gate_piecewise(reservoir_gate_t04)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(reservoir_gate_t04)_CC_constraint5_:
+1 gate_piecewise(reservoir_gate_t04)_CC_bin_y(1)
+1 gate_piecewise(reservoir_gate_t04)_CC_bin_y(2)
+1 gate_piecewise(reservoir_gate_t04)_CC_bin_y(3)
= 1

c_e_gate_piecewise(reservoir_gate_t05)_CC_constraint1_:
+1 gate_opening(reservoir_gate_t05)
-0.4 gate_piecewise(reservoir_gate_t05)_CC_lambda(2)
-0.7 gate_piecewise(reservoir_gate_t05)_CC_lambda(3)
-1 gate_piecewise(reservoir_gate_t05)_CC_lambda(4)
= 0

c_e_gate_piecewise(reservoir_gate_t05)_CC_constraint2_:
+1 gate_capacity(reservoir_gate_t05)
-45.0 gate_piecewise(reservoir_gate_t05)_CC_lambda(2)
-80.0 gate_piecewise(reservoir_gate_t05)_CC_lambda(3)
-120.0 gate_piecewise(reservoir_gate_t05)_CC_lambda(4)
= 0

c_e_gate_piecewise(reservoir_gate_t05)_CC_constraint3_:
+1 gate_piecewise(reservoir_gate_t05)_CC_lambda(1)
+1 gate_piecewise(reservoir_gate_t05)_CC_lambda(2)
+1 gate_piecewise(reservoir_gate_t05)_CC_lambda(3)
+1 gate_piecewise(reservoir_gate_t05)_CC_lambda(4)
= 1

c_u_gate_piecewise(reservoir_gate_t05)_CC_constraint4(1)_:
+1 gate_piecewise(reservoir_gate_t05)_CC_lambda(1)
-1 gate_piecewise(reservoir_gate_t05)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(reservoir_gate_t05)_CC_constraint4(2)_:
+1 gate_piecewise(reservoir_gate_t05)_CC_lambda(2)
-1 gate_piecewise(reservoir_gate_t05)_CC_bin_y(1)
-1 gate_piecewise(reservoir_gate_t05)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(reservoir_gate_t05)_CC_constraint4(3)_:
+1 gate_piecewise(reservoir_gate_t05)_CC_lambda(3)
-1 gate_piecewise(reservoir_gate_t05)_CC_bin_y(2)
-1 gate_piecewise(reservoir_gate_t05)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(reservoir_gate_t05)_CC_constraint4(4)_:
+1 gate_piecewise(reservoir_gate_t05)_CC_lambda(4)
-1 gate_piecewise(reservoir_gate_t05)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(reservoir_gate_t05)_CC_constraint5_:
+1 gate_piecewise(reservoir_gate_t05)_CC_bin_y(1)
+1 gate_piecewise(reservoir_gate_t05)_CC_bin_y(2)
+1 gate_piecewise(reservoir_gate_t05)_CC_bin_y(3)
= 1

c_e_gate_piecewise(reservoir_gate_t06)_CC_constraint1_:
+1 gate_opening(reservoir_gate_t06)
-0.4 gate_piecewise(reservoir_gate_t06)_CC_lambda(2)
-0.7 gate_piecewise(reservoir_gate_t06)_CC_lambda(3)
-1 gate_piecewise(reservoir_gate_t06)_CC_lambda(4)
= 0

c_e_gate_piecewise(reservoir_gate_t06)_CC_constraint2_:
+1 gate_capacity(reservoir_gate_t06)
-45.0 gate_piecewise(reservoir_gate_t06)_CC_lambda(2)
-80.0 gate_piecewise(reservoir_gate_t06)_CC_lambda(3)
-120.0 gate_piecewise(reservoir_gate_t06)_CC_lambda(4)
= 0

c_e_gate_piecewise(reservoir_gate_t06)_CC_constraint3_:
+1 gate_piecewise(reservoir_gate_t06)_CC_lambda(1)
+1 gate_piecewise(reservoir_gate_t06)_CC_lambda(2)
+1 gate_piecewise(reservoir_gate_t06)_CC_lambda(3)
+1 gate_piecewise(reservoir_gate_t06)_CC_lambda(4)
= 1

c_u_gate_piecewise(reservoir_gate_t06)_CC_constraint4(1)_:
+1 gate_piecewise(reservoir_gate_t06)_CC_lambda(1)
-1 gate_piecewise(reservoir_gate_t06)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(reservoir_gate_t06)_CC_constraint4(2)_:
+1 gate_piecewise(reservoir_gate_t06)_CC_lambda(2)
-1 gate_piecewise(reservoir_gate_t06)_CC_bin_y(1)
-1 gate_piecewise(reservoir_gate_t06)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(reservoir_gate_t06)_CC_constraint4(3)_:
+1 gate_piecewise(reservoir_gate_t06)_CC_lambda(3)
-1 gate_piecewise(reservoir_gate_t06)_CC_bin_y(2)
-1 gate_piecewise(reservoir_gate_t06)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(reservoir_gate_t06)_CC_constraint4(4)_:
+1 gate_piecewise(reservoir_gate_t06)_CC_lambda(4)
-1 gate_piecewise(reservoir_gate_t06)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(reservoir_gate_t06)_CC_constraint5_:
+1 gate_piecewise(reservoir_gate_t06)_CC_bin_y(1)
+1 gate_piecewise(reservoir_gate_t06)_CC_bin_y(2)
+1 gate_piecewise(reservoir_gate_t06)_CC_bin_y(3)
= 1

c_e_gate_piecewise(reservoir_gate_t07)_CC_constraint1_:
+1 gate_opening(reservoir_gate_t07)
-0.4 gate_piecewise(reservoir_gate_t07)_CC_lambda(2)
-0.7 gate_piecewise(reservoir_gate_t07)_CC_lambda(3)
-1 gate_piecewise(reservoir_gate_t07)_CC_lambda(4)
= 0

c_e_gate_piecewise(reservoir_gate_t07)_CC_constraint2_:
+1 gate_capacity(reservoir_gate_t07)
-45.0 gate_piecewise(reservoir_gate_t07)_CC_lambda(2)
-80.0 gate_piecewise(reservoir_gate_t07)_CC_lambda(3)
-120.0 gate_piecewise(reservoir_gate_t07)_CC_lambda(4)
= 0

c_e_gate_piecewise(reservoir_gate_t07)_CC_constraint3_:
+1 gate_piecewise(reservoir_gate_t07)_CC_lambda(1)
+1 gate_piecewise(reservoir_gate_t07)_CC_lambda(2)
+1 gate_piecewise(reservoir_gate_t07)_CC_lambda(3)
+1 gate_piecewise(reservoir_gate_t07)_CC_lambda(4)
= 1

c_u_gate_piecewise(reservoir_gate_t07)_CC_constraint4(1)_:
+1 gate_piecewise(reservoir_gate_t07)_CC_lambda(1)
-1 gate_piecewise(reservoir_gate_t07)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(reservoir_gate_t07)_CC_constraint4(2)_:
+1 gate_piecewise(reservoir_gate_t07)_CC_lambda(2)
-1 gate_piecewise(reservoir_gate_t07)_CC_bin_y(1)
-1 gate_piecewise(reservoir_gate_t07)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(reservoir_gate_t07)_CC_constraint4(3)_:
+1 gate_piecewise(reservoir_gate_t07)_CC_lambda(3)
-1 gate_piecewise(reservoir_gate_t07)_CC_bin_y(2)
-1 gate_piecewise(reservoir_gate_t07)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(reservoir_gate_t07)_CC_constraint4(4)_:
+1 gate_piecewise(reservoir_gate_t07)_CC_lambda(4)
-1 gate_piecewise(reservoir_gate_t07)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(reservoir_gate_t07)_CC_constraint5_:
+1 gate_piecewise(reservoir_gate_t07)_CC_bin_y(1)
+1 gate_piecewise(reservoir_gate_t07)_CC_bin_y(2)
+1 gate_piecewise(reservoir_gate_t07)_CC_bin_y(3)
= 1

c_e_gate_piecewise(reservoir_gate_t08)_CC_constraint1_:
+1 gate_opening(reservoir_gate_t08)
-0.4 gate_piecewise(reservoir_gate_t08)_CC_lambda(2)
-0.7 gate_piecewise(reservoir_gate_t08)_CC_lambda(3)
-1 gate_piecewise(reservoir_gate_t08)_CC_lambda(4)
= 0

c_e_gate_piecewise(reservoir_gate_t08)_CC_constraint2_:
+1 gate_capacity(reservoir_gate_t08)
-45.0 gate_piecewise(reservoir_gate_t08)_CC_lambda(2)
-80.0 gate_piecewise(reservoir_gate_t08)_CC_lambda(3)
-120.0 gate_piecewise(reservoir_gate_t08)_CC_lambda(4)
= 0

c_e_gate_piecewise(reservoir_gate_t08)_CC_constraint3_:
+1 gate_piecewise(reservoir_gate_t08)_CC_lambda(1)
+1 gate_piecewise(reservoir_gate_t08)_CC_lambda(2)
+1 gate_piecewise(reservoir_gate_t08)_CC_lambda(3)
+1 gate_piecewise(reservoir_gate_t08)_CC_lambda(4)
= 1

c_u_gate_piecewise(reservoir_gate_t08)_CC_constraint4(1)_:
+1 gate_piecewise(reservoir_gate_t08)_CC_lambda(1)
-1 gate_piecewise(reservoir_gate_t08)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(reservoir_gate_t08)_CC_constraint4(2)_:
+1 gate_piecewise(reservoir_gate_t08)_CC_lambda(2)
-1 gate_piecewise(reservoir_gate_t08)_CC_bin_y(1)
-1 gate_piecewise(reservoir_gate_t08)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(reservoir_gate_t08)_CC_constraint4(3)_:
+1 gate_piecewise(reservoir_gate_t08)_CC_lambda(3)
-1 gate_piecewise(reservoir_gate_t08)_CC_bin_y(2)
-1 gate_piecewise(reservoir_gate_t08)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(reservoir_gate_t08)_CC_constraint4(4)_:
+1 gate_piecewise(reservoir_gate_t08)_CC_lambda(4)
-1 gate_piecewise(reservoir_gate_t08)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(reservoir_gate_t08)_CC_constraint5_:
+1 gate_piecewise(reservoir_gate_t08)_CC_bin_y(1)
+1 gate_piecewise(reservoir_gate_t08)_CC_bin_y(2)
+1 gate_piecewise(reservoir_gate_t08)_CC_bin_y(3)
= 1

c_e_gate_piecewise(reservoir_gate_t09)_CC_constraint1_:
+1 gate_opening(reservoir_gate_t09)
-0.4 gate_piecewise(reservoir_gate_t09)_CC_lambda(2)
-0.7 gate_piecewise(reservoir_gate_t09)_CC_lambda(3)
-1 gate_piecewise(reservoir_gate_t09)_CC_lambda(4)
= 0

c_e_gate_piecewise(reservoir_gate_t09)_CC_constraint2_:
+1 gate_capacity(reservoir_gate_t09)
-45.0 gate_piecewise(reservoir_gate_t09)_CC_lambda(2)
-80.0 gate_piecewise(reservoir_gate_t09)_CC_lambda(3)
-120.0 gate_piecewise(reservoir_gate_t09)_CC_lambda(4)
= 0

c_e_gate_piecewise(reservoir_gate_t09)_CC_constraint3_:
+1 gate_piecewise(reservoir_gate_t09)_CC_lambda(1)
+1 gate_piecewise(reservoir_gate_t09)_CC_lambda(2)
+1 gate_piecewise(reservoir_gate_t09)_CC_lambda(3)
+1 gate_piecewise(reservoir_gate_t09)_CC_lambda(4)
= 1

c_u_gate_piecewise(reservoir_gate_t09)_CC_constraint4(1)_:
+1 gate_piecewise(reservoir_gate_t09)_CC_lambda(1)
-1 gate_piecewise(reservoir_gate_t09)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(reservoir_gate_t09)_CC_constraint4(2)_:
+1 gate_piecewise(reservoir_gate_t09)_CC_lambda(2)
-1 gate_piecewise(reservoir_gate_t09)_CC_bin_y(1)
-1 gate_piecewise(reservoir_gate_t09)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(reservoir_gate_t09)_CC_constraint4(3)_:
+1 gate_piecewise(reservoir_gate_t09)_CC_lambda(3)
-1 gate_piecewise(reservoir_gate_t09)_CC_bin_y(2)
-1 gate_piecewise(reservoir_gate_t09)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(reservoir_gate_t09)_CC_constraint4(4)_:
+1 gate_piecewise(reservoir_gate_t09)_CC_lambda(4)
-1 gate_piecewise(reservoir_gate_t09)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(reservoir_gate_t09)_CC_constraint5_:
+1 gate_piecewise(reservoir_gate_t09)_CC_bin_y(1)
+1 gate_piecewise(reservoir_gate_t09)_CC_bin_y(2)
+1 gate_piecewise(reservoir_gate_t09)_CC_bin_y(3)
= 1

c_e_gate_piecewise(reservoir_gate_t10)_CC_constraint1_:
+1 gate_opening(reservoir_gate_t10)
-0.4 gate_piecewise(reservoir_gate_t10)_CC_lambda(2)
-0.7 gate_piecewise(reservoir_gate_t10)_CC_lambda(3)
-1 gate_piecewise(reservoir_gate_t10)_CC_lambda(4)
= 0

c_e_gate_piecewise(reservoir_gate_t10)_CC_constraint2_:
+1 gate_capacity(reservoir_gate_t10)
-45.0 gate_piecewise(reservoir_gate_t10)_CC_lambda(2)
-80.0 gate_piecewise(reservoir_gate_t10)_CC_lambda(3)
-120.0 gate_piecewise(reservoir_gate_t10)_CC_lambda(4)
= 0

c_e_gate_piecewise(reservoir_gate_t10)_CC_constraint3_:
+1 gate_piecewise(reservoir_gate_t10)_CC_lambda(1)
+1 gate_piecewise(reservoir_gate_t10)_CC_lambda(2)
+1 gate_piecewise(reservoir_gate_t10)_CC_lambda(3)
+1 gate_piecewise(reservoir_gate_t10)_CC_lambda(4)
= 1

c_u_gate_piecewise(reservoir_gate_t10)_CC_constraint4(1)_:
+1 gate_piecewise(reservoir_gate_t10)_CC_lambda(1)
-1 gate_piecewise(reservoir_gate_t10)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(reservoir_gate_t10)_CC_constraint4(2)_:
+1 gate_piecewise(reservoir_gate_t10)_CC_lambda(2)
-1 gate_piecewise(reservoir_gate_t10)_CC_bin_y(1)
-1 gate_piecewise(reservoir_gate_t10)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(reservoir_gate_t10)_CC_constraint4(3)_:
+1 gate_piecewise(reservoir_gate_t10)_CC_lambda(3)
-1 gate_piecewise(reservoir_gate_t10)_CC_bin_y(2)
-1 gate_piecewise(reservoir_gate_t10)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(reservoir_gate_t10)_CC_constraint4(4)_:
+1 gate_piecewise(reservoir_gate_t10)_CC_lambda(4)
-1 gate_piecewise(reservoir_gate_t10)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(reservoir_gate_t10)_CC_constraint5_:
+1 gate_piecewise(reservoir_gate_t10)_CC_bin_y(1)
+1 gate_piecewise(reservoir_gate_t10)_CC_bin_y(2)
+1 gate_piecewise(reservoir_gate_t10)_CC_bin_y(3)
= 1

c_e_gate_piecewise(reservoir_gate_t11)_CC_constraint1_:
+1 gate_opening(reservoir_gate_t11)
-0.4 gate_piecewise(reservoir_gate_t11)_CC_lambda(2)
-0.7 gate_piecewise(reservoir_gate_t11)_CC_lambda(3)
-1 gate_piecewise(reservoir_gate_t11)_CC_lambda(4)
= 0

c_e_gate_piecewise(reservoir_gate_t11)_CC_constraint2_:
+1 gate_capacity(reservoir_gate_t11)
-45.0 gate_piecewise(reservoir_gate_t11)_CC_lambda(2)
-80.0 gate_piecewise(reservoir_gate_t11)_CC_lambda(3)
-120.0 gate_piecewise(reservoir_gate_t11)_CC_lambda(4)
= 0

c_e_gate_piecewise(reservoir_gate_t11)_CC_constraint3_:
+1 gate_piecewise(reservoir_gate_t11)_CC_lambda(1)
+1 gate_piecewise(reservoir_gate_t11)_CC_lambda(2)
+1 gate_piecewise(reservoir_gate_t11)_CC_lambda(3)
+1 gate_piecewise(reservoir_gate_t11)_CC_lambda(4)
= 1

c_u_gate_piecewise(reservoir_gate_t11)_CC_constraint4(1)_:
+1 gate_piecewise(reservoir_gate_t11)_CC_lambda(1)
-1 gate_piecewise(reservoir_gate_t11)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(reservoir_gate_t11)_CC_constraint4(2)_:
+1 gate_piecewise(reservoir_gate_t11)_CC_lambda(2)
-1 gate_piecewise(reservoir_gate_t11)_CC_bin_y(1)
-1 gate_piecewise(reservoir_gate_t11)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(reservoir_gate_t11)_CC_constraint4(3)_:
+1 gate_piecewise(reservoir_gate_t11)_CC_lambda(3)
-1 gate_piecewise(reservoir_gate_t11)_CC_bin_y(2)
-1 gate_piecewise(reservoir_gate_t11)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(reservoir_gate_t11)_CC_constraint4(4)_:
+1 gate_piecewise(reservoir_gate_t11)_CC_lambda(4)
-1 gate_piecewise(reservoir_gate_t11)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(reservoir_gate_t11)_CC_constraint5_:
+1 gate_piecewise(reservoir_gate_t11)_CC_bin_y(1)
+1 gate_piecewise(reservoir_gate_t11)_CC_bin_y(2)
+1 gate_piecewise(reservoir_gate_t11)_CC_bin_y(3)
= 1

c_e_gate_piecewise(reservoir_gate_t12)_CC_constraint1_:
+1 gate_opening(reservoir_gate_t12)
-0.4 gate_piecewise(reservoir_gate_t12)_CC_lambda(2)
-0.7 gate_piecewise(reservoir_gate_t12)_CC_lambda(3)
-1 gate_piecewise(reservoir_gate_t12)_CC_lambda(4)
= 0

c_e_gate_piecewise(reservoir_gate_t12)_CC_constraint2_:
+1 gate_capacity(reservoir_gate_t12)
-45.0 gate_piecewise(reservoir_gate_t12)_CC_lambda(2)
-80.0 gate_piecewise(reservoir_gate_t12)_CC_lambda(3)
-120.0 gate_piecewise(reservoir_gate_t12)_CC_lambda(4)
= 0

c_e_gate_piecewise(reservoir_gate_t12)_CC_constraint3_:
+1 gate_piecewise(reservoir_gate_t12)_CC_lambda(1)
+1 gate_piecewise(reservoir_gate_t12)_CC_lambda(2)
+1 gate_piecewise(reservoir_gate_t12)_CC_lambda(3)
+1 gate_piecewise(reservoir_gate_t12)_CC_lambda(4)
= 1

c_u_gate_piecewise(reservoir_gate_t12)_CC_constraint4(1)_:
+1 gate_piecewise(reservoir_gate_t12)_CC_lambda(1)
-1 gate_piecewise(reservoir_gate_t12)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(reservoir_gate_t12)_CC_constraint4(2)_:
+1 gate_piecewise(reservoir_gate_t12)_CC_lambda(2)
-1 gate_piecewise(reservoir_gate_t12)_CC_bin_y(1)
-1 gate_piecewise(reservoir_gate_t12)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(reservoir_gate_t12)_CC_constraint4(3)_:
+1 gate_piecewise(reservoir_gate_t12)_CC_lambda(3)
-1 gate_piecewise(reservoir_gate_t12)_CC_bin_y(2)
-1 gate_piecewise(reservoir_gate_t12)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(reservoir_gate_t12)_CC_constraint4(4)_:
+1 gate_piecewise(reservoir_gate_t12)_CC_lambda(4)
-1 gate_piecewise(reservoir_gate_t12)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(reservoir_gate_t12)_CC_constraint5_:
+1 gate_piecewise(reservoir_gate_t12)_CC_bin_y(1)
+1 gate_piecewise(reservoir_gate_t12)_CC_bin_y(2)
+1 gate_piecewise(reservoir_gate_t12)_CC_bin_y(3)
= 1

c_e_sos2_piecewise(tailwater_pump_t01)_CC_constraint1_:
+1 flow(tailwater_pump_t01)
-31.499999999999996 sos2_piecewise(tailwater_pump_t01)_CC_lambda(2)
-62.99999999999999 sos2_piecewise(tailwater_pump_t01)_CC_lambda(3)
-90.0 sos2_piecewise(tailwater_pump_t01)_CC_lambda(4)
= 0

c_e_sos2_piecewise(tailwater_pump_t01)_CC_constraint2_:
+1 sos2_cost(tailwater_pump_t01)
-188.99999999999997 sos2_piecewise(tailwater_pump_t01)_CC_lambda(2)
-692.9999999999999 sos2_piecewise(tailwater_pump_t01)_CC_lambda(3)
-1485.0 sos2_piecewise(tailwater_pump_t01)_CC_lambda(4)
= 0

c_e_sos2_piecewise(tailwater_pump_t01)_CC_constraint3_:
+1 sos2_piecewise(tailwater_pump_t01)_CC_lambda(1)
+1 sos2_piecewise(tailwater_pump_t01)_CC_lambda(2)
+1 sos2_piecewise(tailwater_pump_t01)_CC_lambda(3)
+1 sos2_piecewise(tailwater_pump_t01)_CC_lambda(4)
= 1

c_u_sos2_piecewise(tailwater_pump_t01)_CC_constraint4(1)_:
+1 sos2_piecewise(tailwater_pump_t01)_CC_lambda(1)
-1 sos2_piecewise(tailwater_pump_t01)_CC_bin_y(1)
<= 0

c_u_sos2_piecewise(tailwater_pump_t01)_CC_constraint4(2)_:
+1 sos2_piecewise(tailwater_pump_t01)_CC_lambda(2)
-1 sos2_piecewise(tailwater_pump_t01)_CC_bin_y(1)
-1 sos2_piecewise(tailwater_pump_t01)_CC_bin_y(2)
<= 0

c_u_sos2_piecewise(tailwater_pump_t01)_CC_constraint4(3)_:
+1 sos2_piecewise(tailwater_pump_t01)_CC_lambda(3)
-1 sos2_piecewise(tailwater_pump_t01)_CC_bin_y(2)
-1 sos2_piecewise(tailwater_pump_t01)_CC_bin_y(3)
<= 0

c_u_sos2_piecewise(tailwater_pump_t01)_CC_constraint4(4)_:
+1 sos2_piecewise(tailwater_pump_t01)_CC_lambda(4)
-1 sos2_piecewise(tailwater_pump_t01)_CC_bin_y(3)
<= 0

c_e_sos2_piecewise(tailwater_pump_t01)_CC_constraint5_:
+1 sos2_piecewise(tailwater_pump_t01)_CC_bin_y(1)
+1 sos2_piecewise(tailwater_pump_t01)_CC_bin_y(2)
+1 sos2_piecewise(tailwater_pump_t01)_CC_bin_y(3)
= 1

c_e_sos2_piecewise(tailwater_pump_t02)_CC_constraint1_:
+1 flow(tailwater_pump_t02)
-31.499999999999996 sos2_piecewise(tailwater_pump_t02)_CC_lambda(2)
-62.99999999999999 sos2_piecewise(tailwater_pump_t02)_CC_lambda(3)
-90.0 sos2_piecewise(tailwater_pump_t02)_CC_lambda(4)
= 0

c_e_sos2_piecewise(tailwater_pump_t02)_CC_constraint2_:
+1 sos2_cost(tailwater_pump_t02)
-188.99999999999997 sos2_piecewise(tailwater_pump_t02)_CC_lambda(2)
-692.9999999999999 sos2_piecewise(tailwater_pump_t02)_CC_lambda(3)
-1485.0 sos2_piecewise(tailwater_pump_t02)_CC_lambda(4)
= 0

c_e_sos2_piecewise(tailwater_pump_t02)_CC_constraint3_:
+1 sos2_piecewise(tailwater_pump_t02)_CC_lambda(1)
+1 sos2_piecewise(tailwater_pump_t02)_CC_lambda(2)
+1 sos2_piecewise(tailwater_pump_t02)_CC_lambda(3)
+1 sos2_piecewise(tailwater_pump_t02)_CC_lambda(4)
= 1

c_u_sos2_piecewise(tailwater_pump_t02)_CC_constraint4(1)_:
+1 sos2_piecewise(tailwater_pump_t02)_CC_lambda(1)
-1 sos2_piecewise(tailwater_pump_t02)_CC_bin_y(1)
<= 0

c_u_sos2_piecewise(tailwater_pump_t02)_CC_constraint4(2)_:
+1 sos2_piecewise(tailwater_pump_t02)_CC_lambda(2)
-1 sos2_piecewise(tailwater_pump_t02)_CC_bin_y(1)
-1 sos2_piecewise(tailwater_pump_t02)_CC_bin_y(2)
<= 0

c_u_sos2_piecewise(tailwater_pump_t02)_CC_constraint4(3)_:
+1 sos2_piecewise(tailwater_pump_t02)_CC_lambda(3)
-1 sos2_piecewise(tailwater_pump_t02)_CC_bin_y(2)
-1 sos2_piecewise(tailwater_pump_t02)_CC_bin_y(3)
<= 0

c_u_sos2_piecewise(tailwater_pump_t02)_CC_constraint4(4)_:
+1 sos2_piecewise(tailwater_pump_t02)_CC_lambda(4)
-1 sos2_piecewise(tailwater_pump_t02)_CC_bin_y(3)
<= 0

c_e_sos2_piecewise(tailwater_pump_t02)_CC_constraint5_:
+1 sos2_piecewise(tailwater_pump_t02)_CC_bin_y(1)
+1 sos2_piecewise(tailwater_pump_t02)_CC_bin_y(2)
+1 sos2_piecewise(tailwater_pump_t02)_CC_bin_y(3)
= 1

c_e_sos2_piecewise(tailwater_pump_t03)_CC_constraint1_:
+1 flow(tailwater_pump_t03)
-31.499999999999996 sos2_piecewise(tailwater_pump_t03)_CC_lambda(2)
-62.99999999999999 sos2_piecewise(tailwater_pump_t03)_CC_lambda(3)
-90.0 sos2_piecewise(tailwater_pump_t03)_CC_lambda(4)
= 0

c_e_sos2_piecewise(tailwater_pump_t03)_CC_constraint2_:
+1 sos2_cost(tailwater_pump_t03)
-188.99999999999997 sos2_piecewise(tailwater_pump_t03)_CC_lambda(2)
-692.9999999999999 sos2_piecewise(tailwater_pump_t03)_CC_lambda(3)
-1485.0 sos2_piecewise(tailwater_pump_t03)_CC_lambda(4)
= 0

c_e_sos2_piecewise(tailwater_pump_t03)_CC_constraint3_:
+1 sos2_piecewise(tailwater_pump_t03)_CC_lambda(1)
+1 sos2_piecewise(tailwater_pump_t03)_CC_lambda(2)
+1 sos2_piecewise(tailwater_pump_t03)_CC_lambda(3)
+1 sos2_piecewise(tailwater_pump_t03)_CC_lambda(4)
= 1

c_u_sos2_piecewise(tailwater_pump_t03)_CC_constraint4(1)_:
+1 sos2_piecewise(tailwater_pump_t03)_CC_lambda(1)
-1 sos2_piecewise(tailwater_pump_t03)_CC_bin_y(1)
<= 0

c_u_sos2_piecewise(tailwater_pump_t03)_CC_constraint4(2)_:
+1 sos2_piecewise(tailwater_pump_t03)_CC_lambda(2)
-1 sos2_piecewise(tailwater_pump_t03)_CC_bin_y(1)
-1 sos2_piecewise(tailwater_pump_t03)_CC_bin_y(2)
<= 0

c_u_sos2_piecewise(tailwater_pump_t03)_CC_constraint4(3)_:
+1 sos2_piecewise(tailwater_pump_t03)_CC_lambda(3)
-1 sos2_piecewise(tailwater_pump_t03)_CC_bin_y(2)
-1 sos2_piecewise(tailwater_pump_t03)_CC_bin_y(3)
<= 0

c_u_sos2_piecewise(tailwater_pump_t03)_CC_constraint4(4)_:
+1 sos2_piecewise(tailwater_pump_t03)_CC_lambda(4)
-1 sos2_piecewise(tailwater_pump_t03)_CC_bin_y(3)
<= 0

c_e_sos2_piecewise(tailwater_pump_t03)_CC_constraint5_:
+1 sos2_piecewise(tailwater_pump_t03)_CC_bin_y(1)
+1 sos2_piecewise(tailwater_pump_t03)_CC_bin_y(2)
+1 sos2_piecewise(tailwater_pump_t03)_CC_bin_y(3)
= 1

c_e_sos2_piecewise(tailwater_pump_t04)_CC_constraint1_:
+1 flow(tailwater_pump_t04)
-31.499999999999996 sos2_piecewise(tailwater_pump_t04)_CC_lambda(2)
-62.99999999999999 sos2_piecewise(tailwater_pump_t04)_CC_lambda(3)
-90.0 sos2_piecewise(tailwater_pump_t04)_CC_lambda(4)
= 0

c_e_sos2_piecewise(tailwater_pump_t04)_CC_constraint2_:
+1 sos2_cost(tailwater_pump_t04)
-188.99999999999997 sos2_piecewise(tailwater_pump_t04)_CC_lambda(2)
-692.9999999999999 sos2_piecewise(tailwater_pump_t04)_CC_lambda(3)
-1485.0 sos2_piecewise(tailwater_pump_t04)_CC_lambda(4)
= 0

c_e_sos2_piecewise(tailwater_pump_t04)_CC_constraint3_:
+1 sos2_piecewise(tailwater_pump_t04)_CC_lambda(1)
+1 sos2_piecewise(tailwater_pump_t04)_CC_lambda(2)
+1 sos2_piecewise(tailwater_pump_t04)_CC_lambda(3)
+1 sos2_piecewise(tailwater_pump_t04)_CC_lambda(4)
= 1

c_u_sos2_piecewise(tailwater_pump_t04)_CC_constraint4(1)_:
+1 sos2_piecewise(tailwater_pump_t04)_CC_lambda(1)
-1 sos2_piecewise(tailwater_pump_t04)_CC_bin_y(1)
<= 0

c_u_sos2_piecewise(tailwater_pump_t04)_CC_constraint4(2)_:
+1 sos2_piecewise(tailwater_pump_t04)_CC_lambda(2)
-1 sos2_piecewise(tailwater_pump_t04)_CC_bin_y(1)
-1 sos2_piecewise(tailwater_pump_t04)_CC_bin_y(2)
<= 0

c_u_sos2_piecewise(tailwater_pump_t04)_CC_constraint4(3)_:
+1 sos2_piecewise(tailwater_pump_t04)_CC_lambda(3)
-1 sos2_piecewise(tailwater_pump_t04)_CC_bin_y(2)
-1 sos2_piecewise(tailwater_pump_t04)_CC_bin_y(3)
<= 0

c_u_sos2_piecewise(tailwater_pump_t04)_CC_constraint4(4)_:
+1 sos2_piecewise(tailwater_pump_t04)_CC_lambda(4)
-1 sos2_piecewise(tailwater_pump_t04)_CC_bin_y(3)
<= 0

c_e_sos2_piecewise(tailwater_pump_t04)_CC_constraint5_:
+1 sos2_piecewise(tailwater_pump_t04)_CC_bin_y(1)
+1 sos2_piecewise(tailwater_pump_t04)_CC_bin_y(2)
+1 sos2_piecewise(tailwater_pump_t04)_CC_bin_y(3)
= 1

c_e_sos2_piecewise(tailwater_pump_t05)_CC_constraint1_:
+1 flow(tailwater_pump_t05)
-31.499999999999996 sos2_piecewise(tailwater_pump_t05)_CC_lambda(2)
-62.99999999999999 sos2_piecewise(tailwater_pump_t05)_CC_lambda(3)
-90.0 sos2_piecewise(tailwater_pump_t05)_CC_lambda(4)
= 0

c_e_sos2_piecewise(tailwater_pump_t05)_CC_constraint2_:
+1 sos2_cost(tailwater_pump_t05)
-188.99999999999997 sos2_piecewise(tailwater_pump_t05)_CC_lambda(2)
-692.9999999999999 sos2_piecewise(tailwater_pump_t05)_CC_lambda(3)
-1485.0 sos2_piecewise(tailwater_pump_t05)_CC_lambda(4)
= 0

c_e_sos2_piecewise(tailwater_pump_t05)_CC_constraint3_:
+1 sos2_piecewise(tailwater_pump_t05)_CC_lambda(1)
+1 sos2_piecewise(tailwater_pump_t05)_CC_lambda(2)
+1 sos2_piecewise(tailwater_pump_t05)_CC_lambda(3)
+1 sos2_piecewise(tailwater_pump_t05)_CC_lambda(4)
= 1

c_u_sos2_piecewise(tailwater_pump_t05)_CC_constraint4(1)_:
+1 sos2_piecewise(tailwater_pump_t05)_CC_lambda(1)
-1 sos2_piecewise(tailwater_pump_t05)_CC_bin_y(1)
<= 0

c_u_sos2_piecewise(tailwater_pump_t05)_CC_constraint4(2)_:
+1 sos2_piecewise(tailwater_pump_t05)_CC_lambda(2)
-1 sos2_piecewise(tailwater_pump_t05)_CC_bin_y(1)
-1 sos2_piecewise(tailwater_pump_t05)_CC_bin_y(2)
<= 0

c_u_sos2_piecewise(tailwater_pump_t05)_CC_constraint4(3)_:
+1 sos2_piecewise(tailwater_pump_t05)_CC_lambda(3)
-1 sos2_piecewise(tailwater_pump_t05)_CC_bin_y(2)
-1 sos2_piecewise(tailwater_pump_t05)_CC_bin_y(3)
<= 0

c_u_sos2_piecewise(tailwater_pump_t05)_CC_constraint4(4)_:
+1 sos2_piecewise(tailwater_pump_t05)_CC_lambda(4)
-1 sos2_piecewise(tailwater_pump_t05)_CC_bin_y(3)
<= 0

c_e_sos2_piecewise(tailwater_pump_t05)_CC_constraint5_:
+1 sos2_piecewise(tailwater_pump_t05)_CC_bin_y(1)
+1 sos2_piecewise(tailwater_pump_t05)_CC_bin_y(2)
+1 sos2_piecewise(tailwater_pump_t05)_CC_bin_y(3)
= 1

c_e_sos2_piecewise(tailwater_pump_t06)_CC_constraint1_:
+1 flow(tailwater_pump_t06)
-31.499999999999996 sos2_piecewise(tailwater_pump_t06)_CC_lambda(2)
-62.99999999999999 sos2_piecewise(tailwater_pump_t06)_CC_lambda(3)
-90.0 sos2_piecewise(tailwater_pump_t06)_CC_lambda(4)
= 0

c_e_sos2_piecewise(tailwater_pump_t06)_CC_constraint2_:
+1 sos2_cost(tailwater_pump_t06)
-188.99999999999997 sos2_piecewise(tailwater_pump_t06)_CC_lambda(2)
-692.9999999999999 sos2_piecewise(tailwater_pump_t06)_CC_lambda(3)
-1485.0 sos2_piecewise(tailwater_pump_t06)_CC_lambda(4)
= 0

c_e_sos2_piecewise(tailwater_pump_t06)_CC_constraint3_:
+1 sos2_piecewise(tailwater_pump_t06)_CC_lambda(1)
+1 sos2_piecewise(tailwater_pump_t06)_CC_lambda(2)
+1 sos2_piecewise(tailwater_pump_t06)_CC_lambda(3)
+1 sos2_piecewise(tailwater_pump_t06)_CC_lambda(4)
= 1

c_u_sos2_piecewise(tailwater_pump_t06)_CC_constraint4(1)_:
+1 sos2_piecewise(tailwater_pump_t06)_CC_lambda(1)
-1 sos2_piecewise(tailwater_pump_t06)_CC_bin_y(1)
<= 0

c_u_sos2_piecewise(tailwater_pump_t06)_CC_constraint4(2)_:
+1 sos2_piecewise(tailwater_pump_t06)_CC_lambda(2)
-1 sos2_piecewise(tailwater_pump_t06)_CC_bin_y(1)
-1 sos2_piecewise(tailwater_pump_t06)_CC_bin_y(2)
<= 0

c_u_sos2_piecewise(tailwater_pump_t06)_CC_constraint4(3)_:
+1 sos2_piecewise(tailwater_pump_t06)_CC_lambda(3)
-1 sos2_piecewise(tailwater_pump_t06)_CC_bin_y(2)
-1 sos2_piecewise(tailwater_pump_t06)_CC_bin_y(3)
<= 0

c_u_sos2_piecewise(tailwater_pump_t06)_CC_constraint4(4)_:
+1 sos2_piecewise(tailwater_pump_t06)_CC_lambda(4)
-1 sos2_piecewise(tailwater_pump_t06)_CC_bin_y(3)
<= 0

c_e_sos2_piecewise(tailwater_pump_t06)_CC_constraint5_:
+1 sos2_piecewise(tailwater_pump_t06)_CC_bin_y(1)
+1 sos2_piecewise(tailwater_pump_t06)_CC_bin_y(2)
+1 sos2_piecewise(tailwater_pump_t06)_CC_bin_y(3)
= 1

c_e_sos2_piecewise(tailwater_pump_t07)_CC_constraint1_:
+1 flow(tailwater_pump_t07)
-31.499999999999996 sos2_piecewise(tailwater_pump_t07)_CC_lambda(2)
-62.99999999999999 sos2_piecewise(tailwater_pump_t07)_CC_lambda(3)
-90.0 sos2_piecewise(tailwater_pump_t07)_CC_lambda(4)
= 0

c_e_sos2_piecewise(tailwater_pump_t07)_CC_constraint2_:
+1 sos2_cost(tailwater_pump_t07)
-188.99999999999997 sos2_piecewise(tailwater_pump_t07)_CC_lambda(2)
-692.9999999999999 sos2_piecewise(tailwater_pump_t07)_CC_lambda(3)
-1485.0 sos2_piecewise(tailwater_pump_t07)_CC_lambda(4)
= 0

c_e_sos2_piecewise(tailwater_pump_t07)_CC_constraint3_:
+1 sos2_piecewise(tailwater_pump_t07)_CC_lambda(1)
+1 sos2_piecewise(tailwater_pump_t07)_CC_lambda(2)
+1 sos2_piecewise(tailwater_pump_t07)_CC_lambda(3)
+1 sos2_piecewise(tailwater_pump_t07)_CC_lambda(4)
= 1

c_u_sos2_piecewise(tailwater_pump_t07)_CC_constraint4(1)_:
+1 sos2_piecewise(tailwater_pump_t07)_CC_lambda(1)
-1 sos2_piecewise(tailwater_pump_t07)_CC_bin_y(1)
<= 0

c_u_sos2_piecewise(tailwater_pump_t07)_CC_constraint4(2)_:
+1 sos2_piecewise(tailwater_pump_t07)_CC_lambda(2)
-1 sos2_piecewise(tailwater_pump_t07)_CC_bin_y(1)
-1 sos2_piecewise(tailwater_pump_t07)_CC_bin_y(2)
<= 0

c_u_sos2_piecewise(tailwater_pump_t07)_CC_constraint4(3)_:
+1 sos2_piecewise(tailwater_pump_t07)_CC_lambda(3)
-1 sos2_piecewise(tailwater_pump_t07)_CC_bin_y(2)
-1 sos2_piecewise(tailwater_pump_t07)_CC_bin_y(3)
<= 0

c_u_sos2_piecewise(tailwater_pump_t07)_CC_constraint4(4)_:
+1 sos2_piecewise(tailwater_pump_t07)_CC_lambda(4)
-1 sos2_piecewise(tailwater_pump_t07)_CC_bin_y(3)
<= 0

c_e_sos2_piecewise(tailwater_pump_t07)_CC_constraint5_:
+1 sos2_piecewise(tailwater_pump_t07)_CC_bin_y(1)
+1 sos2_piecewise(tailwater_pump_t07)_CC_bin_y(2)
+1 sos2_piecewise(tailwater_pump_t07)_CC_bin_y(3)
= 1

c_e_sos2_piecewise(tailwater_pump_t08)_CC_constraint1_:
+1 flow(tailwater_pump_t08)
-31.499999999999996 sos2_piecewise(tailwater_pump_t08)_CC_lambda(2)
-62.99999999999999 sos2_piecewise(tailwater_pump_t08)_CC_lambda(3)
-90.0 sos2_piecewise(tailwater_pump_t08)_CC_lambda(4)
= 0

c_e_sos2_piecewise(tailwater_pump_t08)_CC_constraint2_:
+1 sos2_cost(tailwater_pump_t08)
-188.99999999999997 sos2_piecewise(tailwater_pump_t08)_CC_lambda(2)
-692.9999999999999 sos2_piecewise(tailwater_pump_t08)_CC_lambda(3)
-1485.0 sos2_piecewise(tailwater_pump_t08)_CC_lambda(4)
= 0

c_e_sos2_piecewise(tailwater_pump_t08)_CC_constraint3_:
+1 sos2_piecewise(tailwater_pump_t08)_CC_lambda(1)
+1 sos2_piecewise(tailwater_pump_t08)_CC_lambda(2)
+1 sos2_piecewise(tailwater_pump_t08)_CC_lambda(3)
+1 sos2_piecewise(tailwater_pump_t08)_CC_lambda(4)
= 1

c_u_sos2_piecewise(tailwater_pump_t08)_CC_constraint4(1)_:
+1 sos2_piecewise(tailwater_pump_t08)_CC_lambda(1)
-1 sos2_piecewise(tailwater_pump_t08)_CC_bin_y(1)
<= 0

c_u_sos2_piecewise(tailwater_pump_t08)_CC_constraint4(2)_:
+1 sos2_piecewise(tailwater_pump_t08)_CC_lambda(2)
-1 sos2_piecewise(tailwater_pump_t08)_CC_bin_y(1)
-1 sos2_piecewise(tailwater_pump_t08)_CC_bin_y(2)
<= 0

c_u_sos2_piecewise(tailwater_pump_t08)_CC_constraint4(3)_:
+1 sos2_piecewise(tailwater_pump_t08)_CC_lambda(3)
-1 sos2_piecewise(tailwater_pump_t08)_CC_bin_y(2)
-1 sos2_piecewise(tailwater_pump_t08)_CC_bin_y(3)
<= 0

c_u_sos2_piecewise(tailwater_pump_t08)_CC_constraint4(4)_:
+1 sos2_piecewise(tailwater_pump_t08)_CC_lambda(4)
-1 sos2_piecewise(tailwater_pump_t08)_CC_bin_y(3)
<= 0

c_e_sos2_piecewise(tailwater_pump_t08)_CC_constraint5_:
+1 sos2_piecewise(tailwater_pump_t08)_CC_bin_y(1)
+1 sos2_piecewise(tailwater_pump_t08)_CC_bin_y(2)
+1 sos2_piecewise(tailwater_pump_t08)_CC_bin_y(3)
= 1

c_e_sos2_piecewise(tailwater_pump_t09)_CC_constraint1_:
+1 flow(tailwater_pump_t09)
-31.499999999999996 sos2_piecewise(tailwater_pump_t09)_CC_lambda(2)
-62.99999999999999 sos2_piecewise(tailwater_pump_t09)_CC_lambda(3)
-90.0 sos2_piecewise(tailwater_pump_t09)_CC_lambda(4)
= 0

c_e_sos2_piecewise(tailwater_pump_t09)_CC_constraint2_:
+1 sos2_cost(tailwater_pump_t09)
-188.99999999999997 sos2_piecewise(tailwater_pump_t09)_CC_lambda(2)
-692.9999999999999 sos2_piecewise(tailwater_pump_t09)_CC_lambda(3)
-1485.0 sos2_piecewise(tailwater_pump_t09)_CC_lambda(4)
= 0

c_e_sos2_piecewise(tailwater_pump_t09)_CC_constraint3_:
+1 sos2_piecewise(tailwater_pump_t09)_CC_lambda(1)
+1 sos2_piecewise(tailwater_pump_t09)_CC_lambda(2)
+1 sos2_piecewise(tailwater_pump_t09)_CC_lambda(3)
+1 sos2_piecewise(tailwater_pump_t09)_CC_lambda(4)
= 1

c_u_sos2_piecewise(tailwater_pump_t09)_CC_constraint4(1)_:
+1 sos2_piecewise(tailwater_pump_t09)_CC_lambda(1)
-1 sos2_piecewise(tailwater_pump_t09)_CC_bin_y(1)
<= 0

c_u_sos2_piecewise(tailwater_pump_t09)_CC_constraint4(2)_:
+1 sos2_piecewise(tailwater_pump_t09)_CC_lambda(2)
-1 sos2_piecewise(tailwater_pump_t09)_CC_bin_y(1)
-1 sos2_piecewise(tailwater_pump_t09)_CC_bin_y(2)
<= 0

c_u_sos2_piecewise(tailwater_pump_t09)_CC_constraint4(3)_:
+1 sos2_piecewise(tailwater_pump_t09)_CC_lambda(3)
-1 sos2_piecewise(tailwater_pump_t09)_CC_bin_y(2)
-1 sos2_piecewise(tailwater_pump_t09)_CC_bin_y(3)
<= 0

c_u_sos2_piecewise(tailwater_pump_t09)_CC_constraint4(4)_:
+1 sos2_piecewise(tailwater_pump_t09)_CC_lambda(4)
-1 sos2_piecewise(tailwater_pump_t09)_CC_bin_y(3)
<= 0

c_e_sos2_piecewise(tailwater_pump_t09)_CC_constraint5_:
+1 sos2_piecewise(tailwater_pump_t09)_CC_bin_y(1)
+1 sos2_piecewise(tailwater_pump_t09)_CC_bin_y(2)
+1 sos2_piecewise(tailwater_pump_t09)_CC_bin_y(3)
= 1

c_e_sos2_piecewise(tailwater_pump_t10)_CC_constraint1_:
+1 flow(tailwater_pump_t10)
-31.499999999999996 sos2_piecewise(tailwater_pump_t10)_CC_lambda(2)
-62.99999999999999 sos2_piecewise(tailwater_pump_t10)_CC_lambda(3)
-90.0 sos2_piecewise(tailwater_pump_t10)_CC_lambda(4)
= 0

c_e_sos2_piecewise(tailwater_pump_t10)_CC_constraint2_:
+1 sos2_cost(tailwater_pump_t10)
-188.99999999999997 sos2_piecewise(tailwater_pump_t10)_CC_lambda(2)
-692.9999999999999 sos2_piecewise(tailwater_pump_t10)_CC_lambda(3)
-1485.0 sos2_piecewise(tailwater_pump_t10)_CC_lambda(4)
= 0

c_e_sos2_piecewise(tailwater_pump_t10)_CC_constraint3_:
+1 sos2_piecewise(tailwater_pump_t10)_CC_lambda(1)
+1 sos2_piecewise(tailwater_pump_t10)_CC_lambda(2)
+1 sos2_piecewise(tailwater_pump_t10)_CC_lambda(3)
+1 sos2_piecewise(tailwater_pump_t10)_CC_lambda(4)
= 1

c_u_sos2_piecewise(tailwater_pump_t10)_CC_constraint4(1)_:
+1 sos2_piecewise(tailwater_pump_t10)_CC_lambda(1)
-1 sos2_piecewise(tailwater_pump_t10)_CC_bin_y(1)
<= 0

c_u_sos2_piecewise(tailwater_pump_t10)_CC_constraint4(2)_:
+1 sos2_piecewise(tailwater_pump_t10)_CC_lambda(2)
-1 sos2_piecewise(tailwater_pump_t10)_CC_bin_y(1)
-1 sos2_piecewise(tailwater_pump_t10)_CC_bin_y(2)
<= 0

c_u_sos2_piecewise(tailwater_pump_t10)_CC_constraint4(3)_:
+1 sos2_piecewise(tailwater_pump_t10)_CC_lambda(3)
-1 sos2_piecewise(tailwater_pump_t10)_CC_bin_y(2)
-1 sos2_piecewise(tailwater_pump_t10)_CC_bin_y(3)
<= 0

c_u_sos2_piecewise(tailwater_pump_t10)_CC_constraint4(4)_:
+1 sos2_piecewise(tailwater_pump_t10)_CC_lambda(4)
-1 sos2_piecewise(tailwater_pump_t10)_CC_bin_y(3)
<= 0

c_e_sos2_piecewise(tailwater_pump_t10)_CC_constraint5_:
+1 sos2_piecewise(tailwater_pump_t10)_CC_bin_y(1)
+1 sos2_piecewise(tailwater_pump_t10)_CC_bin_y(2)
+1 sos2_piecewise(tailwater_pump_t10)_CC_bin_y(3)
= 1

c_e_sos2_piecewise(tailwater_pump_t11)_CC_constraint1_:
+1 flow(tailwater_pump_t11)
-31.499999999999996 sos2_piecewise(tailwater_pump_t11)_CC_lambda(2)
-62.99999999999999 sos2_piecewise(tailwater_pump_t11)_CC_lambda(3)
-90.0 sos2_piecewise(tailwater_pump_t11)_CC_lambda(4)
= 0

c_e_sos2_piecewise(tailwater_pump_t11)_CC_constraint2_:
+1 sos2_cost(tailwater_pump_t11)
-188.99999999999997 sos2_piecewise(tailwater_pump_t11)_CC_lambda(2)
-692.9999999999999 sos2_piecewise(tailwater_pump_t11)_CC_lambda(3)
-1485.0 sos2_piecewise(tailwater_pump_t11)_CC_lambda(4)
= 0

c_e_sos2_piecewise(tailwater_pump_t11)_CC_constraint3_:
+1 sos2_piecewise(tailwater_pump_t11)_CC_lambda(1)
+1 sos2_piecewise(tailwater_pump_t11)_CC_lambda(2)
+1 sos2_piecewise(tailwater_pump_t11)_CC_lambda(3)
+1 sos2_piecewise(tailwater_pump_t11)_CC_lambda(4)
= 1

c_u_sos2_piecewise(tailwater_pump_t11)_CC_constraint4(1)_:
+1 sos2_piecewise(tailwater_pump_t11)_CC_lambda(1)
-1 sos2_piecewise(tailwater_pump_t11)_CC_bin_y(1)
<= 0

c_u_sos2_piecewise(tailwater_pump_t11)_CC_constraint4(2)_:
+1 sos2_piecewise(tailwater_pump_t11)_CC_lambda(2)
-1 sos2_piecewise(tailwater_pump_t11)_CC_bin_y(1)
-1 sos2_piecewise(tailwater_pump_t11)_CC_bin_y(2)
<= 0

c_u_sos2_piecewise(tailwater_pump_t11)_CC_constraint4(3)_:
+1 sos2_piecewise(tailwater_pump_t11)_CC_lambda(3)
-1 sos2_piecewise(tailwater_pump_t11)_CC_bin_y(2)
-1 sos2_piecewise(tailwater_pump_t11)_CC_bin_y(3)
<= 0

c_u_sos2_piecewise(tailwater_pump_t11)_CC_constraint4(4)_:
+1 sos2_piecewise(tailwater_pump_t11)_CC_lambda(4)
-1 sos2_piecewise(tailwater_pump_t11)_CC_bin_y(3)
<= 0

c_e_sos2_piecewise(tailwater_pump_t11)_CC_constraint5_:
+1 sos2_piecewise(tailwater_pump_t11)_CC_bin_y(1)
+1 sos2_piecewise(tailwater_pump_t11)_CC_bin_y(2)
+1 sos2_piecewise(tailwater_pump_t11)_CC_bin_y(3)
= 1

c_e_sos2_piecewise(tailwater_pump_t12)_CC_constraint1_:
+1 flow(tailwater_pump_t12)
-31.499999999999996 sos2_piecewise(tailwater_pump_t12)_CC_lambda(2)
-62.99999999999999 sos2_piecewise(tailwater_pump_t12)_CC_lambda(3)
-90.0 sos2_piecewise(tailwater_pump_t12)_CC_lambda(4)
= 0

c_e_sos2_piecewise(tailwater_pump_t12)_CC_constraint2_:
+1 sos2_cost(tailwater_pump_t12)
-188.99999999999997 sos2_piecewise(tailwater_pump_t12)_CC_lambda(2)
-692.9999999999999 sos2_piecewise(tailwater_pump_t12)_CC_lambda(3)
-1485.0 sos2_piecewise(tailwater_pump_t12)_CC_lambda(4)
= 0

c_e_sos2_piecewise(tailwater_pump_t12)_CC_constraint3_:
+1 sos2_piecewise(tailwater_pump_t12)_CC_lambda(1)
+1 sos2_piecewise(tailwater_pump_t12)_CC_lambda(2)
+1 sos2_piecewise(tailwater_pump_t12)_CC_lambda(3)
+1 sos2_piecewise(tailwater_pump_t12)_CC_lambda(4)
= 1

c_u_sos2_piecewise(tailwater_pump_t12)_CC_constraint4(1)_:
+1 sos2_piecewise(tailwater_pump_t12)_CC_lambda(1)
-1 sos2_piecewise(tailwater_pump_t12)_CC_bin_y(1)
<= 0

c_u_sos2_piecewise(tailwater_pump_t12)_CC_constraint4(2)_:
+1 sos2_piecewise(tailwater_pump_t12)_CC_lambda(2)
-1 sos2_piecewise(tailwater_pump_t12)_CC_bin_y(1)
-1 sos2_piecewise(tailwater_pump_t12)_CC_bin_y(2)
<= 0

c_u_sos2_piecewise(tailwater_pump_t12)_CC_constraint4(3)_:
+1 sos2_piecewise(tailwater_pump_t12)_CC_lambda(3)
-1 sos2_piecewise(tailwater_pump_t12)_CC_bin_y(2)
-1 sos2_piecewise(tailwater_pump_t12)_CC_bin_y(3)
<= 0

c_u_sos2_piecewise(tailwater_pump_t12)_CC_constraint4(4)_:
+1 sos2_piecewise(tailwater_pump_t12)_CC_lambda(4)
-1 sos2_piecewise(tailwater_pump_t12)_CC_bin_y(3)
<= 0

c_e_sos2_piecewise(tailwater_pump_t12)_CC_constraint5_:
+1 sos2_piecewise(tailwater_pump_t12)_CC_bin_y(1)
+1 sos2_piecewise(tailwater_pump_t12)_CC_bin_y(2)
+1 sos2_piecewise(tailwater_pump_t12)_CC_bin_y(3)
= 1

bounds
   -inf <= sos2_cost(tailwater_pump_t01) <= +inf
   -inf <= sos2_cost(tailwater_pump_t02) <= +inf
   -inf <= sos2_cost(tailwater_pump_t03) <= +inf
   -inf <= sos2_cost(tailwater_pump_t04) <= +inf
   -inf <= sos2_cost(tailwater_pump_t05) <= +inf
   -inf <= sos2_cost(tailwater_pump_t06) <= +inf
   -inf <= sos2_cost(tailwater_pump_t07) <= +inf
   -inf <= sos2_cost(tailwater_pump_t08) <= +inf
   -inf <= sos2_cost(tailwater_pump_t09) <= +inf
   -inf <= sos2_cost(tailwater_pump_t10) <= +inf
   -inf <= sos2_cost(tailwater_pump_t11) <= +inf
   -inf <= sos2_cost(tailwater_pump_t12) <= +inf
   0 <= shortage(demand_t01) <= +inf
   0 <= shortage(demand_t02) <= +inf
   0 <= shortage(demand_t03) <= +inf
   0 <= shortage(demand_t04) <= +inf
   0 <= shortage(demand_t05) <= +inf
   0 <= shortage(demand_t06) <= +inf
   0 <= shortage(demand_t07) <= +inf
   0 <= shortage(demand_t08) <= +inf
   0 <= shortage(demand_t09) <= +inf
   0 <= shortage(demand_t10) <= +inf
   0 <= shortage(demand_t11) <= +inf
   0 <= shortage(demand_t12) <= +inf
   0 <= flow(reservoir_gate_t01) <= +inf
   0 <= flow(reservoir_gate_t02) <= +inf
   0 <= flow(reservoir_gate_t03) <= +inf
   0 <= flow(reservoir_gate_t04) <= +inf
   0 <= flow(reservoir_gate_t05) <= +inf
   0 <= flow(reservoir_gate_t06) <= +inf
   0 <= flow(reservoir_gate_t07) <= +inf
   0 <= flow(reservoir_gate_t08) <= +inf
   0 <= flow(reservoir_gate_t09) <= +inf
   0 <= flow(reservoir_gate_t10) <= +inf
   0 <= flow(reservoir_gate_t11) <= +inf
   0 <= flow(reservoir_gate_t12) <= +inf
   0 <= flow(tailwater_pump_t01) <= +inf
   0 <= flow(tailwater_pump_t02) <= +inf
   0 <= flow(tailwater_pump_t03) <= +inf
   0 <= flow(tailwater_pump_t04) <= +inf
   0 <= flow(tailwater_pump_t05) <= +inf
   0 <= flow(tailwater_pump_t06) <= +inf
   0 <= flow(tailwater_pump_t07) <= +inf
   0 <= flow(tailwater_pump_t08) <= +inf
   0 <= flow(tailwater_pump_t09) <= +inf
   0 <= flow(tailwater_pump_t10) <= +inf
   0 <= flow(tailwater_pump_t11) <= +inf
   0 <= flow(tailwater_pump_t12) <= +inf
   0 <= gate_capacity(reservoir_gate_t01) <= +inf
   0 <= gate_capacity(reservoir_gate_t02) <= +inf
   0 <= gate_capacity(reservoir_gate_t03) <= +inf
   0 <= gate_capacity(reservoir_gate_t04) <= +inf
   0 <= gate_capacity(reservoir_gate_t05) <= +inf
   0 <= gate_capacity(reservoir_gate_t06) <= +inf
   0 <= gate_capacity(reservoir_gate_t07) <= +inf
   0 <= gate_capacity(reservoir_gate_t08) <= +inf
   0 <= gate_capacity(reservoir_gate_t09) <= +inf
   0 <= gate_capacity(reservoir_gate_t10) <= +inf
   0 <= gate_capacity(reservoir_gate_t11) <= +inf
   0 <= gate_capacity(reservoir_gate_t12) <= +inf
   250.0 <= state(reservoir_storage_t01) <= 600.0
   250.0 <= state(reservoir_storage_t02) <= 600.0
   250.0 <= state(reservoir_storage_t03) <= 600.0
   250.0 <= state(reservoir_storage_t04) <= 600.0
   250.0 <= state(reservoir_storage_t05) <= 600.0
   250.0 <= state(reservoir_storage_t06) <= 600.0
   250.0 <= state(reservoir_storage_t07) <= 600.0
   250.0 <= state(reservoir_storage_t08) <= 600.0
   250.0 <= state(reservoir_storage_t09) <= 600.0
   250.0 <= state(reservoir_storage_t10) <= 600.0
   250.0 <= state(reservoir_storage_t11) <= 600.0
   250.0 <= state(reservoir_storage_t12) <= 600.0
   200.0 <= state(reservoir_level_t01) <= 230.0
   200.0 <= state(reservoir_level_t02) <= 230.0
   200.0 <= state(reservoir_level_t03) <= 230.0
   200.0 <= state(reservoir_level_t04) <= 230.0
   200.0 <= state(reservoir_level_t05) <= 230.0
   200.0 <= state(reservoir_level_t06) <= 230.0
   200.0 <= state(reservoir_level_t07) <= 230.0
   200.0 <= state(reservoir_level_t08) <= 230.0
   200.0 <= state(reservoir_level_t09) <= 230.0
   200.0 <= state(reservoir_level_t10) <= 230.0
   200.0 <= state(reservoir_level_t11) <= 230.0
   200.0 <= state(reservoir_level_t12) <= 230.0
   -200.0 <= state(tailwater_storage_t01) <= 500.0
   -200.0 <= state(tailwater_storage_t02) <= 500.0
   -200.0 <= state(tailwater_storage_t03) <= 500.0
   -200.0 <= state(tailwater_storage_t04) <= 500.0
   -200.0 <= state(tailwater_storage_t05) <= 500.0
   -200.0 <= state(tailwater_storage_t06) <= 500.0
   -200.0 <= state(tailwater_storage_t07) <= 500.0
   -200.0 <= state(tailwater_storage_t08) <= 500.0
   -200.0 <= state(tailwater_storage_t09) <= 500.0
   -200.0 <= state(tailwater_storage_t10) <= 500.0
   -200.0 <= state(tailwater_storage_t11) <= 500.0
   -200.0 <= state(tailwater_storage_t12) <= 500.0
   110.0 <= state(tailwater_stage_t01) <= 140.0
   110.0 <= state(tailwater_stage_t02) <= 140.0
   110.0 <= state(tailwater_stage_t03) <= 140.0
   110.0 <= state(tailwater_stage_t04) <= 140.0
   110.0 <= state(tailwater_stage_t05) <= 140.0
   110.0 <= state(tailwater_stage_t06) <= 140.0
   110.0 <= state(tailwater_stage_t07) <= 140.0
   110.0 <= state(tailwater_stage_t08) <= 140.0
   110.0 <= state(tailwater_stage_t09) <= 140.0
   110.0 <= state(tailwater_stage_t10) <= 140.0
   110.0 <= state(tailwater_stage_t11) <= 140.0
   110.0 <= state(tailwater_stage_t12) <= 140.0
   0 <= state_piecewise(0)_pw_t01_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t01_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t01_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t01_CC_lambda(4) <= +inf
   0 <= state_piecewise(0)_pw_t01_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t01_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t01_CC_bin_y(3) <= 1
   0 <= state_piecewise(0)_pw_t02_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t02_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t02_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t02_CC_lambda(4) <= +inf
   0 <= state_piecewise(0)_pw_t02_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t02_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t02_CC_bin_y(3) <= 1
   0 <= state_piecewise(0)_pw_t03_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t03_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t03_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t03_CC_lambda(4) <= +inf
   0 <= state_piecewise(0)_pw_t03_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t03_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t03_CC_bin_y(3) <= 1
   0 <= state_piecewise(0)_pw_t04_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t04_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t04_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t04_CC_lambda(4) <= +inf
   0 <= state_piecewise(0)_pw_t04_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t04_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t04_CC_bin_y(3) <= 1
   0 <= state_piecewise(0)_pw_t05_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t05_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t05_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t05_CC_lambda(4) <= +inf
   0 <= state_piecewise(0)_pw_t05_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t05_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t05_CC_bin_y(3) <= 1
   0 <= state_piecewise(0)_pw_t06_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t06_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t06_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t06_CC_lambda(4) <= +inf
   0 <= state_piecewise(0)_pw_t06_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t06_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t06_CC_bin_y(3) <= 1
   0 <= state_piecewise(0)_pw_t07_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t07_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t07_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t07_CC_lambda(4) <= +inf
   0 <= state_piecewise(0)_pw_t07_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t07_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t07_CC_bin_y(3) <= 1
   0 <= state_piecewise(0)_pw_t08_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t08_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t08_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t08_CC_lambda(4) <= +inf
   0 <= state_piecewise(0)_pw_t08_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t08_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t08_CC_bin_y(3) <= 1
   0 <= state_piecewise(0)_pw_t09_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t09_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t09_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t09_CC_lambda(4) <= +inf
   0 <= state_piecewise(0)_pw_t09_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t09_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t09_CC_bin_y(3) <= 1
   0 <= state_piecewise(0)_pw_t10_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t10_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t10_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t10_CC_lambda(4) <= +inf
   0 <= state_piecewise(0)_pw_t10_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t10_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t10_CC_bin_y(3) <= 1
   0 <= state_piecewise(0)_pw_t11_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t11_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t11_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t11_CC_lambda(4) <= +inf
   0 <= state_piecewise(0)_pw_t11_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t11_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t11_CC_bin_y(3) <= 1
   0 <= state_piecewise(0)_pw_t12_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t12_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t12_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t12_CC_lambda(4) <= +inf
   0 <= state_piecewise(0)_pw_t12_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t12_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t12_CC_bin_y(3) <= 1
   0 <= state_piecewise(1)_pw_t01_CC_lambda(1) <= +inf
   0 <= state_piecewise(1)_pw_t01_CC_lambda(2) <= +inf
   0 <= state_piecewise(1)_pw_t01_CC_lambda(3) <= +inf
   0 <= state_piecewise(1)_pw_t01_CC_bin_y(1) <= 1
   0 <= state_piecewise(1)_pw_t01_CC_bin_y(2) <= 1
   0 <= state_piecewise(1)_pw_t02_CC_lambda(1) <= +inf
   0 <= state_piecewise(1)_pw_t02_CC_lambda(2) <= +inf
   0 <= state_piecewise(1)_pw_t02_CC_lambda(3) <= +inf
   0 <= state_piecewise(1)_pw_t02_CC_bin_y(1) <= 1
   0 <= state_piecewise(1)_pw_t02_CC_bin_y(2) <= 1
   0 <= state_piecewise(1)_pw_t03_CC_lambda(1) <= +inf
   0 <= state_piecewise(1)_pw_t03_CC_lambda(2) <= +inf
   0 <= state_piecewise(1)_pw_t03_CC_lambda(3) <= +inf
   0 <= state_piecewise(1)_pw_t03_CC_bin_y(1) <= 1
   0 <= state_piecewise(1)_pw_t03_CC_bin_y(2) <= 1
   0 <= state_piecewise(1)_pw_t04_CC_lambda(1) <= +inf
   0 <= state_piecewise(1)_pw_t04_CC_lambda(2) <= +inf
   0 <= state_piecewise(1)_pw_t04_CC_lambda(3) <= +inf
   0 <= state_piecewise(1)_pw_t04_CC_bin_y(1) <= 1
   0 <= state_piecewise(1)_pw_t04_CC_bin_y(2) <= 1
   0 <= state_piecewise(1)_pw_t05_CC_lambda(1) <= +inf
   0 <= state_piecewise(1)_pw_t05_CC_lambda(2) <= +inf
   0 <= state_piecewise(1)_pw_t05_CC_lambda(3) <= +inf
   0 <= state_piecewise(1)_pw_t05_CC_bin_y(1) <= 1
   0 <= state_piecewise(1)_pw_t05_CC_bin_y(2) <= 1
   0 <= state_piecewise(1)_pw_t06_CC_lambda(1) <= +inf
   0 <= state_piecewise(1)_pw_t06_CC_lambda(2) <= +inf
   0 <= state_piecewise(1)_pw_t06_CC_lambda(3) <= +inf
   0 <= state_piecewise(1)_pw_t06_CC_bin_y(1) <= 1
   0 <= state_piecewise(1)_pw_t06_CC_bin_y(2) <= 1
   0 <= state_piecewise(1)_pw_t07_CC_lambda(1) <= +inf
   0 <= state_piecewise(1)_pw_t07_CC_lambda(2) <= +inf
   0 <= state_piecewise(1)_pw_t07_CC_lambda(3) <= +inf
   0 <= state_piecewise(1)_pw_t07_CC_bin_y(1) <= 1
   0 <= state_piecewise(1)_pw_t07_CC_bin_y(2) <= 1
   0 <= state_piecewise(1)_pw_t08_CC_lambda(1) <= +inf
   0 <= state_piecewise(1)_pw_t08_CC_lambda(2) <= +inf
   0 <= state_piecewise(1)_pw_t08_CC_lambda(3) <= +inf
   0 <= state_piecewise(1)_pw_t08_CC_bin_y(1) <= 1
   0 <= state_piecewise(1)_pw_t08_CC_bin_y(2) <= 1
   0 <= state_piecewise(1)_pw_t09_CC_lambda(1) <= +inf
   0 <= state_piecewise(1)_pw_t09_CC_lambda(2) <= +inf
   0 <= state_piecewise(1)_pw_t09_CC_lambda(3) <= +inf
   0 <= state_piecewise(1)_pw_t09_CC_bin_y(1) <= 1
   0 <= state_piecewise(1)_pw_t09_CC_bin_y(2) <= 1
   0 <= state_piecewise(1)_pw_t10_CC_lambda(1) <= +inf
   0 <= state_piecewise(1)_pw_t10_CC_lambda(2) <= +inf
   0 <= state_piecewise(1)_pw_t10_CC_lambda(3) <= +inf
   0 <= state_piecewise(1)_pw_t10_CC_bin_y(1) <= 1
   0 <= state_piecewise(1)_pw_t10_CC_bin_y(2) <= 1
   0 <= state_piecewise(1)_pw_t11_CC_lambda(1) <= +inf
   0 <= state_piecewise(1)_pw_t11_CC_lambda(2) <= +inf
   0 <= state_piecewise(1)_pw_t11_CC_lambda(3) <= +inf
   0 <= state_piecewise(1)_pw_t11_CC_bin_y(1) <= 1
   0 <= state_piecewise(1)_pw_t11_CC_bin_y(2) <= 1
   0 <= state_piecewise(1)_pw_t12_CC_lambda(1) <= +inf
   0 <= state_piecewise(1)_pw_t12_CC_lambda(2) <= +inf
   0 <= state_piecewise(1)_pw_t12_CC_lambda(3) <= +inf
   0 <= state_piecewise(1)_pw_t12_CC_bin_y(1) <= 1
   0 <= state_piecewise(1)_pw_t12_CC_bin_y(2) <= 1
   0.0 <= gate_opening(reservoir_gate_t01) <= 1.0
   0.0 <= gate_opening(reservoir_gate_t02) <= 1.0
   0.0 <= gate_opening(reservoir_gate_t03) <= 1.0
   0.0 <= gate_opening(reservoir_gate_t04) <= 1.0
   0.0 <= gate_opening(reservoir_gate_t05) <= 1.0
   0.0 <= gate_opening(reservoir_gate_t06) <= 1.0
   0.0 <= gate_opening(reservoir_gate_t07) <= 1.0
   0.0 <= gate_opening(reservoir_gate_t08) <= 1.0
   0.0 <= gate_opening(reservoir_gate_t09) <= 1.0
   0.0 <= gate_opening(reservoir_gate_t10) <= 1.0
   0.0 <= gate_opening(reservoir_gate_t11) <= 1.0
   0.0 <= gate_opening(reservoir_gate_t12) <= 1.0
   0 <= gate_piecewise(reservoir_gate_t01)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(reservoir_gate_t01)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(reservoir_gate_t01)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(reservoir_gate_t01)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(reservoir_gate_t01)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(reservoir_gate_t01)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(reservoir_gate_t01)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(reservoir_gate_t02)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(reservoir_gate_t02)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(reservoir_gate_t02)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(reservoir_gate_t02)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(reservoir_gate_t02)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(reservoir_gate_t02)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(reservoir_gate_t02)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(reservoir_gate_t03)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(reservoir_gate_t03)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(reservoir_gate_t03)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(reservoir_gate_t03)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(reservoir_gate_t03)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(reservoir_gate_t03)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(reservoir_gate_t03)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(reservoir_gate_t04)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(reservoir_gate_t04)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(reservoir_gate_t04)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(reservoir_gate_t04)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(reservoir_gate_t04)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(reservoir_gate_t04)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(reservoir_gate_t04)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(reservoir_gate_t05)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(reservoir_gate_t05)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(reservoir_gate_t05)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(reservoir_gate_t05)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(reservoir_gate_t05)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(reservoir_gate_t05)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(reservoir_gate_t05)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(reservoir_gate_t06)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(reservoir_gate_t06)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(reservoir_gate_t06)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(reservoir_gate_t06)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(reservoir_gate_t06)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(reservoir_gate_t06)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(reservoir_gate_t06)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(reservoir_gate_t07)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(reservoir_gate_t07)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(reservoir_gate_t07)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(reservoir_gate_t07)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(reservoir_gate_t07)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(reservoir_gate_t07)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(reservoir_gate_t07)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(reservoir_gate_t08)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(reservoir_gate_t08)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(reservoir_gate_t08)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(reservoir_gate_t08)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(reservoir_gate_t08)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(reservoir_gate_t08)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(reservoir_gate_t08)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(reservoir_gate_t09)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(reservoir_gate_t09)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(reservoir_gate_t09)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(reservoir_gate_t09)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(reservoir_gate_t09)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(reservoir_gate_t09)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(reservoir_gate_t09)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(reservoir_gate_t10)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(reservoir_gate_t10)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(reservoir_gate_t10)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(reservoir_gate_t10)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(reservoir_gate_t10)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(reservoir_gate_t10)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(reservoir_gate_t10)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(reservoir_gate_t11)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(reservoir_gate_t11)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(reservoir_gate_t11)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(reservoir_gate_t11)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(reservoir_gate_t11)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(reservoir_gate_t11)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(reservoir_gate_t11)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(reservoir_gate_t12)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(reservoir_gate_t12)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(reservoir_gate_t12)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(reservoir_gate_t12)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(reservoir_gate_t12)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(reservoir_gate_t12)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(reservoir_gate_t12)_CC_bin_y(3) <= 1
   0 <= sos2_piecewise(tailwater_pump_t01)_CC_lambda(1) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t01)_CC_lambda(2) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t01)_CC_lambda(3) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t01)_CC_lambda(4) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t01)_CC_bin_y(1) <= 1
   0 <= sos2_piecewise(tailwater_pump_t01)_CC_bin_y(2) <= 1
   0 <= sos2_piecewise(tailwater_pump_t01)_CC_bin_y(3) <= 1
   0 <= sos2_piecewise(tailwater_pump_t02)_CC_lambda(1) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t02)_CC_lambda(2) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t02)_CC_lambda(3) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t02)_CC_lambda(4) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t02)_CC_bin_y(1) <= 1
   0 <= sos2_piecewise(tailwater_pump_t02)_CC_bin_y(2) <= 1
   0 <= sos2_piecewise(tailwater_pump_t02)_CC_bin_y(3) <= 1
   0 <= sos2_piecewise(tailwater_pump_t03)_CC_lambda(1) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t03)_CC_lambda(2) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t03)_CC_lambda(3) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t03)_CC_lambda(4) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t03)_CC_bin_y(1) <= 1
   0 <= sos2_piecewise(tailwater_pump_t03)_CC_bin_y(2) <= 1
   0 <= sos2_piecewise(tailwater_pump_t03)_CC_bin_y(3) <= 1
   0 <= sos2_piecewise(tailwater_pump_t04)_CC_lambda(1) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t04)_CC_lambda(2) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t04)_CC_lambda(3) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t04)_CC_lambda(4) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t04)_CC_bin_y(1) <= 1
   0 <= sos2_piecewise(tailwater_pump_t04)_CC_bin_y(2) <= 1
   0 <= sos2_piecewise(tailwater_pump_t04)_CC_bin_y(3) <= 1
   0 <= sos2_piecewise(tailwater_pump_t05)_CC_lambda(1) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t05)_CC_lambda(2) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t05)_CC_lambda(3) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t05)_CC_lambda(4) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t05)_CC_bin_y(1) <= 1
   0 <= sos2_piecewise(tailwater_pump_t05)_CC_bin_y(2) <= 1
   0 <= sos2_piecewise(tailwater_pump_t05)_CC_bin_y(3) <= 1
   0 <= sos2_piecewise(tailwater_pump_t06)_CC_lambda(1) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t06)_CC_lambda(2) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t06)_CC_lambda(3) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t06)_CC_lambda(4) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t06)_CC_bin_y(1) <= 1
   0 <= sos2_piecewise(tailwater_pump_t06)_CC_bin_y(2) <= 1
   0 <= sos2_piecewise(tailwater_pump_t06)_CC_bin_y(3) <= 1
   0 <= sos2_piecewise(tailwater_pump_t07)_CC_lambda(1) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t07)_CC_lambda(2) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t07)_CC_lambda(3) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t07)_CC_lambda(4) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t07)_CC_bin_y(1) <= 1
   0 <= sos2_piecewise(tailwater_pump_t07)_CC_bin_y(2) <= 1
   0 <= sos2_piecewise(tailwater_pump_t07)_CC_bin_y(3) <= 1
   0 <= sos2_piecewise(tailwater_pump_t08)_CC_lambda(1) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t08)_CC_lambda(2) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t08)_CC_lambda(3) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t08)_CC_lambda(4) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t08)_CC_bin_y(1) <= 1
   0 <= sos2_piecewise(tailwater_pump_t08)_CC_bin_y(2) <= 1
   0 <= sos2_piecewise(tailwater_pump_t08)_CC_bin_y(3) <= 1
   0 <= sos2_piecewise(tailwater_pump_t09)_CC_lambda(1) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t09)_CC_lambda(2) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t09)_CC_lambda(3) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t09)_CC_lambda(4) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t09)_CC_bin_y(1) <= 1
   0 <= sos2_piecewise(tailwater_pump_t09)_CC_bin_y(2) <= 1
   0 <= sos2_piecewise(tailwater_pump_t09)_CC_bin_y(3) <= 1
   0 <= sos2_piecewise(tailwater_pump_t10)_CC_lambda(1) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t10)_CC_lambda(2) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t10)_CC_lambda(3) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t10)_CC_lambda(4) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t10)_CC_bin_y(1) <= 1
   0 <= sos2_piecewise(tailwater_pump_t10)_CC_bin_y(2) <= 1
   0 <= sos2_piecewise(tailwater_pump_t10)_CC_bin_y(3) <= 1
   0 <= sos2_piecewise(tailwater_pump_t11)_CC_lambda(1) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t11)_CC_lambda(2) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t11)_CC_lambda(3) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t11)_CC_lambda(4) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t11)_CC_bin_y(1) <= 1
   0 <= sos2_piecewise(tailwater_pump_t11)_CC_bin_y(2) <= 1
   0 <= sos2_piecewise(tailwater_pump_t11)_CC_bin_y(3) <= 1
   0 <= sos2_piecewise(tailwater_pump_t12)_CC_lambda(1) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t12)_CC_lambda(2) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t12)_CC_lambda(3) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t12)_CC_lambda(4) <= +inf
   0 <= sos2_piecewise(tailwater_pump_t12)_CC_bin_y(1) <= 1
   0 <= sos2_piecewise(tailwater_pump_t12)_CC_bin_y(2) <= 1
   0 <= sos2_piecewise(tailwater_pump_t12)_CC_bin_y(3) <= 1
binary
  state_piecewise(0)_pw_t01_CC_bin_y(1)
  state_piecewise(0)_pw_t01_CC_bin_y(2)
  state_piecewise(0)_pw_t01_CC_bin_y(3)
  state_piecewise(0)_pw_t02_CC_bin_y(1)
  state_piecewise(0)_pw_t02_CC_bin_y(2)
  state_piecewise(0)_pw_t02_CC_bin_y(3)
  state_piecewise(0)_pw_t03_CC_bin_y(1)
  state_piecewise(0)_pw_t03_CC_bin_y(2)
  state_piecewise(0)_pw_t03_CC_bin_y(3)
  state_piecewise(0)_pw_t04_CC_bin_y(1)
  state_piecewise(0)_pw_t04_CC_bin_y(2)
  state_piecewise(0)_pw_t04_CC_bin_y(3)
  state_piecewise(0)_pw_t05_CC_bin_y(1)
  state_piecewise(0)_pw_t05_CC_bin_y(2)
  state_piecewise(0)_pw_t05_CC_bin_y(3)
  state_piecewise(0)_pw_t06_CC_bin_y(1)
  state_piecewise(0)_pw_t06_CC_bin_y(2)
  state_piecewise(0)_pw_t06_CC_bin_y(3)
  state_piecewise(0)_pw_t07_CC_bin_y(1)
  state_piecewise(0)_pw_t07_CC_bin_y(2)
  state_piecewise(0)_pw_t07_CC_bin_y(3)
  state_piecewise(0)_pw_t08_CC_bin_y(1)
  state_piecewise(0)_pw_t08_CC_bin_y(2)
  state_piecewise(0)_pw_t08_CC_bin_y(3)
  state_piecewise(0)_pw_t09_CC_bin_y(1)
  state_piecewise(0)_pw_t09_CC_bin_y(2)
  state_piecewise(0)_pw_t09_CC_bin_y(3)
  state_piecewise(0)_pw_t10_CC_bin_y(1)
  state_piecewise(0)_pw_t10_CC_bin_y(2)
  state_piecewise(0)_pw_t10_CC_bin_y(3)
  state_piecewise(0)_pw_t11_CC_bin_y(1)
  state_piecewise(0)_pw_t11_CC_bin_y(2)
  state_piecewise(0)_pw_t11_CC_bin_y(3)
  state_piecewise(0)_pw_t12_CC_bin_y(1)
  state_piecewise(0)_pw_t12_CC_bin_y(2)
  state_piecewise(0)_pw_t12_CC_bin_y(3)
  state_piecewise(1)_pw_t01_CC_bin_y(1)
  state_piecewise(1)_pw_t01_CC_bin_y(2)
  state_piecewise(1)_pw_t02_CC_bin_y(1)
  state_piecewise(1)_pw_t02_CC_bin_y(2)
  state_piecewise(1)_pw_t03_CC_bin_y(1)
  state_piecewise(1)_pw_t03_CC_bin_y(2)
  state_piecewise(1)_pw_t04_CC_bin_y(1)
  state_piecewise(1)_pw_t04_CC_bin_y(2)
  state_piecewise(1)_pw_t05_CC_bin_y(1)
  state_piecewise(1)_pw_t05_CC_bin_y(2)
  state_piecewise(1)_pw_t06_CC_bin_y(1)
  state_piecewise(1)_pw_t06_CC_bin_y(2)
  state_piecewise(1)_pw_t07_CC_bin_y(1)
  state_piecewise(1)_pw_t07_CC_bin_y(2)
  state_piecewise(1)_pw_t08_CC_bin_y(1)
  state_piecewise(1)_pw_t08_CC_bin_y(2)
  state_piecewise(1)_pw_t09_CC_bin_y(1)
  state_piecewise(1)_pw_t09_CC_bin_y(2)
  state_piecewise(1)_pw_t10_CC_bin_y(1)
  state_piecewise(1)_pw_t10_CC_bin_y(2)
  state_piecewise(1)_pw_t11_CC_bin_y(1)
  state_piecewise(1)_pw_t11_CC_bin_y(2)
  state_piecewise(1)_pw_t12_CC_bin_y(1)
  state_piecewise(1)_pw_t12_CC_bin_y(2)
  gate_piecewise(reservoir_gate_t01)_CC_bin_y(1)
  gate_piecewise(reservoir_gate_t01)_CC_bin_y(2)
  gate_piecewise(reservoir_gate_t01)_CC_bin_y(3)
  gate_piecewise(reservoir_gate_t02)_CC_bin_y(1)
  gate_piecewise(reservoir_gate_t02)_CC_bin_y(2)
  gate_piecewise(reservoir_gate_t02)_CC_bin_y(3)
  gate_piecewise(reservoir_gate_t03)_CC_bin_y(1)
  gate_piecewise(reservoir_gate_t03)_CC_bin_y(2)
  gate_piecewise(reservoir_gate_t03)_CC_bin_y(3)
  gate_piecewise(reservoir_gate_t04)_CC_bin_y(1)
  gate_piecewise(reservoir_gate_t04)_CC_bin_y(2)
  gate_piecewise(reservoir_gate_t04)_CC_bin_y(3)
  gate_piecewise(reservoir_gate_t05)_CC_bin_y(1)
  gate_piecewise(reservoir_gate_t05)_CC_bin_y(2)
  gate_piecewise(reservoir_gate_t05)_CC_bin_y(3)
  gate_piecewise(reservoir_gate_t06)_CC_bin_y(1)
  gate_piecewise(reservoir_gate_t06)_CC_bin_y(2)
  gate_piecewise(reservoir_gate_t06)_CC_bin_y(3)
  gate_piecewise(reservoir_gate_t07)_CC_bin_y(1)
  gate_piecewise(reservoir_gate_t07)_CC_bin_y(2)
  gate_piecewise(reservoir_gate_t07)_CC_bin_y(3)
  gate_piecewise(reservoir_gate_t08)_CC_bin_y(1)
  gate_piecewise(reservoir_gate_t08)_CC_bin_y(2)
  gate_piecewise(reservoir_gate_t08)_CC_bin_y(3)
  gate_piecewise(reservoir_gate_t09)_CC_bin_y(1)
  gate_piecewise(reservoir_gate_t09)_CC_bin_y(2)
  gate_piecewise(reservoir_gate_t09)_CC_bin_y(3)
  gate_piecewise(reservoir_gate_t10)_CC_bin_y(1)
  gate_piecewise(reservoir_gate_t10)_CC_bin_y(2)
  gate_piecewise(reservoir_gate_t10)_CC_bin_y(3)
  gate_piecewise(reservoir_gate_t11)_CC_bin_y(1)
  gate_piecewise(reservoir_gate_t11)_CC_bin_y(2)
  gate_piecewise(reservoir_gate_t11)_CC_bin_y(3)
  gate_piecewise(reservoir_gate_t12)_CC_bin_y(1)
  gate_piecewise(reservoir_gate_t12)_CC_bin_y(2)
  gate_piecewise(reservoir_gate_t12)_CC_bin_y(3)
  sos2_piecewise(tailwater_pump_t01)_CC_bin_y(1)
  sos2_piecewise(tailwater_pump_t01)_CC_bin_y(2)
  sos2_piecewise(tailwater_pump_t01)_CC_bin_y(3)
  sos2_piecewise(tailwater_pump_t02)_CC_bin_y(1)
  sos2_piecewise(tailwater_pump_t02)_CC_bin_y(2)
  sos2_piecewise(tailwater_pump_t02)_CC_bin_y(3)
  sos2_piecewise(tailwater_pump_t03)_CC_bin_y(1)
  sos2_piecewise(tailwater_pump_t03)_CC_bin_y(2)
  sos2_piecewise(tailwater_pump_t03)_CC_bin_y(3)
  sos2_piecewise(tailwater_pump_t04)_CC_bin_y(1)
  sos2_piecewise(tailwater_pump_t04)_CC_bin_y(2)
  sos2_piecewise(tailwater_pump_t04)_CC_bin_y(3)
  sos2_piecewise(tailwater_pump_t05)_CC_bin_y(1)
  sos2_piecewise(tailwater_pump_t05)_CC_bin_y(2)
  sos2_piecewise(tailwater_pump_t05)_CC_bin_y(3)
  sos2_piecewise(tailwater_pump_t06)_CC_bin_y(1)
  sos2_piecewise(tailwater_pump_t06)_CC_bin_y(2)
  sos2_piecewise(tailwater_pump_t06)_CC_bin_y(3)
  sos2_piecewise(tailwater_pump_t07)_CC_bin_y(1)
  sos2_piecewise(tailwater_pump_t07)_CC_bin_y(2)
  sos2_piecewise(tailwater_pump_t07)_CC_bin_y(3)
  sos2_piecewise(tailwater_pump_t08)_CC_bin_y(1)
  sos2_piecewise(tailwater_pump_t08)_CC_bin_y(2)
  sos2_piecewise(tailwater_pump_t08)_CC_bin_y(3)
  sos2_piecewise(tailwater_pump_t09)_CC_bin_y(1)
  sos2_piecewise(tailwater_pump_t09)_CC_bin_y(2)
  sos2_piecewise(tailwater_pump_t09)_CC_bin_y(3)
  sos2_piecewise(tailwater_pump_t10)_CC_bin_y(1)
  sos2_piecewise(tailwater_pump_t10)_CC_bin_y(2)
  sos2_piecewise(tailwater_pump_t10)_CC_bin_y(3)
  sos2_piecewise(tailwater_pump_t11)_CC_bin_y(1)
  sos2_piecewise(tailwater_pump_t11)_CC_bin_y(2)
  sos2_piecewise(tailwater_pump_t11)_CC_bin_y(3)
  sos2_piecewise(tailwater_pump_t12)_CC_bin_y(1)
  sos2_piecewise(tailwater_pump_t12)_CC_bin_y(2)
  sos2_piecewise(tailwater_pump_t12)_CC_bin_y(3)
end
