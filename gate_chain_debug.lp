\* Source Pyomo model name=GenericWaterNetwork *\

min 
obj:
+2500.0 shortage(outlet_t01)
+2500.0 shortage(outlet_t02)
+2500.0 shortage(outlet_t03)
+2500.0 shortage(outlet_t04)
+2500.0 shortage(outlet_t05)
+2500.0 shortage(outlet_t06)
+2500.0 shortage(outlet_t07)
+2500.0 shortage(outlet_t08)
+2500.0 shortage(outlet_t09)
+2500.0 shortage(outlet_t10)
+2500.0 shortage(outlet_t11)
+2500.0 shortage(outlet_t12)
+2500.0 shortage(outlet_t13)
+2500.0 shortage(outlet_t14)
+2500.0 shortage(outlet_t15)
+2500.0 shortage(outlet_t16)
+2500.0 shortage(outlet_t17)
+2500.0 shortage(outlet_t18)
+2500.0 shortage(outlet_t19)
+2500.0 shortage(outlet_t20)
+2500.0 shortage(outlet_t21)
+2500.0 shortage(outlet_t22)
+2500.0 shortage(outlet_t23)
+2500.0 shortage(outlet_t24)

s.t.

c_u_gate_flow_limit(gate1_t01)_:
+1 flow(gate1_t01)
-1 gate_capacity(gate1_t01)
<= 0

c_u_gate_flow_limit(gate1_t02)_:
+1 flow(gate1_t02)
-1 gate_capacity(gate1_t02)
<= 0

c_u_gate_flow_limit(gate1_t03)_:
+1 flow(gate1_t03)
-1 gate_capacity(gate1_t03)
<= 0

c_u_gate_flow_limit(gate1_t04)_:
+1 flow(gate1_t04)
-1 gate_capacity(gate1_t04)
<= 0

c_u_gate_flow_limit(gate1_t05)_:
+1 flow(gate1_t05)
-1 gate_capacity(gate1_t05)
<= 0

c_u_gate_flow_limit(gate1_t06)_:
+1 flow(gate1_t06)
-1 gate_capacity(gate1_t06)
<= 0

c_u_gate_flow_limit(gate1_t07)_:
+1 flow(gate1_t07)
-1 gate_capacity(gate1_t07)
<= 0

c_u_gate_flow_limit(gate1_t08)_:
+1 flow(gate1_t08)
-1 gate_capacity(gate1_t08)
<= 0

c_u_gate_flow_limit(gate1_t09)_:
+1 flow(gate1_t09)
-1 gate_capacity(gate1_t09)
<= 0

c_u_gate_flow_limit(gate1_t10)_:
+1 flow(gate1_t10)
-1 gate_capacity(gate1_t10)
<= 0

c_u_gate_flow_limit(gate1_t11)_:
+1 flow(gate1_t11)
-1 gate_capacity(gate1_t11)
<= 0

c_u_gate_flow_limit(gate1_t12)_:
+1 flow(gate1_t12)
-1 gate_capacity(gate1_t12)
<= 0

c_u_gate_flow_limit(gate1_t13)_:
+1 flow(gate1_t13)
-1 gate_capacity(gate1_t13)
<= 0

c_u_gate_flow_limit(gate1_t14)_:
+1 flow(gate1_t14)
-1 gate_capacity(gate1_t14)
<= 0

c_u_gate_flow_limit(gate1_t15)_:
+1 flow(gate1_t15)
-1 gate_capacity(gate1_t15)
<= 0

c_u_gate_flow_limit(gate1_t16)_:
+1 flow(gate1_t16)
-1 gate_capacity(gate1_t16)
<= 0

c_u_gate_flow_limit(gate1_t17)_:
+1 flow(gate1_t17)
-1 gate_capacity(gate1_t17)
<= 0

c_u_gate_flow_limit(gate1_t18)_:
+1 flow(gate1_t18)
-1 gate_capacity(gate1_t18)
<= 0

c_u_gate_flow_limit(gate1_t19)_:
+1 flow(gate1_t19)
-1 gate_capacity(gate1_t19)
<= 0

c_u_gate_flow_limit(gate1_t20)_:
+1 flow(gate1_t20)
-1 gate_capacity(gate1_t20)
<= 0

c_u_gate_flow_limit(gate1_t21)_:
+1 flow(gate1_t21)
-1 gate_capacity(gate1_t21)
<= 0

c_u_gate_flow_limit(gate1_t22)_:
+1 flow(gate1_t22)
-1 gate_capacity(gate1_t22)
<= 0

c_u_gate_flow_limit(gate1_t23)_:
+1 flow(gate1_t23)
-1 gate_capacity(gate1_t23)
<= 0

c_u_gate_flow_limit(gate1_t24)_:
+1 flow(gate1_t24)
-1 gate_capacity(gate1_t24)
<= 0

c_u_gate_flow_limit(gate2_t01)_:
+1 flow(gate2_t01)
-1 gate_capacity(gate2_t01)
<= 0

c_u_gate_flow_limit(gate2_t02)_:
+1 flow(gate2_t02)
-1 gate_capacity(gate2_t02)
<= 0

c_u_gate_flow_limit(gate2_t03)_:
+1 flow(gate2_t03)
-1 gate_capacity(gate2_t03)
<= 0

c_u_gate_flow_limit(gate2_t04)_:
+1 flow(gate2_t04)
-1 gate_capacity(gate2_t04)
<= 0

c_u_gate_flow_limit(gate2_t05)_:
+1 flow(gate2_t05)
-1 gate_capacity(gate2_t05)
<= 0

c_u_gate_flow_limit(gate2_t06)_:
+1 flow(gate2_t06)
-1 gate_capacity(gate2_t06)
<= 0

c_u_gate_flow_limit(gate2_t07)_:
+1 flow(gate2_t07)
-1 gate_capacity(gate2_t07)
<= 0

c_u_gate_flow_limit(gate2_t08)_:
+1 flow(gate2_t08)
-1 gate_capacity(gate2_t08)
<= 0

c_u_gate_flow_limit(gate2_t09)_:
+1 flow(gate2_t09)
-1 gate_capacity(gate2_t09)
<= 0

c_u_gate_flow_limit(gate2_t10)_:
+1 flow(gate2_t10)
-1 gate_capacity(gate2_t10)
<= 0

c_u_gate_flow_limit(gate2_t11)_:
+1 flow(gate2_t11)
-1 gate_capacity(gate2_t11)
<= 0

c_u_gate_flow_limit(gate2_t12)_:
+1 flow(gate2_t12)
-1 gate_capacity(gate2_t12)
<= 0

c_u_gate_flow_limit(gate2_t13)_:
+1 flow(gate2_t13)
-1 gate_capacity(gate2_t13)
<= 0

c_u_gate_flow_limit(gate2_t14)_:
+1 flow(gate2_t14)
-1 gate_capacity(gate2_t14)
<= 0

c_u_gate_flow_limit(gate2_t15)_:
+1 flow(gate2_t15)
-1 gate_capacity(gate2_t15)
<= 0

c_u_gate_flow_limit(gate2_t16)_:
+1 flow(gate2_t16)
-1 gate_capacity(gate2_t16)
<= 0

c_u_gate_flow_limit(gate2_t17)_:
+1 flow(gate2_t17)
-1 gate_capacity(gate2_t17)
<= 0

c_u_gate_flow_limit(gate2_t18)_:
+1 flow(gate2_t18)
-1 gate_capacity(gate2_t18)
<= 0

c_u_gate_flow_limit(gate2_t19)_:
+1 flow(gate2_t19)
-1 gate_capacity(gate2_t19)
<= 0

c_u_gate_flow_limit(gate2_t20)_:
+1 flow(gate2_t20)
-1 gate_capacity(gate2_t20)
<= 0

c_u_gate_flow_limit(gate2_t21)_:
+1 flow(gate2_t21)
-1 gate_capacity(gate2_t21)
<= 0

c_u_gate_flow_limit(gate2_t22)_:
+1 flow(gate2_t22)
-1 gate_capacity(gate2_t22)
<= 0

c_u_gate_flow_limit(gate2_t23)_:
+1 flow(gate2_t23)
-1 gate_capacity(gate2_t23)
<= 0

c_u_gate_flow_limit(gate2_t24)_:
+1 flow(gate2_t24)
-1 gate_capacity(gate2_t24)
<= 0

c_u_gate_head_linear(gate1_t01)_:
+1 flow(gate1_t01)
-1.9016725270140493 gate_opening(gate1_t01)
-10.459198898577272 state(reservoir_level_t01)
+10.459198898577272 state(reach1_level_upstream_t01)
<= -0.5229599449288634

c_u_gate_head_linear(gate1_t02)_:
+1 flow(gate1_t02)
-1.9016725270140493 gate_opening(gate1_t02)
-10.459198898577272 state(reservoir_level_t02)
+10.459198898577272 state(reach1_level_upstream_t02)
<= -0.5229599449288634

c_u_gate_head_linear(gate1_t03)_:
+1 flow(gate1_t03)
-1.9016725270140493 gate_opening(gate1_t03)
-10.459198898577272 state(reservoir_level_t03)
+10.459198898577272 state(reach1_level_upstream_t03)
<= -0.5229599449288634

c_u_gate_head_linear(gate1_t04)_:
+1 flow(gate1_t04)
-1.9016725270140493 gate_opening(gate1_t04)
-10.459198898577272 state(reservoir_level_t04)
+10.459198898577272 state(reach1_level_upstream_t04)
<= -0.5229599449288634

c_u_gate_head_linear(gate1_t05)_:
+1 flow(gate1_t05)
-1.9016725270140493 gate_opening(gate1_t05)
-10.459198898577272 state(reservoir_level_t05)
+10.459198898577272 state(reach1_level_upstream_t05)
<= -0.5229599449288634

c_u_gate_head_linear(gate1_t06)_:
+1 flow(gate1_t06)
-1.9016725270140493 gate_opening(gate1_t06)
-10.459198898577272 state(reservoir_level_t06)
+10.459198898577272 state(reach1_level_upstream_t06)
<= -0.5229599449288634

c_u_gate_head_linear(gate1_t07)_:
+1 flow(gate1_t07)
-1.9016725270140493 gate_opening(gate1_t07)
-10.459198898577272 state(reservoir_level_t07)
+10.459198898577272 state(reach1_level_upstream_t07)
<= -0.5229599449288634

c_u_gate_head_linear(gate1_t08)_:
+1 flow(gate1_t08)
-1.9016725270140493 gate_opening(gate1_t08)
-10.459198898577272 state(reservoir_level_t08)
+10.459198898577272 state(reach1_level_upstream_t08)
<= -0.5229599449288634

c_u_gate_head_linear(gate1_t09)_:
+1 flow(gate1_t09)
-1.9016725270140493 gate_opening(gate1_t09)
-10.459198898577272 state(reservoir_level_t09)
+10.459198898577272 state(reach1_level_upstream_t09)
<= -0.5229599449288634

c_u_gate_head_linear(gate1_t10)_:
+1 flow(gate1_t10)
-1.9016725270140493 gate_opening(gate1_t10)
-10.459198898577272 state(reservoir_level_t10)
+10.459198898577272 state(reach1_level_upstream_t10)
<= -0.5229599449288634

c_u_gate_head_linear(gate1_t11)_:
+1 flow(gate1_t11)
-1.9016725270140493 gate_opening(gate1_t11)
-10.459198898577272 state(reservoir_level_t11)
+10.459198898577272 state(reach1_level_upstream_t11)
<= -0.5229599449288634

c_u_gate_head_linear(gate1_t12)_:
+1 flow(gate1_t12)
-1.9016725270140493 gate_opening(gate1_t12)
-10.459198898577272 state(reservoir_level_t12)
+10.459198898577272 state(reach1_level_upstream_t12)
<= -0.5229599449288634

c_u_gate_head_linear(gate1_t13)_:
+1 flow(gate1_t13)
-1.9016725270140493 gate_opening(gate1_t13)
-10.459198898577272 state(reservoir_level_t13)
+10.459198898577272 state(reach1_level_upstream_t13)
<= -0.5229599449288634

c_u_gate_head_linear(gate1_t14)_:
+1 flow(gate1_t14)
-1.9016725270140493 gate_opening(gate1_t14)
-10.459198898577272 state(reservoir_level_t14)
+10.459198898577272 state(reach1_level_upstream_t14)
<= -0.5229599449288634

c_u_gate_head_linear(gate1_t15)_:
+1 flow(gate1_t15)
-1.9016725270140493 gate_opening(gate1_t15)
-10.459198898577272 state(reservoir_level_t15)
+10.459198898577272 state(reach1_level_upstream_t15)
<= -0.5229599449288634

c_u_gate_head_linear(gate1_t16)_:
+1 flow(gate1_t16)
-1.9016725270140493 gate_opening(gate1_t16)
-10.459198898577272 state(reservoir_level_t16)
+10.459198898577272 state(reach1_level_upstream_t16)
<= -0.5229599449288634

c_u_gate_head_linear(gate1_t17)_:
+1 flow(gate1_t17)
-1.9016725270140493 gate_opening(gate1_t17)
-10.459198898577272 state(reservoir_level_t17)
+10.459198898577272 state(reach1_level_upstream_t17)
<= -0.5229599449288634

c_u_gate_head_linear(gate1_t18)_:
+1 flow(gate1_t18)
-1.9016725270140493 gate_opening(gate1_t18)
-10.459198898577272 state(reservoir_level_t18)
+10.459198898577272 state(reach1_level_upstream_t18)
<= -0.5229599449288634

c_u_gate_head_linear(gate1_t19)_:
+1 flow(gate1_t19)
-1.9016725270140493 gate_opening(gate1_t19)
-10.459198898577272 state(reservoir_level_t19)
+10.459198898577272 state(reach1_level_upstream_t19)
<= -0.5229599449288634

c_u_gate_head_linear(gate1_t20)_:
+1 flow(gate1_t20)
-1.9016725270140493 gate_opening(gate1_t20)
-10.459198898577272 state(reservoir_level_t20)
+10.459198898577272 state(reach1_level_upstream_t20)
<= -0.5229599449288634

c_u_gate_head_linear(gate1_t21)_:
+1 flow(gate1_t21)
-1.9016725270140493 gate_opening(gate1_t21)
-10.459198898577272 state(reservoir_level_t21)
+10.459198898577272 state(reach1_level_upstream_t21)
<= -0.5229599449288634

c_u_gate_head_linear(gate1_t22)_:
+1 flow(gate1_t22)
-1.9016725270140493 gate_opening(gate1_t22)
-10.459198898577272 state(reservoir_level_t22)
+10.459198898577272 state(reach1_level_upstream_t22)
<= -0.5229599449288634

c_u_gate_head_linear(gate1_t23)_:
+1 flow(gate1_t23)
-1.9016725270140493 gate_opening(gate1_t23)
-10.459198898577272 state(reservoir_level_t23)
+10.459198898577272 state(reach1_level_upstream_t23)
<= -0.5229599449288634

c_u_gate_head_linear(gate1_t24)_:
+1 flow(gate1_t24)
-1.9016725270140493 gate_opening(gate1_t24)
-10.459198898577272 state(reservoir_level_t24)
+10.459198898577272 state(reach1_level_upstream_t24)
<= -0.5229599449288634

c_u_gate_head_linear(gate2_t01)_:
+1 flow(gate2_t01)
-1.5451089281989152 gate_opening(gate2_t01)
-7.725544640994576 state(reach1_level_downstream_t01)
+7.725544640994576 state(reach2_level_upstream_t01)
<= -0.3862772320497288

c_u_gate_head_linear(gate2_t02)_:
+1 flow(gate2_t02)
-1.5451089281989152 gate_opening(gate2_t02)
-7.725544640994576 state(reach1_level_downstream_t02)
+7.725544640994576 state(reach2_level_upstream_t02)
<= -0.3862772320497288

c_u_gate_head_linear(gate2_t03)_:
+1 flow(gate2_t03)
-1.5451089281989152 gate_opening(gate2_t03)
-7.725544640994576 state(reach1_level_downstream_t03)
+7.725544640994576 state(reach2_level_upstream_t03)
<= -0.3862772320497288

c_u_gate_head_linear(gate2_t04)_:
+1 flow(gate2_t04)
-1.5451089281989152 gate_opening(gate2_t04)
-7.725544640994576 state(reach1_level_downstream_t04)
+7.725544640994576 state(reach2_level_upstream_t04)
<= -0.3862772320497288

c_u_gate_head_linear(gate2_t05)_:
+1 flow(gate2_t05)
-1.5451089281989152 gate_opening(gate2_t05)
-7.725544640994576 state(reach1_level_downstream_t05)
+7.725544640994576 state(reach2_level_upstream_t05)
<= -0.3862772320497288

c_u_gate_head_linear(gate2_t06)_:
+1 flow(gate2_t06)
-1.5451089281989152 gate_opening(gate2_t06)
-7.725544640994576 state(reach1_level_downstream_t06)
+7.725544640994576 state(reach2_level_upstream_t06)
<= -0.3862772320497288

c_u_gate_head_linear(gate2_t07)_:
+1 flow(gate2_t07)
-1.5451089281989152 gate_opening(gate2_t07)
-7.725544640994576 state(reach1_level_downstream_t07)
+7.725544640994576 state(reach2_level_upstream_t07)
<= -0.3862772320497288

c_u_gate_head_linear(gate2_t08)_:
+1 flow(gate2_t08)
-1.5451089281989152 gate_opening(gate2_t08)
-7.725544640994576 state(reach1_level_downstream_t08)
+7.725544640994576 state(reach2_level_upstream_t08)
<= -0.3862772320497288

c_u_gate_head_linear(gate2_t09)_:
+1 flow(gate2_t09)
-1.5451089281989152 gate_opening(gate2_t09)
-7.725544640994576 state(reach1_level_downstream_t09)
+7.725544640994576 state(reach2_level_upstream_t09)
<= -0.3862772320497288

c_u_gate_head_linear(gate2_t10)_:
+1 flow(gate2_t10)
-1.5451089281989152 gate_opening(gate2_t10)
-7.725544640994576 state(reach1_level_downstream_t10)
+7.725544640994576 state(reach2_level_upstream_t10)
<= -0.3862772320497288

c_u_gate_head_linear(gate2_t11)_:
+1 flow(gate2_t11)
-1.5451089281989152 gate_opening(gate2_t11)
-7.725544640994576 state(reach1_level_downstream_t11)
+7.725544640994576 state(reach2_level_upstream_t11)
<= -0.3862772320497288

c_u_gate_head_linear(gate2_t12)_:
+1 flow(gate2_t12)
-1.5451089281989152 gate_opening(gate2_t12)
-7.725544640994576 state(reach1_level_downstream_t12)
+7.725544640994576 state(reach2_level_upstream_t12)
<= -0.3862772320497288

c_u_gate_head_linear(gate2_t13)_:
+1 flow(gate2_t13)
-1.5451089281989152 gate_opening(gate2_t13)
-7.725544640994576 state(reach1_level_downstream_t13)
+7.725544640994576 state(reach2_level_upstream_t13)
<= -0.3862772320497288

c_u_gate_head_linear(gate2_t14)_:
+1 flow(gate2_t14)
-1.5451089281989152 gate_opening(gate2_t14)
-7.725544640994576 state(reach1_level_downstream_t14)
+7.725544640994576 state(reach2_level_upstream_t14)
<= -0.3862772320497288

c_u_gate_head_linear(gate2_t15)_:
+1 flow(gate2_t15)
-1.5451089281989152 gate_opening(gate2_t15)
-7.725544640994576 state(reach1_level_downstream_t15)
+7.725544640994576 state(reach2_level_upstream_t15)
<= -0.3862772320497288

c_u_gate_head_linear(gate2_t16)_:
+1 flow(gate2_t16)
-1.5451089281989152 gate_opening(gate2_t16)
-7.725544640994576 state(reach1_level_downstream_t16)
+7.725544640994576 state(reach2_level_upstream_t16)
<= -0.3862772320497288

c_u_gate_head_linear(gate2_t17)_:
+1 flow(gate2_t17)
-1.5451089281989152 gate_opening(gate2_t17)
-7.725544640994576 state(reach1_level_downstream_t17)
+7.725544640994576 state(reach2_level_upstream_t17)
<= -0.3862772320497288

c_u_gate_head_linear(gate2_t18)_:
+1 flow(gate2_t18)
-1.5451089281989152 gate_opening(gate2_t18)
-7.725544640994576 state(reach1_level_downstream_t18)
+7.725544640994576 state(reach2_level_upstream_t18)
<= -0.3862772320497288

c_u_gate_head_linear(gate2_t19)_:
+1 flow(gate2_t19)
-1.5451089281989152 gate_opening(gate2_t19)
-7.725544640994576 state(reach1_level_downstream_t19)
+7.725544640994576 state(reach2_level_upstream_t19)
<= -0.3862772320497288

c_u_gate_head_linear(gate2_t20)_:
+1 flow(gate2_t20)
-1.5451089281989152 gate_opening(gate2_t20)
-7.725544640994576 state(reach1_level_downstream_t20)
+7.725544640994576 state(reach2_level_upstream_t20)
<= -0.3862772320497288

c_u_gate_head_linear(gate2_t21)_:
+1 flow(gate2_t21)
-1.5451089281989152 gate_opening(gate2_t21)
-7.725544640994576 state(reach1_level_downstream_t21)
+7.725544640994576 state(reach2_level_upstream_t21)
<= -0.3862772320497288

c_u_gate_head_linear(gate2_t22)_:
+1 flow(gate2_t22)
-1.5451089281989152 gate_opening(gate2_t22)
-7.725544640994576 state(reach1_level_downstream_t22)
+7.725544640994576 state(reach2_level_upstream_t22)
<= -0.3862772320497288

c_u_gate_head_linear(gate2_t23)_:
+1 flow(gate2_t23)
-1.5451089281989152 gate_opening(gate2_t23)
-7.725544640994576 state(reach1_level_downstream_t23)
+7.725544640994576 state(reach2_level_upstream_t23)
<= -0.3862772320497288

c_u_gate_head_linear(gate2_t24)_:
+1 flow(gate2_t24)
-1.5451089281989152 gate_opening(gate2_t24)
-7.725544640994576 state(reach1_level_downstream_t24)
+7.725544640994576 state(reach2_level_upstream_t24)
<= -0.3862772320497288

c_e_channel_downstream_discharge_alignment(reach1_t01)_:
-1 flow(gate2_t01)
+1 state(reach1_discharge_downstream_t01)
= 0

c_e_channel_downstream_discharge_alignment(reach1_t02)_:
-1 flow(gate2_t02)
+1 state(reach1_discharge_downstream_t02)
= 0

c_e_channel_downstream_discharge_alignment(reach1_t03)_:
-1 flow(gate2_t03)
+1 state(reach1_discharge_downstream_t03)
= 0

c_e_channel_downstream_discharge_alignment(reach1_t04)_:
-1 flow(gate2_t04)
+1 state(reach1_discharge_downstream_t04)
= 0

c_e_channel_downstream_discharge_alignment(reach1_t05)_:
-1 flow(gate2_t05)
+1 state(reach1_discharge_downstream_t05)
= 0

c_e_channel_downstream_discharge_alignment(reach1_t06)_:
-1 flow(gate2_t06)
+1 state(reach1_discharge_downstream_t06)
= 0

c_e_channel_downstream_discharge_alignment(reach1_t07)_:
-1 flow(gate2_t07)
+1 state(reach1_discharge_downstream_t07)
= 0

c_e_channel_downstream_discharge_alignment(reach1_t08)_:
-1 flow(gate2_t08)
+1 state(reach1_discharge_downstream_t08)
= 0

c_e_channel_downstream_discharge_alignment(reach1_t09)_:
-1 flow(gate2_t09)
+1 state(reach1_discharge_downstream_t09)
= 0

c_e_channel_downstream_discharge_alignment(reach1_t10)_:
-1 flow(gate2_t10)
+1 state(reach1_discharge_downstream_t10)
= 0

c_e_channel_downstream_discharge_alignment(reach1_t11)_:
-1 flow(gate2_t11)
+1 state(reach1_discharge_downstream_t11)
= 0

c_e_channel_downstream_discharge_alignment(reach1_t12)_:
-1 flow(gate2_t12)
+1 state(reach1_discharge_downstream_t12)
= 0

c_e_channel_downstream_discharge_alignment(reach1_t13)_:
-1 flow(gate2_t13)
+1 state(reach1_discharge_downstream_t13)
= 0

c_e_channel_downstream_discharge_alignment(reach1_t14)_:
-1 flow(gate2_t14)
+1 state(reach1_discharge_downstream_t14)
= 0

c_e_channel_downstream_discharge_alignment(reach1_t15)_:
-1 flow(gate2_t15)
+1 state(reach1_discharge_downstream_t15)
= 0

c_e_channel_downstream_discharge_alignment(reach1_t16)_:
-1 flow(gate2_t16)
+1 state(reach1_discharge_downstream_t16)
= 0

c_e_channel_downstream_discharge_alignment(reach1_t17)_:
-1 flow(gate2_t17)
+1 state(reach1_discharge_downstream_t17)
= 0

c_e_channel_downstream_discharge_alignment(reach1_t18)_:
-1 flow(gate2_t18)
+1 state(reach1_discharge_downstream_t18)
= 0

c_e_channel_downstream_discharge_alignment(reach1_t19)_:
-1 flow(gate2_t19)
+1 state(reach1_discharge_downstream_t19)
= 0

c_e_channel_downstream_discharge_alignment(reach1_t20)_:
-1 flow(gate2_t20)
+1 state(reach1_discharge_downstream_t20)
= 0

c_e_channel_downstream_discharge_alignment(reach1_t21)_:
-1 flow(gate2_t21)
+1 state(reach1_discharge_downstream_t21)
= 0

c_e_channel_downstream_discharge_alignment(reach1_t22)_:
-1 flow(gate2_t22)
+1 state(reach1_discharge_downstream_t22)
= 0

c_e_channel_downstream_discharge_alignment(reach1_t23)_:
-1 flow(gate2_t23)
+1 state(reach1_discharge_downstream_t23)
= 0

c_e_channel_downstream_discharge_alignment(reach1_t24)_:
-1 flow(gate2_t24)
+1 state(reach1_discharge_downstream_t24)
= 0

c_e_channel_downstream_discharge_alignment(reach2_t01)_:
-1 flow(channel23_t01)
+1 state(reach2_discharge_downstream_t01)
= 0

c_e_channel_downstream_discharge_alignment(reach2_t02)_:
-1 flow(channel23_t02)
+1 state(reach2_discharge_downstream_t02)
= 0

c_e_channel_downstream_discharge_alignment(reach2_t03)_:
-1 flow(channel23_t03)
+1 state(reach2_discharge_downstream_t03)
= 0

c_e_channel_downstream_discharge_alignment(reach2_t04)_:
-1 flow(channel23_t04)
+1 state(reach2_discharge_downstream_t04)
= 0

c_e_channel_downstream_discharge_alignment(reach2_t05)_:
-1 flow(channel23_t05)
+1 state(reach2_discharge_downstream_t05)
= 0

c_e_channel_downstream_discharge_alignment(reach2_t06)_:
-1 flow(channel23_t06)
+1 state(reach2_discharge_downstream_t06)
= 0

c_e_channel_downstream_discharge_alignment(reach2_t07)_:
-1 flow(channel23_t07)
+1 state(reach2_discharge_downstream_t07)
= 0

c_e_channel_downstream_discharge_alignment(reach2_t08)_:
-1 flow(channel23_t08)
+1 state(reach2_discharge_downstream_t08)
= 0

c_e_channel_downstream_discharge_alignment(reach2_t09)_:
-1 flow(channel23_t09)
+1 state(reach2_discharge_downstream_t09)
= 0

c_e_channel_downstream_discharge_alignment(reach2_t10)_:
-1 flow(channel23_t10)
+1 state(reach2_discharge_downstream_t10)
= 0

c_e_channel_downstream_discharge_alignment(reach2_t11)_:
-1 flow(channel23_t11)
+1 state(reach2_discharge_downstream_t11)
= 0

c_e_channel_downstream_discharge_alignment(reach2_t12)_:
-1 flow(channel23_t12)
+1 state(reach2_discharge_downstream_t12)
= 0

c_e_channel_downstream_discharge_alignment(reach2_t13)_:
-1 flow(channel23_t13)
+1 state(reach2_discharge_downstream_t13)
= 0

c_e_channel_downstream_discharge_alignment(reach2_t14)_:
-1 flow(channel23_t14)
+1 state(reach2_discharge_downstream_t14)
= 0

c_e_channel_downstream_discharge_alignment(reach2_t15)_:
-1 flow(channel23_t15)
+1 state(reach2_discharge_downstream_t15)
= 0

c_e_channel_downstream_discharge_alignment(reach2_t16)_:
-1 flow(channel23_t16)
+1 state(reach2_discharge_downstream_t16)
= 0

c_e_channel_downstream_discharge_alignment(reach2_t17)_:
-1 flow(channel23_t17)
+1 state(reach2_discharge_downstream_t17)
= 0

c_e_channel_downstream_discharge_alignment(reach2_t18)_:
-1 flow(channel23_t18)
+1 state(reach2_discharge_downstream_t18)
= 0

c_e_channel_downstream_discharge_alignment(reach2_t19)_:
-1 flow(channel23_t19)
+1 state(reach2_discharge_downstream_t19)
= 0

c_e_channel_downstream_discharge_alignment(reach2_t20)_:
-1 flow(channel23_t20)
+1 state(reach2_discharge_downstream_t20)
= 0

c_e_channel_downstream_discharge_alignment(reach2_t21)_:
-1 flow(channel23_t21)
+1 state(reach2_discharge_downstream_t21)
= 0

c_e_channel_downstream_discharge_alignment(reach2_t22)_:
-1 flow(channel23_t22)
+1 state(reach2_discharge_downstream_t22)
= 0

c_e_channel_downstream_discharge_alignment(reach2_t23)_:
-1 flow(channel23_t23)
+1 state(reach2_discharge_downstream_t23)
= 0

c_e_channel_downstream_discharge_alignment(reach2_t24)_:
-1 flow(channel23_t24)
+1 state(reach2_discharge_downstream_t24)
= 0

c_e_channel_upstream_discharge_alignment(reach1_t01)_:
-1 flow(gate1_t01)
+1 state(reach1_discharge_upstream_t01)
= 0

c_e_channel_upstream_discharge_alignment(reach1_t02)_:
-1 flow(gate1_t02)
+1 state(reach1_discharge_upstream_t02)
= 0

c_e_channel_upstream_discharge_alignment(reach1_t03)_:
-1 flow(gate1_t03)
+1 state(reach1_discharge_upstream_t03)
= 0

c_e_channel_upstream_discharge_alignment(reach1_t04)_:
-1 flow(gate1_t04)
+1 state(reach1_discharge_upstream_t04)
= 0

c_e_channel_upstream_discharge_alignment(reach1_t05)_:
-1 flow(gate1_t05)
+1 state(reach1_discharge_upstream_t05)
= 0

c_e_channel_upstream_discharge_alignment(reach1_t06)_:
-1 flow(gate1_t06)
+1 state(reach1_discharge_upstream_t06)
= 0

c_e_channel_upstream_discharge_alignment(reach1_t07)_:
-1 flow(gate1_t07)
+1 state(reach1_discharge_upstream_t07)
= 0

c_e_channel_upstream_discharge_alignment(reach1_t08)_:
-1 flow(gate1_t08)
+1 state(reach1_discharge_upstream_t08)
= 0

c_e_channel_upstream_discharge_alignment(reach1_t09)_:
-1 flow(gate1_t09)
+1 state(reach1_discharge_upstream_t09)
= 0

c_e_channel_upstream_discharge_alignment(reach1_t10)_:
-1 flow(gate1_t10)
+1 state(reach1_discharge_upstream_t10)
= 0

c_e_channel_upstream_discharge_alignment(reach1_t11)_:
-1 flow(gate1_t11)
+1 state(reach1_discharge_upstream_t11)
= 0

c_e_channel_upstream_discharge_alignment(reach1_t12)_:
-1 flow(gate1_t12)
+1 state(reach1_discharge_upstream_t12)
= 0

c_e_channel_upstream_discharge_alignment(reach1_t13)_:
-1 flow(gate1_t13)
+1 state(reach1_discharge_upstream_t13)
= 0

c_e_channel_upstream_discharge_alignment(reach1_t14)_:
-1 flow(gate1_t14)
+1 state(reach1_discharge_upstream_t14)
= 0

c_e_channel_upstream_discharge_alignment(reach1_t15)_:
-1 flow(gate1_t15)
+1 state(reach1_discharge_upstream_t15)
= 0

c_e_channel_upstream_discharge_alignment(reach1_t16)_:
-1 flow(gate1_t16)
+1 state(reach1_discharge_upstream_t16)
= 0

c_e_channel_upstream_discharge_alignment(reach1_t17)_:
-1 flow(gate1_t17)
+1 state(reach1_discharge_upstream_t17)
= 0

c_e_channel_upstream_discharge_alignment(reach1_t18)_:
-1 flow(gate1_t18)
+1 state(reach1_discharge_upstream_t18)
= 0

c_e_channel_upstream_discharge_alignment(reach1_t19)_:
-1 flow(gate1_t19)
+1 state(reach1_discharge_upstream_t19)
= 0

c_e_channel_upstream_discharge_alignment(reach1_t20)_:
-1 flow(gate1_t20)
+1 state(reach1_discharge_upstream_t20)
= 0

c_e_channel_upstream_discharge_alignment(reach1_t21)_:
-1 flow(gate1_t21)
+1 state(reach1_discharge_upstream_t21)
= 0

c_e_channel_upstream_discharge_alignment(reach1_t22)_:
-1 flow(gate1_t22)
+1 state(reach1_discharge_upstream_t22)
= 0

c_e_channel_upstream_discharge_alignment(reach1_t23)_:
-1 flow(gate1_t23)
+1 state(reach1_discharge_upstream_t23)
= 0

c_e_channel_upstream_discharge_alignment(reach1_t24)_:
-1 flow(gate1_t24)
+1 state(reach1_discharge_upstream_t24)
= 0

c_e_channel_upstream_discharge_alignment(reach2_t01)_:
-1 flow(gate2_t01)
+1 state(reach2_discharge_upstream_t01)
= 0

c_e_channel_upstream_discharge_alignment(reach2_t02)_:
-1 flow(gate2_t02)
+1 state(reach2_discharge_upstream_t02)
= 0

c_e_channel_upstream_discharge_alignment(reach2_t03)_:
-1 flow(gate2_t03)
+1 state(reach2_discharge_upstream_t03)
= 0

c_e_channel_upstream_discharge_alignment(reach2_t04)_:
-1 flow(gate2_t04)
+1 state(reach2_discharge_upstream_t04)
= 0

c_e_channel_upstream_discharge_alignment(reach2_t05)_:
-1 flow(gate2_t05)
+1 state(reach2_discharge_upstream_t05)
= 0

c_e_channel_upstream_discharge_alignment(reach2_t06)_:
-1 flow(gate2_t06)
+1 state(reach2_discharge_upstream_t06)
= 0

c_e_channel_upstream_discharge_alignment(reach2_t07)_:
-1 flow(gate2_t07)
+1 state(reach2_discharge_upstream_t07)
= 0

c_e_channel_upstream_discharge_alignment(reach2_t08)_:
-1 flow(gate2_t08)
+1 state(reach2_discharge_upstream_t08)
= 0

c_e_channel_upstream_discharge_alignment(reach2_t09)_:
-1 flow(gate2_t09)
+1 state(reach2_discharge_upstream_t09)
= 0

c_e_channel_upstream_discharge_alignment(reach2_t10)_:
-1 flow(gate2_t10)
+1 state(reach2_discharge_upstream_t10)
= 0

c_e_channel_upstream_discharge_alignment(reach2_t11)_:
-1 flow(gate2_t11)
+1 state(reach2_discharge_upstream_t11)
= 0

c_e_channel_upstream_discharge_alignment(reach2_t12)_:
-1 flow(gate2_t12)
+1 state(reach2_discharge_upstream_t12)
= 0

c_e_channel_upstream_discharge_alignment(reach2_t13)_:
-1 flow(gate2_t13)
+1 state(reach2_discharge_upstream_t13)
= 0

c_e_channel_upstream_discharge_alignment(reach2_t14)_:
-1 flow(gate2_t14)
+1 state(reach2_discharge_upstream_t14)
= 0

c_e_channel_upstream_discharge_alignment(reach2_t15)_:
-1 flow(gate2_t15)
+1 state(reach2_discharge_upstream_t15)
= 0

c_e_channel_upstream_discharge_alignment(reach2_t16)_:
-1 flow(gate2_t16)
+1 state(reach2_discharge_upstream_t16)
= 0

c_e_channel_upstream_discharge_alignment(reach2_t17)_:
-1 flow(gate2_t17)
+1 state(reach2_discharge_upstream_t17)
= 0

c_e_channel_upstream_discharge_alignment(reach2_t18)_:
-1 flow(gate2_t18)
+1 state(reach2_discharge_upstream_t18)
= 0

c_e_channel_upstream_discharge_alignment(reach2_t19)_:
-1 flow(gate2_t19)
+1 state(reach2_discharge_upstream_t19)
= 0

c_e_channel_upstream_discharge_alignment(reach2_t20)_:
-1 flow(gate2_t20)
+1 state(reach2_discharge_upstream_t20)
= 0

c_e_channel_upstream_discharge_alignment(reach2_t21)_:
-1 flow(gate2_t21)
+1 state(reach2_discharge_upstream_t21)
= 0

c_e_channel_upstream_discharge_alignment(reach2_t22)_:
-1 flow(gate2_t22)
+1 state(reach2_discharge_upstream_t22)
= 0

c_e_channel_upstream_discharge_alignment(reach2_t23)_:
-1 flow(gate2_t23)
+1 state(reach2_discharge_upstream_t23)
= 0

c_e_channel_upstream_discharge_alignment(reach2_t24)_:
-1 flow(gate2_t24)
+1 state(reach2_discharge_upstream_t24)
= 0

c_e_idz_discharge_dynamics(reach1_t04)_:
-0.125 state(reach1_discharge_upstream_t01)
-0.125 state(reach1_discharge_upstream_t02)
-0.75 state(reach1_discharge_downstream_t03)
+1 state(reach1_discharge_downstream_t04)
= 0

c_e_idz_discharge_dynamics(reach1_t05)_:
-0.125 state(reach1_discharge_upstream_t02)
-0.125 state(reach1_discharge_upstream_t03)
-0.75 state(reach1_discharge_downstream_t04)
+1 state(reach1_discharge_downstream_t05)
= 0

c_e_idz_discharge_dynamics(reach1_t06)_:
-0.125 state(reach1_discharge_upstream_t03)
-0.125 state(reach1_discharge_upstream_t04)
-0.75 state(reach1_discharge_downstream_t05)
+1 state(reach1_discharge_downstream_t06)
= 0

c_e_idz_discharge_dynamics(reach1_t07)_:
-0.125 state(reach1_discharge_upstream_t04)
-0.125 state(reach1_discharge_upstream_t05)
-0.75 state(reach1_discharge_downstream_t06)
+1 state(reach1_discharge_downstream_t07)
= 0

c_e_idz_discharge_dynamics(reach1_t08)_:
-0.125 state(reach1_discharge_upstream_t05)
-0.125 state(reach1_discharge_upstream_t06)
-0.75 state(reach1_discharge_downstream_t07)
+1 state(reach1_discharge_downstream_t08)
= 0

c_e_idz_discharge_dynamics(reach1_t09)_:
-0.125 state(reach1_discharge_upstream_t06)
-0.125 state(reach1_discharge_upstream_t07)
-0.75 state(reach1_discharge_downstream_t08)
+1 state(reach1_discharge_downstream_t09)
= 0

c_e_idz_discharge_dynamics(reach1_t10)_:
-0.125 state(reach1_discharge_upstream_t07)
-0.125 state(reach1_discharge_upstream_t08)
-0.75 state(reach1_discharge_downstream_t09)
+1 state(reach1_discharge_downstream_t10)
= 0

c_e_idz_discharge_dynamics(reach1_t11)_:
-0.125 state(reach1_discharge_upstream_t08)
-0.125 state(reach1_discharge_upstream_t09)
-0.75 state(reach1_discharge_downstream_t10)
+1 state(reach1_discharge_downstream_t11)
= 0

c_e_idz_discharge_dynamics(reach1_t12)_:
-0.125 state(reach1_discharge_upstream_t09)
-0.125 state(reach1_discharge_upstream_t10)
-0.75 state(reach1_discharge_downstream_t11)
+1 state(reach1_discharge_downstream_t12)
= 0

c_e_idz_discharge_dynamics(reach1_t13)_:
-0.125 state(reach1_discharge_upstream_t10)
-0.125 state(reach1_discharge_upstream_t11)
-0.75 state(reach1_discharge_downstream_t12)
+1 state(reach1_discharge_downstream_t13)
= 0

c_e_idz_discharge_dynamics(reach1_t14)_:
-0.125 state(reach1_discharge_upstream_t11)
-0.125 state(reach1_discharge_upstream_t12)
-0.75 state(reach1_discharge_downstream_t13)
+1 state(reach1_discharge_downstream_t14)
= 0

c_e_idz_discharge_dynamics(reach1_t15)_:
-0.125 state(reach1_discharge_upstream_t12)
-0.125 state(reach1_discharge_upstream_t13)
-0.75 state(reach1_discharge_downstream_t14)
+1 state(reach1_discharge_downstream_t15)
= 0

c_e_idz_discharge_dynamics(reach1_t16)_:
-0.125 state(reach1_discharge_upstream_t13)
-0.125 state(reach1_discharge_upstream_t14)
-0.75 state(reach1_discharge_downstream_t15)
+1 state(reach1_discharge_downstream_t16)
= 0

c_e_idz_discharge_dynamics(reach1_t17)_:
-0.125 state(reach1_discharge_upstream_t14)
-0.125 state(reach1_discharge_upstream_t15)
-0.75 state(reach1_discharge_downstream_t16)
+1 state(reach1_discharge_downstream_t17)
= 0

c_e_idz_discharge_dynamics(reach1_t18)_:
-0.125 state(reach1_discharge_upstream_t15)
-0.125 state(reach1_discharge_upstream_t16)
-0.75 state(reach1_discharge_downstream_t17)
+1 state(reach1_discharge_downstream_t18)
= 0

c_e_idz_discharge_dynamics(reach1_t19)_:
-0.125 state(reach1_discharge_upstream_t16)
-0.125 state(reach1_discharge_upstream_t17)
-0.75 state(reach1_discharge_downstream_t18)
+1 state(reach1_discharge_downstream_t19)
= 0

c_e_idz_discharge_dynamics(reach1_t20)_:
-0.125 state(reach1_discharge_upstream_t17)
-0.125 state(reach1_discharge_upstream_t18)
-0.75 state(reach1_discharge_downstream_t19)
+1 state(reach1_discharge_downstream_t20)
= 0

c_e_idz_discharge_dynamics(reach1_t21)_:
-0.125 state(reach1_discharge_upstream_t18)
-0.125 state(reach1_discharge_upstream_t19)
-0.75 state(reach1_discharge_downstream_t20)
+1 state(reach1_discharge_downstream_t21)
= 0

c_e_idz_discharge_dynamics(reach1_t22)_:
-0.125 state(reach1_discharge_upstream_t19)
-0.125 state(reach1_discharge_upstream_t20)
-0.75 state(reach1_discharge_downstream_t21)
+1 state(reach1_discharge_downstream_t22)
= 0

c_e_idz_discharge_dynamics(reach1_t23)_:
-0.125 state(reach1_discharge_upstream_t20)
-0.125 state(reach1_discharge_upstream_t21)
-0.75 state(reach1_discharge_downstream_t22)
+1 state(reach1_discharge_downstream_t23)
= 0

c_e_idz_discharge_dynamics(reach1_t24)_:
-0.125 state(reach1_discharge_upstream_t21)
-0.125 state(reach1_discharge_upstream_t22)
-0.75 state(reach1_discharge_downstream_t23)
+1 state(reach1_discharge_downstream_t24)
= 0

c_e_idz_discharge_dynamics(reach2_t06)_:
-0.03333333333333333 state(reach2_discharge_upstream_t01)
-0.16666666666666669 state(reach2_discharge_upstream_t02)
-0.8 state(reach2_discharge_downstream_t05)
+1 state(reach2_discharge_downstream_t06)
= 0

c_e_idz_discharge_dynamics(reach2_t07)_:
-0.03333333333333333 state(reach2_discharge_upstream_t02)
-0.16666666666666669 state(reach2_discharge_upstream_t03)
-0.8 state(reach2_discharge_downstream_t06)
+1 state(reach2_discharge_downstream_t07)
= 0

c_e_idz_discharge_dynamics(reach2_t08)_:
-0.03333333333333333 state(reach2_discharge_upstream_t03)
-0.16666666666666669 state(reach2_discharge_upstream_t04)
-0.8 state(reach2_discharge_downstream_t07)
+1 state(reach2_discharge_downstream_t08)
= 0

c_e_idz_discharge_dynamics(reach2_t09)_:
-0.03333333333333333 state(reach2_discharge_upstream_t04)
-0.16666666666666669 state(reach2_discharge_upstream_t05)
-0.8 state(reach2_discharge_downstream_t08)
+1 state(reach2_discharge_downstream_t09)
= 0

c_e_idz_discharge_dynamics(reach2_t10)_:
-0.03333333333333333 state(reach2_discharge_upstream_t05)
-0.16666666666666669 state(reach2_discharge_upstream_t06)
-0.8 state(reach2_discharge_downstream_t09)
+1 state(reach2_discharge_downstream_t10)
= 0

c_e_idz_discharge_dynamics(reach2_t11)_:
-0.03333333333333333 state(reach2_discharge_upstream_t06)
-0.16666666666666669 state(reach2_discharge_upstream_t07)
-0.8 state(reach2_discharge_downstream_t10)
+1 state(reach2_discharge_downstream_t11)
= 0

c_e_idz_discharge_dynamics(reach2_t12)_:
-0.03333333333333333 state(reach2_discharge_upstream_t07)
-0.16666666666666669 state(reach2_discharge_upstream_t08)
-0.8 state(reach2_discharge_downstream_t11)
+1 state(reach2_discharge_downstream_t12)
= 0

c_e_idz_discharge_dynamics(reach2_t13)_:
-0.03333333333333333 state(reach2_discharge_upstream_t08)
-0.16666666666666669 state(reach2_discharge_upstream_t09)
-0.8 state(reach2_discharge_downstream_t12)
+1 state(reach2_discharge_downstream_t13)
= 0

c_e_idz_discharge_dynamics(reach2_t14)_:
-0.03333333333333333 state(reach2_discharge_upstream_t09)
-0.16666666666666669 state(reach2_discharge_upstream_t10)
-0.8 state(reach2_discharge_downstream_t13)
+1 state(reach2_discharge_downstream_t14)
= 0

c_e_idz_discharge_dynamics(reach2_t15)_:
-0.03333333333333333 state(reach2_discharge_upstream_t10)
-0.16666666666666669 state(reach2_discharge_upstream_t11)
-0.8 state(reach2_discharge_downstream_t14)
+1 state(reach2_discharge_downstream_t15)
= 0

c_e_idz_discharge_dynamics(reach2_t16)_:
-0.03333333333333333 state(reach2_discharge_upstream_t11)
-0.16666666666666669 state(reach2_discharge_upstream_t12)
-0.8 state(reach2_discharge_downstream_t15)
+1 state(reach2_discharge_downstream_t16)
= 0

c_e_idz_discharge_dynamics(reach2_t17)_:
-0.03333333333333333 state(reach2_discharge_upstream_t12)
-0.16666666666666669 state(reach2_discharge_upstream_t13)
-0.8 state(reach2_discharge_downstream_t16)
+1 state(reach2_discharge_downstream_t17)
= 0

c_e_idz_discharge_dynamics(reach2_t18)_:
-0.03333333333333333 state(reach2_discharge_upstream_t13)
-0.16666666666666669 state(reach2_discharge_upstream_t14)
-0.8 state(reach2_discharge_downstream_t17)
+1 state(reach2_discharge_downstream_t18)
= 0

c_e_idz_discharge_dynamics(reach2_t19)_:
-0.03333333333333333 state(reach2_discharge_upstream_t14)
-0.16666666666666669 state(reach2_discharge_upstream_t15)
-0.8 state(reach2_discharge_downstream_t18)
+1 state(reach2_discharge_downstream_t19)
= 0

c_e_idz_discharge_dynamics(reach2_t20)_:
-0.03333333333333333 state(reach2_discharge_upstream_t15)
-0.16666666666666669 state(reach2_discharge_upstream_t16)
-0.8 state(reach2_discharge_downstream_t19)
+1 state(reach2_discharge_downstream_t20)
= 0

c_e_idz_discharge_dynamics(reach2_t21)_:
-0.03333333333333333 state(reach2_discharge_upstream_t16)
-0.16666666666666669 state(reach2_discharge_upstream_t17)
-0.8 state(reach2_discharge_downstream_t20)
+1 state(reach2_discharge_downstream_t21)
= 0

c_e_idz_discharge_dynamics(reach2_t22)_:
-0.03333333333333333 state(reach2_discharge_upstream_t17)
-0.16666666666666669 state(reach2_discharge_upstream_t18)
-0.8 state(reach2_discharge_downstream_t21)
+1 state(reach2_discharge_downstream_t22)
= 0

c_e_idz_discharge_dynamics(reach2_t23)_:
-0.03333333333333333 state(reach2_discharge_upstream_t18)
-0.16666666666666669 state(reach2_discharge_upstream_t19)
-0.8 state(reach2_discharge_downstream_t22)
+1 state(reach2_discharge_downstream_t23)
= 0

c_e_idz_discharge_dynamics(reach2_t24)_:
-0.03333333333333333 state(reach2_discharge_upstream_t19)
-0.16666666666666669 state(reach2_discharge_upstream_t20)
-0.8 state(reach2_discharge_downstream_t23)
+1 state(reach2_discharge_downstream_t24)
= 0

c_e_mass_balance(reservoir_t01)_:
+1 flow(gate1_t01)
+1 state(reservoir_storage_t01)
= 555.0

c_e_mass_balance(reservoir_t02)_:
+1 flow(gate1_t02)
-1 state(reservoir_storage_t01)
+1 state(reservoir_storage_t02)
= 54.0

c_e_mass_balance(reservoir_t03)_:
+1 flow(gate1_t03)
-1 state(reservoir_storage_t02)
+1 state(reservoir_storage_t03)
= 53.0

c_e_mass_balance(reservoir_t04)_:
+1 flow(gate1_t04)
-1 state(reservoir_storage_t03)
+1 state(reservoir_storage_t04)
= 52.0

c_e_mass_balance(reservoir_t05)_:
+1 flow(gate1_t05)
-1 state(reservoir_storage_t04)
+1 state(reservoir_storage_t05)
= 51.0

c_e_mass_balance(reservoir_t06)_:
+1 flow(gate1_t06)
-1 state(reservoir_storage_t05)
+1 state(reservoir_storage_t06)
= 50.0

c_e_mass_balance(reservoir_t07)_:
+1 flow(gate1_t07)
-1 state(reservoir_storage_t06)
+1 state(reservoir_storage_t07)
= 50.0

c_e_mass_balance(reservoir_t08)_:
+1 flow(gate1_t08)
-1 state(reservoir_storage_t07)
+1 state(reservoir_storage_t08)
= 49.0

c_e_mass_balance(reservoir_t09)_:
+1 flow(gate1_t09)
-1 state(reservoir_storage_t08)
+1 state(reservoir_storage_t09)
= 48.0

c_e_mass_balance(reservoir_t10)_:
+1 flow(gate1_t10)
-1 state(reservoir_storage_t09)
+1 state(reservoir_storage_t10)
= 47.5

c_e_mass_balance(reservoir_t11)_:
+1 flow(gate1_t11)
-1 state(reservoir_storage_t10)
+1 state(reservoir_storage_t11)
= 47.0

c_e_mass_balance(reservoir_t12)_:
+1 flow(gate1_t12)
-1 state(reservoir_storage_t11)
+1 state(reservoir_storage_t12)
= 46.0

c_e_mass_balance(reservoir_t13)_:
+1 flow(gate1_t13)
-1 state(reservoir_storage_t12)
+1 state(reservoir_storage_t13)
= 45.0

c_e_mass_balance(reservoir_t14)_:
+1 flow(gate1_t14)
-1 state(reservoir_storage_t13)
+1 state(reservoir_storage_t14)
= 45.0

c_e_mass_balance(reservoir_t15)_:
+1 flow(gate1_t15)
-1 state(reservoir_storage_t14)
+1 state(reservoir_storage_t15)
= 44.0

c_e_mass_balance(reservoir_t16)_:
+1 flow(gate1_t16)
-1 state(reservoir_storage_t15)
+1 state(reservoir_storage_t16)
= 44.0

c_e_mass_balance(reservoir_t17)_:
+1 flow(gate1_t17)
-1 state(reservoir_storage_t16)
+1 state(reservoir_storage_t17)
= 43.5

c_e_mass_balance(reservoir_t18)_:
+1 flow(gate1_t18)
-1 state(reservoir_storage_t17)
+1 state(reservoir_storage_t18)
= 43.0

c_e_mass_balance(reservoir_t19)_:
+1 flow(gate1_t19)
-1 state(reservoir_storage_t18)
+1 state(reservoir_storage_t19)
= 42.5

c_e_mass_balance(reservoir_t20)_:
+1 flow(gate1_t20)
-1 state(reservoir_storage_t19)
+1 state(reservoir_storage_t20)
= 42.0

c_e_mass_balance(reservoir_t21)_:
+1 flow(gate1_t21)
-1 state(reservoir_storage_t20)
+1 state(reservoir_storage_t21)
= 42.0

c_e_mass_balance(reservoir_t22)_:
+1 flow(gate1_t22)
-1 state(reservoir_storage_t21)
+1 state(reservoir_storage_t22)
= 41.5

c_e_mass_balance(reservoir_t23)_:
+1 flow(gate1_t23)
-1 state(reservoir_storage_t22)
+1 state(reservoir_storage_t23)
= 41.0

c_e_mass_balance(reservoir_t24)_:
+1 flow(gate1_t24)
-1 state(reservoir_storage_t23)
+1 state(reservoir_storage_t24)
= 41.0

c_e_mass_balance(reach1_t01)_:
-1 flow(gate1_t01)
+1 flow(gate2_t01)
+1 state(reach1_storage_t01)
= 259.8

c_e_mass_balance(reach1_t02)_:
-1 flow(gate1_t02)
+1 flow(gate2_t02)
-1 state(reach1_storage_t01)
+1 state(reach1_storage_t02)
= -0.2

c_e_mass_balance(reach1_t03)_:
-1 flow(gate1_t03)
+1 flow(gate2_t03)
-1 state(reach1_storage_t02)
+1 state(reach1_storage_t03)
= -0.2

c_e_mass_balance(reach1_t04)_:
-1 flow(gate1_t04)
+1 flow(gate2_t04)
-1 state(reach1_storage_t03)
+1 state(reach1_storage_t04)
= -0.2

c_e_mass_balance(reach1_t05)_:
-1 flow(gate1_t05)
+1 flow(gate2_t05)
-1 state(reach1_storage_t04)
+1 state(reach1_storage_t05)
= -0.2

c_e_mass_balance(reach1_t06)_:
-1 flow(gate1_t06)
+1 flow(gate2_t06)
-1 state(reach1_storage_t05)
+1 state(reach1_storage_t06)
= -0.2

c_e_mass_balance(reach1_t07)_:
-1 flow(gate1_t07)
+1 flow(gate2_t07)
-1 state(reach1_storage_t06)
+1 state(reach1_storage_t07)
= -0.2

c_e_mass_balance(reach1_t08)_:
-1 flow(gate1_t08)
+1 flow(gate2_t08)
-1 state(reach1_storage_t07)
+1 state(reach1_storage_t08)
= -0.2

c_e_mass_balance(reach1_t09)_:
-1 flow(gate1_t09)
+1 flow(gate2_t09)
-1 state(reach1_storage_t08)
+1 state(reach1_storage_t09)
= -1.0

c_e_mass_balance(reach1_t10)_:
-1 flow(gate1_t10)
+1 flow(gate2_t10)
-1 state(reach1_storage_t09)
+1 state(reach1_storage_t10)
= -1.0

c_e_mass_balance(reach1_t11)_:
-1 flow(gate1_t11)
+1 flow(gate2_t11)
-1 state(reach1_storage_t10)
+1 state(reach1_storage_t11)
= -1.0

c_e_mass_balance(reach1_t12)_:
-1 flow(gate1_t12)
+1 flow(gate2_t12)
-1 state(reach1_storage_t11)
+1 state(reach1_storage_t12)
= -1.0

c_e_mass_balance(reach1_t13)_:
-1 flow(gate1_t13)
+1 flow(gate2_t13)
-1 state(reach1_storage_t12)
+1 state(reach1_storage_t13)
= -1.0

c_e_mass_balance(reach1_t14)_:
-1 flow(gate1_t14)
+1 flow(gate2_t14)
-1 state(reach1_storage_t13)
+1 state(reach1_storage_t14)
= -1.0

c_e_mass_balance(reach1_t15)_:
-1 flow(gate1_t15)
+1 flow(gate2_t15)
-1 state(reach1_storage_t14)
+1 state(reach1_storage_t15)
= -1.0

c_e_mass_balance(reach1_t16)_:
-1 flow(gate1_t16)
+1 flow(gate2_t16)
-1 state(reach1_storage_t15)
+1 state(reach1_storage_t16)
= -1.0

c_e_mass_balance(reach1_t17)_:
-1 flow(gate1_t17)
+1 flow(gate2_t17)
-1 state(reach1_storage_t16)
+1 state(reach1_storage_t17)
= -0.35

c_e_mass_balance(reach1_t18)_:
-1 flow(gate1_t18)
+1 flow(gate2_t18)
-1 state(reach1_storage_t17)
+1 state(reach1_storage_t18)
= -0.35

c_e_mass_balance(reach1_t19)_:
-1 flow(gate1_t19)
+1 flow(gate2_t19)
-1 state(reach1_storage_t18)
+1 state(reach1_storage_t19)
= -0.35

c_e_mass_balance(reach1_t20)_:
-1 flow(gate1_t20)
+1 flow(gate2_t20)
-1 state(reach1_storage_t19)
+1 state(reach1_storage_t20)
= -0.35

c_e_mass_balance(reach1_t21)_:
-1 flow(gate1_t21)
+1 flow(gate2_t21)
-1 state(reach1_storage_t20)
+1 state(reach1_storage_t21)
= -0.35

c_e_mass_balance(reach1_t22)_:
-1 flow(gate1_t22)
+1 flow(gate2_t22)
-1 state(reach1_storage_t21)
+1 state(reach1_storage_t22)
= -0.35

c_e_mass_balance(reach1_t23)_:
-1 flow(gate1_t23)
+1 flow(gate2_t23)
-1 state(reach1_storage_t22)
+1 state(reach1_storage_t23)
= -0.35

c_e_mass_balance(reach1_t24)_:
-1 flow(gate1_t24)
+1 flow(gate2_t24)
-1 state(reach1_storage_t23)
+1 state(reach1_storage_t24)
= -0.35

c_e_mass_balance(reach2_t01)_:
-1 flow(gate2_t01)
+1 flow(channel23_t01)
+1 state(reach2_storage_t01)
= 329.75

c_e_mass_balance(reach2_t02)_:
-1 flow(gate2_t02)
+1 flow(channel23_t02)
-1 state(reach2_storage_t01)
+1 state(reach2_storage_t02)
= -0.25

c_e_mass_balance(reach2_t03)_:
-1 flow(gate2_t03)
+1 flow(channel23_t03)
-1 state(reach2_storage_t02)
+1 state(reach2_storage_t03)
= -0.25

c_e_mass_balance(reach2_t04)_:
-1 flow(gate2_t04)
+1 flow(channel23_t04)
-1 state(reach2_storage_t03)
+1 state(reach2_storage_t04)
= -0.25

c_e_mass_balance(reach2_t05)_:
-1 flow(gate2_t05)
+1 flow(channel23_t05)
-1 state(reach2_storage_t04)
+1 state(reach2_storage_t05)
= -0.25

c_e_mass_balance(reach2_t06)_:
-1 flow(gate2_t06)
+1 flow(channel23_t06)
-1 state(reach2_storage_t05)
+1 state(reach2_storage_t06)
= -0.25

c_e_mass_balance(reach2_t07)_:
-1 flow(gate2_t07)
+1 flow(channel23_t07)
-1 state(reach2_storage_t06)
+1 state(reach2_storage_t07)
= -0.25

c_e_mass_balance(reach2_t08)_:
-1 flow(gate2_t08)
+1 flow(channel23_t08)
-1 state(reach2_storage_t07)
+1 state(reach2_storage_t08)
= -0.25

c_e_mass_balance(reach2_t09)_:
-1 flow(gate2_t09)
+1 flow(channel23_t09)
-1 state(reach2_storage_t08)
+1 state(reach2_storage_t09)
= -0.75

c_e_mass_balance(reach2_t10)_:
-1 flow(gate2_t10)
+1 flow(channel23_t10)
-1 state(reach2_storage_t09)
+1 state(reach2_storage_t10)
= -0.75

c_e_mass_balance(reach2_t11)_:
-1 flow(gate2_t11)
+1 flow(channel23_t11)
-1 state(reach2_storage_t10)
+1 state(reach2_storage_t11)
= -0.75

c_e_mass_balance(reach2_t12)_:
-1 flow(gate2_t12)
+1 flow(channel23_t12)
-1 state(reach2_storage_t11)
+1 state(reach2_storage_t12)
= -0.75

c_e_mass_balance(reach2_t13)_:
-1 flow(gate2_t13)
+1 flow(channel23_t13)
-1 state(reach2_storage_t12)
+1 state(reach2_storage_t13)
= -0.75

c_e_mass_balance(reach2_t14)_:
-1 flow(gate2_t14)
+1 flow(channel23_t14)
-1 state(reach2_storage_t13)
+1 state(reach2_storage_t14)
= -0.75

c_e_mass_balance(reach2_t15)_:
-1 flow(gate2_t15)
+1 flow(channel23_t15)
-1 state(reach2_storage_t14)
+1 state(reach2_storage_t15)
= -0.75

c_e_mass_balance(reach2_t16)_:
-1 flow(gate2_t16)
+1 flow(channel23_t16)
-1 state(reach2_storage_t15)
+1 state(reach2_storage_t16)
= -0.75

c_e_mass_balance(reach2_t17)_:
-1 flow(gate2_t17)
+1 flow(channel23_t17)
-1 state(reach2_storage_t16)
+1 state(reach2_storage_t17)
= -0.25

c_e_mass_balance(reach2_t18)_:
-1 flow(gate2_t18)
+1 flow(channel23_t18)
-1 state(reach2_storage_t17)
+1 state(reach2_storage_t18)
= -0.25

c_e_mass_balance(reach2_t19)_:
-1 flow(gate2_t19)
+1 flow(channel23_t19)
-1 state(reach2_storage_t18)
+1 state(reach2_storage_t19)
= -0.25

c_e_mass_balance(reach2_t20)_:
-1 flow(gate2_t20)
+1 flow(channel23_t20)
-1 state(reach2_storage_t19)
+1 state(reach2_storage_t20)
= -0.25

c_e_mass_balance(reach2_t21)_:
-1 flow(gate2_t21)
+1 flow(channel23_t21)
-1 state(reach2_storage_t20)
+1 state(reach2_storage_t21)
= -0.25

c_e_mass_balance(reach2_t22)_:
-1 flow(gate2_t22)
+1 flow(channel23_t22)
-1 state(reach2_storage_t21)
+1 state(reach2_storage_t22)
= -0.25

c_e_mass_balance(reach2_t23)_:
-1 flow(gate2_t23)
+1 flow(channel23_t23)
-1 state(reach2_storage_t22)
+1 state(reach2_storage_t23)
= -0.25

c_e_mass_balance(reach2_t24)_:
-1 flow(gate2_t24)
+1 flow(channel23_t24)
-1 state(reach2_storage_t23)
+1 state(reach2_storage_t24)
= -0.25

c_e_mass_balance(outlet_t01)_:
+1 shortage(outlet_t01)
+1 flow(channel23_t01)
= 30.0

c_e_mass_balance(outlet_t02)_:
+1 shortage(outlet_t02)
+1 flow(channel23_t02)
= 30.0

c_e_mass_balance(outlet_t03)_:
+1 shortage(outlet_t03)
+1 flow(channel23_t03)
= 30.0

c_e_mass_balance(outlet_t04)_:
+1 shortage(outlet_t04)
+1 flow(channel23_t04)
= 32.0

c_e_mass_balance(outlet_t05)_:
+1 shortage(outlet_t05)
+1 flow(channel23_t05)
= 32.0

c_e_mass_balance(outlet_t06)_:
+1 shortage(outlet_t06)
+1 flow(channel23_t06)
= 35.0

c_e_mass_balance(outlet_t07)_:
+1 shortage(outlet_t07)
+1 flow(channel23_t07)
= 35.0

c_e_mass_balance(outlet_t08)_:
+1 shortage(outlet_t08)
+1 flow(channel23_t08)
= 36.0

c_e_mass_balance(outlet_t09)_:
+1 shortage(outlet_t09)
+1 flow(channel23_t09)
= 38.0

c_e_mass_balance(outlet_t10)_:
+1 shortage(outlet_t10)
+1 flow(channel23_t10)
= 40.0

c_e_mass_balance(outlet_t11)_:
+1 shortage(outlet_t11)
+1 flow(channel23_t11)
= 40.0

c_e_mass_balance(outlet_t12)_:
+1 shortage(outlet_t12)
+1 flow(channel23_t12)
= 42.0

c_e_mass_balance(outlet_t13)_:
+1 shortage(outlet_t13)
+1 flow(channel23_t13)
= 45.0

c_e_mass_balance(outlet_t14)_:
+1 shortage(outlet_t14)
+1 flow(channel23_t14)
= 45.0

c_e_mass_balance(outlet_t15)_:
+1 shortage(outlet_t15)
+1 flow(channel23_t15)
= 44.0

c_e_mass_balance(outlet_t16)_:
+1 shortage(outlet_t16)
+1 flow(channel23_t16)
= 42.0

c_e_mass_balance(outlet_t17)_:
+1 shortage(outlet_t17)
+1 flow(channel23_t17)
= 40.0

c_e_mass_balance(outlet_t18)_:
+1 shortage(outlet_t18)
+1 flow(channel23_t18)
= 38.0

c_e_mass_balance(outlet_t19)_:
+1 shortage(outlet_t19)
+1 flow(channel23_t19)
= 36.0

c_e_mass_balance(outlet_t20)_:
+1 shortage(outlet_t20)
+1 flow(channel23_t20)
= 34.0

c_e_mass_balance(outlet_t21)_:
+1 shortage(outlet_t21)
+1 flow(channel23_t21)
= 32.0

c_e_mass_balance(outlet_t22)_:
+1 shortage(outlet_t22)
+1 flow(channel23_t22)
= 30.0

c_e_mass_balance(outlet_t23)_:
+1 shortage(outlet_t23)
+1 flow(channel23_t23)
= 30.0

c_e_mass_balance(outlet_t24)_:
+1 shortage(outlet_t24)
+1 flow(channel23_t24)
= 30.0

c_u_flow_capacity(gate1_t01)_:
+1 flow(gate1_t01)
<= 18.0

c_u_flow_capacity(gate1_t02)_:
+1 flow(gate1_t02)
<= 18.0

c_u_flow_capacity(gate1_t03)_:
+1 flow(gate1_t03)
<= 18.0

c_u_flow_capacity(gate1_t04)_:
+1 flow(gate1_t04)
<= 18.0

c_u_flow_capacity(gate1_t05)_:
+1 flow(gate1_t05)
<= 18.0

c_u_flow_capacity(gate1_t06)_:
+1 flow(gate1_t06)
<= 18.0

c_u_flow_capacity(gate1_t07)_:
+1 flow(gate1_t07)
<= 18.0

c_u_flow_capacity(gate1_t08)_:
+1 flow(gate1_t08)
<= 18.0

c_u_flow_capacity(gate1_t09)_:
+1 flow(gate1_t09)
<= 18.0

c_u_flow_capacity(gate1_t10)_:
+1 flow(gate1_t10)
<= 18.0

c_u_flow_capacity(gate1_t11)_:
+1 flow(gate1_t11)
<= 18.0

c_u_flow_capacity(gate1_t12)_:
+1 flow(gate1_t12)
<= 18.0

c_u_flow_capacity(gate1_t13)_:
+1 flow(gate1_t13)
<= 18.0

c_u_flow_capacity(gate1_t14)_:
+1 flow(gate1_t14)
<= 18.0

c_u_flow_capacity(gate1_t15)_:
+1 flow(gate1_t15)
<= 18.0

c_u_flow_capacity(gate1_t16)_:
+1 flow(gate1_t16)
<= 18.0

c_u_flow_capacity(gate1_t17)_:
+1 flow(gate1_t17)
<= 18.0

c_u_flow_capacity(gate1_t18)_:
+1 flow(gate1_t18)
<= 18.0

c_u_flow_capacity(gate1_t19)_:
+1 flow(gate1_t19)
<= 18.0

c_u_flow_capacity(gate1_t20)_:
+1 flow(gate1_t20)
<= 18.0

c_u_flow_capacity(gate1_t21)_:
+1 flow(gate1_t21)
<= 18.0

c_u_flow_capacity(gate1_t22)_:
+1 flow(gate1_t22)
<= 18.0

c_u_flow_capacity(gate1_t23)_:
+1 flow(gate1_t23)
<= 18.0

c_u_flow_capacity(gate1_t24)_:
+1 flow(gate1_t24)
<= 18.0

c_u_flow_capacity(gate2_t01)_:
+1 flow(gate2_t01)
<= 15.0

c_u_flow_capacity(gate2_t02)_:
+1 flow(gate2_t02)
<= 15.0

c_u_flow_capacity(gate2_t03)_:
+1 flow(gate2_t03)
<= 15.0

c_u_flow_capacity(gate2_t04)_:
+1 flow(gate2_t04)
<= 15.0

c_u_flow_capacity(gate2_t05)_:
+1 flow(gate2_t05)
<= 15.0

c_u_flow_capacity(gate2_t06)_:
+1 flow(gate2_t06)
<= 15.0

c_u_flow_capacity(gate2_t07)_:
+1 flow(gate2_t07)
<= 15.0

c_u_flow_capacity(gate2_t08)_:
+1 flow(gate2_t08)
<= 15.0

c_u_flow_capacity(gate2_t09)_:
+1 flow(gate2_t09)
<= 15.0

c_u_flow_capacity(gate2_t10)_:
+1 flow(gate2_t10)
<= 15.0

c_u_flow_capacity(gate2_t11)_:
+1 flow(gate2_t11)
<= 15.0

c_u_flow_capacity(gate2_t12)_:
+1 flow(gate2_t12)
<= 15.0

c_u_flow_capacity(gate2_t13)_:
+1 flow(gate2_t13)
<= 15.0

c_u_flow_capacity(gate2_t14)_:
+1 flow(gate2_t14)
<= 15.0

c_u_flow_capacity(gate2_t15)_:
+1 flow(gate2_t15)
<= 15.0

c_u_flow_capacity(gate2_t16)_:
+1 flow(gate2_t16)
<= 15.0

c_u_flow_capacity(gate2_t17)_:
+1 flow(gate2_t17)
<= 15.0

c_u_flow_capacity(gate2_t18)_:
+1 flow(gate2_t18)
<= 15.0

c_u_flow_capacity(gate2_t19)_:
+1 flow(gate2_t19)
<= 15.0

c_u_flow_capacity(gate2_t20)_:
+1 flow(gate2_t20)
<= 15.0

c_u_flow_capacity(gate2_t21)_:
+1 flow(gate2_t21)
<= 15.0

c_u_flow_capacity(gate2_t22)_:
+1 flow(gate2_t22)
<= 15.0

c_u_flow_capacity(gate2_t23)_:
+1 flow(gate2_t23)
<= 15.0

c_u_flow_capacity(gate2_t24)_:
+1 flow(gate2_t24)
<= 15.0

c_u_flow_capacity(channel23_t01)_:
+1 flow(channel23_t01)
<= 15.0

c_u_flow_capacity(channel23_t02)_:
+1 flow(channel23_t02)
<= 15.0

c_u_flow_capacity(channel23_t03)_:
+1 flow(channel23_t03)
<= 15.0

c_u_flow_capacity(channel23_t04)_:
+1 flow(channel23_t04)
<= 15.0

c_u_flow_capacity(channel23_t05)_:
+1 flow(channel23_t05)
<= 15.0

c_u_flow_capacity(channel23_t06)_:
+1 flow(channel23_t06)
<= 15.0

c_u_flow_capacity(channel23_t07)_:
+1 flow(channel23_t07)
<= 15.0

c_u_flow_capacity(channel23_t08)_:
+1 flow(channel23_t08)
<= 15.0

c_u_flow_capacity(channel23_t09)_:
+1 flow(channel23_t09)
<= 15.0

c_u_flow_capacity(channel23_t10)_:
+1 flow(channel23_t10)
<= 15.0

c_u_flow_capacity(channel23_t11)_:
+1 flow(channel23_t11)
<= 15.0

c_u_flow_capacity(channel23_t12)_:
+1 flow(channel23_t12)
<= 15.0

c_u_flow_capacity(channel23_t13)_:
+1 flow(channel23_t13)
<= 15.0

c_u_flow_capacity(channel23_t14)_:
+1 flow(channel23_t14)
<= 15.0

c_u_flow_capacity(channel23_t15)_:
+1 flow(channel23_t15)
<= 15.0

c_u_flow_capacity(channel23_t16)_:
+1 flow(channel23_t16)
<= 15.0

c_u_flow_capacity(channel23_t17)_:
+1 flow(channel23_t17)
<= 15.0

c_u_flow_capacity(channel23_t18)_:
+1 flow(channel23_t18)
<= 15.0

c_u_flow_capacity(channel23_t19)_:
+1 flow(channel23_t19)
<= 15.0

c_u_flow_capacity(channel23_t20)_:
+1 flow(channel23_t20)
<= 15.0

c_u_flow_capacity(channel23_t21)_:
+1 flow(channel23_t21)
<= 15.0

c_u_flow_capacity(channel23_t22)_:
+1 flow(channel23_t22)
<= 15.0

c_u_flow_capacity(channel23_t23)_:
+1 flow(channel23_t23)
<= 15.0

c_u_flow_capacity(channel23_t24)_:
+1 flow(channel23_t24)
<= 15.0

c_e_state_piecewise(0)_pw_t01_CC_constraint1_:
+1 state(reservoir_storage_t01)
-300.0 state_piecewise(0)_pw_t01_CC_lambda(1)
-500.0 state_piecewise(0)_pw_t01_CC_lambda(2)
-900.0 state_piecewise(0)_pw_t01_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t01_CC_constraint2_:
+1 state(reservoir_level_t01)
-2.0 state_piecewise(0)_pw_t01_CC_lambda(1)
-2.2 state_piecewise(0)_pw_t01_CC_lambda(2)
-2.35 state_piecewise(0)_pw_t01_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t01_CC_constraint3_:
+1 state_piecewise(0)_pw_t01_CC_lambda(1)
+1 state_piecewise(0)_pw_t01_CC_lambda(2)
+1 state_piecewise(0)_pw_t01_CC_lambda(3)
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
<= 0

c_e_state_piecewise(0)_pw_t01_CC_constraint5_:
+1 state_piecewise(0)_pw_t01_CC_bin_y(1)
+1 state_piecewise(0)_pw_t01_CC_bin_y(2)
= 1

c_e_state_piecewise(0)_pw_t02_CC_constraint1_:
+1 state(reservoir_storage_t02)
-300.0 state_piecewise(0)_pw_t02_CC_lambda(1)
-500.0 state_piecewise(0)_pw_t02_CC_lambda(2)
-900.0 state_piecewise(0)_pw_t02_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t02_CC_constraint2_:
+1 state(reservoir_level_t02)
-2.0 state_piecewise(0)_pw_t02_CC_lambda(1)
-2.2 state_piecewise(0)_pw_t02_CC_lambda(2)
-2.35 state_piecewise(0)_pw_t02_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t02_CC_constraint3_:
+1 state_piecewise(0)_pw_t02_CC_lambda(1)
+1 state_piecewise(0)_pw_t02_CC_lambda(2)
+1 state_piecewise(0)_pw_t02_CC_lambda(3)
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
<= 0

c_e_state_piecewise(0)_pw_t02_CC_constraint5_:
+1 state_piecewise(0)_pw_t02_CC_bin_y(1)
+1 state_piecewise(0)_pw_t02_CC_bin_y(2)
= 1

c_e_state_piecewise(0)_pw_t03_CC_constraint1_:
+1 state(reservoir_storage_t03)
-300.0 state_piecewise(0)_pw_t03_CC_lambda(1)
-500.0 state_piecewise(0)_pw_t03_CC_lambda(2)
-900.0 state_piecewise(0)_pw_t03_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t03_CC_constraint2_:
+1 state(reservoir_level_t03)
-2.0 state_piecewise(0)_pw_t03_CC_lambda(1)
-2.2 state_piecewise(0)_pw_t03_CC_lambda(2)
-2.35 state_piecewise(0)_pw_t03_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t03_CC_constraint3_:
+1 state_piecewise(0)_pw_t03_CC_lambda(1)
+1 state_piecewise(0)_pw_t03_CC_lambda(2)
+1 state_piecewise(0)_pw_t03_CC_lambda(3)
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
<= 0

c_e_state_piecewise(0)_pw_t03_CC_constraint5_:
+1 state_piecewise(0)_pw_t03_CC_bin_y(1)
+1 state_piecewise(0)_pw_t03_CC_bin_y(2)
= 1

c_e_state_piecewise(0)_pw_t04_CC_constraint1_:
+1 state(reservoir_storage_t04)
-300.0 state_piecewise(0)_pw_t04_CC_lambda(1)
-500.0 state_piecewise(0)_pw_t04_CC_lambda(2)
-900.0 state_piecewise(0)_pw_t04_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t04_CC_constraint2_:
+1 state(reservoir_level_t04)
-2.0 state_piecewise(0)_pw_t04_CC_lambda(1)
-2.2 state_piecewise(0)_pw_t04_CC_lambda(2)
-2.35 state_piecewise(0)_pw_t04_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t04_CC_constraint3_:
+1 state_piecewise(0)_pw_t04_CC_lambda(1)
+1 state_piecewise(0)_pw_t04_CC_lambda(2)
+1 state_piecewise(0)_pw_t04_CC_lambda(3)
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
<= 0

c_e_state_piecewise(0)_pw_t04_CC_constraint5_:
+1 state_piecewise(0)_pw_t04_CC_bin_y(1)
+1 state_piecewise(0)_pw_t04_CC_bin_y(2)
= 1

c_e_state_piecewise(0)_pw_t05_CC_constraint1_:
+1 state(reservoir_storage_t05)
-300.0 state_piecewise(0)_pw_t05_CC_lambda(1)
-500.0 state_piecewise(0)_pw_t05_CC_lambda(2)
-900.0 state_piecewise(0)_pw_t05_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t05_CC_constraint2_:
+1 state(reservoir_level_t05)
-2.0 state_piecewise(0)_pw_t05_CC_lambda(1)
-2.2 state_piecewise(0)_pw_t05_CC_lambda(2)
-2.35 state_piecewise(0)_pw_t05_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t05_CC_constraint3_:
+1 state_piecewise(0)_pw_t05_CC_lambda(1)
+1 state_piecewise(0)_pw_t05_CC_lambda(2)
+1 state_piecewise(0)_pw_t05_CC_lambda(3)
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
<= 0

c_e_state_piecewise(0)_pw_t05_CC_constraint5_:
+1 state_piecewise(0)_pw_t05_CC_bin_y(1)
+1 state_piecewise(0)_pw_t05_CC_bin_y(2)
= 1

c_e_state_piecewise(0)_pw_t06_CC_constraint1_:
+1 state(reservoir_storage_t06)
-300.0 state_piecewise(0)_pw_t06_CC_lambda(1)
-500.0 state_piecewise(0)_pw_t06_CC_lambda(2)
-900.0 state_piecewise(0)_pw_t06_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t06_CC_constraint2_:
+1 state(reservoir_level_t06)
-2.0 state_piecewise(0)_pw_t06_CC_lambda(1)
-2.2 state_piecewise(0)_pw_t06_CC_lambda(2)
-2.35 state_piecewise(0)_pw_t06_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t06_CC_constraint3_:
+1 state_piecewise(0)_pw_t06_CC_lambda(1)
+1 state_piecewise(0)_pw_t06_CC_lambda(2)
+1 state_piecewise(0)_pw_t06_CC_lambda(3)
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
<= 0

c_e_state_piecewise(0)_pw_t06_CC_constraint5_:
+1 state_piecewise(0)_pw_t06_CC_bin_y(1)
+1 state_piecewise(0)_pw_t06_CC_bin_y(2)
= 1

c_e_state_piecewise(0)_pw_t07_CC_constraint1_:
+1 state(reservoir_storage_t07)
-300.0 state_piecewise(0)_pw_t07_CC_lambda(1)
-500.0 state_piecewise(0)_pw_t07_CC_lambda(2)
-900.0 state_piecewise(0)_pw_t07_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t07_CC_constraint2_:
+1 state(reservoir_level_t07)
-2.0 state_piecewise(0)_pw_t07_CC_lambda(1)
-2.2 state_piecewise(0)_pw_t07_CC_lambda(2)
-2.35 state_piecewise(0)_pw_t07_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t07_CC_constraint3_:
+1 state_piecewise(0)_pw_t07_CC_lambda(1)
+1 state_piecewise(0)_pw_t07_CC_lambda(2)
+1 state_piecewise(0)_pw_t07_CC_lambda(3)
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
<= 0

c_e_state_piecewise(0)_pw_t07_CC_constraint5_:
+1 state_piecewise(0)_pw_t07_CC_bin_y(1)
+1 state_piecewise(0)_pw_t07_CC_bin_y(2)
= 1

c_e_state_piecewise(0)_pw_t08_CC_constraint1_:
+1 state(reservoir_storage_t08)
-300.0 state_piecewise(0)_pw_t08_CC_lambda(1)
-500.0 state_piecewise(0)_pw_t08_CC_lambda(2)
-900.0 state_piecewise(0)_pw_t08_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t08_CC_constraint2_:
+1 state(reservoir_level_t08)
-2.0 state_piecewise(0)_pw_t08_CC_lambda(1)
-2.2 state_piecewise(0)_pw_t08_CC_lambda(2)
-2.35 state_piecewise(0)_pw_t08_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t08_CC_constraint3_:
+1 state_piecewise(0)_pw_t08_CC_lambda(1)
+1 state_piecewise(0)_pw_t08_CC_lambda(2)
+1 state_piecewise(0)_pw_t08_CC_lambda(3)
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
<= 0

c_e_state_piecewise(0)_pw_t08_CC_constraint5_:
+1 state_piecewise(0)_pw_t08_CC_bin_y(1)
+1 state_piecewise(0)_pw_t08_CC_bin_y(2)
= 1

c_e_state_piecewise(0)_pw_t09_CC_constraint1_:
+1 state(reservoir_storage_t09)
-300.0 state_piecewise(0)_pw_t09_CC_lambda(1)
-500.0 state_piecewise(0)_pw_t09_CC_lambda(2)
-900.0 state_piecewise(0)_pw_t09_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t09_CC_constraint2_:
+1 state(reservoir_level_t09)
-2.0 state_piecewise(0)_pw_t09_CC_lambda(1)
-2.2 state_piecewise(0)_pw_t09_CC_lambda(2)
-2.35 state_piecewise(0)_pw_t09_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t09_CC_constraint3_:
+1 state_piecewise(0)_pw_t09_CC_lambda(1)
+1 state_piecewise(0)_pw_t09_CC_lambda(2)
+1 state_piecewise(0)_pw_t09_CC_lambda(3)
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
<= 0

c_e_state_piecewise(0)_pw_t09_CC_constraint5_:
+1 state_piecewise(0)_pw_t09_CC_bin_y(1)
+1 state_piecewise(0)_pw_t09_CC_bin_y(2)
= 1

c_e_state_piecewise(0)_pw_t10_CC_constraint1_:
+1 state(reservoir_storage_t10)
-300.0 state_piecewise(0)_pw_t10_CC_lambda(1)
-500.0 state_piecewise(0)_pw_t10_CC_lambda(2)
-900.0 state_piecewise(0)_pw_t10_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t10_CC_constraint2_:
+1 state(reservoir_level_t10)
-2.0 state_piecewise(0)_pw_t10_CC_lambda(1)
-2.2 state_piecewise(0)_pw_t10_CC_lambda(2)
-2.35 state_piecewise(0)_pw_t10_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t10_CC_constraint3_:
+1 state_piecewise(0)_pw_t10_CC_lambda(1)
+1 state_piecewise(0)_pw_t10_CC_lambda(2)
+1 state_piecewise(0)_pw_t10_CC_lambda(3)
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
<= 0

c_e_state_piecewise(0)_pw_t10_CC_constraint5_:
+1 state_piecewise(0)_pw_t10_CC_bin_y(1)
+1 state_piecewise(0)_pw_t10_CC_bin_y(2)
= 1

c_e_state_piecewise(0)_pw_t11_CC_constraint1_:
+1 state(reservoir_storage_t11)
-300.0 state_piecewise(0)_pw_t11_CC_lambda(1)
-500.0 state_piecewise(0)_pw_t11_CC_lambda(2)
-900.0 state_piecewise(0)_pw_t11_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t11_CC_constraint2_:
+1 state(reservoir_level_t11)
-2.0 state_piecewise(0)_pw_t11_CC_lambda(1)
-2.2 state_piecewise(0)_pw_t11_CC_lambda(2)
-2.35 state_piecewise(0)_pw_t11_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t11_CC_constraint3_:
+1 state_piecewise(0)_pw_t11_CC_lambda(1)
+1 state_piecewise(0)_pw_t11_CC_lambda(2)
+1 state_piecewise(0)_pw_t11_CC_lambda(3)
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
<= 0

c_e_state_piecewise(0)_pw_t11_CC_constraint5_:
+1 state_piecewise(0)_pw_t11_CC_bin_y(1)
+1 state_piecewise(0)_pw_t11_CC_bin_y(2)
= 1

c_e_state_piecewise(0)_pw_t12_CC_constraint1_:
+1 state(reservoir_storage_t12)
-300.0 state_piecewise(0)_pw_t12_CC_lambda(1)
-500.0 state_piecewise(0)_pw_t12_CC_lambda(2)
-900.0 state_piecewise(0)_pw_t12_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t12_CC_constraint2_:
+1 state(reservoir_level_t12)
-2.0 state_piecewise(0)_pw_t12_CC_lambda(1)
-2.2 state_piecewise(0)_pw_t12_CC_lambda(2)
-2.35 state_piecewise(0)_pw_t12_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t12_CC_constraint3_:
+1 state_piecewise(0)_pw_t12_CC_lambda(1)
+1 state_piecewise(0)_pw_t12_CC_lambda(2)
+1 state_piecewise(0)_pw_t12_CC_lambda(3)
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
<= 0

c_e_state_piecewise(0)_pw_t12_CC_constraint5_:
+1 state_piecewise(0)_pw_t12_CC_bin_y(1)
+1 state_piecewise(0)_pw_t12_CC_bin_y(2)
= 1

c_e_state_piecewise(0)_pw_t13_CC_constraint1_:
+1 state(reservoir_storage_t13)
-300.0 state_piecewise(0)_pw_t13_CC_lambda(1)
-500.0 state_piecewise(0)_pw_t13_CC_lambda(2)
-900.0 state_piecewise(0)_pw_t13_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t13_CC_constraint2_:
+1 state(reservoir_level_t13)
-2.0 state_piecewise(0)_pw_t13_CC_lambda(1)
-2.2 state_piecewise(0)_pw_t13_CC_lambda(2)
-2.35 state_piecewise(0)_pw_t13_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t13_CC_constraint3_:
+1 state_piecewise(0)_pw_t13_CC_lambda(1)
+1 state_piecewise(0)_pw_t13_CC_lambda(2)
+1 state_piecewise(0)_pw_t13_CC_lambda(3)
= 1

c_u_state_piecewise(0)_pw_t13_CC_constraint4(1)_:
+1 state_piecewise(0)_pw_t13_CC_lambda(1)
-1 state_piecewise(0)_pw_t13_CC_bin_y(1)
<= 0

c_u_state_piecewise(0)_pw_t13_CC_constraint4(2)_:
+1 state_piecewise(0)_pw_t13_CC_lambda(2)
-1 state_piecewise(0)_pw_t13_CC_bin_y(1)
-1 state_piecewise(0)_pw_t13_CC_bin_y(2)
<= 0

c_u_state_piecewise(0)_pw_t13_CC_constraint4(3)_:
+1 state_piecewise(0)_pw_t13_CC_lambda(3)
-1 state_piecewise(0)_pw_t13_CC_bin_y(2)
<= 0

c_e_state_piecewise(0)_pw_t13_CC_constraint5_:
+1 state_piecewise(0)_pw_t13_CC_bin_y(1)
+1 state_piecewise(0)_pw_t13_CC_bin_y(2)
= 1

c_e_state_piecewise(0)_pw_t14_CC_constraint1_:
+1 state(reservoir_storage_t14)
-300.0 state_piecewise(0)_pw_t14_CC_lambda(1)
-500.0 state_piecewise(0)_pw_t14_CC_lambda(2)
-900.0 state_piecewise(0)_pw_t14_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t14_CC_constraint2_:
+1 state(reservoir_level_t14)
-2.0 state_piecewise(0)_pw_t14_CC_lambda(1)
-2.2 state_piecewise(0)_pw_t14_CC_lambda(2)
-2.35 state_piecewise(0)_pw_t14_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t14_CC_constraint3_:
+1 state_piecewise(0)_pw_t14_CC_lambda(1)
+1 state_piecewise(0)_pw_t14_CC_lambda(2)
+1 state_piecewise(0)_pw_t14_CC_lambda(3)
= 1

c_u_state_piecewise(0)_pw_t14_CC_constraint4(1)_:
+1 state_piecewise(0)_pw_t14_CC_lambda(1)
-1 state_piecewise(0)_pw_t14_CC_bin_y(1)
<= 0

c_u_state_piecewise(0)_pw_t14_CC_constraint4(2)_:
+1 state_piecewise(0)_pw_t14_CC_lambda(2)
-1 state_piecewise(0)_pw_t14_CC_bin_y(1)
-1 state_piecewise(0)_pw_t14_CC_bin_y(2)
<= 0

c_u_state_piecewise(0)_pw_t14_CC_constraint4(3)_:
+1 state_piecewise(0)_pw_t14_CC_lambda(3)
-1 state_piecewise(0)_pw_t14_CC_bin_y(2)
<= 0

c_e_state_piecewise(0)_pw_t14_CC_constraint5_:
+1 state_piecewise(0)_pw_t14_CC_bin_y(1)
+1 state_piecewise(0)_pw_t14_CC_bin_y(2)
= 1

c_e_state_piecewise(0)_pw_t15_CC_constraint1_:
+1 state(reservoir_storage_t15)
-300.0 state_piecewise(0)_pw_t15_CC_lambda(1)
-500.0 state_piecewise(0)_pw_t15_CC_lambda(2)
-900.0 state_piecewise(0)_pw_t15_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t15_CC_constraint2_:
+1 state(reservoir_level_t15)
-2.0 state_piecewise(0)_pw_t15_CC_lambda(1)
-2.2 state_piecewise(0)_pw_t15_CC_lambda(2)
-2.35 state_piecewise(0)_pw_t15_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t15_CC_constraint3_:
+1 state_piecewise(0)_pw_t15_CC_lambda(1)
+1 state_piecewise(0)_pw_t15_CC_lambda(2)
+1 state_piecewise(0)_pw_t15_CC_lambda(3)
= 1

c_u_state_piecewise(0)_pw_t15_CC_constraint4(1)_:
+1 state_piecewise(0)_pw_t15_CC_lambda(1)
-1 state_piecewise(0)_pw_t15_CC_bin_y(1)
<= 0

c_u_state_piecewise(0)_pw_t15_CC_constraint4(2)_:
+1 state_piecewise(0)_pw_t15_CC_lambda(2)
-1 state_piecewise(0)_pw_t15_CC_bin_y(1)
-1 state_piecewise(0)_pw_t15_CC_bin_y(2)
<= 0

c_u_state_piecewise(0)_pw_t15_CC_constraint4(3)_:
+1 state_piecewise(0)_pw_t15_CC_lambda(3)
-1 state_piecewise(0)_pw_t15_CC_bin_y(2)
<= 0

c_e_state_piecewise(0)_pw_t15_CC_constraint5_:
+1 state_piecewise(0)_pw_t15_CC_bin_y(1)
+1 state_piecewise(0)_pw_t15_CC_bin_y(2)
= 1

c_e_state_piecewise(0)_pw_t16_CC_constraint1_:
+1 state(reservoir_storage_t16)
-300.0 state_piecewise(0)_pw_t16_CC_lambda(1)
-500.0 state_piecewise(0)_pw_t16_CC_lambda(2)
-900.0 state_piecewise(0)_pw_t16_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t16_CC_constraint2_:
+1 state(reservoir_level_t16)
-2.0 state_piecewise(0)_pw_t16_CC_lambda(1)
-2.2 state_piecewise(0)_pw_t16_CC_lambda(2)
-2.35 state_piecewise(0)_pw_t16_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t16_CC_constraint3_:
+1 state_piecewise(0)_pw_t16_CC_lambda(1)
+1 state_piecewise(0)_pw_t16_CC_lambda(2)
+1 state_piecewise(0)_pw_t16_CC_lambda(3)
= 1

c_u_state_piecewise(0)_pw_t16_CC_constraint4(1)_:
+1 state_piecewise(0)_pw_t16_CC_lambda(1)
-1 state_piecewise(0)_pw_t16_CC_bin_y(1)
<= 0

c_u_state_piecewise(0)_pw_t16_CC_constraint4(2)_:
+1 state_piecewise(0)_pw_t16_CC_lambda(2)
-1 state_piecewise(0)_pw_t16_CC_bin_y(1)
-1 state_piecewise(0)_pw_t16_CC_bin_y(2)
<= 0

c_u_state_piecewise(0)_pw_t16_CC_constraint4(3)_:
+1 state_piecewise(0)_pw_t16_CC_lambda(3)
-1 state_piecewise(0)_pw_t16_CC_bin_y(2)
<= 0

c_e_state_piecewise(0)_pw_t16_CC_constraint5_:
+1 state_piecewise(0)_pw_t16_CC_bin_y(1)
+1 state_piecewise(0)_pw_t16_CC_bin_y(2)
= 1

c_e_state_piecewise(0)_pw_t17_CC_constraint1_:
+1 state(reservoir_storage_t17)
-300.0 state_piecewise(0)_pw_t17_CC_lambda(1)
-500.0 state_piecewise(0)_pw_t17_CC_lambda(2)
-900.0 state_piecewise(0)_pw_t17_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t17_CC_constraint2_:
+1 state(reservoir_level_t17)
-2.0 state_piecewise(0)_pw_t17_CC_lambda(1)
-2.2 state_piecewise(0)_pw_t17_CC_lambda(2)
-2.35 state_piecewise(0)_pw_t17_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t17_CC_constraint3_:
+1 state_piecewise(0)_pw_t17_CC_lambda(1)
+1 state_piecewise(0)_pw_t17_CC_lambda(2)
+1 state_piecewise(0)_pw_t17_CC_lambda(3)
= 1

c_u_state_piecewise(0)_pw_t17_CC_constraint4(1)_:
+1 state_piecewise(0)_pw_t17_CC_lambda(1)
-1 state_piecewise(0)_pw_t17_CC_bin_y(1)
<= 0

c_u_state_piecewise(0)_pw_t17_CC_constraint4(2)_:
+1 state_piecewise(0)_pw_t17_CC_lambda(2)
-1 state_piecewise(0)_pw_t17_CC_bin_y(1)
-1 state_piecewise(0)_pw_t17_CC_bin_y(2)
<= 0

c_u_state_piecewise(0)_pw_t17_CC_constraint4(3)_:
+1 state_piecewise(0)_pw_t17_CC_lambda(3)
-1 state_piecewise(0)_pw_t17_CC_bin_y(2)
<= 0

c_e_state_piecewise(0)_pw_t17_CC_constraint5_:
+1 state_piecewise(0)_pw_t17_CC_bin_y(1)
+1 state_piecewise(0)_pw_t17_CC_bin_y(2)
= 1

c_e_state_piecewise(0)_pw_t18_CC_constraint1_:
+1 state(reservoir_storage_t18)
-300.0 state_piecewise(0)_pw_t18_CC_lambda(1)
-500.0 state_piecewise(0)_pw_t18_CC_lambda(2)
-900.0 state_piecewise(0)_pw_t18_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t18_CC_constraint2_:
+1 state(reservoir_level_t18)
-2.0 state_piecewise(0)_pw_t18_CC_lambda(1)
-2.2 state_piecewise(0)_pw_t18_CC_lambda(2)
-2.35 state_piecewise(0)_pw_t18_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t18_CC_constraint3_:
+1 state_piecewise(0)_pw_t18_CC_lambda(1)
+1 state_piecewise(0)_pw_t18_CC_lambda(2)
+1 state_piecewise(0)_pw_t18_CC_lambda(3)
= 1

c_u_state_piecewise(0)_pw_t18_CC_constraint4(1)_:
+1 state_piecewise(0)_pw_t18_CC_lambda(1)
-1 state_piecewise(0)_pw_t18_CC_bin_y(1)
<= 0

c_u_state_piecewise(0)_pw_t18_CC_constraint4(2)_:
+1 state_piecewise(0)_pw_t18_CC_lambda(2)
-1 state_piecewise(0)_pw_t18_CC_bin_y(1)
-1 state_piecewise(0)_pw_t18_CC_bin_y(2)
<= 0

c_u_state_piecewise(0)_pw_t18_CC_constraint4(3)_:
+1 state_piecewise(0)_pw_t18_CC_lambda(3)
-1 state_piecewise(0)_pw_t18_CC_bin_y(2)
<= 0

c_e_state_piecewise(0)_pw_t18_CC_constraint5_:
+1 state_piecewise(0)_pw_t18_CC_bin_y(1)
+1 state_piecewise(0)_pw_t18_CC_bin_y(2)
= 1

c_e_state_piecewise(0)_pw_t19_CC_constraint1_:
+1 state(reservoir_storage_t19)
-300.0 state_piecewise(0)_pw_t19_CC_lambda(1)
-500.0 state_piecewise(0)_pw_t19_CC_lambda(2)
-900.0 state_piecewise(0)_pw_t19_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t19_CC_constraint2_:
+1 state(reservoir_level_t19)
-2.0 state_piecewise(0)_pw_t19_CC_lambda(1)
-2.2 state_piecewise(0)_pw_t19_CC_lambda(2)
-2.35 state_piecewise(0)_pw_t19_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t19_CC_constraint3_:
+1 state_piecewise(0)_pw_t19_CC_lambda(1)
+1 state_piecewise(0)_pw_t19_CC_lambda(2)
+1 state_piecewise(0)_pw_t19_CC_lambda(3)
= 1

c_u_state_piecewise(0)_pw_t19_CC_constraint4(1)_:
+1 state_piecewise(0)_pw_t19_CC_lambda(1)
-1 state_piecewise(0)_pw_t19_CC_bin_y(1)
<= 0

c_u_state_piecewise(0)_pw_t19_CC_constraint4(2)_:
+1 state_piecewise(0)_pw_t19_CC_lambda(2)
-1 state_piecewise(0)_pw_t19_CC_bin_y(1)
-1 state_piecewise(0)_pw_t19_CC_bin_y(2)
<= 0

c_u_state_piecewise(0)_pw_t19_CC_constraint4(3)_:
+1 state_piecewise(0)_pw_t19_CC_lambda(3)
-1 state_piecewise(0)_pw_t19_CC_bin_y(2)
<= 0

c_e_state_piecewise(0)_pw_t19_CC_constraint5_:
+1 state_piecewise(0)_pw_t19_CC_bin_y(1)
+1 state_piecewise(0)_pw_t19_CC_bin_y(2)
= 1

c_e_state_piecewise(0)_pw_t20_CC_constraint1_:
+1 state(reservoir_storage_t20)
-300.0 state_piecewise(0)_pw_t20_CC_lambda(1)
-500.0 state_piecewise(0)_pw_t20_CC_lambda(2)
-900.0 state_piecewise(0)_pw_t20_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t20_CC_constraint2_:
+1 state(reservoir_level_t20)
-2.0 state_piecewise(0)_pw_t20_CC_lambda(1)
-2.2 state_piecewise(0)_pw_t20_CC_lambda(2)
-2.35 state_piecewise(0)_pw_t20_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t20_CC_constraint3_:
+1 state_piecewise(0)_pw_t20_CC_lambda(1)
+1 state_piecewise(0)_pw_t20_CC_lambda(2)
+1 state_piecewise(0)_pw_t20_CC_lambda(3)
= 1

c_u_state_piecewise(0)_pw_t20_CC_constraint4(1)_:
+1 state_piecewise(0)_pw_t20_CC_lambda(1)
-1 state_piecewise(0)_pw_t20_CC_bin_y(1)
<= 0

c_u_state_piecewise(0)_pw_t20_CC_constraint4(2)_:
+1 state_piecewise(0)_pw_t20_CC_lambda(2)
-1 state_piecewise(0)_pw_t20_CC_bin_y(1)
-1 state_piecewise(0)_pw_t20_CC_bin_y(2)
<= 0

c_u_state_piecewise(0)_pw_t20_CC_constraint4(3)_:
+1 state_piecewise(0)_pw_t20_CC_lambda(3)
-1 state_piecewise(0)_pw_t20_CC_bin_y(2)
<= 0

c_e_state_piecewise(0)_pw_t20_CC_constraint5_:
+1 state_piecewise(0)_pw_t20_CC_bin_y(1)
+1 state_piecewise(0)_pw_t20_CC_bin_y(2)
= 1

c_e_state_piecewise(0)_pw_t21_CC_constraint1_:
+1 state(reservoir_storage_t21)
-300.0 state_piecewise(0)_pw_t21_CC_lambda(1)
-500.0 state_piecewise(0)_pw_t21_CC_lambda(2)
-900.0 state_piecewise(0)_pw_t21_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t21_CC_constraint2_:
+1 state(reservoir_level_t21)
-2.0 state_piecewise(0)_pw_t21_CC_lambda(1)
-2.2 state_piecewise(0)_pw_t21_CC_lambda(2)
-2.35 state_piecewise(0)_pw_t21_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t21_CC_constraint3_:
+1 state_piecewise(0)_pw_t21_CC_lambda(1)
+1 state_piecewise(0)_pw_t21_CC_lambda(2)
+1 state_piecewise(0)_pw_t21_CC_lambda(3)
= 1

c_u_state_piecewise(0)_pw_t21_CC_constraint4(1)_:
+1 state_piecewise(0)_pw_t21_CC_lambda(1)
-1 state_piecewise(0)_pw_t21_CC_bin_y(1)
<= 0

c_u_state_piecewise(0)_pw_t21_CC_constraint4(2)_:
+1 state_piecewise(0)_pw_t21_CC_lambda(2)
-1 state_piecewise(0)_pw_t21_CC_bin_y(1)
-1 state_piecewise(0)_pw_t21_CC_bin_y(2)
<= 0

c_u_state_piecewise(0)_pw_t21_CC_constraint4(3)_:
+1 state_piecewise(0)_pw_t21_CC_lambda(3)
-1 state_piecewise(0)_pw_t21_CC_bin_y(2)
<= 0

c_e_state_piecewise(0)_pw_t21_CC_constraint5_:
+1 state_piecewise(0)_pw_t21_CC_bin_y(1)
+1 state_piecewise(0)_pw_t21_CC_bin_y(2)
= 1

c_e_state_piecewise(0)_pw_t22_CC_constraint1_:
+1 state(reservoir_storage_t22)
-300.0 state_piecewise(0)_pw_t22_CC_lambda(1)
-500.0 state_piecewise(0)_pw_t22_CC_lambda(2)
-900.0 state_piecewise(0)_pw_t22_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t22_CC_constraint2_:
+1 state(reservoir_level_t22)
-2.0 state_piecewise(0)_pw_t22_CC_lambda(1)
-2.2 state_piecewise(0)_pw_t22_CC_lambda(2)
-2.35 state_piecewise(0)_pw_t22_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t22_CC_constraint3_:
+1 state_piecewise(0)_pw_t22_CC_lambda(1)
+1 state_piecewise(0)_pw_t22_CC_lambda(2)
+1 state_piecewise(0)_pw_t22_CC_lambda(3)
= 1

c_u_state_piecewise(0)_pw_t22_CC_constraint4(1)_:
+1 state_piecewise(0)_pw_t22_CC_lambda(1)
-1 state_piecewise(0)_pw_t22_CC_bin_y(1)
<= 0

c_u_state_piecewise(0)_pw_t22_CC_constraint4(2)_:
+1 state_piecewise(0)_pw_t22_CC_lambda(2)
-1 state_piecewise(0)_pw_t22_CC_bin_y(1)
-1 state_piecewise(0)_pw_t22_CC_bin_y(2)
<= 0

c_u_state_piecewise(0)_pw_t22_CC_constraint4(3)_:
+1 state_piecewise(0)_pw_t22_CC_lambda(3)
-1 state_piecewise(0)_pw_t22_CC_bin_y(2)
<= 0

c_e_state_piecewise(0)_pw_t22_CC_constraint5_:
+1 state_piecewise(0)_pw_t22_CC_bin_y(1)
+1 state_piecewise(0)_pw_t22_CC_bin_y(2)
= 1

c_e_state_piecewise(0)_pw_t23_CC_constraint1_:
+1 state(reservoir_storage_t23)
-300.0 state_piecewise(0)_pw_t23_CC_lambda(1)
-500.0 state_piecewise(0)_pw_t23_CC_lambda(2)
-900.0 state_piecewise(0)_pw_t23_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t23_CC_constraint2_:
+1 state(reservoir_level_t23)
-2.0 state_piecewise(0)_pw_t23_CC_lambda(1)
-2.2 state_piecewise(0)_pw_t23_CC_lambda(2)
-2.35 state_piecewise(0)_pw_t23_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t23_CC_constraint3_:
+1 state_piecewise(0)_pw_t23_CC_lambda(1)
+1 state_piecewise(0)_pw_t23_CC_lambda(2)
+1 state_piecewise(0)_pw_t23_CC_lambda(3)
= 1

c_u_state_piecewise(0)_pw_t23_CC_constraint4(1)_:
+1 state_piecewise(0)_pw_t23_CC_lambda(1)
-1 state_piecewise(0)_pw_t23_CC_bin_y(1)
<= 0

c_u_state_piecewise(0)_pw_t23_CC_constraint4(2)_:
+1 state_piecewise(0)_pw_t23_CC_lambda(2)
-1 state_piecewise(0)_pw_t23_CC_bin_y(1)
-1 state_piecewise(0)_pw_t23_CC_bin_y(2)
<= 0

c_u_state_piecewise(0)_pw_t23_CC_constraint4(3)_:
+1 state_piecewise(0)_pw_t23_CC_lambda(3)
-1 state_piecewise(0)_pw_t23_CC_bin_y(2)
<= 0

c_e_state_piecewise(0)_pw_t23_CC_constraint5_:
+1 state_piecewise(0)_pw_t23_CC_bin_y(1)
+1 state_piecewise(0)_pw_t23_CC_bin_y(2)
= 1

c_e_state_piecewise(0)_pw_t24_CC_constraint1_:
+1 state(reservoir_storage_t24)
-300.0 state_piecewise(0)_pw_t24_CC_lambda(1)
-500.0 state_piecewise(0)_pw_t24_CC_lambda(2)
-900.0 state_piecewise(0)_pw_t24_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t24_CC_constraint2_:
+1 state(reservoir_level_t24)
-2.0 state_piecewise(0)_pw_t24_CC_lambda(1)
-2.2 state_piecewise(0)_pw_t24_CC_lambda(2)
-2.35 state_piecewise(0)_pw_t24_CC_lambda(3)
= 0

c_e_state_piecewise(0)_pw_t24_CC_constraint3_:
+1 state_piecewise(0)_pw_t24_CC_lambda(1)
+1 state_piecewise(0)_pw_t24_CC_lambda(2)
+1 state_piecewise(0)_pw_t24_CC_lambda(3)
= 1

c_u_state_piecewise(0)_pw_t24_CC_constraint4(1)_:
+1 state_piecewise(0)_pw_t24_CC_lambda(1)
-1 state_piecewise(0)_pw_t24_CC_bin_y(1)
<= 0

c_u_state_piecewise(0)_pw_t24_CC_constraint4(2)_:
+1 state_piecewise(0)_pw_t24_CC_lambda(2)
-1 state_piecewise(0)_pw_t24_CC_bin_y(1)
-1 state_piecewise(0)_pw_t24_CC_bin_y(2)
<= 0

c_u_state_piecewise(0)_pw_t24_CC_constraint4(3)_:
+1 state_piecewise(0)_pw_t24_CC_lambda(3)
-1 state_piecewise(0)_pw_t24_CC_bin_y(2)
<= 0

c_e_state_piecewise(0)_pw_t24_CC_constraint5_:
+1 state_piecewise(0)_pw_t24_CC_bin_y(1)
+1 state_piecewise(0)_pw_t24_CC_bin_y(2)
= 1

c_e_state_piecewise(1)_pw_t01_CC_constraint1_:
+1 state(reach1_discharge_upstream_t01)
-3.8 state_piecewise(1)_pw_t01_CC_lambda(1)
-9.5 state_piecewise(1)_pw_t01_CC_lambda(2)
-13.0 state_piecewise(1)_pw_t01_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t01_CC_constraint2_:
+1 state(reach1_level_upstream_t01)
-2.05 state_piecewise(1)_pw_t01_CC_lambda(1)
-1.9849999999999999 state_piecewise(1)_pw_t01_CC_lambda(2)
-1.92 state_piecewise(1)_pw_t01_CC_lambda(3)
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
+1 state(reach1_discharge_upstream_t02)
-3.8 state_piecewise(1)_pw_t02_CC_lambda(1)
-9.5 state_piecewise(1)_pw_t02_CC_lambda(2)
-13.0 state_piecewise(1)_pw_t02_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t02_CC_constraint2_:
+1 state(reach1_level_upstream_t02)
-2.05 state_piecewise(1)_pw_t02_CC_lambda(1)
-1.9849999999999999 state_piecewise(1)_pw_t02_CC_lambda(2)
-1.92 state_piecewise(1)_pw_t02_CC_lambda(3)
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
+1 state(reach1_discharge_upstream_t03)
-3.8 state_piecewise(1)_pw_t03_CC_lambda(1)
-9.5 state_piecewise(1)_pw_t03_CC_lambda(2)
-13.0 state_piecewise(1)_pw_t03_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t03_CC_constraint2_:
+1 state(reach1_level_upstream_t03)
-2.05 state_piecewise(1)_pw_t03_CC_lambda(1)
-1.9849999999999999 state_piecewise(1)_pw_t03_CC_lambda(2)
-1.92 state_piecewise(1)_pw_t03_CC_lambda(3)
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
+1 state(reach1_discharge_upstream_t04)
-3.8 state_piecewise(1)_pw_t04_CC_lambda(1)
-9.5 state_piecewise(1)_pw_t04_CC_lambda(2)
-13.0 state_piecewise(1)_pw_t04_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t04_CC_constraint2_:
+1 state(reach1_level_upstream_t04)
-2.05 state_piecewise(1)_pw_t04_CC_lambda(1)
-1.9849999999999999 state_piecewise(1)_pw_t04_CC_lambda(2)
-1.92 state_piecewise(1)_pw_t04_CC_lambda(3)
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
+1 state(reach1_discharge_upstream_t05)
-3.8 state_piecewise(1)_pw_t05_CC_lambda(1)
-9.5 state_piecewise(1)_pw_t05_CC_lambda(2)
-13.0 state_piecewise(1)_pw_t05_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t05_CC_constraint2_:
+1 state(reach1_level_upstream_t05)
-2.05 state_piecewise(1)_pw_t05_CC_lambda(1)
-1.9849999999999999 state_piecewise(1)_pw_t05_CC_lambda(2)
-1.92 state_piecewise(1)_pw_t05_CC_lambda(3)
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
+1 state(reach1_discharge_upstream_t06)
-3.8 state_piecewise(1)_pw_t06_CC_lambda(1)
-9.5 state_piecewise(1)_pw_t06_CC_lambda(2)
-13.0 state_piecewise(1)_pw_t06_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t06_CC_constraint2_:
+1 state(reach1_level_upstream_t06)
-2.05 state_piecewise(1)_pw_t06_CC_lambda(1)
-1.9849999999999999 state_piecewise(1)_pw_t06_CC_lambda(2)
-1.92 state_piecewise(1)_pw_t06_CC_lambda(3)
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
+1 state(reach1_discharge_upstream_t07)
-3.8 state_piecewise(1)_pw_t07_CC_lambda(1)
-9.5 state_piecewise(1)_pw_t07_CC_lambda(2)
-13.0 state_piecewise(1)_pw_t07_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t07_CC_constraint2_:
+1 state(reach1_level_upstream_t07)
-2.05 state_piecewise(1)_pw_t07_CC_lambda(1)
-1.9849999999999999 state_piecewise(1)_pw_t07_CC_lambda(2)
-1.92 state_piecewise(1)_pw_t07_CC_lambda(3)
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
+1 state(reach1_discharge_upstream_t08)
-3.8 state_piecewise(1)_pw_t08_CC_lambda(1)
-9.5 state_piecewise(1)_pw_t08_CC_lambda(2)
-13.0 state_piecewise(1)_pw_t08_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t08_CC_constraint2_:
+1 state(reach1_level_upstream_t08)
-2.05 state_piecewise(1)_pw_t08_CC_lambda(1)
-1.9849999999999999 state_piecewise(1)_pw_t08_CC_lambda(2)
-1.92 state_piecewise(1)_pw_t08_CC_lambda(3)
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
+1 state(reach1_discharge_upstream_t09)
-3.8 state_piecewise(1)_pw_t09_CC_lambda(1)
-9.5 state_piecewise(1)_pw_t09_CC_lambda(2)
-13.0 state_piecewise(1)_pw_t09_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t09_CC_constraint2_:
+1 state(reach1_level_upstream_t09)
-2.05 state_piecewise(1)_pw_t09_CC_lambda(1)
-1.9849999999999999 state_piecewise(1)_pw_t09_CC_lambda(2)
-1.92 state_piecewise(1)_pw_t09_CC_lambda(3)
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
+1 state(reach1_discharge_upstream_t10)
-3.8 state_piecewise(1)_pw_t10_CC_lambda(1)
-9.5 state_piecewise(1)_pw_t10_CC_lambda(2)
-13.0 state_piecewise(1)_pw_t10_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t10_CC_constraint2_:
+1 state(reach1_level_upstream_t10)
-2.05 state_piecewise(1)_pw_t10_CC_lambda(1)
-1.9849999999999999 state_piecewise(1)_pw_t10_CC_lambda(2)
-1.92 state_piecewise(1)_pw_t10_CC_lambda(3)
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
+1 state(reach1_discharge_upstream_t11)
-3.8 state_piecewise(1)_pw_t11_CC_lambda(1)
-9.5 state_piecewise(1)_pw_t11_CC_lambda(2)
-13.0 state_piecewise(1)_pw_t11_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t11_CC_constraint2_:
+1 state(reach1_level_upstream_t11)
-2.05 state_piecewise(1)_pw_t11_CC_lambda(1)
-1.9849999999999999 state_piecewise(1)_pw_t11_CC_lambda(2)
-1.92 state_piecewise(1)_pw_t11_CC_lambda(3)
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
+1 state(reach1_discharge_upstream_t12)
-3.8 state_piecewise(1)_pw_t12_CC_lambda(1)
-9.5 state_piecewise(1)_pw_t12_CC_lambda(2)
-13.0 state_piecewise(1)_pw_t12_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t12_CC_constraint2_:
+1 state(reach1_level_upstream_t12)
-2.05 state_piecewise(1)_pw_t12_CC_lambda(1)
-1.9849999999999999 state_piecewise(1)_pw_t12_CC_lambda(2)
-1.92 state_piecewise(1)_pw_t12_CC_lambda(3)
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

c_e_state_piecewise(1)_pw_t13_CC_constraint1_:
+1 state(reach1_discharge_upstream_t13)
-3.8 state_piecewise(1)_pw_t13_CC_lambda(1)
-9.5 state_piecewise(1)_pw_t13_CC_lambda(2)
-13.0 state_piecewise(1)_pw_t13_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t13_CC_constraint2_:
+1 state(reach1_level_upstream_t13)
-2.05 state_piecewise(1)_pw_t13_CC_lambda(1)
-1.9849999999999999 state_piecewise(1)_pw_t13_CC_lambda(2)
-1.92 state_piecewise(1)_pw_t13_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t13_CC_constraint3_:
+1 state_piecewise(1)_pw_t13_CC_lambda(1)
+1 state_piecewise(1)_pw_t13_CC_lambda(2)
+1 state_piecewise(1)_pw_t13_CC_lambda(3)
= 1

c_u_state_piecewise(1)_pw_t13_CC_constraint4(1)_:
+1 state_piecewise(1)_pw_t13_CC_lambda(1)
-1 state_piecewise(1)_pw_t13_CC_bin_y(1)
<= 0

c_u_state_piecewise(1)_pw_t13_CC_constraint4(2)_:
+1 state_piecewise(1)_pw_t13_CC_lambda(2)
-1 state_piecewise(1)_pw_t13_CC_bin_y(1)
-1 state_piecewise(1)_pw_t13_CC_bin_y(2)
<= 0

c_u_state_piecewise(1)_pw_t13_CC_constraint4(3)_:
+1 state_piecewise(1)_pw_t13_CC_lambda(3)
-1 state_piecewise(1)_pw_t13_CC_bin_y(2)
<= 0

c_e_state_piecewise(1)_pw_t13_CC_constraint5_:
+1 state_piecewise(1)_pw_t13_CC_bin_y(1)
+1 state_piecewise(1)_pw_t13_CC_bin_y(2)
= 1

c_e_state_piecewise(1)_pw_t14_CC_constraint1_:
+1 state(reach1_discharge_upstream_t14)
-3.8 state_piecewise(1)_pw_t14_CC_lambda(1)
-9.5 state_piecewise(1)_pw_t14_CC_lambda(2)
-13.0 state_piecewise(1)_pw_t14_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t14_CC_constraint2_:
+1 state(reach1_level_upstream_t14)
-2.05 state_piecewise(1)_pw_t14_CC_lambda(1)
-1.9849999999999999 state_piecewise(1)_pw_t14_CC_lambda(2)
-1.92 state_piecewise(1)_pw_t14_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t14_CC_constraint3_:
+1 state_piecewise(1)_pw_t14_CC_lambda(1)
+1 state_piecewise(1)_pw_t14_CC_lambda(2)
+1 state_piecewise(1)_pw_t14_CC_lambda(3)
= 1

c_u_state_piecewise(1)_pw_t14_CC_constraint4(1)_:
+1 state_piecewise(1)_pw_t14_CC_lambda(1)
-1 state_piecewise(1)_pw_t14_CC_bin_y(1)
<= 0

c_u_state_piecewise(1)_pw_t14_CC_constraint4(2)_:
+1 state_piecewise(1)_pw_t14_CC_lambda(2)
-1 state_piecewise(1)_pw_t14_CC_bin_y(1)
-1 state_piecewise(1)_pw_t14_CC_bin_y(2)
<= 0

c_u_state_piecewise(1)_pw_t14_CC_constraint4(3)_:
+1 state_piecewise(1)_pw_t14_CC_lambda(3)
-1 state_piecewise(1)_pw_t14_CC_bin_y(2)
<= 0

c_e_state_piecewise(1)_pw_t14_CC_constraint5_:
+1 state_piecewise(1)_pw_t14_CC_bin_y(1)
+1 state_piecewise(1)_pw_t14_CC_bin_y(2)
= 1

c_e_state_piecewise(1)_pw_t15_CC_constraint1_:
+1 state(reach1_discharge_upstream_t15)
-3.8 state_piecewise(1)_pw_t15_CC_lambda(1)
-9.5 state_piecewise(1)_pw_t15_CC_lambda(2)
-13.0 state_piecewise(1)_pw_t15_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t15_CC_constraint2_:
+1 state(reach1_level_upstream_t15)
-2.05 state_piecewise(1)_pw_t15_CC_lambda(1)
-1.9849999999999999 state_piecewise(1)_pw_t15_CC_lambda(2)
-1.92 state_piecewise(1)_pw_t15_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t15_CC_constraint3_:
+1 state_piecewise(1)_pw_t15_CC_lambda(1)
+1 state_piecewise(1)_pw_t15_CC_lambda(2)
+1 state_piecewise(1)_pw_t15_CC_lambda(3)
= 1

c_u_state_piecewise(1)_pw_t15_CC_constraint4(1)_:
+1 state_piecewise(1)_pw_t15_CC_lambda(1)
-1 state_piecewise(1)_pw_t15_CC_bin_y(1)
<= 0

c_u_state_piecewise(1)_pw_t15_CC_constraint4(2)_:
+1 state_piecewise(1)_pw_t15_CC_lambda(2)
-1 state_piecewise(1)_pw_t15_CC_bin_y(1)
-1 state_piecewise(1)_pw_t15_CC_bin_y(2)
<= 0

c_u_state_piecewise(1)_pw_t15_CC_constraint4(3)_:
+1 state_piecewise(1)_pw_t15_CC_lambda(3)
-1 state_piecewise(1)_pw_t15_CC_bin_y(2)
<= 0

c_e_state_piecewise(1)_pw_t15_CC_constraint5_:
+1 state_piecewise(1)_pw_t15_CC_bin_y(1)
+1 state_piecewise(1)_pw_t15_CC_bin_y(2)
= 1

c_e_state_piecewise(1)_pw_t16_CC_constraint1_:
+1 state(reach1_discharge_upstream_t16)
-3.8 state_piecewise(1)_pw_t16_CC_lambda(1)
-9.5 state_piecewise(1)_pw_t16_CC_lambda(2)
-13.0 state_piecewise(1)_pw_t16_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t16_CC_constraint2_:
+1 state(reach1_level_upstream_t16)
-2.05 state_piecewise(1)_pw_t16_CC_lambda(1)
-1.9849999999999999 state_piecewise(1)_pw_t16_CC_lambda(2)
-1.92 state_piecewise(1)_pw_t16_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t16_CC_constraint3_:
+1 state_piecewise(1)_pw_t16_CC_lambda(1)
+1 state_piecewise(1)_pw_t16_CC_lambda(2)
+1 state_piecewise(1)_pw_t16_CC_lambda(3)
= 1

c_u_state_piecewise(1)_pw_t16_CC_constraint4(1)_:
+1 state_piecewise(1)_pw_t16_CC_lambda(1)
-1 state_piecewise(1)_pw_t16_CC_bin_y(1)
<= 0

c_u_state_piecewise(1)_pw_t16_CC_constraint4(2)_:
+1 state_piecewise(1)_pw_t16_CC_lambda(2)
-1 state_piecewise(1)_pw_t16_CC_bin_y(1)
-1 state_piecewise(1)_pw_t16_CC_bin_y(2)
<= 0

c_u_state_piecewise(1)_pw_t16_CC_constraint4(3)_:
+1 state_piecewise(1)_pw_t16_CC_lambda(3)
-1 state_piecewise(1)_pw_t16_CC_bin_y(2)
<= 0

c_e_state_piecewise(1)_pw_t16_CC_constraint5_:
+1 state_piecewise(1)_pw_t16_CC_bin_y(1)
+1 state_piecewise(1)_pw_t16_CC_bin_y(2)
= 1

c_e_state_piecewise(1)_pw_t17_CC_constraint1_:
+1 state(reach1_discharge_upstream_t17)
-3.8 state_piecewise(1)_pw_t17_CC_lambda(1)
-9.5 state_piecewise(1)_pw_t17_CC_lambda(2)
-13.0 state_piecewise(1)_pw_t17_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t17_CC_constraint2_:
+1 state(reach1_level_upstream_t17)
-2.05 state_piecewise(1)_pw_t17_CC_lambda(1)
-1.9849999999999999 state_piecewise(1)_pw_t17_CC_lambda(2)
-1.92 state_piecewise(1)_pw_t17_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t17_CC_constraint3_:
+1 state_piecewise(1)_pw_t17_CC_lambda(1)
+1 state_piecewise(1)_pw_t17_CC_lambda(2)
+1 state_piecewise(1)_pw_t17_CC_lambda(3)
= 1

c_u_state_piecewise(1)_pw_t17_CC_constraint4(1)_:
+1 state_piecewise(1)_pw_t17_CC_lambda(1)
-1 state_piecewise(1)_pw_t17_CC_bin_y(1)
<= 0

c_u_state_piecewise(1)_pw_t17_CC_constraint4(2)_:
+1 state_piecewise(1)_pw_t17_CC_lambda(2)
-1 state_piecewise(1)_pw_t17_CC_bin_y(1)
-1 state_piecewise(1)_pw_t17_CC_bin_y(2)
<= 0

c_u_state_piecewise(1)_pw_t17_CC_constraint4(3)_:
+1 state_piecewise(1)_pw_t17_CC_lambda(3)
-1 state_piecewise(1)_pw_t17_CC_bin_y(2)
<= 0

c_e_state_piecewise(1)_pw_t17_CC_constraint5_:
+1 state_piecewise(1)_pw_t17_CC_bin_y(1)
+1 state_piecewise(1)_pw_t17_CC_bin_y(2)
= 1

c_e_state_piecewise(1)_pw_t18_CC_constraint1_:
+1 state(reach1_discharge_upstream_t18)
-3.8 state_piecewise(1)_pw_t18_CC_lambda(1)
-9.5 state_piecewise(1)_pw_t18_CC_lambda(2)
-13.0 state_piecewise(1)_pw_t18_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t18_CC_constraint2_:
+1 state(reach1_level_upstream_t18)
-2.05 state_piecewise(1)_pw_t18_CC_lambda(1)
-1.9849999999999999 state_piecewise(1)_pw_t18_CC_lambda(2)
-1.92 state_piecewise(1)_pw_t18_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t18_CC_constraint3_:
+1 state_piecewise(1)_pw_t18_CC_lambda(1)
+1 state_piecewise(1)_pw_t18_CC_lambda(2)
+1 state_piecewise(1)_pw_t18_CC_lambda(3)
= 1

c_u_state_piecewise(1)_pw_t18_CC_constraint4(1)_:
+1 state_piecewise(1)_pw_t18_CC_lambda(1)
-1 state_piecewise(1)_pw_t18_CC_bin_y(1)
<= 0

c_u_state_piecewise(1)_pw_t18_CC_constraint4(2)_:
+1 state_piecewise(1)_pw_t18_CC_lambda(2)
-1 state_piecewise(1)_pw_t18_CC_bin_y(1)
-1 state_piecewise(1)_pw_t18_CC_bin_y(2)
<= 0

c_u_state_piecewise(1)_pw_t18_CC_constraint4(3)_:
+1 state_piecewise(1)_pw_t18_CC_lambda(3)
-1 state_piecewise(1)_pw_t18_CC_bin_y(2)
<= 0

c_e_state_piecewise(1)_pw_t18_CC_constraint5_:
+1 state_piecewise(1)_pw_t18_CC_bin_y(1)
+1 state_piecewise(1)_pw_t18_CC_bin_y(2)
= 1

c_e_state_piecewise(1)_pw_t19_CC_constraint1_:
+1 state(reach1_discharge_upstream_t19)
-3.8 state_piecewise(1)_pw_t19_CC_lambda(1)
-9.5 state_piecewise(1)_pw_t19_CC_lambda(2)
-13.0 state_piecewise(1)_pw_t19_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t19_CC_constraint2_:
+1 state(reach1_level_upstream_t19)
-2.05 state_piecewise(1)_pw_t19_CC_lambda(1)
-1.9849999999999999 state_piecewise(1)_pw_t19_CC_lambda(2)
-1.92 state_piecewise(1)_pw_t19_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t19_CC_constraint3_:
+1 state_piecewise(1)_pw_t19_CC_lambda(1)
+1 state_piecewise(1)_pw_t19_CC_lambda(2)
+1 state_piecewise(1)_pw_t19_CC_lambda(3)
= 1

c_u_state_piecewise(1)_pw_t19_CC_constraint4(1)_:
+1 state_piecewise(1)_pw_t19_CC_lambda(1)
-1 state_piecewise(1)_pw_t19_CC_bin_y(1)
<= 0

c_u_state_piecewise(1)_pw_t19_CC_constraint4(2)_:
+1 state_piecewise(1)_pw_t19_CC_lambda(2)
-1 state_piecewise(1)_pw_t19_CC_bin_y(1)
-1 state_piecewise(1)_pw_t19_CC_bin_y(2)
<= 0

c_u_state_piecewise(1)_pw_t19_CC_constraint4(3)_:
+1 state_piecewise(1)_pw_t19_CC_lambda(3)
-1 state_piecewise(1)_pw_t19_CC_bin_y(2)
<= 0

c_e_state_piecewise(1)_pw_t19_CC_constraint5_:
+1 state_piecewise(1)_pw_t19_CC_bin_y(1)
+1 state_piecewise(1)_pw_t19_CC_bin_y(2)
= 1

c_e_state_piecewise(1)_pw_t20_CC_constraint1_:
+1 state(reach1_discharge_upstream_t20)
-3.8 state_piecewise(1)_pw_t20_CC_lambda(1)
-9.5 state_piecewise(1)_pw_t20_CC_lambda(2)
-13.0 state_piecewise(1)_pw_t20_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t20_CC_constraint2_:
+1 state(reach1_level_upstream_t20)
-2.05 state_piecewise(1)_pw_t20_CC_lambda(1)
-1.9849999999999999 state_piecewise(1)_pw_t20_CC_lambda(2)
-1.92 state_piecewise(1)_pw_t20_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t20_CC_constraint3_:
+1 state_piecewise(1)_pw_t20_CC_lambda(1)
+1 state_piecewise(1)_pw_t20_CC_lambda(2)
+1 state_piecewise(1)_pw_t20_CC_lambda(3)
= 1

c_u_state_piecewise(1)_pw_t20_CC_constraint4(1)_:
+1 state_piecewise(1)_pw_t20_CC_lambda(1)
-1 state_piecewise(1)_pw_t20_CC_bin_y(1)
<= 0

c_u_state_piecewise(1)_pw_t20_CC_constraint4(2)_:
+1 state_piecewise(1)_pw_t20_CC_lambda(2)
-1 state_piecewise(1)_pw_t20_CC_bin_y(1)
-1 state_piecewise(1)_pw_t20_CC_bin_y(2)
<= 0

c_u_state_piecewise(1)_pw_t20_CC_constraint4(3)_:
+1 state_piecewise(1)_pw_t20_CC_lambda(3)
-1 state_piecewise(1)_pw_t20_CC_bin_y(2)
<= 0

c_e_state_piecewise(1)_pw_t20_CC_constraint5_:
+1 state_piecewise(1)_pw_t20_CC_bin_y(1)
+1 state_piecewise(1)_pw_t20_CC_bin_y(2)
= 1

c_e_state_piecewise(1)_pw_t21_CC_constraint1_:
+1 state(reach1_discharge_upstream_t21)
-3.8 state_piecewise(1)_pw_t21_CC_lambda(1)
-9.5 state_piecewise(1)_pw_t21_CC_lambda(2)
-13.0 state_piecewise(1)_pw_t21_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t21_CC_constraint2_:
+1 state(reach1_level_upstream_t21)
-2.05 state_piecewise(1)_pw_t21_CC_lambda(1)
-1.9849999999999999 state_piecewise(1)_pw_t21_CC_lambda(2)
-1.92 state_piecewise(1)_pw_t21_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t21_CC_constraint3_:
+1 state_piecewise(1)_pw_t21_CC_lambda(1)
+1 state_piecewise(1)_pw_t21_CC_lambda(2)
+1 state_piecewise(1)_pw_t21_CC_lambda(3)
= 1

c_u_state_piecewise(1)_pw_t21_CC_constraint4(1)_:
+1 state_piecewise(1)_pw_t21_CC_lambda(1)
-1 state_piecewise(1)_pw_t21_CC_bin_y(1)
<= 0

c_u_state_piecewise(1)_pw_t21_CC_constraint4(2)_:
+1 state_piecewise(1)_pw_t21_CC_lambda(2)
-1 state_piecewise(1)_pw_t21_CC_bin_y(1)
-1 state_piecewise(1)_pw_t21_CC_bin_y(2)
<= 0

c_u_state_piecewise(1)_pw_t21_CC_constraint4(3)_:
+1 state_piecewise(1)_pw_t21_CC_lambda(3)
-1 state_piecewise(1)_pw_t21_CC_bin_y(2)
<= 0

c_e_state_piecewise(1)_pw_t21_CC_constraint5_:
+1 state_piecewise(1)_pw_t21_CC_bin_y(1)
+1 state_piecewise(1)_pw_t21_CC_bin_y(2)
= 1

c_e_state_piecewise(1)_pw_t22_CC_constraint1_:
+1 state(reach1_discharge_upstream_t22)
-3.8 state_piecewise(1)_pw_t22_CC_lambda(1)
-9.5 state_piecewise(1)_pw_t22_CC_lambda(2)
-13.0 state_piecewise(1)_pw_t22_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t22_CC_constraint2_:
+1 state(reach1_level_upstream_t22)
-2.05 state_piecewise(1)_pw_t22_CC_lambda(1)
-1.9849999999999999 state_piecewise(1)_pw_t22_CC_lambda(2)
-1.92 state_piecewise(1)_pw_t22_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t22_CC_constraint3_:
+1 state_piecewise(1)_pw_t22_CC_lambda(1)
+1 state_piecewise(1)_pw_t22_CC_lambda(2)
+1 state_piecewise(1)_pw_t22_CC_lambda(3)
= 1

c_u_state_piecewise(1)_pw_t22_CC_constraint4(1)_:
+1 state_piecewise(1)_pw_t22_CC_lambda(1)
-1 state_piecewise(1)_pw_t22_CC_bin_y(1)
<= 0

c_u_state_piecewise(1)_pw_t22_CC_constraint4(2)_:
+1 state_piecewise(1)_pw_t22_CC_lambda(2)
-1 state_piecewise(1)_pw_t22_CC_bin_y(1)
-1 state_piecewise(1)_pw_t22_CC_bin_y(2)
<= 0

c_u_state_piecewise(1)_pw_t22_CC_constraint4(3)_:
+1 state_piecewise(1)_pw_t22_CC_lambda(3)
-1 state_piecewise(1)_pw_t22_CC_bin_y(2)
<= 0

c_e_state_piecewise(1)_pw_t22_CC_constraint5_:
+1 state_piecewise(1)_pw_t22_CC_bin_y(1)
+1 state_piecewise(1)_pw_t22_CC_bin_y(2)
= 1

c_e_state_piecewise(1)_pw_t23_CC_constraint1_:
+1 state(reach1_discharge_upstream_t23)
-3.8 state_piecewise(1)_pw_t23_CC_lambda(1)
-9.5 state_piecewise(1)_pw_t23_CC_lambda(2)
-13.0 state_piecewise(1)_pw_t23_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t23_CC_constraint2_:
+1 state(reach1_level_upstream_t23)
-2.05 state_piecewise(1)_pw_t23_CC_lambda(1)
-1.9849999999999999 state_piecewise(1)_pw_t23_CC_lambda(2)
-1.92 state_piecewise(1)_pw_t23_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t23_CC_constraint3_:
+1 state_piecewise(1)_pw_t23_CC_lambda(1)
+1 state_piecewise(1)_pw_t23_CC_lambda(2)
+1 state_piecewise(1)_pw_t23_CC_lambda(3)
= 1

c_u_state_piecewise(1)_pw_t23_CC_constraint4(1)_:
+1 state_piecewise(1)_pw_t23_CC_lambda(1)
-1 state_piecewise(1)_pw_t23_CC_bin_y(1)
<= 0

c_u_state_piecewise(1)_pw_t23_CC_constraint4(2)_:
+1 state_piecewise(1)_pw_t23_CC_lambda(2)
-1 state_piecewise(1)_pw_t23_CC_bin_y(1)
-1 state_piecewise(1)_pw_t23_CC_bin_y(2)
<= 0

c_u_state_piecewise(1)_pw_t23_CC_constraint4(3)_:
+1 state_piecewise(1)_pw_t23_CC_lambda(3)
-1 state_piecewise(1)_pw_t23_CC_bin_y(2)
<= 0

c_e_state_piecewise(1)_pw_t23_CC_constraint5_:
+1 state_piecewise(1)_pw_t23_CC_bin_y(1)
+1 state_piecewise(1)_pw_t23_CC_bin_y(2)
= 1

c_e_state_piecewise(1)_pw_t24_CC_constraint1_:
+1 state(reach1_discharge_upstream_t24)
-3.8 state_piecewise(1)_pw_t24_CC_lambda(1)
-9.5 state_piecewise(1)_pw_t24_CC_lambda(2)
-13.0 state_piecewise(1)_pw_t24_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t24_CC_constraint2_:
+1 state(reach1_level_upstream_t24)
-2.05 state_piecewise(1)_pw_t24_CC_lambda(1)
-1.9849999999999999 state_piecewise(1)_pw_t24_CC_lambda(2)
-1.92 state_piecewise(1)_pw_t24_CC_lambda(3)
= 0

c_e_state_piecewise(1)_pw_t24_CC_constraint3_:
+1 state_piecewise(1)_pw_t24_CC_lambda(1)
+1 state_piecewise(1)_pw_t24_CC_lambda(2)
+1 state_piecewise(1)_pw_t24_CC_lambda(3)
= 1

c_u_state_piecewise(1)_pw_t24_CC_constraint4(1)_:
+1 state_piecewise(1)_pw_t24_CC_lambda(1)
-1 state_piecewise(1)_pw_t24_CC_bin_y(1)
<= 0

c_u_state_piecewise(1)_pw_t24_CC_constraint4(2)_:
+1 state_piecewise(1)_pw_t24_CC_lambda(2)
-1 state_piecewise(1)_pw_t24_CC_bin_y(1)
-1 state_piecewise(1)_pw_t24_CC_bin_y(2)
<= 0

c_u_state_piecewise(1)_pw_t24_CC_constraint4(3)_:
+1 state_piecewise(1)_pw_t24_CC_lambda(3)
-1 state_piecewise(1)_pw_t24_CC_bin_y(2)
<= 0

c_e_state_piecewise(1)_pw_t24_CC_constraint5_:
+1 state_piecewise(1)_pw_t24_CC_bin_y(1)
+1 state_piecewise(1)_pw_t24_CC_bin_y(2)
= 1

c_e_state_piecewise(2)_pw_t01_CC_constraint1_:
+1 state(reach1_discharge_downstream_t01)
-3.8 state_piecewise(2)_pw_t01_CC_lambda(1)
-9.5 state_piecewise(2)_pw_t01_CC_lambda(2)
-13.0 state_piecewise(2)_pw_t01_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t01_CC_constraint2_:
+1 state(reach1_level_downstream_t01)
-1.92 state_piecewise(2)_pw_t01_CC_lambda(1)
-1.8599999999999999 state_piecewise(2)_pw_t01_CC_lambda(2)
-1.8 state_piecewise(2)_pw_t01_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t01_CC_constraint3_:
+1 state_piecewise(2)_pw_t01_CC_lambda(1)
+1 state_piecewise(2)_pw_t01_CC_lambda(2)
+1 state_piecewise(2)_pw_t01_CC_lambda(3)
= 1

c_u_state_piecewise(2)_pw_t01_CC_constraint4(1)_:
+1 state_piecewise(2)_pw_t01_CC_lambda(1)
-1 state_piecewise(2)_pw_t01_CC_bin_y(1)
<= 0

c_u_state_piecewise(2)_pw_t01_CC_constraint4(2)_:
+1 state_piecewise(2)_pw_t01_CC_lambda(2)
-1 state_piecewise(2)_pw_t01_CC_bin_y(1)
-1 state_piecewise(2)_pw_t01_CC_bin_y(2)
<= 0

c_u_state_piecewise(2)_pw_t01_CC_constraint4(3)_:
+1 state_piecewise(2)_pw_t01_CC_lambda(3)
-1 state_piecewise(2)_pw_t01_CC_bin_y(2)
<= 0

c_e_state_piecewise(2)_pw_t01_CC_constraint5_:
+1 state_piecewise(2)_pw_t01_CC_bin_y(1)
+1 state_piecewise(2)_pw_t01_CC_bin_y(2)
= 1

c_e_state_piecewise(2)_pw_t02_CC_constraint1_:
+1 state(reach1_discharge_downstream_t02)
-3.8 state_piecewise(2)_pw_t02_CC_lambda(1)
-9.5 state_piecewise(2)_pw_t02_CC_lambda(2)
-13.0 state_piecewise(2)_pw_t02_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t02_CC_constraint2_:
+1 state(reach1_level_downstream_t02)
-1.92 state_piecewise(2)_pw_t02_CC_lambda(1)
-1.8599999999999999 state_piecewise(2)_pw_t02_CC_lambda(2)
-1.8 state_piecewise(2)_pw_t02_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t02_CC_constraint3_:
+1 state_piecewise(2)_pw_t02_CC_lambda(1)
+1 state_piecewise(2)_pw_t02_CC_lambda(2)
+1 state_piecewise(2)_pw_t02_CC_lambda(3)
= 1

c_u_state_piecewise(2)_pw_t02_CC_constraint4(1)_:
+1 state_piecewise(2)_pw_t02_CC_lambda(1)
-1 state_piecewise(2)_pw_t02_CC_bin_y(1)
<= 0

c_u_state_piecewise(2)_pw_t02_CC_constraint4(2)_:
+1 state_piecewise(2)_pw_t02_CC_lambda(2)
-1 state_piecewise(2)_pw_t02_CC_bin_y(1)
-1 state_piecewise(2)_pw_t02_CC_bin_y(2)
<= 0

c_u_state_piecewise(2)_pw_t02_CC_constraint4(3)_:
+1 state_piecewise(2)_pw_t02_CC_lambda(3)
-1 state_piecewise(2)_pw_t02_CC_bin_y(2)
<= 0

c_e_state_piecewise(2)_pw_t02_CC_constraint5_:
+1 state_piecewise(2)_pw_t02_CC_bin_y(1)
+1 state_piecewise(2)_pw_t02_CC_bin_y(2)
= 1

c_e_state_piecewise(2)_pw_t03_CC_constraint1_:
+1 state(reach1_discharge_downstream_t03)
-3.8 state_piecewise(2)_pw_t03_CC_lambda(1)
-9.5 state_piecewise(2)_pw_t03_CC_lambda(2)
-13.0 state_piecewise(2)_pw_t03_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t03_CC_constraint2_:
+1 state(reach1_level_downstream_t03)
-1.92 state_piecewise(2)_pw_t03_CC_lambda(1)
-1.8599999999999999 state_piecewise(2)_pw_t03_CC_lambda(2)
-1.8 state_piecewise(2)_pw_t03_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t03_CC_constraint3_:
+1 state_piecewise(2)_pw_t03_CC_lambda(1)
+1 state_piecewise(2)_pw_t03_CC_lambda(2)
+1 state_piecewise(2)_pw_t03_CC_lambda(3)
= 1

c_u_state_piecewise(2)_pw_t03_CC_constraint4(1)_:
+1 state_piecewise(2)_pw_t03_CC_lambda(1)
-1 state_piecewise(2)_pw_t03_CC_bin_y(1)
<= 0

c_u_state_piecewise(2)_pw_t03_CC_constraint4(2)_:
+1 state_piecewise(2)_pw_t03_CC_lambda(2)
-1 state_piecewise(2)_pw_t03_CC_bin_y(1)
-1 state_piecewise(2)_pw_t03_CC_bin_y(2)
<= 0

c_u_state_piecewise(2)_pw_t03_CC_constraint4(3)_:
+1 state_piecewise(2)_pw_t03_CC_lambda(3)
-1 state_piecewise(2)_pw_t03_CC_bin_y(2)
<= 0

c_e_state_piecewise(2)_pw_t03_CC_constraint5_:
+1 state_piecewise(2)_pw_t03_CC_bin_y(1)
+1 state_piecewise(2)_pw_t03_CC_bin_y(2)
= 1

c_e_state_piecewise(2)_pw_t04_CC_constraint1_:
+1 state(reach1_discharge_downstream_t04)
-3.8 state_piecewise(2)_pw_t04_CC_lambda(1)
-9.5 state_piecewise(2)_pw_t04_CC_lambda(2)
-13.0 state_piecewise(2)_pw_t04_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t04_CC_constraint2_:
+1 state(reach1_level_downstream_t04)
-1.92 state_piecewise(2)_pw_t04_CC_lambda(1)
-1.8599999999999999 state_piecewise(2)_pw_t04_CC_lambda(2)
-1.8 state_piecewise(2)_pw_t04_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t04_CC_constraint3_:
+1 state_piecewise(2)_pw_t04_CC_lambda(1)
+1 state_piecewise(2)_pw_t04_CC_lambda(2)
+1 state_piecewise(2)_pw_t04_CC_lambda(3)
= 1

c_u_state_piecewise(2)_pw_t04_CC_constraint4(1)_:
+1 state_piecewise(2)_pw_t04_CC_lambda(1)
-1 state_piecewise(2)_pw_t04_CC_bin_y(1)
<= 0

c_u_state_piecewise(2)_pw_t04_CC_constraint4(2)_:
+1 state_piecewise(2)_pw_t04_CC_lambda(2)
-1 state_piecewise(2)_pw_t04_CC_bin_y(1)
-1 state_piecewise(2)_pw_t04_CC_bin_y(2)
<= 0

c_u_state_piecewise(2)_pw_t04_CC_constraint4(3)_:
+1 state_piecewise(2)_pw_t04_CC_lambda(3)
-1 state_piecewise(2)_pw_t04_CC_bin_y(2)
<= 0

c_e_state_piecewise(2)_pw_t04_CC_constraint5_:
+1 state_piecewise(2)_pw_t04_CC_bin_y(1)
+1 state_piecewise(2)_pw_t04_CC_bin_y(2)
= 1

c_e_state_piecewise(2)_pw_t05_CC_constraint1_:
+1 state(reach1_discharge_downstream_t05)
-3.8 state_piecewise(2)_pw_t05_CC_lambda(1)
-9.5 state_piecewise(2)_pw_t05_CC_lambda(2)
-13.0 state_piecewise(2)_pw_t05_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t05_CC_constraint2_:
+1 state(reach1_level_downstream_t05)
-1.92 state_piecewise(2)_pw_t05_CC_lambda(1)
-1.8599999999999999 state_piecewise(2)_pw_t05_CC_lambda(2)
-1.8 state_piecewise(2)_pw_t05_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t05_CC_constraint3_:
+1 state_piecewise(2)_pw_t05_CC_lambda(1)
+1 state_piecewise(2)_pw_t05_CC_lambda(2)
+1 state_piecewise(2)_pw_t05_CC_lambda(3)
= 1

c_u_state_piecewise(2)_pw_t05_CC_constraint4(1)_:
+1 state_piecewise(2)_pw_t05_CC_lambda(1)
-1 state_piecewise(2)_pw_t05_CC_bin_y(1)
<= 0

c_u_state_piecewise(2)_pw_t05_CC_constraint4(2)_:
+1 state_piecewise(2)_pw_t05_CC_lambda(2)
-1 state_piecewise(2)_pw_t05_CC_bin_y(1)
-1 state_piecewise(2)_pw_t05_CC_bin_y(2)
<= 0

c_u_state_piecewise(2)_pw_t05_CC_constraint4(3)_:
+1 state_piecewise(2)_pw_t05_CC_lambda(3)
-1 state_piecewise(2)_pw_t05_CC_bin_y(2)
<= 0

c_e_state_piecewise(2)_pw_t05_CC_constraint5_:
+1 state_piecewise(2)_pw_t05_CC_bin_y(1)
+1 state_piecewise(2)_pw_t05_CC_bin_y(2)
= 1

c_e_state_piecewise(2)_pw_t06_CC_constraint1_:
+1 state(reach1_discharge_downstream_t06)
-3.8 state_piecewise(2)_pw_t06_CC_lambda(1)
-9.5 state_piecewise(2)_pw_t06_CC_lambda(2)
-13.0 state_piecewise(2)_pw_t06_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t06_CC_constraint2_:
+1 state(reach1_level_downstream_t06)
-1.92 state_piecewise(2)_pw_t06_CC_lambda(1)
-1.8599999999999999 state_piecewise(2)_pw_t06_CC_lambda(2)
-1.8 state_piecewise(2)_pw_t06_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t06_CC_constraint3_:
+1 state_piecewise(2)_pw_t06_CC_lambda(1)
+1 state_piecewise(2)_pw_t06_CC_lambda(2)
+1 state_piecewise(2)_pw_t06_CC_lambda(3)
= 1

c_u_state_piecewise(2)_pw_t06_CC_constraint4(1)_:
+1 state_piecewise(2)_pw_t06_CC_lambda(1)
-1 state_piecewise(2)_pw_t06_CC_bin_y(1)
<= 0

c_u_state_piecewise(2)_pw_t06_CC_constraint4(2)_:
+1 state_piecewise(2)_pw_t06_CC_lambda(2)
-1 state_piecewise(2)_pw_t06_CC_bin_y(1)
-1 state_piecewise(2)_pw_t06_CC_bin_y(2)
<= 0

c_u_state_piecewise(2)_pw_t06_CC_constraint4(3)_:
+1 state_piecewise(2)_pw_t06_CC_lambda(3)
-1 state_piecewise(2)_pw_t06_CC_bin_y(2)
<= 0

c_e_state_piecewise(2)_pw_t06_CC_constraint5_:
+1 state_piecewise(2)_pw_t06_CC_bin_y(1)
+1 state_piecewise(2)_pw_t06_CC_bin_y(2)
= 1

c_e_state_piecewise(2)_pw_t07_CC_constraint1_:
+1 state(reach1_discharge_downstream_t07)
-3.8 state_piecewise(2)_pw_t07_CC_lambda(1)
-9.5 state_piecewise(2)_pw_t07_CC_lambda(2)
-13.0 state_piecewise(2)_pw_t07_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t07_CC_constraint2_:
+1 state(reach1_level_downstream_t07)
-1.92 state_piecewise(2)_pw_t07_CC_lambda(1)
-1.8599999999999999 state_piecewise(2)_pw_t07_CC_lambda(2)
-1.8 state_piecewise(2)_pw_t07_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t07_CC_constraint3_:
+1 state_piecewise(2)_pw_t07_CC_lambda(1)
+1 state_piecewise(2)_pw_t07_CC_lambda(2)
+1 state_piecewise(2)_pw_t07_CC_lambda(3)
= 1

c_u_state_piecewise(2)_pw_t07_CC_constraint4(1)_:
+1 state_piecewise(2)_pw_t07_CC_lambda(1)
-1 state_piecewise(2)_pw_t07_CC_bin_y(1)
<= 0

c_u_state_piecewise(2)_pw_t07_CC_constraint4(2)_:
+1 state_piecewise(2)_pw_t07_CC_lambda(2)
-1 state_piecewise(2)_pw_t07_CC_bin_y(1)
-1 state_piecewise(2)_pw_t07_CC_bin_y(2)
<= 0

c_u_state_piecewise(2)_pw_t07_CC_constraint4(3)_:
+1 state_piecewise(2)_pw_t07_CC_lambda(3)
-1 state_piecewise(2)_pw_t07_CC_bin_y(2)
<= 0

c_e_state_piecewise(2)_pw_t07_CC_constraint5_:
+1 state_piecewise(2)_pw_t07_CC_bin_y(1)
+1 state_piecewise(2)_pw_t07_CC_bin_y(2)
= 1

c_e_state_piecewise(2)_pw_t08_CC_constraint1_:
+1 state(reach1_discharge_downstream_t08)
-3.8 state_piecewise(2)_pw_t08_CC_lambda(1)
-9.5 state_piecewise(2)_pw_t08_CC_lambda(2)
-13.0 state_piecewise(2)_pw_t08_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t08_CC_constraint2_:
+1 state(reach1_level_downstream_t08)
-1.92 state_piecewise(2)_pw_t08_CC_lambda(1)
-1.8599999999999999 state_piecewise(2)_pw_t08_CC_lambda(2)
-1.8 state_piecewise(2)_pw_t08_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t08_CC_constraint3_:
+1 state_piecewise(2)_pw_t08_CC_lambda(1)
+1 state_piecewise(2)_pw_t08_CC_lambda(2)
+1 state_piecewise(2)_pw_t08_CC_lambda(3)
= 1

c_u_state_piecewise(2)_pw_t08_CC_constraint4(1)_:
+1 state_piecewise(2)_pw_t08_CC_lambda(1)
-1 state_piecewise(2)_pw_t08_CC_bin_y(1)
<= 0

c_u_state_piecewise(2)_pw_t08_CC_constraint4(2)_:
+1 state_piecewise(2)_pw_t08_CC_lambda(2)
-1 state_piecewise(2)_pw_t08_CC_bin_y(1)
-1 state_piecewise(2)_pw_t08_CC_bin_y(2)
<= 0

c_u_state_piecewise(2)_pw_t08_CC_constraint4(3)_:
+1 state_piecewise(2)_pw_t08_CC_lambda(3)
-1 state_piecewise(2)_pw_t08_CC_bin_y(2)
<= 0

c_e_state_piecewise(2)_pw_t08_CC_constraint5_:
+1 state_piecewise(2)_pw_t08_CC_bin_y(1)
+1 state_piecewise(2)_pw_t08_CC_bin_y(2)
= 1

c_e_state_piecewise(2)_pw_t09_CC_constraint1_:
+1 state(reach1_discharge_downstream_t09)
-3.8 state_piecewise(2)_pw_t09_CC_lambda(1)
-9.5 state_piecewise(2)_pw_t09_CC_lambda(2)
-13.0 state_piecewise(2)_pw_t09_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t09_CC_constraint2_:
+1 state(reach1_level_downstream_t09)
-1.92 state_piecewise(2)_pw_t09_CC_lambda(1)
-1.8599999999999999 state_piecewise(2)_pw_t09_CC_lambda(2)
-1.8 state_piecewise(2)_pw_t09_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t09_CC_constraint3_:
+1 state_piecewise(2)_pw_t09_CC_lambda(1)
+1 state_piecewise(2)_pw_t09_CC_lambda(2)
+1 state_piecewise(2)_pw_t09_CC_lambda(3)
= 1

c_u_state_piecewise(2)_pw_t09_CC_constraint4(1)_:
+1 state_piecewise(2)_pw_t09_CC_lambda(1)
-1 state_piecewise(2)_pw_t09_CC_bin_y(1)
<= 0

c_u_state_piecewise(2)_pw_t09_CC_constraint4(2)_:
+1 state_piecewise(2)_pw_t09_CC_lambda(2)
-1 state_piecewise(2)_pw_t09_CC_bin_y(1)
-1 state_piecewise(2)_pw_t09_CC_bin_y(2)
<= 0

c_u_state_piecewise(2)_pw_t09_CC_constraint4(3)_:
+1 state_piecewise(2)_pw_t09_CC_lambda(3)
-1 state_piecewise(2)_pw_t09_CC_bin_y(2)
<= 0

c_e_state_piecewise(2)_pw_t09_CC_constraint5_:
+1 state_piecewise(2)_pw_t09_CC_bin_y(1)
+1 state_piecewise(2)_pw_t09_CC_bin_y(2)
= 1

c_e_state_piecewise(2)_pw_t10_CC_constraint1_:
+1 state(reach1_discharge_downstream_t10)
-3.8 state_piecewise(2)_pw_t10_CC_lambda(1)
-9.5 state_piecewise(2)_pw_t10_CC_lambda(2)
-13.0 state_piecewise(2)_pw_t10_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t10_CC_constraint2_:
+1 state(reach1_level_downstream_t10)
-1.92 state_piecewise(2)_pw_t10_CC_lambda(1)
-1.8599999999999999 state_piecewise(2)_pw_t10_CC_lambda(2)
-1.8 state_piecewise(2)_pw_t10_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t10_CC_constraint3_:
+1 state_piecewise(2)_pw_t10_CC_lambda(1)
+1 state_piecewise(2)_pw_t10_CC_lambda(2)
+1 state_piecewise(2)_pw_t10_CC_lambda(3)
= 1

c_u_state_piecewise(2)_pw_t10_CC_constraint4(1)_:
+1 state_piecewise(2)_pw_t10_CC_lambda(1)
-1 state_piecewise(2)_pw_t10_CC_bin_y(1)
<= 0

c_u_state_piecewise(2)_pw_t10_CC_constraint4(2)_:
+1 state_piecewise(2)_pw_t10_CC_lambda(2)
-1 state_piecewise(2)_pw_t10_CC_bin_y(1)
-1 state_piecewise(2)_pw_t10_CC_bin_y(2)
<= 0

c_u_state_piecewise(2)_pw_t10_CC_constraint4(3)_:
+1 state_piecewise(2)_pw_t10_CC_lambda(3)
-1 state_piecewise(2)_pw_t10_CC_bin_y(2)
<= 0

c_e_state_piecewise(2)_pw_t10_CC_constraint5_:
+1 state_piecewise(2)_pw_t10_CC_bin_y(1)
+1 state_piecewise(2)_pw_t10_CC_bin_y(2)
= 1

c_e_state_piecewise(2)_pw_t11_CC_constraint1_:
+1 state(reach1_discharge_downstream_t11)
-3.8 state_piecewise(2)_pw_t11_CC_lambda(1)
-9.5 state_piecewise(2)_pw_t11_CC_lambda(2)
-13.0 state_piecewise(2)_pw_t11_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t11_CC_constraint2_:
+1 state(reach1_level_downstream_t11)
-1.92 state_piecewise(2)_pw_t11_CC_lambda(1)
-1.8599999999999999 state_piecewise(2)_pw_t11_CC_lambda(2)
-1.8 state_piecewise(2)_pw_t11_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t11_CC_constraint3_:
+1 state_piecewise(2)_pw_t11_CC_lambda(1)
+1 state_piecewise(2)_pw_t11_CC_lambda(2)
+1 state_piecewise(2)_pw_t11_CC_lambda(3)
= 1

c_u_state_piecewise(2)_pw_t11_CC_constraint4(1)_:
+1 state_piecewise(2)_pw_t11_CC_lambda(1)
-1 state_piecewise(2)_pw_t11_CC_bin_y(1)
<= 0

c_u_state_piecewise(2)_pw_t11_CC_constraint4(2)_:
+1 state_piecewise(2)_pw_t11_CC_lambda(2)
-1 state_piecewise(2)_pw_t11_CC_bin_y(1)
-1 state_piecewise(2)_pw_t11_CC_bin_y(2)
<= 0

c_u_state_piecewise(2)_pw_t11_CC_constraint4(3)_:
+1 state_piecewise(2)_pw_t11_CC_lambda(3)
-1 state_piecewise(2)_pw_t11_CC_bin_y(2)
<= 0

c_e_state_piecewise(2)_pw_t11_CC_constraint5_:
+1 state_piecewise(2)_pw_t11_CC_bin_y(1)
+1 state_piecewise(2)_pw_t11_CC_bin_y(2)
= 1

c_e_state_piecewise(2)_pw_t12_CC_constraint1_:
+1 state(reach1_discharge_downstream_t12)
-3.8 state_piecewise(2)_pw_t12_CC_lambda(1)
-9.5 state_piecewise(2)_pw_t12_CC_lambda(2)
-13.0 state_piecewise(2)_pw_t12_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t12_CC_constraint2_:
+1 state(reach1_level_downstream_t12)
-1.92 state_piecewise(2)_pw_t12_CC_lambda(1)
-1.8599999999999999 state_piecewise(2)_pw_t12_CC_lambda(2)
-1.8 state_piecewise(2)_pw_t12_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t12_CC_constraint3_:
+1 state_piecewise(2)_pw_t12_CC_lambda(1)
+1 state_piecewise(2)_pw_t12_CC_lambda(2)
+1 state_piecewise(2)_pw_t12_CC_lambda(3)
= 1

c_u_state_piecewise(2)_pw_t12_CC_constraint4(1)_:
+1 state_piecewise(2)_pw_t12_CC_lambda(1)
-1 state_piecewise(2)_pw_t12_CC_bin_y(1)
<= 0

c_u_state_piecewise(2)_pw_t12_CC_constraint4(2)_:
+1 state_piecewise(2)_pw_t12_CC_lambda(2)
-1 state_piecewise(2)_pw_t12_CC_bin_y(1)
-1 state_piecewise(2)_pw_t12_CC_bin_y(2)
<= 0

c_u_state_piecewise(2)_pw_t12_CC_constraint4(3)_:
+1 state_piecewise(2)_pw_t12_CC_lambda(3)
-1 state_piecewise(2)_pw_t12_CC_bin_y(2)
<= 0

c_e_state_piecewise(2)_pw_t12_CC_constraint5_:
+1 state_piecewise(2)_pw_t12_CC_bin_y(1)
+1 state_piecewise(2)_pw_t12_CC_bin_y(2)
= 1

c_e_state_piecewise(2)_pw_t13_CC_constraint1_:
+1 state(reach1_discharge_downstream_t13)
-3.8 state_piecewise(2)_pw_t13_CC_lambda(1)
-9.5 state_piecewise(2)_pw_t13_CC_lambda(2)
-13.0 state_piecewise(2)_pw_t13_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t13_CC_constraint2_:
+1 state(reach1_level_downstream_t13)
-1.92 state_piecewise(2)_pw_t13_CC_lambda(1)
-1.8599999999999999 state_piecewise(2)_pw_t13_CC_lambda(2)
-1.8 state_piecewise(2)_pw_t13_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t13_CC_constraint3_:
+1 state_piecewise(2)_pw_t13_CC_lambda(1)
+1 state_piecewise(2)_pw_t13_CC_lambda(2)
+1 state_piecewise(2)_pw_t13_CC_lambda(3)
= 1

c_u_state_piecewise(2)_pw_t13_CC_constraint4(1)_:
+1 state_piecewise(2)_pw_t13_CC_lambda(1)
-1 state_piecewise(2)_pw_t13_CC_bin_y(1)
<= 0

c_u_state_piecewise(2)_pw_t13_CC_constraint4(2)_:
+1 state_piecewise(2)_pw_t13_CC_lambda(2)
-1 state_piecewise(2)_pw_t13_CC_bin_y(1)
-1 state_piecewise(2)_pw_t13_CC_bin_y(2)
<= 0

c_u_state_piecewise(2)_pw_t13_CC_constraint4(3)_:
+1 state_piecewise(2)_pw_t13_CC_lambda(3)
-1 state_piecewise(2)_pw_t13_CC_bin_y(2)
<= 0

c_e_state_piecewise(2)_pw_t13_CC_constraint5_:
+1 state_piecewise(2)_pw_t13_CC_bin_y(1)
+1 state_piecewise(2)_pw_t13_CC_bin_y(2)
= 1

c_e_state_piecewise(2)_pw_t14_CC_constraint1_:
+1 state(reach1_discharge_downstream_t14)
-3.8 state_piecewise(2)_pw_t14_CC_lambda(1)
-9.5 state_piecewise(2)_pw_t14_CC_lambda(2)
-13.0 state_piecewise(2)_pw_t14_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t14_CC_constraint2_:
+1 state(reach1_level_downstream_t14)
-1.92 state_piecewise(2)_pw_t14_CC_lambda(1)
-1.8599999999999999 state_piecewise(2)_pw_t14_CC_lambda(2)
-1.8 state_piecewise(2)_pw_t14_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t14_CC_constraint3_:
+1 state_piecewise(2)_pw_t14_CC_lambda(1)
+1 state_piecewise(2)_pw_t14_CC_lambda(2)
+1 state_piecewise(2)_pw_t14_CC_lambda(3)
= 1

c_u_state_piecewise(2)_pw_t14_CC_constraint4(1)_:
+1 state_piecewise(2)_pw_t14_CC_lambda(1)
-1 state_piecewise(2)_pw_t14_CC_bin_y(1)
<= 0

c_u_state_piecewise(2)_pw_t14_CC_constraint4(2)_:
+1 state_piecewise(2)_pw_t14_CC_lambda(2)
-1 state_piecewise(2)_pw_t14_CC_bin_y(1)
-1 state_piecewise(2)_pw_t14_CC_bin_y(2)
<= 0

c_u_state_piecewise(2)_pw_t14_CC_constraint4(3)_:
+1 state_piecewise(2)_pw_t14_CC_lambda(3)
-1 state_piecewise(2)_pw_t14_CC_bin_y(2)
<= 0

c_e_state_piecewise(2)_pw_t14_CC_constraint5_:
+1 state_piecewise(2)_pw_t14_CC_bin_y(1)
+1 state_piecewise(2)_pw_t14_CC_bin_y(2)
= 1

c_e_state_piecewise(2)_pw_t15_CC_constraint1_:
+1 state(reach1_discharge_downstream_t15)
-3.8 state_piecewise(2)_pw_t15_CC_lambda(1)
-9.5 state_piecewise(2)_pw_t15_CC_lambda(2)
-13.0 state_piecewise(2)_pw_t15_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t15_CC_constraint2_:
+1 state(reach1_level_downstream_t15)
-1.92 state_piecewise(2)_pw_t15_CC_lambda(1)
-1.8599999999999999 state_piecewise(2)_pw_t15_CC_lambda(2)
-1.8 state_piecewise(2)_pw_t15_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t15_CC_constraint3_:
+1 state_piecewise(2)_pw_t15_CC_lambda(1)
+1 state_piecewise(2)_pw_t15_CC_lambda(2)
+1 state_piecewise(2)_pw_t15_CC_lambda(3)
= 1

c_u_state_piecewise(2)_pw_t15_CC_constraint4(1)_:
+1 state_piecewise(2)_pw_t15_CC_lambda(1)
-1 state_piecewise(2)_pw_t15_CC_bin_y(1)
<= 0

c_u_state_piecewise(2)_pw_t15_CC_constraint4(2)_:
+1 state_piecewise(2)_pw_t15_CC_lambda(2)
-1 state_piecewise(2)_pw_t15_CC_bin_y(1)
-1 state_piecewise(2)_pw_t15_CC_bin_y(2)
<= 0

c_u_state_piecewise(2)_pw_t15_CC_constraint4(3)_:
+1 state_piecewise(2)_pw_t15_CC_lambda(3)
-1 state_piecewise(2)_pw_t15_CC_bin_y(2)
<= 0

c_e_state_piecewise(2)_pw_t15_CC_constraint5_:
+1 state_piecewise(2)_pw_t15_CC_bin_y(1)
+1 state_piecewise(2)_pw_t15_CC_bin_y(2)
= 1

c_e_state_piecewise(2)_pw_t16_CC_constraint1_:
+1 state(reach1_discharge_downstream_t16)
-3.8 state_piecewise(2)_pw_t16_CC_lambda(1)
-9.5 state_piecewise(2)_pw_t16_CC_lambda(2)
-13.0 state_piecewise(2)_pw_t16_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t16_CC_constraint2_:
+1 state(reach1_level_downstream_t16)
-1.92 state_piecewise(2)_pw_t16_CC_lambda(1)
-1.8599999999999999 state_piecewise(2)_pw_t16_CC_lambda(2)
-1.8 state_piecewise(2)_pw_t16_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t16_CC_constraint3_:
+1 state_piecewise(2)_pw_t16_CC_lambda(1)
+1 state_piecewise(2)_pw_t16_CC_lambda(2)
+1 state_piecewise(2)_pw_t16_CC_lambda(3)
= 1

c_u_state_piecewise(2)_pw_t16_CC_constraint4(1)_:
+1 state_piecewise(2)_pw_t16_CC_lambda(1)
-1 state_piecewise(2)_pw_t16_CC_bin_y(1)
<= 0

c_u_state_piecewise(2)_pw_t16_CC_constraint4(2)_:
+1 state_piecewise(2)_pw_t16_CC_lambda(2)
-1 state_piecewise(2)_pw_t16_CC_bin_y(1)
-1 state_piecewise(2)_pw_t16_CC_bin_y(2)
<= 0

c_u_state_piecewise(2)_pw_t16_CC_constraint4(3)_:
+1 state_piecewise(2)_pw_t16_CC_lambda(3)
-1 state_piecewise(2)_pw_t16_CC_bin_y(2)
<= 0

c_e_state_piecewise(2)_pw_t16_CC_constraint5_:
+1 state_piecewise(2)_pw_t16_CC_bin_y(1)
+1 state_piecewise(2)_pw_t16_CC_bin_y(2)
= 1

c_e_state_piecewise(2)_pw_t17_CC_constraint1_:
+1 state(reach1_discharge_downstream_t17)
-3.8 state_piecewise(2)_pw_t17_CC_lambda(1)
-9.5 state_piecewise(2)_pw_t17_CC_lambda(2)
-13.0 state_piecewise(2)_pw_t17_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t17_CC_constraint2_:
+1 state(reach1_level_downstream_t17)
-1.92 state_piecewise(2)_pw_t17_CC_lambda(1)
-1.8599999999999999 state_piecewise(2)_pw_t17_CC_lambda(2)
-1.8 state_piecewise(2)_pw_t17_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t17_CC_constraint3_:
+1 state_piecewise(2)_pw_t17_CC_lambda(1)
+1 state_piecewise(2)_pw_t17_CC_lambda(2)
+1 state_piecewise(2)_pw_t17_CC_lambda(3)
= 1

c_u_state_piecewise(2)_pw_t17_CC_constraint4(1)_:
+1 state_piecewise(2)_pw_t17_CC_lambda(1)
-1 state_piecewise(2)_pw_t17_CC_bin_y(1)
<= 0

c_u_state_piecewise(2)_pw_t17_CC_constraint4(2)_:
+1 state_piecewise(2)_pw_t17_CC_lambda(2)
-1 state_piecewise(2)_pw_t17_CC_bin_y(1)
-1 state_piecewise(2)_pw_t17_CC_bin_y(2)
<= 0

c_u_state_piecewise(2)_pw_t17_CC_constraint4(3)_:
+1 state_piecewise(2)_pw_t17_CC_lambda(3)
-1 state_piecewise(2)_pw_t17_CC_bin_y(2)
<= 0

c_e_state_piecewise(2)_pw_t17_CC_constraint5_:
+1 state_piecewise(2)_pw_t17_CC_bin_y(1)
+1 state_piecewise(2)_pw_t17_CC_bin_y(2)
= 1

c_e_state_piecewise(2)_pw_t18_CC_constraint1_:
+1 state(reach1_discharge_downstream_t18)
-3.8 state_piecewise(2)_pw_t18_CC_lambda(1)
-9.5 state_piecewise(2)_pw_t18_CC_lambda(2)
-13.0 state_piecewise(2)_pw_t18_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t18_CC_constraint2_:
+1 state(reach1_level_downstream_t18)
-1.92 state_piecewise(2)_pw_t18_CC_lambda(1)
-1.8599999999999999 state_piecewise(2)_pw_t18_CC_lambda(2)
-1.8 state_piecewise(2)_pw_t18_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t18_CC_constraint3_:
+1 state_piecewise(2)_pw_t18_CC_lambda(1)
+1 state_piecewise(2)_pw_t18_CC_lambda(2)
+1 state_piecewise(2)_pw_t18_CC_lambda(3)
= 1

c_u_state_piecewise(2)_pw_t18_CC_constraint4(1)_:
+1 state_piecewise(2)_pw_t18_CC_lambda(1)
-1 state_piecewise(2)_pw_t18_CC_bin_y(1)
<= 0

c_u_state_piecewise(2)_pw_t18_CC_constraint4(2)_:
+1 state_piecewise(2)_pw_t18_CC_lambda(2)
-1 state_piecewise(2)_pw_t18_CC_bin_y(1)
-1 state_piecewise(2)_pw_t18_CC_bin_y(2)
<= 0

c_u_state_piecewise(2)_pw_t18_CC_constraint4(3)_:
+1 state_piecewise(2)_pw_t18_CC_lambda(3)
-1 state_piecewise(2)_pw_t18_CC_bin_y(2)
<= 0

c_e_state_piecewise(2)_pw_t18_CC_constraint5_:
+1 state_piecewise(2)_pw_t18_CC_bin_y(1)
+1 state_piecewise(2)_pw_t18_CC_bin_y(2)
= 1

c_e_state_piecewise(2)_pw_t19_CC_constraint1_:
+1 state(reach1_discharge_downstream_t19)
-3.8 state_piecewise(2)_pw_t19_CC_lambda(1)
-9.5 state_piecewise(2)_pw_t19_CC_lambda(2)
-13.0 state_piecewise(2)_pw_t19_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t19_CC_constraint2_:
+1 state(reach1_level_downstream_t19)
-1.92 state_piecewise(2)_pw_t19_CC_lambda(1)
-1.8599999999999999 state_piecewise(2)_pw_t19_CC_lambda(2)
-1.8 state_piecewise(2)_pw_t19_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t19_CC_constraint3_:
+1 state_piecewise(2)_pw_t19_CC_lambda(1)
+1 state_piecewise(2)_pw_t19_CC_lambda(2)
+1 state_piecewise(2)_pw_t19_CC_lambda(3)
= 1

c_u_state_piecewise(2)_pw_t19_CC_constraint4(1)_:
+1 state_piecewise(2)_pw_t19_CC_lambda(1)
-1 state_piecewise(2)_pw_t19_CC_bin_y(1)
<= 0

c_u_state_piecewise(2)_pw_t19_CC_constraint4(2)_:
+1 state_piecewise(2)_pw_t19_CC_lambda(2)
-1 state_piecewise(2)_pw_t19_CC_bin_y(1)
-1 state_piecewise(2)_pw_t19_CC_bin_y(2)
<= 0

c_u_state_piecewise(2)_pw_t19_CC_constraint4(3)_:
+1 state_piecewise(2)_pw_t19_CC_lambda(3)
-1 state_piecewise(2)_pw_t19_CC_bin_y(2)
<= 0

c_e_state_piecewise(2)_pw_t19_CC_constraint5_:
+1 state_piecewise(2)_pw_t19_CC_bin_y(1)
+1 state_piecewise(2)_pw_t19_CC_bin_y(2)
= 1

c_e_state_piecewise(2)_pw_t20_CC_constraint1_:
+1 state(reach1_discharge_downstream_t20)
-3.8 state_piecewise(2)_pw_t20_CC_lambda(1)
-9.5 state_piecewise(2)_pw_t20_CC_lambda(2)
-13.0 state_piecewise(2)_pw_t20_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t20_CC_constraint2_:
+1 state(reach1_level_downstream_t20)
-1.92 state_piecewise(2)_pw_t20_CC_lambda(1)
-1.8599999999999999 state_piecewise(2)_pw_t20_CC_lambda(2)
-1.8 state_piecewise(2)_pw_t20_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t20_CC_constraint3_:
+1 state_piecewise(2)_pw_t20_CC_lambda(1)
+1 state_piecewise(2)_pw_t20_CC_lambda(2)
+1 state_piecewise(2)_pw_t20_CC_lambda(3)
= 1

c_u_state_piecewise(2)_pw_t20_CC_constraint4(1)_:
+1 state_piecewise(2)_pw_t20_CC_lambda(1)
-1 state_piecewise(2)_pw_t20_CC_bin_y(1)
<= 0

c_u_state_piecewise(2)_pw_t20_CC_constraint4(2)_:
+1 state_piecewise(2)_pw_t20_CC_lambda(2)
-1 state_piecewise(2)_pw_t20_CC_bin_y(1)
-1 state_piecewise(2)_pw_t20_CC_bin_y(2)
<= 0

c_u_state_piecewise(2)_pw_t20_CC_constraint4(3)_:
+1 state_piecewise(2)_pw_t20_CC_lambda(3)
-1 state_piecewise(2)_pw_t20_CC_bin_y(2)
<= 0

c_e_state_piecewise(2)_pw_t20_CC_constraint5_:
+1 state_piecewise(2)_pw_t20_CC_bin_y(1)
+1 state_piecewise(2)_pw_t20_CC_bin_y(2)
= 1

c_e_state_piecewise(2)_pw_t21_CC_constraint1_:
+1 state(reach1_discharge_downstream_t21)
-3.8 state_piecewise(2)_pw_t21_CC_lambda(1)
-9.5 state_piecewise(2)_pw_t21_CC_lambda(2)
-13.0 state_piecewise(2)_pw_t21_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t21_CC_constraint2_:
+1 state(reach1_level_downstream_t21)
-1.92 state_piecewise(2)_pw_t21_CC_lambda(1)
-1.8599999999999999 state_piecewise(2)_pw_t21_CC_lambda(2)
-1.8 state_piecewise(2)_pw_t21_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t21_CC_constraint3_:
+1 state_piecewise(2)_pw_t21_CC_lambda(1)
+1 state_piecewise(2)_pw_t21_CC_lambda(2)
+1 state_piecewise(2)_pw_t21_CC_lambda(3)
= 1

c_u_state_piecewise(2)_pw_t21_CC_constraint4(1)_:
+1 state_piecewise(2)_pw_t21_CC_lambda(1)
-1 state_piecewise(2)_pw_t21_CC_bin_y(1)
<= 0

c_u_state_piecewise(2)_pw_t21_CC_constraint4(2)_:
+1 state_piecewise(2)_pw_t21_CC_lambda(2)
-1 state_piecewise(2)_pw_t21_CC_bin_y(1)
-1 state_piecewise(2)_pw_t21_CC_bin_y(2)
<= 0

c_u_state_piecewise(2)_pw_t21_CC_constraint4(3)_:
+1 state_piecewise(2)_pw_t21_CC_lambda(3)
-1 state_piecewise(2)_pw_t21_CC_bin_y(2)
<= 0

c_e_state_piecewise(2)_pw_t21_CC_constraint5_:
+1 state_piecewise(2)_pw_t21_CC_bin_y(1)
+1 state_piecewise(2)_pw_t21_CC_bin_y(2)
= 1

c_e_state_piecewise(2)_pw_t22_CC_constraint1_:
+1 state(reach1_discharge_downstream_t22)
-3.8 state_piecewise(2)_pw_t22_CC_lambda(1)
-9.5 state_piecewise(2)_pw_t22_CC_lambda(2)
-13.0 state_piecewise(2)_pw_t22_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t22_CC_constraint2_:
+1 state(reach1_level_downstream_t22)
-1.92 state_piecewise(2)_pw_t22_CC_lambda(1)
-1.8599999999999999 state_piecewise(2)_pw_t22_CC_lambda(2)
-1.8 state_piecewise(2)_pw_t22_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t22_CC_constraint3_:
+1 state_piecewise(2)_pw_t22_CC_lambda(1)
+1 state_piecewise(2)_pw_t22_CC_lambda(2)
+1 state_piecewise(2)_pw_t22_CC_lambda(3)
= 1

c_u_state_piecewise(2)_pw_t22_CC_constraint4(1)_:
+1 state_piecewise(2)_pw_t22_CC_lambda(1)
-1 state_piecewise(2)_pw_t22_CC_bin_y(1)
<= 0

c_u_state_piecewise(2)_pw_t22_CC_constraint4(2)_:
+1 state_piecewise(2)_pw_t22_CC_lambda(2)
-1 state_piecewise(2)_pw_t22_CC_bin_y(1)
-1 state_piecewise(2)_pw_t22_CC_bin_y(2)
<= 0

c_u_state_piecewise(2)_pw_t22_CC_constraint4(3)_:
+1 state_piecewise(2)_pw_t22_CC_lambda(3)
-1 state_piecewise(2)_pw_t22_CC_bin_y(2)
<= 0

c_e_state_piecewise(2)_pw_t22_CC_constraint5_:
+1 state_piecewise(2)_pw_t22_CC_bin_y(1)
+1 state_piecewise(2)_pw_t22_CC_bin_y(2)
= 1

c_e_state_piecewise(2)_pw_t23_CC_constraint1_:
+1 state(reach1_discharge_downstream_t23)
-3.8 state_piecewise(2)_pw_t23_CC_lambda(1)
-9.5 state_piecewise(2)_pw_t23_CC_lambda(2)
-13.0 state_piecewise(2)_pw_t23_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t23_CC_constraint2_:
+1 state(reach1_level_downstream_t23)
-1.92 state_piecewise(2)_pw_t23_CC_lambda(1)
-1.8599999999999999 state_piecewise(2)_pw_t23_CC_lambda(2)
-1.8 state_piecewise(2)_pw_t23_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t23_CC_constraint3_:
+1 state_piecewise(2)_pw_t23_CC_lambda(1)
+1 state_piecewise(2)_pw_t23_CC_lambda(2)
+1 state_piecewise(2)_pw_t23_CC_lambda(3)
= 1

c_u_state_piecewise(2)_pw_t23_CC_constraint4(1)_:
+1 state_piecewise(2)_pw_t23_CC_lambda(1)
-1 state_piecewise(2)_pw_t23_CC_bin_y(1)
<= 0

c_u_state_piecewise(2)_pw_t23_CC_constraint4(2)_:
+1 state_piecewise(2)_pw_t23_CC_lambda(2)
-1 state_piecewise(2)_pw_t23_CC_bin_y(1)
-1 state_piecewise(2)_pw_t23_CC_bin_y(2)
<= 0

c_u_state_piecewise(2)_pw_t23_CC_constraint4(3)_:
+1 state_piecewise(2)_pw_t23_CC_lambda(3)
-1 state_piecewise(2)_pw_t23_CC_bin_y(2)
<= 0

c_e_state_piecewise(2)_pw_t23_CC_constraint5_:
+1 state_piecewise(2)_pw_t23_CC_bin_y(1)
+1 state_piecewise(2)_pw_t23_CC_bin_y(2)
= 1

c_e_state_piecewise(2)_pw_t24_CC_constraint1_:
+1 state(reach1_discharge_downstream_t24)
-3.8 state_piecewise(2)_pw_t24_CC_lambda(1)
-9.5 state_piecewise(2)_pw_t24_CC_lambda(2)
-13.0 state_piecewise(2)_pw_t24_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t24_CC_constraint2_:
+1 state(reach1_level_downstream_t24)
-1.92 state_piecewise(2)_pw_t24_CC_lambda(1)
-1.8599999999999999 state_piecewise(2)_pw_t24_CC_lambda(2)
-1.8 state_piecewise(2)_pw_t24_CC_lambda(3)
= 0

c_e_state_piecewise(2)_pw_t24_CC_constraint3_:
+1 state_piecewise(2)_pw_t24_CC_lambda(1)
+1 state_piecewise(2)_pw_t24_CC_lambda(2)
+1 state_piecewise(2)_pw_t24_CC_lambda(3)
= 1

c_u_state_piecewise(2)_pw_t24_CC_constraint4(1)_:
+1 state_piecewise(2)_pw_t24_CC_lambda(1)
-1 state_piecewise(2)_pw_t24_CC_bin_y(1)
<= 0

c_u_state_piecewise(2)_pw_t24_CC_constraint4(2)_:
+1 state_piecewise(2)_pw_t24_CC_lambda(2)
-1 state_piecewise(2)_pw_t24_CC_bin_y(1)
-1 state_piecewise(2)_pw_t24_CC_bin_y(2)
<= 0

c_u_state_piecewise(2)_pw_t24_CC_constraint4(3)_:
+1 state_piecewise(2)_pw_t24_CC_lambda(3)
-1 state_piecewise(2)_pw_t24_CC_bin_y(2)
<= 0

c_e_state_piecewise(2)_pw_t24_CC_constraint5_:
+1 state_piecewise(2)_pw_t24_CC_bin_y(1)
+1 state_piecewise(2)_pw_t24_CC_bin_y(2)
= 1

c_e_state_piecewise(3)_pw_t01_CC_constraint1_:
+1 state(reach1_storage_t01)
-200.0 state_piecewise(3)_pw_t01_CC_lambda(1)
-260.0 state_piecewise(3)_pw_t01_CC_lambda(2)
-320.0 state_piecewise(3)_pw_t01_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t01_CC_constraint2_:
+1 state(reach1_discharge_downstream_t01)
-3.8 state_piecewise(3)_pw_t01_CC_lambda(1)
-9.5 state_piecewise(3)_pw_t01_CC_lambda(2)
-13.0 state_piecewise(3)_pw_t01_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t01_CC_constraint3_:
+1 state_piecewise(3)_pw_t01_CC_lambda(1)
+1 state_piecewise(3)_pw_t01_CC_lambda(2)
+1 state_piecewise(3)_pw_t01_CC_lambda(3)
= 1

c_u_state_piecewise(3)_pw_t01_CC_constraint4(1)_:
+1 state_piecewise(3)_pw_t01_CC_lambda(1)
-1 state_piecewise(3)_pw_t01_CC_bin_y(1)
<= 0

c_u_state_piecewise(3)_pw_t01_CC_constraint4(2)_:
+1 state_piecewise(3)_pw_t01_CC_lambda(2)
-1 state_piecewise(3)_pw_t01_CC_bin_y(1)
-1 state_piecewise(3)_pw_t01_CC_bin_y(2)
<= 0

c_u_state_piecewise(3)_pw_t01_CC_constraint4(3)_:
+1 state_piecewise(3)_pw_t01_CC_lambda(3)
-1 state_piecewise(3)_pw_t01_CC_bin_y(2)
<= 0

c_e_state_piecewise(3)_pw_t01_CC_constraint5_:
+1 state_piecewise(3)_pw_t01_CC_bin_y(1)
+1 state_piecewise(3)_pw_t01_CC_bin_y(2)
= 1

c_e_state_piecewise(3)_pw_t02_CC_constraint1_:
+1 state(reach1_storage_t02)
-200.0 state_piecewise(3)_pw_t02_CC_lambda(1)
-260.0 state_piecewise(3)_pw_t02_CC_lambda(2)
-320.0 state_piecewise(3)_pw_t02_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t02_CC_constraint2_:
+1 state(reach1_discharge_downstream_t02)
-3.8 state_piecewise(3)_pw_t02_CC_lambda(1)
-9.5 state_piecewise(3)_pw_t02_CC_lambda(2)
-13.0 state_piecewise(3)_pw_t02_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t02_CC_constraint3_:
+1 state_piecewise(3)_pw_t02_CC_lambda(1)
+1 state_piecewise(3)_pw_t02_CC_lambda(2)
+1 state_piecewise(3)_pw_t02_CC_lambda(3)
= 1

c_u_state_piecewise(3)_pw_t02_CC_constraint4(1)_:
+1 state_piecewise(3)_pw_t02_CC_lambda(1)
-1 state_piecewise(3)_pw_t02_CC_bin_y(1)
<= 0

c_u_state_piecewise(3)_pw_t02_CC_constraint4(2)_:
+1 state_piecewise(3)_pw_t02_CC_lambda(2)
-1 state_piecewise(3)_pw_t02_CC_bin_y(1)
-1 state_piecewise(3)_pw_t02_CC_bin_y(2)
<= 0

c_u_state_piecewise(3)_pw_t02_CC_constraint4(3)_:
+1 state_piecewise(3)_pw_t02_CC_lambda(3)
-1 state_piecewise(3)_pw_t02_CC_bin_y(2)
<= 0

c_e_state_piecewise(3)_pw_t02_CC_constraint5_:
+1 state_piecewise(3)_pw_t02_CC_bin_y(1)
+1 state_piecewise(3)_pw_t02_CC_bin_y(2)
= 1

c_e_state_piecewise(3)_pw_t03_CC_constraint1_:
+1 state(reach1_storage_t03)
-200.0 state_piecewise(3)_pw_t03_CC_lambda(1)
-260.0 state_piecewise(3)_pw_t03_CC_lambda(2)
-320.0 state_piecewise(3)_pw_t03_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t03_CC_constraint2_:
+1 state(reach1_discharge_downstream_t03)
-3.8 state_piecewise(3)_pw_t03_CC_lambda(1)
-9.5 state_piecewise(3)_pw_t03_CC_lambda(2)
-13.0 state_piecewise(3)_pw_t03_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t03_CC_constraint3_:
+1 state_piecewise(3)_pw_t03_CC_lambda(1)
+1 state_piecewise(3)_pw_t03_CC_lambda(2)
+1 state_piecewise(3)_pw_t03_CC_lambda(3)
= 1

c_u_state_piecewise(3)_pw_t03_CC_constraint4(1)_:
+1 state_piecewise(3)_pw_t03_CC_lambda(1)
-1 state_piecewise(3)_pw_t03_CC_bin_y(1)
<= 0

c_u_state_piecewise(3)_pw_t03_CC_constraint4(2)_:
+1 state_piecewise(3)_pw_t03_CC_lambda(2)
-1 state_piecewise(3)_pw_t03_CC_bin_y(1)
-1 state_piecewise(3)_pw_t03_CC_bin_y(2)
<= 0

c_u_state_piecewise(3)_pw_t03_CC_constraint4(3)_:
+1 state_piecewise(3)_pw_t03_CC_lambda(3)
-1 state_piecewise(3)_pw_t03_CC_bin_y(2)
<= 0

c_e_state_piecewise(3)_pw_t03_CC_constraint5_:
+1 state_piecewise(3)_pw_t03_CC_bin_y(1)
+1 state_piecewise(3)_pw_t03_CC_bin_y(2)
= 1

c_e_state_piecewise(3)_pw_t04_CC_constraint1_:
+1 state(reach1_storage_t04)
-200.0 state_piecewise(3)_pw_t04_CC_lambda(1)
-260.0 state_piecewise(3)_pw_t04_CC_lambda(2)
-320.0 state_piecewise(3)_pw_t04_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t04_CC_constraint2_:
+1 state(reach1_discharge_downstream_t04)
-3.8 state_piecewise(3)_pw_t04_CC_lambda(1)
-9.5 state_piecewise(3)_pw_t04_CC_lambda(2)
-13.0 state_piecewise(3)_pw_t04_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t04_CC_constraint3_:
+1 state_piecewise(3)_pw_t04_CC_lambda(1)
+1 state_piecewise(3)_pw_t04_CC_lambda(2)
+1 state_piecewise(3)_pw_t04_CC_lambda(3)
= 1

c_u_state_piecewise(3)_pw_t04_CC_constraint4(1)_:
+1 state_piecewise(3)_pw_t04_CC_lambda(1)
-1 state_piecewise(3)_pw_t04_CC_bin_y(1)
<= 0

c_u_state_piecewise(3)_pw_t04_CC_constraint4(2)_:
+1 state_piecewise(3)_pw_t04_CC_lambda(2)
-1 state_piecewise(3)_pw_t04_CC_bin_y(1)
-1 state_piecewise(3)_pw_t04_CC_bin_y(2)
<= 0

c_u_state_piecewise(3)_pw_t04_CC_constraint4(3)_:
+1 state_piecewise(3)_pw_t04_CC_lambda(3)
-1 state_piecewise(3)_pw_t04_CC_bin_y(2)
<= 0

c_e_state_piecewise(3)_pw_t04_CC_constraint5_:
+1 state_piecewise(3)_pw_t04_CC_bin_y(1)
+1 state_piecewise(3)_pw_t04_CC_bin_y(2)
= 1

c_e_state_piecewise(3)_pw_t05_CC_constraint1_:
+1 state(reach1_storage_t05)
-200.0 state_piecewise(3)_pw_t05_CC_lambda(1)
-260.0 state_piecewise(3)_pw_t05_CC_lambda(2)
-320.0 state_piecewise(3)_pw_t05_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t05_CC_constraint2_:
+1 state(reach1_discharge_downstream_t05)
-3.8 state_piecewise(3)_pw_t05_CC_lambda(1)
-9.5 state_piecewise(3)_pw_t05_CC_lambda(2)
-13.0 state_piecewise(3)_pw_t05_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t05_CC_constraint3_:
+1 state_piecewise(3)_pw_t05_CC_lambda(1)
+1 state_piecewise(3)_pw_t05_CC_lambda(2)
+1 state_piecewise(3)_pw_t05_CC_lambda(3)
= 1

c_u_state_piecewise(3)_pw_t05_CC_constraint4(1)_:
+1 state_piecewise(3)_pw_t05_CC_lambda(1)
-1 state_piecewise(3)_pw_t05_CC_bin_y(1)
<= 0

c_u_state_piecewise(3)_pw_t05_CC_constraint4(2)_:
+1 state_piecewise(3)_pw_t05_CC_lambda(2)
-1 state_piecewise(3)_pw_t05_CC_bin_y(1)
-1 state_piecewise(3)_pw_t05_CC_bin_y(2)
<= 0

c_u_state_piecewise(3)_pw_t05_CC_constraint4(3)_:
+1 state_piecewise(3)_pw_t05_CC_lambda(3)
-1 state_piecewise(3)_pw_t05_CC_bin_y(2)
<= 0

c_e_state_piecewise(3)_pw_t05_CC_constraint5_:
+1 state_piecewise(3)_pw_t05_CC_bin_y(1)
+1 state_piecewise(3)_pw_t05_CC_bin_y(2)
= 1

c_e_state_piecewise(3)_pw_t06_CC_constraint1_:
+1 state(reach1_storage_t06)
-200.0 state_piecewise(3)_pw_t06_CC_lambda(1)
-260.0 state_piecewise(3)_pw_t06_CC_lambda(2)
-320.0 state_piecewise(3)_pw_t06_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t06_CC_constraint2_:
+1 state(reach1_discharge_downstream_t06)
-3.8 state_piecewise(3)_pw_t06_CC_lambda(1)
-9.5 state_piecewise(3)_pw_t06_CC_lambda(2)
-13.0 state_piecewise(3)_pw_t06_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t06_CC_constraint3_:
+1 state_piecewise(3)_pw_t06_CC_lambda(1)
+1 state_piecewise(3)_pw_t06_CC_lambda(2)
+1 state_piecewise(3)_pw_t06_CC_lambda(3)
= 1

c_u_state_piecewise(3)_pw_t06_CC_constraint4(1)_:
+1 state_piecewise(3)_pw_t06_CC_lambda(1)
-1 state_piecewise(3)_pw_t06_CC_bin_y(1)
<= 0

c_u_state_piecewise(3)_pw_t06_CC_constraint4(2)_:
+1 state_piecewise(3)_pw_t06_CC_lambda(2)
-1 state_piecewise(3)_pw_t06_CC_bin_y(1)
-1 state_piecewise(3)_pw_t06_CC_bin_y(2)
<= 0

c_u_state_piecewise(3)_pw_t06_CC_constraint4(3)_:
+1 state_piecewise(3)_pw_t06_CC_lambda(3)
-1 state_piecewise(3)_pw_t06_CC_bin_y(2)
<= 0

c_e_state_piecewise(3)_pw_t06_CC_constraint5_:
+1 state_piecewise(3)_pw_t06_CC_bin_y(1)
+1 state_piecewise(3)_pw_t06_CC_bin_y(2)
= 1

c_e_state_piecewise(3)_pw_t07_CC_constraint1_:
+1 state(reach1_storage_t07)
-200.0 state_piecewise(3)_pw_t07_CC_lambda(1)
-260.0 state_piecewise(3)_pw_t07_CC_lambda(2)
-320.0 state_piecewise(3)_pw_t07_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t07_CC_constraint2_:
+1 state(reach1_discharge_downstream_t07)
-3.8 state_piecewise(3)_pw_t07_CC_lambda(1)
-9.5 state_piecewise(3)_pw_t07_CC_lambda(2)
-13.0 state_piecewise(3)_pw_t07_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t07_CC_constraint3_:
+1 state_piecewise(3)_pw_t07_CC_lambda(1)
+1 state_piecewise(3)_pw_t07_CC_lambda(2)
+1 state_piecewise(3)_pw_t07_CC_lambda(3)
= 1

c_u_state_piecewise(3)_pw_t07_CC_constraint4(1)_:
+1 state_piecewise(3)_pw_t07_CC_lambda(1)
-1 state_piecewise(3)_pw_t07_CC_bin_y(1)
<= 0

c_u_state_piecewise(3)_pw_t07_CC_constraint4(2)_:
+1 state_piecewise(3)_pw_t07_CC_lambda(2)
-1 state_piecewise(3)_pw_t07_CC_bin_y(1)
-1 state_piecewise(3)_pw_t07_CC_bin_y(2)
<= 0

c_u_state_piecewise(3)_pw_t07_CC_constraint4(3)_:
+1 state_piecewise(3)_pw_t07_CC_lambda(3)
-1 state_piecewise(3)_pw_t07_CC_bin_y(2)
<= 0

c_e_state_piecewise(3)_pw_t07_CC_constraint5_:
+1 state_piecewise(3)_pw_t07_CC_bin_y(1)
+1 state_piecewise(3)_pw_t07_CC_bin_y(2)
= 1

c_e_state_piecewise(3)_pw_t08_CC_constraint1_:
+1 state(reach1_storage_t08)
-200.0 state_piecewise(3)_pw_t08_CC_lambda(1)
-260.0 state_piecewise(3)_pw_t08_CC_lambda(2)
-320.0 state_piecewise(3)_pw_t08_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t08_CC_constraint2_:
+1 state(reach1_discharge_downstream_t08)
-3.8 state_piecewise(3)_pw_t08_CC_lambda(1)
-9.5 state_piecewise(3)_pw_t08_CC_lambda(2)
-13.0 state_piecewise(3)_pw_t08_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t08_CC_constraint3_:
+1 state_piecewise(3)_pw_t08_CC_lambda(1)
+1 state_piecewise(3)_pw_t08_CC_lambda(2)
+1 state_piecewise(3)_pw_t08_CC_lambda(3)
= 1

c_u_state_piecewise(3)_pw_t08_CC_constraint4(1)_:
+1 state_piecewise(3)_pw_t08_CC_lambda(1)
-1 state_piecewise(3)_pw_t08_CC_bin_y(1)
<= 0

c_u_state_piecewise(3)_pw_t08_CC_constraint4(2)_:
+1 state_piecewise(3)_pw_t08_CC_lambda(2)
-1 state_piecewise(3)_pw_t08_CC_bin_y(1)
-1 state_piecewise(3)_pw_t08_CC_bin_y(2)
<= 0

c_u_state_piecewise(3)_pw_t08_CC_constraint4(3)_:
+1 state_piecewise(3)_pw_t08_CC_lambda(3)
-1 state_piecewise(3)_pw_t08_CC_bin_y(2)
<= 0

c_e_state_piecewise(3)_pw_t08_CC_constraint5_:
+1 state_piecewise(3)_pw_t08_CC_bin_y(1)
+1 state_piecewise(3)_pw_t08_CC_bin_y(2)
= 1

c_e_state_piecewise(3)_pw_t09_CC_constraint1_:
+1 state(reach1_storage_t09)
-200.0 state_piecewise(3)_pw_t09_CC_lambda(1)
-260.0 state_piecewise(3)_pw_t09_CC_lambda(2)
-320.0 state_piecewise(3)_pw_t09_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t09_CC_constraint2_:
+1 state(reach1_discharge_downstream_t09)
-3.8 state_piecewise(3)_pw_t09_CC_lambda(1)
-9.5 state_piecewise(3)_pw_t09_CC_lambda(2)
-13.0 state_piecewise(3)_pw_t09_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t09_CC_constraint3_:
+1 state_piecewise(3)_pw_t09_CC_lambda(1)
+1 state_piecewise(3)_pw_t09_CC_lambda(2)
+1 state_piecewise(3)_pw_t09_CC_lambda(3)
= 1

c_u_state_piecewise(3)_pw_t09_CC_constraint4(1)_:
+1 state_piecewise(3)_pw_t09_CC_lambda(1)
-1 state_piecewise(3)_pw_t09_CC_bin_y(1)
<= 0

c_u_state_piecewise(3)_pw_t09_CC_constraint4(2)_:
+1 state_piecewise(3)_pw_t09_CC_lambda(2)
-1 state_piecewise(3)_pw_t09_CC_bin_y(1)
-1 state_piecewise(3)_pw_t09_CC_bin_y(2)
<= 0

c_u_state_piecewise(3)_pw_t09_CC_constraint4(3)_:
+1 state_piecewise(3)_pw_t09_CC_lambda(3)
-1 state_piecewise(3)_pw_t09_CC_bin_y(2)
<= 0

c_e_state_piecewise(3)_pw_t09_CC_constraint5_:
+1 state_piecewise(3)_pw_t09_CC_bin_y(1)
+1 state_piecewise(3)_pw_t09_CC_bin_y(2)
= 1

c_e_state_piecewise(3)_pw_t10_CC_constraint1_:
+1 state(reach1_storage_t10)
-200.0 state_piecewise(3)_pw_t10_CC_lambda(1)
-260.0 state_piecewise(3)_pw_t10_CC_lambda(2)
-320.0 state_piecewise(3)_pw_t10_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t10_CC_constraint2_:
+1 state(reach1_discharge_downstream_t10)
-3.8 state_piecewise(3)_pw_t10_CC_lambda(1)
-9.5 state_piecewise(3)_pw_t10_CC_lambda(2)
-13.0 state_piecewise(3)_pw_t10_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t10_CC_constraint3_:
+1 state_piecewise(3)_pw_t10_CC_lambda(1)
+1 state_piecewise(3)_pw_t10_CC_lambda(2)
+1 state_piecewise(3)_pw_t10_CC_lambda(3)
= 1

c_u_state_piecewise(3)_pw_t10_CC_constraint4(1)_:
+1 state_piecewise(3)_pw_t10_CC_lambda(1)
-1 state_piecewise(3)_pw_t10_CC_bin_y(1)
<= 0

c_u_state_piecewise(3)_pw_t10_CC_constraint4(2)_:
+1 state_piecewise(3)_pw_t10_CC_lambda(2)
-1 state_piecewise(3)_pw_t10_CC_bin_y(1)
-1 state_piecewise(3)_pw_t10_CC_bin_y(2)
<= 0

c_u_state_piecewise(3)_pw_t10_CC_constraint4(3)_:
+1 state_piecewise(3)_pw_t10_CC_lambda(3)
-1 state_piecewise(3)_pw_t10_CC_bin_y(2)
<= 0

c_e_state_piecewise(3)_pw_t10_CC_constraint5_:
+1 state_piecewise(3)_pw_t10_CC_bin_y(1)
+1 state_piecewise(3)_pw_t10_CC_bin_y(2)
= 1

c_e_state_piecewise(3)_pw_t11_CC_constraint1_:
+1 state(reach1_storage_t11)
-200.0 state_piecewise(3)_pw_t11_CC_lambda(1)
-260.0 state_piecewise(3)_pw_t11_CC_lambda(2)
-320.0 state_piecewise(3)_pw_t11_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t11_CC_constraint2_:
+1 state(reach1_discharge_downstream_t11)
-3.8 state_piecewise(3)_pw_t11_CC_lambda(1)
-9.5 state_piecewise(3)_pw_t11_CC_lambda(2)
-13.0 state_piecewise(3)_pw_t11_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t11_CC_constraint3_:
+1 state_piecewise(3)_pw_t11_CC_lambda(1)
+1 state_piecewise(3)_pw_t11_CC_lambda(2)
+1 state_piecewise(3)_pw_t11_CC_lambda(3)
= 1

c_u_state_piecewise(3)_pw_t11_CC_constraint4(1)_:
+1 state_piecewise(3)_pw_t11_CC_lambda(1)
-1 state_piecewise(3)_pw_t11_CC_bin_y(1)
<= 0

c_u_state_piecewise(3)_pw_t11_CC_constraint4(2)_:
+1 state_piecewise(3)_pw_t11_CC_lambda(2)
-1 state_piecewise(3)_pw_t11_CC_bin_y(1)
-1 state_piecewise(3)_pw_t11_CC_bin_y(2)
<= 0

c_u_state_piecewise(3)_pw_t11_CC_constraint4(3)_:
+1 state_piecewise(3)_pw_t11_CC_lambda(3)
-1 state_piecewise(3)_pw_t11_CC_bin_y(2)
<= 0

c_e_state_piecewise(3)_pw_t11_CC_constraint5_:
+1 state_piecewise(3)_pw_t11_CC_bin_y(1)
+1 state_piecewise(3)_pw_t11_CC_bin_y(2)
= 1

c_e_state_piecewise(3)_pw_t12_CC_constraint1_:
+1 state(reach1_storage_t12)
-200.0 state_piecewise(3)_pw_t12_CC_lambda(1)
-260.0 state_piecewise(3)_pw_t12_CC_lambda(2)
-320.0 state_piecewise(3)_pw_t12_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t12_CC_constraint2_:
+1 state(reach1_discharge_downstream_t12)
-3.8 state_piecewise(3)_pw_t12_CC_lambda(1)
-9.5 state_piecewise(3)_pw_t12_CC_lambda(2)
-13.0 state_piecewise(3)_pw_t12_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t12_CC_constraint3_:
+1 state_piecewise(3)_pw_t12_CC_lambda(1)
+1 state_piecewise(3)_pw_t12_CC_lambda(2)
+1 state_piecewise(3)_pw_t12_CC_lambda(3)
= 1

c_u_state_piecewise(3)_pw_t12_CC_constraint4(1)_:
+1 state_piecewise(3)_pw_t12_CC_lambda(1)
-1 state_piecewise(3)_pw_t12_CC_bin_y(1)
<= 0

c_u_state_piecewise(3)_pw_t12_CC_constraint4(2)_:
+1 state_piecewise(3)_pw_t12_CC_lambda(2)
-1 state_piecewise(3)_pw_t12_CC_bin_y(1)
-1 state_piecewise(3)_pw_t12_CC_bin_y(2)
<= 0

c_u_state_piecewise(3)_pw_t12_CC_constraint4(3)_:
+1 state_piecewise(3)_pw_t12_CC_lambda(3)
-1 state_piecewise(3)_pw_t12_CC_bin_y(2)
<= 0

c_e_state_piecewise(3)_pw_t12_CC_constraint5_:
+1 state_piecewise(3)_pw_t12_CC_bin_y(1)
+1 state_piecewise(3)_pw_t12_CC_bin_y(2)
= 1

c_e_state_piecewise(3)_pw_t13_CC_constraint1_:
+1 state(reach1_storage_t13)
-200.0 state_piecewise(3)_pw_t13_CC_lambda(1)
-260.0 state_piecewise(3)_pw_t13_CC_lambda(2)
-320.0 state_piecewise(3)_pw_t13_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t13_CC_constraint2_:
+1 state(reach1_discharge_downstream_t13)
-3.8 state_piecewise(3)_pw_t13_CC_lambda(1)
-9.5 state_piecewise(3)_pw_t13_CC_lambda(2)
-13.0 state_piecewise(3)_pw_t13_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t13_CC_constraint3_:
+1 state_piecewise(3)_pw_t13_CC_lambda(1)
+1 state_piecewise(3)_pw_t13_CC_lambda(2)
+1 state_piecewise(3)_pw_t13_CC_lambda(3)
= 1

c_u_state_piecewise(3)_pw_t13_CC_constraint4(1)_:
+1 state_piecewise(3)_pw_t13_CC_lambda(1)
-1 state_piecewise(3)_pw_t13_CC_bin_y(1)
<= 0

c_u_state_piecewise(3)_pw_t13_CC_constraint4(2)_:
+1 state_piecewise(3)_pw_t13_CC_lambda(2)
-1 state_piecewise(3)_pw_t13_CC_bin_y(1)
-1 state_piecewise(3)_pw_t13_CC_bin_y(2)
<= 0

c_u_state_piecewise(3)_pw_t13_CC_constraint4(3)_:
+1 state_piecewise(3)_pw_t13_CC_lambda(3)
-1 state_piecewise(3)_pw_t13_CC_bin_y(2)
<= 0

c_e_state_piecewise(3)_pw_t13_CC_constraint5_:
+1 state_piecewise(3)_pw_t13_CC_bin_y(1)
+1 state_piecewise(3)_pw_t13_CC_bin_y(2)
= 1

c_e_state_piecewise(3)_pw_t14_CC_constraint1_:
+1 state(reach1_storage_t14)
-200.0 state_piecewise(3)_pw_t14_CC_lambda(1)
-260.0 state_piecewise(3)_pw_t14_CC_lambda(2)
-320.0 state_piecewise(3)_pw_t14_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t14_CC_constraint2_:
+1 state(reach1_discharge_downstream_t14)
-3.8 state_piecewise(3)_pw_t14_CC_lambda(1)
-9.5 state_piecewise(3)_pw_t14_CC_lambda(2)
-13.0 state_piecewise(3)_pw_t14_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t14_CC_constraint3_:
+1 state_piecewise(3)_pw_t14_CC_lambda(1)
+1 state_piecewise(3)_pw_t14_CC_lambda(2)
+1 state_piecewise(3)_pw_t14_CC_lambda(3)
= 1

c_u_state_piecewise(3)_pw_t14_CC_constraint4(1)_:
+1 state_piecewise(3)_pw_t14_CC_lambda(1)
-1 state_piecewise(3)_pw_t14_CC_bin_y(1)
<= 0

c_u_state_piecewise(3)_pw_t14_CC_constraint4(2)_:
+1 state_piecewise(3)_pw_t14_CC_lambda(2)
-1 state_piecewise(3)_pw_t14_CC_bin_y(1)
-1 state_piecewise(3)_pw_t14_CC_bin_y(2)
<= 0

c_u_state_piecewise(3)_pw_t14_CC_constraint4(3)_:
+1 state_piecewise(3)_pw_t14_CC_lambda(3)
-1 state_piecewise(3)_pw_t14_CC_bin_y(2)
<= 0

c_e_state_piecewise(3)_pw_t14_CC_constraint5_:
+1 state_piecewise(3)_pw_t14_CC_bin_y(1)
+1 state_piecewise(3)_pw_t14_CC_bin_y(2)
= 1

c_e_state_piecewise(3)_pw_t15_CC_constraint1_:
+1 state(reach1_storage_t15)
-200.0 state_piecewise(3)_pw_t15_CC_lambda(1)
-260.0 state_piecewise(3)_pw_t15_CC_lambda(2)
-320.0 state_piecewise(3)_pw_t15_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t15_CC_constraint2_:
+1 state(reach1_discharge_downstream_t15)
-3.8 state_piecewise(3)_pw_t15_CC_lambda(1)
-9.5 state_piecewise(3)_pw_t15_CC_lambda(2)
-13.0 state_piecewise(3)_pw_t15_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t15_CC_constraint3_:
+1 state_piecewise(3)_pw_t15_CC_lambda(1)
+1 state_piecewise(3)_pw_t15_CC_lambda(2)
+1 state_piecewise(3)_pw_t15_CC_lambda(3)
= 1

c_u_state_piecewise(3)_pw_t15_CC_constraint4(1)_:
+1 state_piecewise(3)_pw_t15_CC_lambda(1)
-1 state_piecewise(3)_pw_t15_CC_bin_y(1)
<= 0

c_u_state_piecewise(3)_pw_t15_CC_constraint4(2)_:
+1 state_piecewise(3)_pw_t15_CC_lambda(2)
-1 state_piecewise(3)_pw_t15_CC_bin_y(1)
-1 state_piecewise(3)_pw_t15_CC_bin_y(2)
<= 0

c_u_state_piecewise(3)_pw_t15_CC_constraint4(3)_:
+1 state_piecewise(3)_pw_t15_CC_lambda(3)
-1 state_piecewise(3)_pw_t15_CC_bin_y(2)
<= 0

c_e_state_piecewise(3)_pw_t15_CC_constraint5_:
+1 state_piecewise(3)_pw_t15_CC_bin_y(1)
+1 state_piecewise(3)_pw_t15_CC_bin_y(2)
= 1

c_e_state_piecewise(3)_pw_t16_CC_constraint1_:
+1 state(reach1_storage_t16)
-200.0 state_piecewise(3)_pw_t16_CC_lambda(1)
-260.0 state_piecewise(3)_pw_t16_CC_lambda(2)
-320.0 state_piecewise(3)_pw_t16_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t16_CC_constraint2_:
+1 state(reach1_discharge_downstream_t16)
-3.8 state_piecewise(3)_pw_t16_CC_lambda(1)
-9.5 state_piecewise(3)_pw_t16_CC_lambda(2)
-13.0 state_piecewise(3)_pw_t16_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t16_CC_constraint3_:
+1 state_piecewise(3)_pw_t16_CC_lambda(1)
+1 state_piecewise(3)_pw_t16_CC_lambda(2)
+1 state_piecewise(3)_pw_t16_CC_lambda(3)
= 1

c_u_state_piecewise(3)_pw_t16_CC_constraint4(1)_:
+1 state_piecewise(3)_pw_t16_CC_lambda(1)
-1 state_piecewise(3)_pw_t16_CC_bin_y(1)
<= 0

c_u_state_piecewise(3)_pw_t16_CC_constraint4(2)_:
+1 state_piecewise(3)_pw_t16_CC_lambda(2)
-1 state_piecewise(3)_pw_t16_CC_bin_y(1)
-1 state_piecewise(3)_pw_t16_CC_bin_y(2)
<= 0

c_u_state_piecewise(3)_pw_t16_CC_constraint4(3)_:
+1 state_piecewise(3)_pw_t16_CC_lambda(3)
-1 state_piecewise(3)_pw_t16_CC_bin_y(2)
<= 0

c_e_state_piecewise(3)_pw_t16_CC_constraint5_:
+1 state_piecewise(3)_pw_t16_CC_bin_y(1)
+1 state_piecewise(3)_pw_t16_CC_bin_y(2)
= 1

c_e_state_piecewise(3)_pw_t17_CC_constraint1_:
+1 state(reach1_storage_t17)
-200.0 state_piecewise(3)_pw_t17_CC_lambda(1)
-260.0 state_piecewise(3)_pw_t17_CC_lambda(2)
-320.0 state_piecewise(3)_pw_t17_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t17_CC_constraint2_:
+1 state(reach1_discharge_downstream_t17)
-3.8 state_piecewise(3)_pw_t17_CC_lambda(1)
-9.5 state_piecewise(3)_pw_t17_CC_lambda(2)
-13.0 state_piecewise(3)_pw_t17_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t17_CC_constraint3_:
+1 state_piecewise(3)_pw_t17_CC_lambda(1)
+1 state_piecewise(3)_pw_t17_CC_lambda(2)
+1 state_piecewise(3)_pw_t17_CC_lambda(3)
= 1

c_u_state_piecewise(3)_pw_t17_CC_constraint4(1)_:
+1 state_piecewise(3)_pw_t17_CC_lambda(1)
-1 state_piecewise(3)_pw_t17_CC_bin_y(1)
<= 0

c_u_state_piecewise(3)_pw_t17_CC_constraint4(2)_:
+1 state_piecewise(3)_pw_t17_CC_lambda(2)
-1 state_piecewise(3)_pw_t17_CC_bin_y(1)
-1 state_piecewise(3)_pw_t17_CC_bin_y(2)
<= 0

c_u_state_piecewise(3)_pw_t17_CC_constraint4(3)_:
+1 state_piecewise(3)_pw_t17_CC_lambda(3)
-1 state_piecewise(3)_pw_t17_CC_bin_y(2)
<= 0

c_e_state_piecewise(3)_pw_t17_CC_constraint5_:
+1 state_piecewise(3)_pw_t17_CC_bin_y(1)
+1 state_piecewise(3)_pw_t17_CC_bin_y(2)
= 1

c_e_state_piecewise(3)_pw_t18_CC_constraint1_:
+1 state(reach1_storage_t18)
-200.0 state_piecewise(3)_pw_t18_CC_lambda(1)
-260.0 state_piecewise(3)_pw_t18_CC_lambda(2)
-320.0 state_piecewise(3)_pw_t18_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t18_CC_constraint2_:
+1 state(reach1_discharge_downstream_t18)
-3.8 state_piecewise(3)_pw_t18_CC_lambda(1)
-9.5 state_piecewise(3)_pw_t18_CC_lambda(2)
-13.0 state_piecewise(3)_pw_t18_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t18_CC_constraint3_:
+1 state_piecewise(3)_pw_t18_CC_lambda(1)
+1 state_piecewise(3)_pw_t18_CC_lambda(2)
+1 state_piecewise(3)_pw_t18_CC_lambda(3)
= 1

c_u_state_piecewise(3)_pw_t18_CC_constraint4(1)_:
+1 state_piecewise(3)_pw_t18_CC_lambda(1)
-1 state_piecewise(3)_pw_t18_CC_bin_y(1)
<= 0

c_u_state_piecewise(3)_pw_t18_CC_constraint4(2)_:
+1 state_piecewise(3)_pw_t18_CC_lambda(2)
-1 state_piecewise(3)_pw_t18_CC_bin_y(1)
-1 state_piecewise(3)_pw_t18_CC_bin_y(2)
<= 0

c_u_state_piecewise(3)_pw_t18_CC_constraint4(3)_:
+1 state_piecewise(3)_pw_t18_CC_lambda(3)
-1 state_piecewise(3)_pw_t18_CC_bin_y(2)
<= 0

c_e_state_piecewise(3)_pw_t18_CC_constraint5_:
+1 state_piecewise(3)_pw_t18_CC_bin_y(1)
+1 state_piecewise(3)_pw_t18_CC_bin_y(2)
= 1

c_e_state_piecewise(3)_pw_t19_CC_constraint1_:
+1 state(reach1_storage_t19)
-200.0 state_piecewise(3)_pw_t19_CC_lambda(1)
-260.0 state_piecewise(3)_pw_t19_CC_lambda(2)
-320.0 state_piecewise(3)_pw_t19_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t19_CC_constraint2_:
+1 state(reach1_discharge_downstream_t19)
-3.8 state_piecewise(3)_pw_t19_CC_lambda(1)
-9.5 state_piecewise(3)_pw_t19_CC_lambda(2)
-13.0 state_piecewise(3)_pw_t19_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t19_CC_constraint3_:
+1 state_piecewise(3)_pw_t19_CC_lambda(1)
+1 state_piecewise(3)_pw_t19_CC_lambda(2)
+1 state_piecewise(3)_pw_t19_CC_lambda(3)
= 1

c_u_state_piecewise(3)_pw_t19_CC_constraint4(1)_:
+1 state_piecewise(3)_pw_t19_CC_lambda(1)
-1 state_piecewise(3)_pw_t19_CC_bin_y(1)
<= 0

c_u_state_piecewise(3)_pw_t19_CC_constraint4(2)_:
+1 state_piecewise(3)_pw_t19_CC_lambda(2)
-1 state_piecewise(3)_pw_t19_CC_bin_y(1)
-1 state_piecewise(3)_pw_t19_CC_bin_y(2)
<= 0

c_u_state_piecewise(3)_pw_t19_CC_constraint4(3)_:
+1 state_piecewise(3)_pw_t19_CC_lambda(3)
-1 state_piecewise(3)_pw_t19_CC_bin_y(2)
<= 0

c_e_state_piecewise(3)_pw_t19_CC_constraint5_:
+1 state_piecewise(3)_pw_t19_CC_bin_y(1)
+1 state_piecewise(3)_pw_t19_CC_bin_y(2)
= 1

c_e_state_piecewise(3)_pw_t20_CC_constraint1_:
+1 state(reach1_storage_t20)
-200.0 state_piecewise(3)_pw_t20_CC_lambda(1)
-260.0 state_piecewise(3)_pw_t20_CC_lambda(2)
-320.0 state_piecewise(3)_pw_t20_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t20_CC_constraint2_:
+1 state(reach1_discharge_downstream_t20)
-3.8 state_piecewise(3)_pw_t20_CC_lambda(1)
-9.5 state_piecewise(3)_pw_t20_CC_lambda(2)
-13.0 state_piecewise(3)_pw_t20_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t20_CC_constraint3_:
+1 state_piecewise(3)_pw_t20_CC_lambda(1)
+1 state_piecewise(3)_pw_t20_CC_lambda(2)
+1 state_piecewise(3)_pw_t20_CC_lambda(3)
= 1

c_u_state_piecewise(3)_pw_t20_CC_constraint4(1)_:
+1 state_piecewise(3)_pw_t20_CC_lambda(1)
-1 state_piecewise(3)_pw_t20_CC_bin_y(1)
<= 0

c_u_state_piecewise(3)_pw_t20_CC_constraint4(2)_:
+1 state_piecewise(3)_pw_t20_CC_lambda(2)
-1 state_piecewise(3)_pw_t20_CC_bin_y(1)
-1 state_piecewise(3)_pw_t20_CC_bin_y(2)
<= 0

c_u_state_piecewise(3)_pw_t20_CC_constraint4(3)_:
+1 state_piecewise(3)_pw_t20_CC_lambda(3)
-1 state_piecewise(3)_pw_t20_CC_bin_y(2)
<= 0

c_e_state_piecewise(3)_pw_t20_CC_constraint5_:
+1 state_piecewise(3)_pw_t20_CC_bin_y(1)
+1 state_piecewise(3)_pw_t20_CC_bin_y(2)
= 1

c_e_state_piecewise(3)_pw_t21_CC_constraint1_:
+1 state(reach1_storage_t21)
-200.0 state_piecewise(3)_pw_t21_CC_lambda(1)
-260.0 state_piecewise(3)_pw_t21_CC_lambda(2)
-320.0 state_piecewise(3)_pw_t21_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t21_CC_constraint2_:
+1 state(reach1_discharge_downstream_t21)
-3.8 state_piecewise(3)_pw_t21_CC_lambda(1)
-9.5 state_piecewise(3)_pw_t21_CC_lambda(2)
-13.0 state_piecewise(3)_pw_t21_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t21_CC_constraint3_:
+1 state_piecewise(3)_pw_t21_CC_lambda(1)
+1 state_piecewise(3)_pw_t21_CC_lambda(2)
+1 state_piecewise(3)_pw_t21_CC_lambda(3)
= 1

c_u_state_piecewise(3)_pw_t21_CC_constraint4(1)_:
+1 state_piecewise(3)_pw_t21_CC_lambda(1)
-1 state_piecewise(3)_pw_t21_CC_bin_y(1)
<= 0

c_u_state_piecewise(3)_pw_t21_CC_constraint4(2)_:
+1 state_piecewise(3)_pw_t21_CC_lambda(2)
-1 state_piecewise(3)_pw_t21_CC_bin_y(1)
-1 state_piecewise(3)_pw_t21_CC_bin_y(2)
<= 0

c_u_state_piecewise(3)_pw_t21_CC_constraint4(3)_:
+1 state_piecewise(3)_pw_t21_CC_lambda(3)
-1 state_piecewise(3)_pw_t21_CC_bin_y(2)
<= 0

c_e_state_piecewise(3)_pw_t21_CC_constraint5_:
+1 state_piecewise(3)_pw_t21_CC_bin_y(1)
+1 state_piecewise(3)_pw_t21_CC_bin_y(2)
= 1

c_e_state_piecewise(3)_pw_t22_CC_constraint1_:
+1 state(reach1_storage_t22)
-200.0 state_piecewise(3)_pw_t22_CC_lambda(1)
-260.0 state_piecewise(3)_pw_t22_CC_lambda(2)
-320.0 state_piecewise(3)_pw_t22_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t22_CC_constraint2_:
+1 state(reach1_discharge_downstream_t22)
-3.8 state_piecewise(3)_pw_t22_CC_lambda(1)
-9.5 state_piecewise(3)_pw_t22_CC_lambda(2)
-13.0 state_piecewise(3)_pw_t22_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t22_CC_constraint3_:
+1 state_piecewise(3)_pw_t22_CC_lambda(1)
+1 state_piecewise(3)_pw_t22_CC_lambda(2)
+1 state_piecewise(3)_pw_t22_CC_lambda(3)
= 1

c_u_state_piecewise(3)_pw_t22_CC_constraint4(1)_:
+1 state_piecewise(3)_pw_t22_CC_lambda(1)
-1 state_piecewise(3)_pw_t22_CC_bin_y(1)
<= 0

c_u_state_piecewise(3)_pw_t22_CC_constraint4(2)_:
+1 state_piecewise(3)_pw_t22_CC_lambda(2)
-1 state_piecewise(3)_pw_t22_CC_bin_y(1)
-1 state_piecewise(3)_pw_t22_CC_bin_y(2)
<= 0

c_u_state_piecewise(3)_pw_t22_CC_constraint4(3)_:
+1 state_piecewise(3)_pw_t22_CC_lambda(3)
-1 state_piecewise(3)_pw_t22_CC_bin_y(2)
<= 0

c_e_state_piecewise(3)_pw_t22_CC_constraint5_:
+1 state_piecewise(3)_pw_t22_CC_bin_y(1)
+1 state_piecewise(3)_pw_t22_CC_bin_y(2)
= 1

c_e_state_piecewise(3)_pw_t23_CC_constraint1_:
+1 state(reach1_storage_t23)
-200.0 state_piecewise(3)_pw_t23_CC_lambda(1)
-260.0 state_piecewise(3)_pw_t23_CC_lambda(2)
-320.0 state_piecewise(3)_pw_t23_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t23_CC_constraint2_:
+1 state(reach1_discharge_downstream_t23)
-3.8 state_piecewise(3)_pw_t23_CC_lambda(1)
-9.5 state_piecewise(3)_pw_t23_CC_lambda(2)
-13.0 state_piecewise(3)_pw_t23_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t23_CC_constraint3_:
+1 state_piecewise(3)_pw_t23_CC_lambda(1)
+1 state_piecewise(3)_pw_t23_CC_lambda(2)
+1 state_piecewise(3)_pw_t23_CC_lambda(3)
= 1

c_u_state_piecewise(3)_pw_t23_CC_constraint4(1)_:
+1 state_piecewise(3)_pw_t23_CC_lambda(1)
-1 state_piecewise(3)_pw_t23_CC_bin_y(1)
<= 0

c_u_state_piecewise(3)_pw_t23_CC_constraint4(2)_:
+1 state_piecewise(3)_pw_t23_CC_lambda(2)
-1 state_piecewise(3)_pw_t23_CC_bin_y(1)
-1 state_piecewise(3)_pw_t23_CC_bin_y(2)
<= 0

c_u_state_piecewise(3)_pw_t23_CC_constraint4(3)_:
+1 state_piecewise(3)_pw_t23_CC_lambda(3)
-1 state_piecewise(3)_pw_t23_CC_bin_y(2)
<= 0

c_e_state_piecewise(3)_pw_t23_CC_constraint5_:
+1 state_piecewise(3)_pw_t23_CC_bin_y(1)
+1 state_piecewise(3)_pw_t23_CC_bin_y(2)
= 1

c_e_state_piecewise(3)_pw_t24_CC_constraint1_:
+1 state(reach1_storage_t24)
-200.0 state_piecewise(3)_pw_t24_CC_lambda(1)
-260.0 state_piecewise(3)_pw_t24_CC_lambda(2)
-320.0 state_piecewise(3)_pw_t24_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t24_CC_constraint2_:
+1 state(reach1_discharge_downstream_t24)
-3.8 state_piecewise(3)_pw_t24_CC_lambda(1)
-9.5 state_piecewise(3)_pw_t24_CC_lambda(2)
-13.0 state_piecewise(3)_pw_t24_CC_lambda(3)
= 0

c_e_state_piecewise(3)_pw_t24_CC_constraint3_:
+1 state_piecewise(3)_pw_t24_CC_lambda(1)
+1 state_piecewise(3)_pw_t24_CC_lambda(2)
+1 state_piecewise(3)_pw_t24_CC_lambda(3)
= 1

c_u_state_piecewise(3)_pw_t24_CC_constraint4(1)_:
+1 state_piecewise(3)_pw_t24_CC_lambda(1)
-1 state_piecewise(3)_pw_t24_CC_bin_y(1)
<= 0

c_u_state_piecewise(3)_pw_t24_CC_constraint4(2)_:
+1 state_piecewise(3)_pw_t24_CC_lambda(2)
-1 state_piecewise(3)_pw_t24_CC_bin_y(1)
-1 state_piecewise(3)_pw_t24_CC_bin_y(2)
<= 0

c_u_state_piecewise(3)_pw_t24_CC_constraint4(3)_:
+1 state_piecewise(3)_pw_t24_CC_lambda(3)
-1 state_piecewise(3)_pw_t24_CC_bin_y(2)
<= 0

c_e_state_piecewise(3)_pw_t24_CC_constraint5_:
+1 state_piecewise(3)_pw_t24_CC_bin_y(1)
+1 state_piecewise(3)_pw_t24_CC_bin_y(2)
= 1

c_e_state_piecewise(4)_pw_t01_CC_constraint1_:
+1 state(reach2_discharge_upstream_t01)
-2.6 state_piecewise(4)_pw_t01_CC_lambda(1)
-7.8 state_piecewise(4)_pw_t01_CC_lambda(2)
-10.8 state_piecewise(4)_pw_t01_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t01_CC_constraint2_:
+1 state(reach2_level_upstream_t01)
-1.85 state_piecewise(4)_pw_t01_CC_lambda(1)
-1.7850000000000001 state_piecewise(4)_pw_t01_CC_lambda(2)
-1.72 state_piecewise(4)_pw_t01_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t01_CC_constraint3_:
+1 state_piecewise(4)_pw_t01_CC_lambda(1)
+1 state_piecewise(4)_pw_t01_CC_lambda(2)
+1 state_piecewise(4)_pw_t01_CC_lambda(3)
= 1

c_u_state_piecewise(4)_pw_t01_CC_constraint4(1)_:
+1 state_piecewise(4)_pw_t01_CC_lambda(1)
-1 state_piecewise(4)_pw_t01_CC_bin_y(1)
<= 0

c_u_state_piecewise(4)_pw_t01_CC_constraint4(2)_:
+1 state_piecewise(4)_pw_t01_CC_lambda(2)
-1 state_piecewise(4)_pw_t01_CC_bin_y(1)
-1 state_piecewise(4)_pw_t01_CC_bin_y(2)
<= 0

c_u_state_piecewise(4)_pw_t01_CC_constraint4(3)_:
+1 state_piecewise(4)_pw_t01_CC_lambda(3)
-1 state_piecewise(4)_pw_t01_CC_bin_y(2)
<= 0

c_e_state_piecewise(4)_pw_t01_CC_constraint5_:
+1 state_piecewise(4)_pw_t01_CC_bin_y(1)
+1 state_piecewise(4)_pw_t01_CC_bin_y(2)
= 1

c_e_state_piecewise(4)_pw_t02_CC_constraint1_:
+1 state(reach2_discharge_upstream_t02)
-2.6 state_piecewise(4)_pw_t02_CC_lambda(1)
-7.8 state_piecewise(4)_pw_t02_CC_lambda(2)
-10.8 state_piecewise(4)_pw_t02_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t02_CC_constraint2_:
+1 state(reach2_level_upstream_t02)
-1.85 state_piecewise(4)_pw_t02_CC_lambda(1)
-1.7850000000000001 state_piecewise(4)_pw_t02_CC_lambda(2)
-1.72 state_piecewise(4)_pw_t02_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t02_CC_constraint3_:
+1 state_piecewise(4)_pw_t02_CC_lambda(1)
+1 state_piecewise(4)_pw_t02_CC_lambda(2)
+1 state_piecewise(4)_pw_t02_CC_lambda(3)
= 1

c_u_state_piecewise(4)_pw_t02_CC_constraint4(1)_:
+1 state_piecewise(4)_pw_t02_CC_lambda(1)
-1 state_piecewise(4)_pw_t02_CC_bin_y(1)
<= 0

c_u_state_piecewise(4)_pw_t02_CC_constraint4(2)_:
+1 state_piecewise(4)_pw_t02_CC_lambda(2)
-1 state_piecewise(4)_pw_t02_CC_bin_y(1)
-1 state_piecewise(4)_pw_t02_CC_bin_y(2)
<= 0

c_u_state_piecewise(4)_pw_t02_CC_constraint4(3)_:
+1 state_piecewise(4)_pw_t02_CC_lambda(3)
-1 state_piecewise(4)_pw_t02_CC_bin_y(2)
<= 0

c_e_state_piecewise(4)_pw_t02_CC_constraint5_:
+1 state_piecewise(4)_pw_t02_CC_bin_y(1)
+1 state_piecewise(4)_pw_t02_CC_bin_y(2)
= 1

c_e_state_piecewise(4)_pw_t03_CC_constraint1_:
+1 state(reach2_discharge_upstream_t03)
-2.6 state_piecewise(4)_pw_t03_CC_lambda(1)
-7.8 state_piecewise(4)_pw_t03_CC_lambda(2)
-10.8 state_piecewise(4)_pw_t03_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t03_CC_constraint2_:
+1 state(reach2_level_upstream_t03)
-1.85 state_piecewise(4)_pw_t03_CC_lambda(1)
-1.7850000000000001 state_piecewise(4)_pw_t03_CC_lambda(2)
-1.72 state_piecewise(4)_pw_t03_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t03_CC_constraint3_:
+1 state_piecewise(4)_pw_t03_CC_lambda(1)
+1 state_piecewise(4)_pw_t03_CC_lambda(2)
+1 state_piecewise(4)_pw_t03_CC_lambda(3)
= 1

c_u_state_piecewise(4)_pw_t03_CC_constraint4(1)_:
+1 state_piecewise(4)_pw_t03_CC_lambda(1)
-1 state_piecewise(4)_pw_t03_CC_bin_y(1)
<= 0

c_u_state_piecewise(4)_pw_t03_CC_constraint4(2)_:
+1 state_piecewise(4)_pw_t03_CC_lambda(2)
-1 state_piecewise(4)_pw_t03_CC_bin_y(1)
-1 state_piecewise(4)_pw_t03_CC_bin_y(2)
<= 0

c_u_state_piecewise(4)_pw_t03_CC_constraint4(3)_:
+1 state_piecewise(4)_pw_t03_CC_lambda(3)
-1 state_piecewise(4)_pw_t03_CC_bin_y(2)
<= 0

c_e_state_piecewise(4)_pw_t03_CC_constraint5_:
+1 state_piecewise(4)_pw_t03_CC_bin_y(1)
+1 state_piecewise(4)_pw_t03_CC_bin_y(2)
= 1

c_e_state_piecewise(4)_pw_t04_CC_constraint1_:
+1 state(reach2_discharge_upstream_t04)
-2.6 state_piecewise(4)_pw_t04_CC_lambda(1)
-7.8 state_piecewise(4)_pw_t04_CC_lambda(2)
-10.8 state_piecewise(4)_pw_t04_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t04_CC_constraint2_:
+1 state(reach2_level_upstream_t04)
-1.85 state_piecewise(4)_pw_t04_CC_lambda(1)
-1.7850000000000001 state_piecewise(4)_pw_t04_CC_lambda(2)
-1.72 state_piecewise(4)_pw_t04_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t04_CC_constraint3_:
+1 state_piecewise(4)_pw_t04_CC_lambda(1)
+1 state_piecewise(4)_pw_t04_CC_lambda(2)
+1 state_piecewise(4)_pw_t04_CC_lambda(3)
= 1

c_u_state_piecewise(4)_pw_t04_CC_constraint4(1)_:
+1 state_piecewise(4)_pw_t04_CC_lambda(1)
-1 state_piecewise(4)_pw_t04_CC_bin_y(1)
<= 0

c_u_state_piecewise(4)_pw_t04_CC_constraint4(2)_:
+1 state_piecewise(4)_pw_t04_CC_lambda(2)
-1 state_piecewise(4)_pw_t04_CC_bin_y(1)
-1 state_piecewise(4)_pw_t04_CC_bin_y(2)
<= 0

c_u_state_piecewise(4)_pw_t04_CC_constraint4(3)_:
+1 state_piecewise(4)_pw_t04_CC_lambda(3)
-1 state_piecewise(4)_pw_t04_CC_bin_y(2)
<= 0

c_e_state_piecewise(4)_pw_t04_CC_constraint5_:
+1 state_piecewise(4)_pw_t04_CC_bin_y(1)
+1 state_piecewise(4)_pw_t04_CC_bin_y(2)
= 1

c_e_state_piecewise(4)_pw_t05_CC_constraint1_:
+1 state(reach2_discharge_upstream_t05)
-2.6 state_piecewise(4)_pw_t05_CC_lambda(1)
-7.8 state_piecewise(4)_pw_t05_CC_lambda(2)
-10.8 state_piecewise(4)_pw_t05_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t05_CC_constraint2_:
+1 state(reach2_level_upstream_t05)
-1.85 state_piecewise(4)_pw_t05_CC_lambda(1)
-1.7850000000000001 state_piecewise(4)_pw_t05_CC_lambda(2)
-1.72 state_piecewise(4)_pw_t05_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t05_CC_constraint3_:
+1 state_piecewise(4)_pw_t05_CC_lambda(1)
+1 state_piecewise(4)_pw_t05_CC_lambda(2)
+1 state_piecewise(4)_pw_t05_CC_lambda(3)
= 1

c_u_state_piecewise(4)_pw_t05_CC_constraint4(1)_:
+1 state_piecewise(4)_pw_t05_CC_lambda(1)
-1 state_piecewise(4)_pw_t05_CC_bin_y(1)
<= 0

c_u_state_piecewise(4)_pw_t05_CC_constraint4(2)_:
+1 state_piecewise(4)_pw_t05_CC_lambda(2)
-1 state_piecewise(4)_pw_t05_CC_bin_y(1)
-1 state_piecewise(4)_pw_t05_CC_bin_y(2)
<= 0

c_u_state_piecewise(4)_pw_t05_CC_constraint4(3)_:
+1 state_piecewise(4)_pw_t05_CC_lambda(3)
-1 state_piecewise(4)_pw_t05_CC_bin_y(2)
<= 0

c_e_state_piecewise(4)_pw_t05_CC_constraint5_:
+1 state_piecewise(4)_pw_t05_CC_bin_y(1)
+1 state_piecewise(4)_pw_t05_CC_bin_y(2)
= 1

c_e_state_piecewise(4)_pw_t06_CC_constraint1_:
+1 state(reach2_discharge_upstream_t06)
-2.6 state_piecewise(4)_pw_t06_CC_lambda(1)
-7.8 state_piecewise(4)_pw_t06_CC_lambda(2)
-10.8 state_piecewise(4)_pw_t06_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t06_CC_constraint2_:
+1 state(reach2_level_upstream_t06)
-1.85 state_piecewise(4)_pw_t06_CC_lambda(1)
-1.7850000000000001 state_piecewise(4)_pw_t06_CC_lambda(2)
-1.72 state_piecewise(4)_pw_t06_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t06_CC_constraint3_:
+1 state_piecewise(4)_pw_t06_CC_lambda(1)
+1 state_piecewise(4)_pw_t06_CC_lambda(2)
+1 state_piecewise(4)_pw_t06_CC_lambda(3)
= 1

c_u_state_piecewise(4)_pw_t06_CC_constraint4(1)_:
+1 state_piecewise(4)_pw_t06_CC_lambda(1)
-1 state_piecewise(4)_pw_t06_CC_bin_y(1)
<= 0

c_u_state_piecewise(4)_pw_t06_CC_constraint4(2)_:
+1 state_piecewise(4)_pw_t06_CC_lambda(2)
-1 state_piecewise(4)_pw_t06_CC_bin_y(1)
-1 state_piecewise(4)_pw_t06_CC_bin_y(2)
<= 0

c_u_state_piecewise(4)_pw_t06_CC_constraint4(3)_:
+1 state_piecewise(4)_pw_t06_CC_lambda(3)
-1 state_piecewise(4)_pw_t06_CC_bin_y(2)
<= 0

c_e_state_piecewise(4)_pw_t06_CC_constraint5_:
+1 state_piecewise(4)_pw_t06_CC_bin_y(1)
+1 state_piecewise(4)_pw_t06_CC_bin_y(2)
= 1

c_e_state_piecewise(4)_pw_t07_CC_constraint1_:
+1 state(reach2_discharge_upstream_t07)
-2.6 state_piecewise(4)_pw_t07_CC_lambda(1)
-7.8 state_piecewise(4)_pw_t07_CC_lambda(2)
-10.8 state_piecewise(4)_pw_t07_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t07_CC_constraint2_:
+1 state(reach2_level_upstream_t07)
-1.85 state_piecewise(4)_pw_t07_CC_lambda(1)
-1.7850000000000001 state_piecewise(4)_pw_t07_CC_lambda(2)
-1.72 state_piecewise(4)_pw_t07_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t07_CC_constraint3_:
+1 state_piecewise(4)_pw_t07_CC_lambda(1)
+1 state_piecewise(4)_pw_t07_CC_lambda(2)
+1 state_piecewise(4)_pw_t07_CC_lambda(3)
= 1

c_u_state_piecewise(4)_pw_t07_CC_constraint4(1)_:
+1 state_piecewise(4)_pw_t07_CC_lambda(1)
-1 state_piecewise(4)_pw_t07_CC_bin_y(1)
<= 0

c_u_state_piecewise(4)_pw_t07_CC_constraint4(2)_:
+1 state_piecewise(4)_pw_t07_CC_lambda(2)
-1 state_piecewise(4)_pw_t07_CC_bin_y(1)
-1 state_piecewise(4)_pw_t07_CC_bin_y(2)
<= 0

c_u_state_piecewise(4)_pw_t07_CC_constraint4(3)_:
+1 state_piecewise(4)_pw_t07_CC_lambda(3)
-1 state_piecewise(4)_pw_t07_CC_bin_y(2)
<= 0

c_e_state_piecewise(4)_pw_t07_CC_constraint5_:
+1 state_piecewise(4)_pw_t07_CC_bin_y(1)
+1 state_piecewise(4)_pw_t07_CC_bin_y(2)
= 1

c_e_state_piecewise(4)_pw_t08_CC_constraint1_:
+1 state(reach2_discharge_upstream_t08)
-2.6 state_piecewise(4)_pw_t08_CC_lambda(1)
-7.8 state_piecewise(4)_pw_t08_CC_lambda(2)
-10.8 state_piecewise(4)_pw_t08_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t08_CC_constraint2_:
+1 state(reach2_level_upstream_t08)
-1.85 state_piecewise(4)_pw_t08_CC_lambda(1)
-1.7850000000000001 state_piecewise(4)_pw_t08_CC_lambda(2)
-1.72 state_piecewise(4)_pw_t08_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t08_CC_constraint3_:
+1 state_piecewise(4)_pw_t08_CC_lambda(1)
+1 state_piecewise(4)_pw_t08_CC_lambda(2)
+1 state_piecewise(4)_pw_t08_CC_lambda(3)
= 1

c_u_state_piecewise(4)_pw_t08_CC_constraint4(1)_:
+1 state_piecewise(4)_pw_t08_CC_lambda(1)
-1 state_piecewise(4)_pw_t08_CC_bin_y(1)
<= 0

c_u_state_piecewise(4)_pw_t08_CC_constraint4(2)_:
+1 state_piecewise(4)_pw_t08_CC_lambda(2)
-1 state_piecewise(4)_pw_t08_CC_bin_y(1)
-1 state_piecewise(4)_pw_t08_CC_bin_y(2)
<= 0

c_u_state_piecewise(4)_pw_t08_CC_constraint4(3)_:
+1 state_piecewise(4)_pw_t08_CC_lambda(3)
-1 state_piecewise(4)_pw_t08_CC_bin_y(2)
<= 0

c_e_state_piecewise(4)_pw_t08_CC_constraint5_:
+1 state_piecewise(4)_pw_t08_CC_bin_y(1)
+1 state_piecewise(4)_pw_t08_CC_bin_y(2)
= 1

c_e_state_piecewise(4)_pw_t09_CC_constraint1_:
+1 state(reach2_discharge_upstream_t09)
-2.6 state_piecewise(4)_pw_t09_CC_lambda(1)
-7.8 state_piecewise(4)_pw_t09_CC_lambda(2)
-10.8 state_piecewise(4)_pw_t09_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t09_CC_constraint2_:
+1 state(reach2_level_upstream_t09)
-1.85 state_piecewise(4)_pw_t09_CC_lambda(1)
-1.7850000000000001 state_piecewise(4)_pw_t09_CC_lambda(2)
-1.72 state_piecewise(4)_pw_t09_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t09_CC_constraint3_:
+1 state_piecewise(4)_pw_t09_CC_lambda(1)
+1 state_piecewise(4)_pw_t09_CC_lambda(2)
+1 state_piecewise(4)_pw_t09_CC_lambda(3)
= 1

c_u_state_piecewise(4)_pw_t09_CC_constraint4(1)_:
+1 state_piecewise(4)_pw_t09_CC_lambda(1)
-1 state_piecewise(4)_pw_t09_CC_bin_y(1)
<= 0

c_u_state_piecewise(4)_pw_t09_CC_constraint4(2)_:
+1 state_piecewise(4)_pw_t09_CC_lambda(2)
-1 state_piecewise(4)_pw_t09_CC_bin_y(1)
-1 state_piecewise(4)_pw_t09_CC_bin_y(2)
<= 0

c_u_state_piecewise(4)_pw_t09_CC_constraint4(3)_:
+1 state_piecewise(4)_pw_t09_CC_lambda(3)
-1 state_piecewise(4)_pw_t09_CC_bin_y(2)
<= 0

c_e_state_piecewise(4)_pw_t09_CC_constraint5_:
+1 state_piecewise(4)_pw_t09_CC_bin_y(1)
+1 state_piecewise(4)_pw_t09_CC_bin_y(2)
= 1

c_e_state_piecewise(4)_pw_t10_CC_constraint1_:
+1 state(reach2_discharge_upstream_t10)
-2.6 state_piecewise(4)_pw_t10_CC_lambda(1)
-7.8 state_piecewise(4)_pw_t10_CC_lambda(2)
-10.8 state_piecewise(4)_pw_t10_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t10_CC_constraint2_:
+1 state(reach2_level_upstream_t10)
-1.85 state_piecewise(4)_pw_t10_CC_lambda(1)
-1.7850000000000001 state_piecewise(4)_pw_t10_CC_lambda(2)
-1.72 state_piecewise(4)_pw_t10_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t10_CC_constraint3_:
+1 state_piecewise(4)_pw_t10_CC_lambda(1)
+1 state_piecewise(4)_pw_t10_CC_lambda(2)
+1 state_piecewise(4)_pw_t10_CC_lambda(3)
= 1

c_u_state_piecewise(4)_pw_t10_CC_constraint4(1)_:
+1 state_piecewise(4)_pw_t10_CC_lambda(1)
-1 state_piecewise(4)_pw_t10_CC_bin_y(1)
<= 0

c_u_state_piecewise(4)_pw_t10_CC_constraint4(2)_:
+1 state_piecewise(4)_pw_t10_CC_lambda(2)
-1 state_piecewise(4)_pw_t10_CC_bin_y(1)
-1 state_piecewise(4)_pw_t10_CC_bin_y(2)
<= 0

c_u_state_piecewise(4)_pw_t10_CC_constraint4(3)_:
+1 state_piecewise(4)_pw_t10_CC_lambda(3)
-1 state_piecewise(4)_pw_t10_CC_bin_y(2)
<= 0

c_e_state_piecewise(4)_pw_t10_CC_constraint5_:
+1 state_piecewise(4)_pw_t10_CC_bin_y(1)
+1 state_piecewise(4)_pw_t10_CC_bin_y(2)
= 1

c_e_state_piecewise(4)_pw_t11_CC_constraint1_:
+1 state(reach2_discharge_upstream_t11)
-2.6 state_piecewise(4)_pw_t11_CC_lambda(1)
-7.8 state_piecewise(4)_pw_t11_CC_lambda(2)
-10.8 state_piecewise(4)_pw_t11_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t11_CC_constraint2_:
+1 state(reach2_level_upstream_t11)
-1.85 state_piecewise(4)_pw_t11_CC_lambda(1)
-1.7850000000000001 state_piecewise(4)_pw_t11_CC_lambda(2)
-1.72 state_piecewise(4)_pw_t11_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t11_CC_constraint3_:
+1 state_piecewise(4)_pw_t11_CC_lambda(1)
+1 state_piecewise(4)_pw_t11_CC_lambda(2)
+1 state_piecewise(4)_pw_t11_CC_lambda(3)
= 1

c_u_state_piecewise(4)_pw_t11_CC_constraint4(1)_:
+1 state_piecewise(4)_pw_t11_CC_lambda(1)
-1 state_piecewise(4)_pw_t11_CC_bin_y(1)
<= 0

c_u_state_piecewise(4)_pw_t11_CC_constraint4(2)_:
+1 state_piecewise(4)_pw_t11_CC_lambda(2)
-1 state_piecewise(4)_pw_t11_CC_bin_y(1)
-1 state_piecewise(4)_pw_t11_CC_bin_y(2)
<= 0

c_u_state_piecewise(4)_pw_t11_CC_constraint4(3)_:
+1 state_piecewise(4)_pw_t11_CC_lambda(3)
-1 state_piecewise(4)_pw_t11_CC_bin_y(2)
<= 0

c_e_state_piecewise(4)_pw_t11_CC_constraint5_:
+1 state_piecewise(4)_pw_t11_CC_bin_y(1)
+1 state_piecewise(4)_pw_t11_CC_bin_y(2)
= 1

c_e_state_piecewise(4)_pw_t12_CC_constraint1_:
+1 state(reach2_discharge_upstream_t12)
-2.6 state_piecewise(4)_pw_t12_CC_lambda(1)
-7.8 state_piecewise(4)_pw_t12_CC_lambda(2)
-10.8 state_piecewise(4)_pw_t12_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t12_CC_constraint2_:
+1 state(reach2_level_upstream_t12)
-1.85 state_piecewise(4)_pw_t12_CC_lambda(1)
-1.7850000000000001 state_piecewise(4)_pw_t12_CC_lambda(2)
-1.72 state_piecewise(4)_pw_t12_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t12_CC_constraint3_:
+1 state_piecewise(4)_pw_t12_CC_lambda(1)
+1 state_piecewise(4)_pw_t12_CC_lambda(2)
+1 state_piecewise(4)_pw_t12_CC_lambda(3)
= 1

c_u_state_piecewise(4)_pw_t12_CC_constraint4(1)_:
+1 state_piecewise(4)_pw_t12_CC_lambda(1)
-1 state_piecewise(4)_pw_t12_CC_bin_y(1)
<= 0

c_u_state_piecewise(4)_pw_t12_CC_constraint4(2)_:
+1 state_piecewise(4)_pw_t12_CC_lambda(2)
-1 state_piecewise(4)_pw_t12_CC_bin_y(1)
-1 state_piecewise(4)_pw_t12_CC_bin_y(2)
<= 0

c_u_state_piecewise(4)_pw_t12_CC_constraint4(3)_:
+1 state_piecewise(4)_pw_t12_CC_lambda(3)
-1 state_piecewise(4)_pw_t12_CC_bin_y(2)
<= 0

c_e_state_piecewise(4)_pw_t12_CC_constraint5_:
+1 state_piecewise(4)_pw_t12_CC_bin_y(1)
+1 state_piecewise(4)_pw_t12_CC_bin_y(2)
= 1

c_e_state_piecewise(4)_pw_t13_CC_constraint1_:
+1 state(reach2_discharge_upstream_t13)
-2.6 state_piecewise(4)_pw_t13_CC_lambda(1)
-7.8 state_piecewise(4)_pw_t13_CC_lambda(2)
-10.8 state_piecewise(4)_pw_t13_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t13_CC_constraint2_:
+1 state(reach2_level_upstream_t13)
-1.85 state_piecewise(4)_pw_t13_CC_lambda(1)
-1.7850000000000001 state_piecewise(4)_pw_t13_CC_lambda(2)
-1.72 state_piecewise(4)_pw_t13_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t13_CC_constraint3_:
+1 state_piecewise(4)_pw_t13_CC_lambda(1)
+1 state_piecewise(4)_pw_t13_CC_lambda(2)
+1 state_piecewise(4)_pw_t13_CC_lambda(3)
= 1

c_u_state_piecewise(4)_pw_t13_CC_constraint4(1)_:
+1 state_piecewise(4)_pw_t13_CC_lambda(1)
-1 state_piecewise(4)_pw_t13_CC_bin_y(1)
<= 0

c_u_state_piecewise(4)_pw_t13_CC_constraint4(2)_:
+1 state_piecewise(4)_pw_t13_CC_lambda(2)
-1 state_piecewise(4)_pw_t13_CC_bin_y(1)
-1 state_piecewise(4)_pw_t13_CC_bin_y(2)
<= 0

c_u_state_piecewise(4)_pw_t13_CC_constraint4(3)_:
+1 state_piecewise(4)_pw_t13_CC_lambda(3)
-1 state_piecewise(4)_pw_t13_CC_bin_y(2)
<= 0

c_e_state_piecewise(4)_pw_t13_CC_constraint5_:
+1 state_piecewise(4)_pw_t13_CC_bin_y(1)
+1 state_piecewise(4)_pw_t13_CC_bin_y(2)
= 1

c_e_state_piecewise(4)_pw_t14_CC_constraint1_:
+1 state(reach2_discharge_upstream_t14)
-2.6 state_piecewise(4)_pw_t14_CC_lambda(1)
-7.8 state_piecewise(4)_pw_t14_CC_lambda(2)
-10.8 state_piecewise(4)_pw_t14_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t14_CC_constraint2_:
+1 state(reach2_level_upstream_t14)
-1.85 state_piecewise(4)_pw_t14_CC_lambda(1)
-1.7850000000000001 state_piecewise(4)_pw_t14_CC_lambda(2)
-1.72 state_piecewise(4)_pw_t14_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t14_CC_constraint3_:
+1 state_piecewise(4)_pw_t14_CC_lambda(1)
+1 state_piecewise(4)_pw_t14_CC_lambda(2)
+1 state_piecewise(4)_pw_t14_CC_lambda(3)
= 1

c_u_state_piecewise(4)_pw_t14_CC_constraint4(1)_:
+1 state_piecewise(4)_pw_t14_CC_lambda(1)
-1 state_piecewise(4)_pw_t14_CC_bin_y(1)
<= 0

c_u_state_piecewise(4)_pw_t14_CC_constraint4(2)_:
+1 state_piecewise(4)_pw_t14_CC_lambda(2)
-1 state_piecewise(4)_pw_t14_CC_bin_y(1)
-1 state_piecewise(4)_pw_t14_CC_bin_y(2)
<= 0

c_u_state_piecewise(4)_pw_t14_CC_constraint4(3)_:
+1 state_piecewise(4)_pw_t14_CC_lambda(3)
-1 state_piecewise(4)_pw_t14_CC_bin_y(2)
<= 0

c_e_state_piecewise(4)_pw_t14_CC_constraint5_:
+1 state_piecewise(4)_pw_t14_CC_bin_y(1)
+1 state_piecewise(4)_pw_t14_CC_bin_y(2)
= 1

c_e_state_piecewise(4)_pw_t15_CC_constraint1_:
+1 state(reach2_discharge_upstream_t15)
-2.6 state_piecewise(4)_pw_t15_CC_lambda(1)
-7.8 state_piecewise(4)_pw_t15_CC_lambda(2)
-10.8 state_piecewise(4)_pw_t15_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t15_CC_constraint2_:
+1 state(reach2_level_upstream_t15)
-1.85 state_piecewise(4)_pw_t15_CC_lambda(1)
-1.7850000000000001 state_piecewise(4)_pw_t15_CC_lambda(2)
-1.72 state_piecewise(4)_pw_t15_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t15_CC_constraint3_:
+1 state_piecewise(4)_pw_t15_CC_lambda(1)
+1 state_piecewise(4)_pw_t15_CC_lambda(2)
+1 state_piecewise(4)_pw_t15_CC_lambda(3)
= 1

c_u_state_piecewise(4)_pw_t15_CC_constraint4(1)_:
+1 state_piecewise(4)_pw_t15_CC_lambda(1)
-1 state_piecewise(4)_pw_t15_CC_bin_y(1)
<= 0

c_u_state_piecewise(4)_pw_t15_CC_constraint4(2)_:
+1 state_piecewise(4)_pw_t15_CC_lambda(2)
-1 state_piecewise(4)_pw_t15_CC_bin_y(1)
-1 state_piecewise(4)_pw_t15_CC_bin_y(2)
<= 0

c_u_state_piecewise(4)_pw_t15_CC_constraint4(3)_:
+1 state_piecewise(4)_pw_t15_CC_lambda(3)
-1 state_piecewise(4)_pw_t15_CC_bin_y(2)
<= 0

c_e_state_piecewise(4)_pw_t15_CC_constraint5_:
+1 state_piecewise(4)_pw_t15_CC_bin_y(1)
+1 state_piecewise(4)_pw_t15_CC_bin_y(2)
= 1

c_e_state_piecewise(4)_pw_t16_CC_constraint1_:
+1 state(reach2_discharge_upstream_t16)
-2.6 state_piecewise(4)_pw_t16_CC_lambda(1)
-7.8 state_piecewise(4)_pw_t16_CC_lambda(2)
-10.8 state_piecewise(4)_pw_t16_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t16_CC_constraint2_:
+1 state(reach2_level_upstream_t16)
-1.85 state_piecewise(4)_pw_t16_CC_lambda(1)
-1.7850000000000001 state_piecewise(4)_pw_t16_CC_lambda(2)
-1.72 state_piecewise(4)_pw_t16_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t16_CC_constraint3_:
+1 state_piecewise(4)_pw_t16_CC_lambda(1)
+1 state_piecewise(4)_pw_t16_CC_lambda(2)
+1 state_piecewise(4)_pw_t16_CC_lambda(3)
= 1

c_u_state_piecewise(4)_pw_t16_CC_constraint4(1)_:
+1 state_piecewise(4)_pw_t16_CC_lambda(1)
-1 state_piecewise(4)_pw_t16_CC_bin_y(1)
<= 0

c_u_state_piecewise(4)_pw_t16_CC_constraint4(2)_:
+1 state_piecewise(4)_pw_t16_CC_lambda(2)
-1 state_piecewise(4)_pw_t16_CC_bin_y(1)
-1 state_piecewise(4)_pw_t16_CC_bin_y(2)
<= 0

c_u_state_piecewise(4)_pw_t16_CC_constraint4(3)_:
+1 state_piecewise(4)_pw_t16_CC_lambda(3)
-1 state_piecewise(4)_pw_t16_CC_bin_y(2)
<= 0

c_e_state_piecewise(4)_pw_t16_CC_constraint5_:
+1 state_piecewise(4)_pw_t16_CC_bin_y(1)
+1 state_piecewise(4)_pw_t16_CC_bin_y(2)
= 1

c_e_state_piecewise(4)_pw_t17_CC_constraint1_:
+1 state(reach2_discharge_upstream_t17)
-2.6 state_piecewise(4)_pw_t17_CC_lambda(1)
-7.8 state_piecewise(4)_pw_t17_CC_lambda(2)
-10.8 state_piecewise(4)_pw_t17_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t17_CC_constraint2_:
+1 state(reach2_level_upstream_t17)
-1.85 state_piecewise(4)_pw_t17_CC_lambda(1)
-1.7850000000000001 state_piecewise(4)_pw_t17_CC_lambda(2)
-1.72 state_piecewise(4)_pw_t17_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t17_CC_constraint3_:
+1 state_piecewise(4)_pw_t17_CC_lambda(1)
+1 state_piecewise(4)_pw_t17_CC_lambda(2)
+1 state_piecewise(4)_pw_t17_CC_lambda(3)
= 1

c_u_state_piecewise(4)_pw_t17_CC_constraint4(1)_:
+1 state_piecewise(4)_pw_t17_CC_lambda(1)
-1 state_piecewise(4)_pw_t17_CC_bin_y(1)
<= 0

c_u_state_piecewise(4)_pw_t17_CC_constraint4(2)_:
+1 state_piecewise(4)_pw_t17_CC_lambda(2)
-1 state_piecewise(4)_pw_t17_CC_bin_y(1)
-1 state_piecewise(4)_pw_t17_CC_bin_y(2)
<= 0

c_u_state_piecewise(4)_pw_t17_CC_constraint4(3)_:
+1 state_piecewise(4)_pw_t17_CC_lambda(3)
-1 state_piecewise(4)_pw_t17_CC_bin_y(2)
<= 0

c_e_state_piecewise(4)_pw_t17_CC_constraint5_:
+1 state_piecewise(4)_pw_t17_CC_bin_y(1)
+1 state_piecewise(4)_pw_t17_CC_bin_y(2)
= 1

c_e_state_piecewise(4)_pw_t18_CC_constraint1_:
+1 state(reach2_discharge_upstream_t18)
-2.6 state_piecewise(4)_pw_t18_CC_lambda(1)
-7.8 state_piecewise(4)_pw_t18_CC_lambda(2)
-10.8 state_piecewise(4)_pw_t18_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t18_CC_constraint2_:
+1 state(reach2_level_upstream_t18)
-1.85 state_piecewise(4)_pw_t18_CC_lambda(1)
-1.7850000000000001 state_piecewise(4)_pw_t18_CC_lambda(2)
-1.72 state_piecewise(4)_pw_t18_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t18_CC_constraint3_:
+1 state_piecewise(4)_pw_t18_CC_lambda(1)
+1 state_piecewise(4)_pw_t18_CC_lambda(2)
+1 state_piecewise(4)_pw_t18_CC_lambda(3)
= 1

c_u_state_piecewise(4)_pw_t18_CC_constraint4(1)_:
+1 state_piecewise(4)_pw_t18_CC_lambda(1)
-1 state_piecewise(4)_pw_t18_CC_bin_y(1)
<= 0

c_u_state_piecewise(4)_pw_t18_CC_constraint4(2)_:
+1 state_piecewise(4)_pw_t18_CC_lambda(2)
-1 state_piecewise(4)_pw_t18_CC_bin_y(1)
-1 state_piecewise(4)_pw_t18_CC_bin_y(2)
<= 0

c_u_state_piecewise(4)_pw_t18_CC_constraint4(3)_:
+1 state_piecewise(4)_pw_t18_CC_lambda(3)
-1 state_piecewise(4)_pw_t18_CC_bin_y(2)
<= 0

c_e_state_piecewise(4)_pw_t18_CC_constraint5_:
+1 state_piecewise(4)_pw_t18_CC_bin_y(1)
+1 state_piecewise(4)_pw_t18_CC_bin_y(2)
= 1

c_e_state_piecewise(4)_pw_t19_CC_constraint1_:
+1 state(reach2_discharge_upstream_t19)
-2.6 state_piecewise(4)_pw_t19_CC_lambda(1)
-7.8 state_piecewise(4)_pw_t19_CC_lambda(2)
-10.8 state_piecewise(4)_pw_t19_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t19_CC_constraint2_:
+1 state(reach2_level_upstream_t19)
-1.85 state_piecewise(4)_pw_t19_CC_lambda(1)
-1.7850000000000001 state_piecewise(4)_pw_t19_CC_lambda(2)
-1.72 state_piecewise(4)_pw_t19_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t19_CC_constraint3_:
+1 state_piecewise(4)_pw_t19_CC_lambda(1)
+1 state_piecewise(4)_pw_t19_CC_lambda(2)
+1 state_piecewise(4)_pw_t19_CC_lambda(3)
= 1

c_u_state_piecewise(4)_pw_t19_CC_constraint4(1)_:
+1 state_piecewise(4)_pw_t19_CC_lambda(1)
-1 state_piecewise(4)_pw_t19_CC_bin_y(1)
<= 0

c_u_state_piecewise(4)_pw_t19_CC_constraint4(2)_:
+1 state_piecewise(4)_pw_t19_CC_lambda(2)
-1 state_piecewise(4)_pw_t19_CC_bin_y(1)
-1 state_piecewise(4)_pw_t19_CC_bin_y(2)
<= 0

c_u_state_piecewise(4)_pw_t19_CC_constraint4(3)_:
+1 state_piecewise(4)_pw_t19_CC_lambda(3)
-1 state_piecewise(4)_pw_t19_CC_bin_y(2)
<= 0

c_e_state_piecewise(4)_pw_t19_CC_constraint5_:
+1 state_piecewise(4)_pw_t19_CC_bin_y(1)
+1 state_piecewise(4)_pw_t19_CC_bin_y(2)
= 1

c_e_state_piecewise(4)_pw_t20_CC_constraint1_:
+1 state(reach2_discharge_upstream_t20)
-2.6 state_piecewise(4)_pw_t20_CC_lambda(1)
-7.8 state_piecewise(4)_pw_t20_CC_lambda(2)
-10.8 state_piecewise(4)_pw_t20_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t20_CC_constraint2_:
+1 state(reach2_level_upstream_t20)
-1.85 state_piecewise(4)_pw_t20_CC_lambda(1)
-1.7850000000000001 state_piecewise(4)_pw_t20_CC_lambda(2)
-1.72 state_piecewise(4)_pw_t20_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t20_CC_constraint3_:
+1 state_piecewise(4)_pw_t20_CC_lambda(1)
+1 state_piecewise(4)_pw_t20_CC_lambda(2)
+1 state_piecewise(4)_pw_t20_CC_lambda(3)
= 1

c_u_state_piecewise(4)_pw_t20_CC_constraint4(1)_:
+1 state_piecewise(4)_pw_t20_CC_lambda(1)
-1 state_piecewise(4)_pw_t20_CC_bin_y(1)
<= 0

c_u_state_piecewise(4)_pw_t20_CC_constraint4(2)_:
+1 state_piecewise(4)_pw_t20_CC_lambda(2)
-1 state_piecewise(4)_pw_t20_CC_bin_y(1)
-1 state_piecewise(4)_pw_t20_CC_bin_y(2)
<= 0

c_u_state_piecewise(4)_pw_t20_CC_constraint4(3)_:
+1 state_piecewise(4)_pw_t20_CC_lambda(3)
-1 state_piecewise(4)_pw_t20_CC_bin_y(2)
<= 0

c_e_state_piecewise(4)_pw_t20_CC_constraint5_:
+1 state_piecewise(4)_pw_t20_CC_bin_y(1)
+1 state_piecewise(4)_pw_t20_CC_bin_y(2)
= 1

c_e_state_piecewise(4)_pw_t21_CC_constraint1_:
+1 state(reach2_discharge_upstream_t21)
-2.6 state_piecewise(4)_pw_t21_CC_lambda(1)
-7.8 state_piecewise(4)_pw_t21_CC_lambda(2)
-10.8 state_piecewise(4)_pw_t21_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t21_CC_constraint2_:
+1 state(reach2_level_upstream_t21)
-1.85 state_piecewise(4)_pw_t21_CC_lambda(1)
-1.7850000000000001 state_piecewise(4)_pw_t21_CC_lambda(2)
-1.72 state_piecewise(4)_pw_t21_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t21_CC_constraint3_:
+1 state_piecewise(4)_pw_t21_CC_lambda(1)
+1 state_piecewise(4)_pw_t21_CC_lambda(2)
+1 state_piecewise(4)_pw_t21_CC_lambda(3)
= 1

c_u_state_piecewise(4)_pw_t21_CC_constraint4(1)_:
+1 state_piecewise(4)_pw_t21_CC_lambda(1)
-1 state_piecewise(4)_pw_t21_CC_bin_y(1)
<= 0

c_u_state_piecewise(4)_pw_t21_CC_constraint4(2)_:
+1 state_piecewise(4)_pw_t21_CC_lambda(2)
-1 state_piecewise(4)_pw_t21_CC_bin_y(1)
-1 state_piecewise(4)_pw_t21_CC_bin_y(2)
<= 0

c_u_state_piecewise(4)_pw_t21_CC_constraint4(3)_:
+1 state_piecewise(4)_pw_t21_CC_lambda(3)
-1 state_piecewise(4)_pw_t21_CC_bin_y(2)
<= 0

c_e_state_piecewise(4)_pw_t21_CC_constraint5_:
+1 state_piecewise(4)_pw_t21_CC_bin_y(1)
+1 state_piecewise(4)_pw_t21_CC_bin_y(2)
= 1

c_e_state_piecewise(4)_pw_t22_CC_constraint1_:
+1 state(reach2_discharge_upstream_t22)
-2.6 state_piecewise(4)_pw_t22_CC_lambda(1)
-7.8 state_piecewise(4)_pw_t22_CC_lambda(2)
-10.8 state_piecewise(4)_pw_t22_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t22_CC_constraint2_:
+1 state(reach2_level_upstream_t22)
-1.85 state_piecewise(4)_pw_t22_CC_lambda(1)
-1.7850000000000001 state_piecewise(4)_pw_t22_CC_lambda(2)
-1.72 state_piecewise(4)_pw_t22_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t22_CC_constraint3_:
+1 state_piecewise(4)_pw_t22_CC_lambda(1)
+1 state_piecewise(4)_pw_t22_CC_lambda(2)
+1 state_piecewise(4)_pw_t22_CC_lambda(3)
= 1

c_u_state_piecewise(4)_pw_t22_CC_constraint4(1)_:
+1 state_piecewise(4)_pw_t22_CC_lambda(1)
-1 state_piecewise(4)_pw_t22_CC_bin_y(1)
<= 0

c_u_state_piecewise(4)_pw_t22_CC_constraint4(2)_:
+1 state_piecewise(4)_pw_t22_CC_lambda(2)
-1 state_piecewise(4)_pw_t22_CC_bin_y(1)
-1 state_piecewise(4)_pw_t22_CC_bin_y(2)
<= 0

c_u_state_piecewise(4)_pw_t22_CC_constraint4(3)_:
+1 state_piecewise(4)_pw_t22_CC_lambda(3)
-1 state_piecewise(4)_pw_t22_CC_bin_y(2)
<= 0

c_e_state_piecewise(4)_pw_t22_CC_constraint5_:
+1 state_piecewise(4)_pw_t22_CC_bin_y(1)
+1 state_piecewise(4)_pw_t22_CC_bin_y(2)
= 1

c_e_state_piecewise(4)_pw_t23_CC_constraint1_:
+1 state(reach2_discharge_upstream_t23)
-2.6 state_piecewise(4)_pw_t23_CC_lambda(1)
-7.8 state_piecewise(4)_pw_t23_CC_lambda(2)
-10.8 state_piecewise(4)_pw_t23_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t23_CC_constraint2_:
+1 state(reach2_level_upstream_t23)
-1.85 state_piecewise(4)_pw_t23_CC_lambda(1)
-1.7850000000000001 state_piecewise(4)_pw_t23_CC_lambda(2)
-1.72 state_piecewise(4)_pw_t23_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t23_CC_constraint3_:
+1 state_piecewise(4)_pw_t23_CC_lambda(1)
+1 state_piecewise(4)_pw_t23_CC_lambda(2)
+1 state_piecewise(4)_pw_t23_CC_lambda(3)
= 1

c_u_state_piecewise(4)_pw_t23_CC_constraint4(1)_:
+1 state_piecewise(4)_pw_t23_CC_lambda(1)
-1 state_piecewise(4)_pw_t23_CC_bin_y(1)
<= 0

c_u_state_piecewise(4)_pw_t23_CC_constraint4(2)_:
+1 state_piecewise(4)_pw_t23_CC_lambda(2)
-1 state_piecewise(4)_pw_t23_CC_bin_y(1)
-1 state_piecewise(4)_pw_t23_CC_bin_y(2)
<= 0

c_u_state_piecewise(4)_pw_t23_CC_constraint4(3)_:
+1 state_piecewise(4)_pw_t23_CC_lambda(3)
-1 state_piecewise(4)_pw_t23_CC_bin_y(2)
<= 0

c_e_state_piecewise(4)_pw_t23_CC_constraint5_:
+1 state_piecewise(4)_pw_t23_CC_bin_y(1)
+1 state_piecewise(4)_pw_t23_CC_bin_y(2)
= 1

c_e_state_piecewise(4)_pw_t24_CC_constraint1_:
+1 state(reach2_discharge_upstream_t24)
-2.6 state_piecewise(4)_pw_t24_CC_lambda(1)
-7.8 state_piecewise(4)_pw_t24_CC_lambda(2)
-10.8 state_piecewise(4)_pw_t24_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t24_CC_constraint2_:
+1 state(reach2_level_upstream_t24)
-1.85 state_piecewise(4)_pw_t24_CC_lambda(1)
-1.7850000000000001 state_piecewise(4)_pw_t24_CC_lambda(2)
-1.72 state_piecewise(4)_pw_t24_CC_lambda(3)
= 0

c_e_state_piecewise(4)_pw_t24_CC_constraint3_:
+1 state_piecewise(4)_pw_t24_CC_lambda(1)
+1 state_piecewise(4)_pw_t24_CC_lambda(2)
+1 state_piecewise(4)_pw_t24_CC_lambda(3)
= 1

c_u_state_piecewise(4)_pw_t24_CC_constraint4(1)_:
+1 state_piecewise(4)_pw_t24_CC_lambda(1)
-1 state_piecewise(4)_pw_t24_CC_bin_y(1)
<= 0

c_u_state_piecewise(4)_pw_t24_CC_constraint4(2)_:
+1 state_piecewise(4)_pw_t24_CC_lambda(2)
-1 state_piecewise(4)_pw_t24_CC_bin_y(1)
-1 state_piecewise(4)_pw_t24_CC_bin_y(2)
<= 0

c_u_state_piecewise(4)_pw_t24_CC_constraint4(3)_:
+1 state_piecewise(4)_pw_t24_CC_lambda(3)
-1 state_piecewise(4)_pw_t24_CC_bin_y(2)
<= 0

c_e_state_piecewise(4)_pw_t24_CC_constraint5_:
+1 state_piecewise(4)_pw_t24_CC_bin_y(1)
+1 state_piecewise(4)_pw_t24_CC_bin_y(2)
= 1

c_e_state_piecewise(5)_pw_t01_CC_constraint1_:
+1 state(reach2_discharge_downstream_t01)
-2.6 state_piecewise(5)_pw_t01_CC_lambda(1)
-7.8 state_piecewise(5)_pw_t01_CC_lambda(2)
-10.8 state_piecewise(5)_pw_t01_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t01_CC_constraint2_:
+1 state(reach2_level_downstream_t01)
-1.72 state_piecewise(5)_pw_t01_CC_lambda(1)
-1.6600000000000001 state_piecewise(5)_pw_t01_CC_lambda(2)
-1.6 state_piecewise(5)_pw_t01_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t01_CC_constraint3_:
+1 state_piecewise(5)_pw_t01_CC_lambda(1)
+1 state_piecewise(5)_pw_t01_CC_lambda(2)
+1 state_piecewise(5)_pw_t01_CC_lambda(3)
= 1

c_u_state_piecewise(5)_pw_t01_CC_constraint4(1)_:
+1 state_piecewise(5)_pw_t01_CC_lambda(1)
-1 state_piecewise(5)_pw_t01_CC_bin_y(1)
<= 0

c_u_state_piecewise(5)_pw_t01_CC_constraint4(2)_:
+1 state_piecewise(5)_pw_t01_CC_lambda(2)
-1 state_piecewise(5)_pw_t01_CC_bin_y(1)
-1 state_piecewise(5)_pw_t01_CC_bin_y(2)
<= 0

c_u_state_piecewise(5)_pw_t01_CC_constraint4(3)_:
+1 state_piecewise(5)_pw_t01_CC_lambda(3)
-1 state_piecewise(5)_pw_t01_CC_bin_y(2)
<= 0

c_e_state_piecewise(5)_pw_t01_CC_constraint5_:
+1 state_piecewise(5)_pw_t01_CC_bin_y(1)
+1 state_piecewise(5)_pw_t01_CC_bin_y(2)
= 1

c_e_state_piecewise(5)_pw_t02_CC_constraint1_:
+1 state(reach2_discharge_downstream_t02)
-2.6 state_piecewise(5)_pw_t02_CC_lambda(1)
-7.8 state_piecewise(5)_pw_t02_CC_lambda(2)
-10.8 state_piecewise(5)_pw_t02_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t02_CC_constraint2_:
+1 state(reach2_level_downstream_t02)
-1.72 state_piecewise(5)_pw_t02_CC_lambda(1)
-1.6600000000000001 state_piecewise(5)_pw_t02_CC_lambda(2)
-1.6 state_piecewise(5)_pw_t02_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t02_CC_constraint3_:
+1 state_piecewise(5)_pw_t02_CC_lambda(1)
+1 state_piecewise(5)_pw_t02_CC_lambda(2)
+1 state_piecewise(5)_pw_t02_CC_lambda(3)
= 1

c_u_state_piecewise(5)_pw_t02_CC_constraint4(1)_:
+1 state_piecewise(5)_pw_t02_CC_lambda(1)
-1 state_piecewise(5)_pw_t02_CC_bin_y(1)
<= 0

c_u_state_piecewise(5)_pw_t02_CC_constraint4(2)_:
+1 state_piecewise(5)_pw_t02_CC_lambda(2)
-1 state_piecewise(5)_pw_t02_CC_bin_y(1)
-1 state_piecewise(5)_pw_t02_CC_bin_y(2)
<= 0

c_u_state_piecewise(5)_pw_t02_CC_constraint4(3)_:
+1 state_piecewise(5)_pw_t02_CC_lambda(3)
-1 state_piecewise(5)_pw_t02_CC_bin_y(2)
<= 0

c_e_state_piecewise(5)_pw_t02_CC_constraint5_:
+1 state_piecewise(5)_pw_t02_CC_bin_y(1)
+1 state_piecewise(5)_pw_t02_CC_bin_y(2)
= 1

c_e_state_piecewise(5)_pw_t03_CC_constraint1_:
+1 state(reach2_discharge_downstream_t03)
-2.6 state_piecewise(5)_pw_t03_CC_lambda(1)
-7.8 state_piecewise(5)_pw_t03_CC_lambda(2)
-10.8 state_piecewise(5)_pw_t03_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t03_CC_constraint2_:
+1 state(reach2_level_downstream_t03)
-1.72 state_piecewise(5)_pw_t03_CC_lambda(1)
-1.6600000000000001 state_piecewise(5)_pw_t03_CC_lambda(2)
-1.6 state_piecewise(5)_pw_t03_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t03_CC_constraint3_:
+1 state_piecewise(5)_pw_t03_CC_lambda(1)
+1 state_piecewise(5)_pw_t03_CC_lambda(2)
+1 state_piecewise(5)_pw_t03_CC_lambda(3)
= 1

c_u_state_piecewise(5)_pw_t03_CC_constraint4(1)_:
+1 state_piecewise(5)_pw_t03_CC_lambda(1)
-1 state_piecewise(5)_pw_t03_CC_bin_y(1)
<= 0

c_u_state_piecewise(5)_pw_t03_CC_constraint4(2)_:
+1 state_piecewise(5)_pw_t03_CC_lambda(2)
-1 state_piecewise(5)_pw_t03_CC_bin_y(1)
-1 state_piecewise(5)_pw_t03_CC_bin_y(2)
<= 0

c_u_state_piecewise(5)_pw_t03_CC_constraint4(3)_:
+1 state_piecewise(5)_pw_t03_CC_lambda(3)
-1 state_piecewise(5)_pw_t03_CC_bin_y(2)
<= 0

c_e_state_piecewise(5)_pw_t03_CC_constraint5_:
+1 state_piecewise(5)_pw_t03_CC_bin_y(1)
+1 state_piecewise(5)_pw_t03_CC_bin_y(2)
= 1

c_e_state_piecewise(5)_pw_t04_CC_constraint1_:
+1 state(reach2_discharge_downstream_t04)
-2.6 state_piecewise(5)_pw_t04_CC_lambda(1)
-7.8 state_piecewise(5)_pw_t04_CC_lambda(2)
-10.8 state_piecewise(5)_pw_t04_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t04_CC_constraint2_:
+1 state(reach2_level_downstream_t04)
-1.72 state_piecewise(5)_pw_t04_CC_lambda(1)
-1.6600000000000001 state_piecewise(5)_pw_t04_CC_lambda(2)
-1.6 state_piecewise(5)_pw_t04_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t04_CC_constraint3_:
+1 state_piecewise(5)_pw_t04_CC_lambda(1)
+1 state_piecewise(5)_pw_t04_CC_lambda(2)
+1 state_piecewise(5)_pw_t04_CC_lambda(3)
= 1

c_u_state_piecewise(5)_pw_t04_CC_constraint4(1)_:
+1 state_piecewise(5)_pw_t04_CC_lambda(1)
-1 state_piecewise(5)_pw_t04_CC_bin_y(1)
<= 0

c_u_state_piecewise(5)_pw_t04_CC_constraint4(2)_:
+1 state_piecewise(5)_pw_t04_CC_lambda(2)
-1 state_piecewise(5)_pw_t04_CC_bin_y(1)
-1 state_piecewise(5)_pw_t04_CC_bin_y(2)
<= 0

c_u_state_piecewise(5)_pw_t04_CC_constraint4(3)_:
+1 state_piecewise(5)_pw_t04_CC_lambda(3)
-1 state_piecewise(5)_pw_t04_CC_bin_y(2)
<= 0

c_e_state_piecewise(5)_pw_t04_CC_constraint5_:
+1 state_piecewise(5)_pw_t04_CC_bin_y(1)
+1 state_piecewise(5)_pw_t04_CC_bin_y(2)
= 1

c_e_state_piecewise(5)_pw_t05_CC_constraint1_:
+1 state(reach2_discharge_downstream_t05)
-2.6 state_piecewise(5)_pw_t05_CC_lambda(1)
-7.8 state_piecewise(5)_pw_t05_CC_lambda(2)
-10.8 state_piecewise(5)_pw_t05_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t05_CC_constraint2_:
+1 state(reach2_level_downstream_t05)
-1.72 state_piecewise(5)_pw_t05_CC_lambda(1)
-1.6600000000000001 state_piecewise(5)_pw_t05_CC_lambda(2)
-1.6 state_piecewise(5)_pw_t05_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t05_CC_constraint3_:
+1 state_piecewise(5)_pw_t05_CC_lambda(1)
+1 state_piecewise(5)_pw_t05_CC_lambda(2)
+1 state_piecewise(5)_pw_t05_CC_lambda(3)
= 1

c_u_state_piecewise(5)_pw_t05_CC_constraint4(1)_:
+1 state_piecewise(5)_pw_t05_CC_lambda(1)
-1 state_piecewise(5)_pw_t05_CC_bin_y(1)
<= 0

c_u_state_piecewise(5)_pw_t05_CC_constraint4(2)_:
+1 state_piecewise(5)_pw_t05_CC_lambda(2)
-1 state_piecewise(5)_pw_t05_CC_bin_y(1)
-1 state_piecewise(5)_pw_t05_CC_bin_y(2)
<= 0

c_u_state_piecewise(5)_pw_t05_CC_constraint4(3)_:
+1 state_piecewise(5)_pw_t05_CC_lambda(3)
-1 state_piecewise(5)_pw_t05_CC_bin_y(2)
<= 0

c_e_state_piecewise(5)_pw_t05_CC_constraint5_:
+1 state_piecewise(5)_pw_t05_CC_bin_y(1)
+1 state_piecewise(5)_pw_t05_CC_bin_y(2)
= 1

c_e_state_piecewise(5)_pw_t06_CC_constraint1_:
+1 state(reach2_discharge_downstream_t06)
-2.6 state_piecewise(5)_pw_t06_CC_lambda(1)
-7.8 state_piecewise(5)_pw_t06_CC_lambda(2)
-10.8 state_piecewise(5)_pw_t06_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t06_CC_constraint2_:
+1 state(reach2_level_downstream_t06)
-1.72 state_piecewise(5)_pw_t06_CC_lambda(1)
-1.6600000000000001 state_piecewise(5)_pw_t06_CC_lambda(2)
-1.6 state_piecewise(5)_pw_t06_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t06_CC_constraint3_:
+1 state_piecewise(5)_pw_t06_CC_lambda(1)
+1 state_piecewise(5)_pw_t06_CC_lambda(2)
+1 state_piecewise(5)_pw_t06_CC_lambda(3)
= 1

c_u_state_piecewise(5)_pw_t06_CC_constraint4(1)_:
+1 state_piecewise(5)_pw_t06_CC_lambda(1)
-1 state_piecewise(5)_pw_t06_CC_bin_y(1)
<= 0

c_u_state_piecewise(5)_pw_t06_CC_constraint4(2)_:
+1 state_piecewise(5)_pw_t06_CC_lambda(2)
-1 state_piecewise(5)_pw_t06_CC_bin_y(1)
-1 state_piecewise(5)_pw_t06_CC_bin_y(2)
<= 0

c_u_state_piecewise(5)_pw_t06_CC_constraint4(3)_:
+1 state_piecewise(5)_pw_t06_CC_lambda(3)
-1 state_piecewise(5)_pw_t06_CC_bin_y(2)
<= 0

c_e_state_piecewise(5)_pw_t06_CC_constraint5_:
+1 state_piecewise(5)_pw_t06_CC_bin_y(1)
+1 state_piecewise(5)_pw_t06_CC_bin_y(2)
= 1

c_e_state_piecewise(5)_pw_t07_CC_constraint1_:
+1 state(reach2_discharge_downstream_t07)
-2.6 state_piecewise(5)_pw_t07_CC_lambda(1)
-7.8 state_piecewise(5)_pw_t07_CC_lambda(2)
-10.8 state_piecewise(5)_pw_t07_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t07_CC_constraint2_:
+1 state(reach2_level_downstream_t07)
-1.72 state_piecewise(5)_pw_t07_CC_lambda(1)
-1.6600000000000001 state_piecewise(5)_pw_t07_CC_lambda(2)
-1.6 state_piecewise(5)_pw_t07_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t07_CC_constraint3_:
+1 state_piecewise(5)_pw_t07_CC_lambda(1)
+1 state_piecewise(5)_pw_t07_CC_lambda(2)
+1 state_piecewise(5)_pw_t07_CC_lambda(3)
= 1

c_u_state_piecewise(5)_pw_t07_CC_constraint4(1)_:
+1 state_piecewise(5)_pw_t07_CC_lambda(1)
-1 state_piecewise(5)_pw_t07_CC_bin_y(1)
<= 0

c_u_state_piecewise(5)_pw_t07_CC_constraint4(2)_:
+1 state_piecewise(5)_pw_t07_CC_lambda(2)
-1 state_piecewise(5)_pw_t07_CC_bin_y(1)
-1 state_piecewise(5)_pw_t07_CC_bin_y(2)
<= 0

c_u_state_piecewise(5)_pw_t07_CC_constraint4(3)_:
+1 state_piecewise(5)_pw_t07_CC_lambda(3)
-1 state_piecewise(5)_pw_t07_CC_bin_y(2)
<= 0

c_e_state_piecewise(5)_pw_t07_CC_constraint5_:
+1 state_piecewise(5)_pw_t07_CC_bin_y(1)
+1 state_piecewise(5)_pw_t07_CC_bin_y(2)
= 1

c_e_state_piecewise(5)_pw_t08_CC_constraint1_:
+1 state(reach2_discharge_downstream_t08)
-2.6 state_piecewise(5)_pw_t08_CC_lambda(1)
-7.8 state_piecewise(5)_pw_t08_CC_lambda(2)
-10.8 state_piecewise(5)_pw_t08_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t08_CC_constraint2_:
+1 state(reach2_level_downstream_t08)
-1.72 state_piecewise(5)_pw_t08_CC_lambda(1)
-1.6600000000000001 state_piecewise(5)_pw_t08_CC_lambda(2)
-1.6 state_piecewise(5)_pw_t08_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t08_CC_constraint3_:
+1 state_piecewise(5)_pw_t08_CC_lambda(1)
+1 state_piecewise(5)_pw_t08_CC_lambda(2)
+1 state_piecewise(5)_pw_t08_CC_lambda(3)
= 1

c_u_state_piecewise(5)_pw_t08_CC_constraint4(1)_:
+1 state_piecewise(5)_pw_t08_CC_lambda(1)
-1 state_piecewise(5)_pw_t08_CC_bin_y(1)
<= 0

c_u_state_piecewise(5)_pw_t08_CC_constraint4(2)_:
+1 state_piecewise(5)_pw_t08_CC_lambda(2)
-1 state_piecewise(5)_pw_t08_CC_bin_y(1)
-1 state_piecewise(5)_pw_t08_CC_bin_y(2)
<= 0

c_u_state_piecewise(5)_pw_t08_CC_constraint4(3)_:
+1 state_piecewise(5)_pw_t08_CC_lambda(3)
-1 state_piecewise(5)_pw_t08_CC_bin_y(2)
<= 0

c_e_state_piecewise(5)_pw_t08_CC_constraint5_:
+1 state_piecewise(5)_pw_t08_CC_bin_y(1)
+1 state_piecewise(5)_pw_t08_CC_bin_y(2)
= 1

c_e_state_piecewise(5)_pw_t09_CC_constraint1_:
+1 state(reach2_discharge_downstream_t09)
-2.6 state_piecewise(5)_pw_t09_CC_lambda(1)
-7.8 state_piecewise(5)_pw_t09_CC_lambda(2)
-10.8 state_piecewise(5)_pw_t09_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t09_CC_constraint2_:
+1 state(reach2_level_downstream_t09)
-1.72 state_piecewise(5)_pw_t09_CC_lambda(1)
-1.6600000000000001 state_piecewise(5)_pw_t09_CC_lambda(2)
-1.6 state_piecewise(5)_pw_t09_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t09_CC_constraint3_:
+1 state_piecewise(5)_pw_t09_CC_lambda(1)
+1 state_piecewise(5)_pw_t09_CC_lambda(2)
+1 state_piecewise(5)_pw_t09_CC_lambda(3)
= 1

c_u_state_piecewise(5)_pw_t09_CC_constraint4(1)_:
+1 state_piecewise(5)_pw_t09_CC_lambda(1)
-1 state_piecewise(5)_pw_t09_CC_bin_y(1)
<= 0

c_u_state_piecewise(5)_pw_t09_CC_constraint4(2)_:
+1 state_piecewise(5)_pw_t09_CC_lambda(2)
-1 state_piecewise(5)_pw_t09_CC_bin_y(1)
-1 state_piecewise(5)_pw_t09_CC_bin_y(2)
<= 0

c_u_state_piecewise(5)_pw_t09_CC_constraint4(3)_:
+1 state_piecewise(5)_pw_t09_CC_lambda(3)
-1 state_piecewise(5)_pw_t09_CC_bin_y(2)
<= 0

c_e_state_piecewise(5)_pw_t09_CC_constraint5_:
+1 state_piecewise(5)_pw_t09_CC_bin_y(1)
+1 state_piecewise(5)_pw_t09_CC_bin_y(2)
= 1

c_e_state_piecewise(5)_pw_t10_CC_constraint1_:
+1 state(reach2_discharge_downstream_t10)
-2.6 state_piecewise(5)_pw_t10_CC_lambda(1)
-7.8 state_piecewise(5)_pw_t10_CC_lambda(2)
-10.8 state_piecewise(5)_pw_t10_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t10_CC_constraint2_:
+1 state(reach2_level_downstream_t10)
-1.72 state_piecewise(5)_pw_t10_CC_lambda(1)
-1.6600000000000001 state_piecewise(5)_pw_t10_CC_lambda(2)
-1.6 state_piecewise(5)_pw_t10_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t10_CC_constraint3_:
+1 state_piecewise(5)_pw_t10_CC_lambda(1)
+1 state_piecewise(5)_pw_t10_CC_lambda(2)
+1 state_piecewise(5)_pw_t10_CC_lambda(3)
= 1

c_u_state_piecewise(5)_pw_t10_CC_constraint4(1)_:
+1 state_piecewise(5)_pw_t10_CC_lambda(1)
-1 state_piecewise(5)_pw_t10_CC_bin_y(1)
<= 0

c_u_state_piecewise(5)_pw_t10_CC_constraint4(2)_:
+1 state_piecewise(5)_pw_t10_CC_lambda(2)
-1 state_piecewise(5)_pw_t10_CC_bin_y(1)
-1 state_piecewise(5)_pw_t10_CC_bin_y(2)
<= 0

c_u_state_piecewise(5)_pw_t10_CC_constraint4(3)_:
+1 state_piecewise(5)_pw_t10_CC_lambda(3)
-1 state_piecewise(5)_pw_t10_CC_bin_y(2)
<= 0

c_e_state_piecewise(5)_pw_t10_CC_constraint5_:
+1 state_piecewise(5)_pw_t10_CC_bin_y(1)
+1 state_piecewise(5)_pw_t10_CC_bin_y(2)
= 1

c_e_state_piecewise(5)_pw_t11_CC_constraint1_:
+1 state(reach2_discharge_downstream_t11)
-2.6 state_piecewise(5)_pw_t11_CC_lambda(1)
-7.8 state_piecewise(5)_pw_t11_CC_lambda(2)
-10.8 state_piecewise(5)_pw_t11_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t11_CC_constraint2_:
+1 state(reach2_level_downstream_t11)
-1.72 state_piecewise(5)_pw_t11_CC_lambda(1)
-1.6600000000000001 state_piecewise(5)_pw_t11_CC_lambda(2)
-1.6 state_piecewise(5)_pw_t11_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t11_CC_constraint3_:
+1 state_piecewise(5)_pw_t11_CC_lambda(1)
+1 state_piecewise(5)_pw_t11_CC_lambda(2)
+1 state_piecewise(5)_pw_t11_CC_lambda(3)
= 1

c_u_state_piecewise(5)_pw_t11_CC_constraint4(1)_:
+1 state_piecewise(5)_pw_t11_CC_lambda(1)
-1 state_piecewise(5)_pw_t11_CC_bin_y(1)
<= 0

c_u_state_piecewise(5)_pw_t11_CC_constraint4(2)_:
+1 state_piecewise(5)_pw_t11_CC_lambda(2)
-1 state_piecewise(5)_pw_t11_CC_bin_y(1)
-1 state_piecewise(5)_pw_t11_CC_bin_y(2)
<= 0

c_u_state_piecewise(5)_pw_t11_CC_constraint4(3)_:
+1 state_piecewise(5)_pw_t11_CC_lambda(3)
-1 state_piecewise(5)_pw_t11_CC_bin_y(2)
<= 0

c_e_state_piecewise(5)_pw_t11_CC_constraint5_:
+1 state_piecewise(5)_pw_t11_CC_bin_y(1)
+1 state_piecewise(5)_pw_t11_CC_bin_y(2)
= 1

c_e_state_piecewise(5)_pw_t12_CC_constraint1_:
+1 state(reach2_discharge_downstream_t12)
-2.6 state_piecewise(5)_pw_t12_CC_lambda(1)
-7.8 state_piecewise(5)_pw_t12_CC_lambda(2)
-10.8 state_piecewise(5)_pw_t12_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t12_CC_constraint2_:
+1 state(reach2_level_downstream_t12)
-1.72 state_piecewise(5)_pw_t12_CC_lambda(1)
-1.6600000000000001 state_piecewise(5)_pw_t12_CC_lambda(2)
-1.6 state_piecewise(5)_pw_t12_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t12_CC_constraint3_:
+1 state_piecewise(5)_pw_t12_CC_lambda(1)
+1 state_piecewise(5)_pw_t12_CC_lambda(2)
+1 state_piecewise(5)_pw_t12_CC_lambda(3)
= 1

c_u_state_piecewise(5)_pw_t12_CC_constraint4(1)_:
+1 state_piecewise(5)_pw_t12_CC_lambda(1)
-1 state_piecewise(5)_pw_t12_CC_bin_y(1)
<= 0

c_u_state_piecewise(5)_pw_t12_CC_constraint4(2)_:
+1 state_piecewise(5)_pw_t12_CC_lambda(2)
-1 state_piecewise(5)_pw_t12_CC_bin_y(1)
-1 state_piecewise(5)_pw_t12_CC_bin_y(2)
<= 0

c_u_state_piecewise(5)_pw_t12_CC_constraint4(3)_:
+1 state_piecewise(5)_pw_t12_CC_lambda(3)
-1 state_piecewise(5)_pw_t12_CC_bin_y(2)
<= 0

c_e_state_piecewise(5)_pw_t12_CC_constraint5_:
+1 state_piecewise(5)_pw_t12_CC_bin_y(1)
+1 state_piecewise(5)_pw_t12_CC_bin_y(2)
= 1

c_e_state_piecewise(5)_pw_t13_CC_constraint1_:
+1 state(reach2_discharge_downstream_t13)
-2.6 state_piecewise(5)_pw_t13_CC_lambda(1)
-7.8 state_piecewise(5)_pw_t13_CC_lambda(2)
-10.8 state_piecewise(5)_pw_t13_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t13_CC_constraint2_:
+1 state(reach2_level_downstream_t13)
-1.72 state_piecewise(5)_pw_t13_CC_lambda(1)
-1.6600000000000001 state_piecewise(5)_pw_t13_CC_lambda(2)
-1.6 state_piecewise(5)_pw_t13_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t13_CC_constraint3_:
+1 state_piecewise(5)_pw_t13_CC_lambda(1)
+1 state_piecewise(5)_pw_t13_CC_lambda(2)
+1 state_piecewise(5)_pw_t13_CC_lambda(3)
= 1

c_u_state_piecewise(5)_pw_t13_CC_constraint4(1)_:
+1 state_piecewise(5)_pw_t13_CC_lambda(1)
-1 state_piecewise(5)_pw_t13_CC_bin_y(1)
<= 0

c_u_state_piecewise(5)_pw_t13_CC_constraint4(2)_:
+1 state_piecewise(5)_pw_t13_CC_lambda(2)
-1 state_piecewise(5)_pw_t13_CC_bin_y(1)
-1 state_piecewise(5)_pw_t13_CC_bin_y(2)
<= 0

c_u_state_piecewise(5)_pw_t13_CC_constraint4(3)_:
+1 state_piecewise(5)_pw_t13_CC_lambda(3)
-1 state_piecewise(5)_pw_t13_CC_bin_y(2)
<= 0

c_e_state_piecewise(5)_pw_t13_CC_constraint5_:
+1 state_piecewise(5)_pw_t13_CC_bin_y(1)
+1 state_piecewise(5)_pw_t13_CC_bin_y(2)
= 1

c_e_state_piecewise(5)_pw_t14_CC_constraint1_:
+1 state(reach2_discharge_downstream_t14)
-2.6 state_piecewise(5)_pw_t14_CC_lambda(1)
-7.8 state_piecewise(5)_pw_t14_CC_lambda(2)
-10.8 state_piecewise(5)_pw_t14_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t14_CC_constraint2_:
+1 state(reach2_level_downstream_t14)
-1.72 state_piecewise(5)_pw_t14_CC_lambda(1)
-1.6600000000000001 state_piecewise(5)_pw_t14_CC_lambda(2)
-1.6 state_piecewise(5)_pw_t14_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t14_CC_constraint3_:
+1 state_piecewise(5)_pw_t14_CC_lambda(1)
+1 state_piecewise(5)_pw_t14_CC_lambda(2)
+1 state_piecewise(5)_pw_t14_CC_lambda(3)
= 1

c_u_state_piecewise(5)_pw_t14_CC_constraint4(1)_:
+1 state_piecewise(5)_pw_t14_CC_lambda(1)
-1 state_piecewise(5)_pw_t14_CC_bin_y(1)
<= 0

c_u_state_piecewise(5)_pw_t14_CC_constraint4(2)_:
+1 state_piecewise(5)_pw_t14_CC_lambda(2)
-1 state_piecewise(5)_pw_t14_CC_bin_y(1)
-1 state_piecewise(5)_pw_t14_CC_bin_y(2)
<= 0

c_u_state_piecewise(5)_pw_t14_CC_constraint4(3)_:
+1 state_piecewise(5)_pw_t14_CC_lambda(3)
-1 state_piecewise(5)_pw_t14_CC_bin_y(2)
<= 0

c_e_state_piecewise(5)_pw_t14_CC_constraint5_:
+1 state_piecewise(5)_pw_t14_CC_bin_y(1)
+1 state_piecewise(5)_pw_t14_CC_bin_y(2)
= 1

c_e_state_piecewise(5)_pw_t15_CC_constraint1_:
+1 state(reach2_discharge_downstream_t15)
-2.6 state_piecewise(5)_pw_t15_CC_lambda(1)
-7.8 state_piecewise(5)_pw_t15_CC_lambda(2)
-10.8 state_piecewise(5)_pw_t15_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t15_CC_constraint2_:
+1 state(reach2_level_downstream_t15)
-1.72 state_piecewise(5)_pw_t15_CC_lambda(1)
-1.6600000000000001 state_piecewise(5)_pw_t15_CC_lambda(2)
-1.6 state_piecewise(5)_pw_t15_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t15_CC_constraint3_:
+1 state_piecewise(5)_pw_t15_CC_lambda(1)
+1 state_piecewise(5)_pw_t15_CC_lambda(2)
+1 state_piecewise(5)_pw_t15_CC_lambda(3)
= 1

c_u_state_piecewise(5)_pw_t15_CC_constraint4(1)_:
+1 state_piecewise(5)_pw_t15_CC_lambda(1)
-1 state_piecewise(5)_pw_t15_CC_bin_y(1)
<= 0

c_u_state_piecewise(5)_pw_t15_CC_constraint4(2)_:
+1 state_piecewise(5)_pw_t15_CC_lambda(2)
-1 state_piecewise(5)_pw_t15_CC_bin_y(1)
-1 state_piecewise(5)_pw_t15_CC_bin_y(2)
<= 0

c_u_state_piecewise(5)_pw_t15_CC_constraint4(3)_:
+1 state_piecewise(5)_pw_t15_CC_lambda(3)
-1 state_piecewise(5)_pw_t15_CC_bin_y(2)
<= 0

c_e_state_piecewise(5)_pw_t15_CC_constraint5_:
+1 state_piecewise(5)_pw_t15_CC_bin_y(1)
+1 state_piecewise(5)_pw_t15_CC_bin_y(2)
= 1

c_e_state_piecewise(5)_pw_t16_CC_constraint1_:
+1 state(reach2_discharge_downstream_t16)
-2.6 state_piecewise(5)_pw_t16_CC_lambda(1)
-7.8 state_piecewise(5)_pw_t16_CC_lambda(2)
-10.8 state_piecewise(5)_pw_t16_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t16_CC_constraint2_:
+1 state(reach2_level_downstream_t16)
-1.72 state_piecewise(5)_pw_t16_CC_lambda(1)
-1.6600000000000001 state_piecewise(5)_pw_t16_CC_lambda(2)
-1.6 state_piecewise(5)_pw_t16_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t16_CC_constraint3_:
+1 state_piecewise(5)_pw_t16_CC_lambda(1)
+1 state_piecewise(5)_pw_t16_CC_lambda(2)
+1 state_piecewise(5)_pw_t16_CC_lambda(3)
= 1

c_u_state_piecewise(5)_pw_t16_CC_constraint4(1)_:
+1 state_piecewise(5)_pw_t16_CC_lambda(1)
-1 state_piecewise(5)_pw_t16_CC_bin_y(1)
<= 0

c_u_state_piecewise(5)_pw_t16_CC_constraint4(2)_:
+1 state_piecewise(5)_pw_t16_CC_lambda(2)
-1 state_piecewise(5)_pw_t16_CC_bin_y(1)
-1 state_piecewise(5)_pw_t16_CC_bin_y(2)
<= 0

c_u_state_piecewise(5)_pw_t16_CC_constraint4(3)_:
+1 state_piecewise(5)_pw_t16_CC_lambda(3)
-1 state_piecewise(5)_pw_t16_CC_bin_y(2)
<= 0

c_e_state_piecewise(5)_pw_t16_CC_constraint5_:
+1 state_piecewise(5)_pw_t16_CC_bin_y(1)
+1 state_piecewise(5)_pw_t16_CC_bin_y(2)
= 1

c_e_state_piecewise(5)_pw_t17_CC_constraint1_:
+1 state(reach2_discharge_downstream_t17)
-2.6 state_piecewise(5)_pw_t17_CC_lambda(1)
-7.8 state_piecewise(5)_pw_t17_CC_lambda(2)
-10.8 state_piecewise(5)_pw_t17_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t17_CC_constraint2_:
+1 state(reach2_level_downstream_t17)
-1.72 state_piecewise(5)_pw_t17_CC_lambda(1)
-1.6600000000000001 state_piecewise(5)_pw_t17_CC_lambda(2)
-1.6 state_piecewise(5)_pw_t17_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t17_CC_constraint3_:
+1 state_piecewise(5)_pw_t17_CC_lambda(1)
+1 state_piecewise(5)_pw_t17_CC_lambda(2)
+1 state_piecewise(5)_pw_t17_CC_lambda(3)
= 1

c_u_state_piecewise(5)_pw_t17_CC_constraint4(1)_:
+1 state_piecewise(5)_pw_t17_CC_lambda(1)
-1 state_piecewise(5)_pw_t17_CC_bin_y(1)
<= 0

c_u_state_piecewise(5)_pw_t17_CC_constraint4(2)_:
+1 state_piecewise(5)_pw_t17_CC_lambda(2)
-1 state_piecewise(5)_pw_t17_CC_bin_y(1)
-1 state_piecewise(5)_pw_t17_CC_bin_y(2)
<= 0

c_u_state_piecewise(5)_pw_t17_CC_constraint4(3)_:
+1 state_piecewise(5)_pw_t17_CC_lambda(3)
-1 state_piecewise(5)_pw_t17_CC_bin_y(2)
<= 0

c_e_state_piecewise(5)_pw_t17_CC_constraint5_:
+1 state_piecewise(5)_pw_t17_CC_bin_y(1)
+1 state_piecewise(5)_pw_t17_CC_bin_y(2)
= 1

c_e_state_piecewise(5)_pw_t18_CC_constraint1_:
+1 state(reach2_discharge_downstream_t18)
-2.6 state_piecewise(5)_pw_t18_CC_lambda(1)
-7.8 state_piecewise(5)_pw_t18_CC_lambda(2)
-10.8 state_piecewise(5)_pw_t18_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t18_CC_constraint2_:
+1 state(reach2_level_downstream_t18)
-1.72 state_piecewise(5)_pw_t18_CC_lambda(1)
-1.6600000000000001 state_piecewise(5)_pw_t18_CC_lambda(2)
-1.6 state_piecewise(5)_pw_t18_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t18_CC_constraint3_:
+1 state_piecewise(5)_pw_t18_CC_lambda(1)
+1 state_piecewise(5)_pw_t18_CC_lambda(2)
+1 state_piecewise(5)_pw_t18_CC_lambda(3)
= 1

c_u_state_piecewise(5)_pw_t18_CC_constraint4(1)_:
+1 state_piecewise(5)_pw_t18_CC_lambda(1)
-1 state_piecewise(5)_pw_t18_CC_bin_y(1)
<= 0

c_u_state_piecewise(5)_pw_t18_CC_constraint4(2)_:
+1 state_piecewise(5)_pw_t18_CC_lambda(2)
-1 state_piecewise(5)_pw_t18_CC_bin_y(1)
-1 state_piecewise(5)_pw_t18_CC_bin_y(2)
<= 0

c_u_state_piecewise(5)_pw_t18_CC_constraint4(3)_:
+1 state_piecewise(5)_pw_t18_CC_lambda(3)
-1 state_piecewise(5)_pw_t18_CC_bin_y(2)
<= 0

c_e_state_piecewise(5)_pw_t18_CC_constraint5_:
+1 state_piecewise(5)_pw_t18_CC_bin_y(1)
+1 state_piecewise(5)_pw_t18_CC_bin_y(2)
= 1

c_e_state_piecewise(5)_pw_t19_CC_constraint1_:
+1 state(reach2_discharge_downstream_t19)
-2.6 state_piecewise(5)_pw_t19_CC_lambda(1)
-7.8 state_piecewise(5)_pw_t19_CC_lambda(2)
-10.8 state_piecewise(5)_pw_t19_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t19_CC_constraint2_:
+1 state(reach2_level_downstream_t19)
-1.72 state_piecewise(5)_pw_t19_CC_lambda(1)
-1.6600000000000001 state_piecewise(5)_pw_t19_CC_lambda(2)
-1.6 state_piecewise(5)_pw_t19_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t19_CC_constraint3_:
+1 state_piecewise(5)_pw_t19_CC_lambda(1)
+1 state_piecewise(5)_pw_t19_CC_lambda(2)
+1 state_piecewise(5)_pw_t19_CC_lambda(3)
= 1

c_u_state_piecewise(5)_pw_t19_CC_constraint4(1)_:
+1 state_piecewise(5)_pw_t19_CC_lambda(1)
-1 state_piecewise(5)_pw_t19_CC_bin_y(1)
<= 0

c_u_state_piecewise(5)_pw_t19_CC_constraint4(2)_:
+1 state_piecewise(5)_pw_t19_CC_lambda(2)
-1 state_piecewise(5)_pw_t19_CC_bin_y(1)
-1 state_piecewise(5)_pw_t19_CC_bin_y(2)
<= 0

c_u_state_piecewise(5)_pw_t19_CC_constraint4(3)_:
+1 state_piecewise(5)_pw_t19_CC_lambda(3)
-1 state_piecewise(5)_pw_t19_CC_bin_y(2)
<= 0

c_e_state_piecewise(5)_pw_t19_CC_constraint5_:
+1 state_piecewise(5)_pw_t19_CC_bin_y(1)
+1 state_piecewise(5)_pw_t19_CC_bin_y(2)
= 1

c_e_state_piecewise(5)_pw_t20_CC_constraint1_:
+1 state(reach2_discharge_downstream_t20)
-2.6 state_piecewise(5)_pw_t20_CC_lambda(1)
-7.8 state_piecewise(5)_pw_t20_CC_lambda(2)
-10.8 state_piecewise(5)_pw_t20_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t20_CC_constraint2_:
+1 state(reach2_level_downstream_t20)
-1.72 state_piecewise(5)_pw_t20_CC_lambda(1)
-1.6600000000000001 state_piecewise(5)_pw_t20_CC_lambda(2)
-1.6 state_piecewise(5)_pw_t20_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t20_CC_constraint3_:
+1 state_piecewise(5)_pw_t20_CC_lambda(1)
+1 state_piecewise(5)_pw_t20_CC_lambda(2)
+1 state_piecewise(5)_pw_t20_CC_lambda(3)
= 1

c_u_state_piecewise(5)_pw_t20_CC_constraint4(1)_:
+1 state_piecewise(5)_pw_t20_CC_lambda(1)
-1 state_piecewise(5)_pw_t20_CC_bin_y(1)
<= 0

c_u_state_piecewise(5)_pw_t20_CC_constraint4(2)_:
+1 state_piecewise(5)_pw_t20_CC_lambda(2)
-1 state_piecewise(5)_pw_t20_CC_bin_y(1)
-1 state_piecewise(5)_pw_t20_CC_bin_y(2)
<= 0

c_u_state_piecewise(5)_pw_t20_CC_constraint4(3)_:
+1 state_piecewise(5)_pw_t20_CC_lambda(3)
-1 state_piecewise(5)_pw_t20_CC_bin_y(2)
<= 0

c_e_state_piecewise(5)_pw_t20_CC_constraint5_:
+1 state_piecewise(5)_pw_t20_CC_bin_y(1)
+1 state_piecewise(5)_pw_t20_CC_bin_y(2)
= 1

c_e_state_piecewise(5)_pw_t21_CC_constraint1_:
+1 state(reach2_discharge_downstream_t21)
-2.6 state_piecewise(5)_pw_t21_CC_lambda(1)
-7.8 state_piecewise(5)_pw_t21_CC_lambda(2)
-10.8 state_piecewise(5)_pw_t21_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t21_CC_constraint2_:
+1 state(reach2_level_downstream_t21)
-1.72 state_piecewise(5)_pw_t21_CC_lambda(1)
-1.6600000000000001 state_piecewise(5)_pw_t21_CC_lambda(2)
-1.6 state_piecewise(5)_pw_t21_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t21_CC_constraint3_:
+1 state_piecewise(5)_pw_t21_CC_lambda(1)
+1 state_piecewise(5)_pw_t21_CC_lambda(2)
+1 state_piecewise(5)_pw_t21_CC_lambda(3)
= 1

c_u_state_piecewise(5)_pw_t21_CC_constraint4(1)_:
+1 state_piecewise(5)_pw_t21_CC_lambda(1)
-1 state_piecewise(5)_pw_t21_CC_bin_y(1)
<= 0

c_u_state_piecewise(5)_pw_t21_CC_constraint4(2)_:
+1 state_piecewise(5)_pw_t21_CC_lambda(2)
-1 state_piecewise(5)_pw_t21_CC_bin_y(1)
-1 state_piecewise(5)_pw_t21_CC_bin_y(2)
<= 0

c_u_state_piecewise(5)_pw_t21_CC_constraint4(3)_:
+1 state_piecewise(5)_pw_t21_CC_lambda(3)
-1 state_piecewise(5)_pw_t21_CC_bin_y(2)
<= 0

c_e_state_piecewise(5)_pw_t21_CC_constraint5_:
+1 state_piecewise(5)_pw_t21_CC_bin_y(1)
+1 state_piecewise(5)_pw_t21_CC_bin_y(2)
= 1

c_e_state_piecewise(5)_pw_t22_CC_constraint1_:
+1 state(reach2_discharge_downstream_t22)
-2.6 state_piecewise(5)_pw_t22_CC_lambda(1)
-7.8 state_piecewise(5)_pw_t22_CC_lambda(2)
-10.8 state_piecewise(5)_pw_t22_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t22_CC_constraint2_:
+1 state(reach2_level_downstream_t22)
-1.72 state_piecewise(5)_pw_t22_CC_lambda(1)
-1.6600000000000001 state_piecewise(5)_pw_t22_CC_lambda(2)
-1.6 state_piecewise(5)_pw_t22_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t22_CC_constraint3_:
+1 state_piecewise(5)_pw_t22_CC_lambda(1)
+1 state_piecewise(5)_pw_t22_CC_lambda(2)
+1 state_piecewise(5)_pw_t22_CC_lambda(3)
= 1

c_u_state_piecewise(5)_pw_t22_CC_constraint4(1)_:
+1 state_piecewise(5)_pw_t22_CC_lambda(1)
-1 state_piecewise(5)_pw_t22_CC_bin_y(1)
<= 0

c_u_state_piecewise(5)_pw_t22_CC_constraint4(2)_:
+1 state_piecewise(5)_pw_t22_CC_lambda(2)
-1 state_piecewise(5)_pw_t22_CC_bin_y(1)
-1 state_piecewise(5)_pw_t22_CC_bin_y(2)
<= 0

c_u_state_piecewise(5)_pw_t22_CC_constraint4(3)_:
+1 state_piecewise(5)_pw_t22_CC_lambda(3)
-1 state_piecewise(5)_pw_t22_CC_bin_y(2)
<= 0

c_e_state_piecewise(5)_pw_t22_CC_constraint5_:
+1 state_piecewise(5)_pw_t22_CC_bin_y(1)
+1 state_piecewise(5)_pw_t22_CC_bin_y(2)
= 1

c_e_state_piecewise(5)_pw_t23_CC_constraint1_:
+1 state(reach2_discharge_downstream_t23)
-2.6 state_piecewise(5)_pw_t23_CC_lambda(1)
-7.8 state_piecewise(5)_pw_t23_CC_lambda(2)
-10.8 state_piecewise(5)_pw_t23_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t23_CC_constraint2_:
+1 state(reach2_level_downstream_t23)
-1.72 state_piecewise(5)_pw_t23_CC_lambda(1)
-1.6600000000000001 state_piecewise(5)_pw_t23_CC_lambda(2)
-1.6 state_piecewise(5)_pw_t23_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t23_CC_constraint3_:
+1 state_piecewise(5)_pw_t23_CC_lambda(1)
+1 state_piecewise(5)_pw_t23_CC_lambda(2)
+1 state_piecewise(5)_pw_t23_CC_lambda(3)
= 1

c_u_state_piecewise(5)_pw_t23_CC_constraint4(1)_:
+1 state_piecewise(5)_pw_t23_CC_lambda(1)
-1 state_piecewise(5)_pw_t23_CC_bin_y(1)
<= 0

c_u_state_piecewise(5)_pw_t23_CC_constraint4(2)_:
+1 state_piecewise(5)_pw_t23_CC_lambda(2)
-1 state_piecewise(5)_pw_t23_CC_bin_y(1)
-1 state_piecewise(5)_pw_t23_CC_bin_y(2)
<= 0

c_u_state_piecewise(5)_pw_t23_CC_constraint4(3)_:
+1 state_piecewise(5)_pw_t23_CC_lambda(3)
-1 state_piecewise(5)_pw_t23_CC_bin_y(2)
<= 0

c_e_state_piecewise(5)_pw_t23_CC_constraint5_:
+1 state_piecewise(5)_pw_t23_CC_bin_y(1)
+1 state_piecewise(5)_pw_t23_CC_bin_y(2)
= 1

c_e_state_piecewise(5)_pw_t24_CC_constraint1_:
+1 state(reach2_discharge_downstream_t24)
-2.6 state_piecewise(5)_pw_t24_CC_lambda(1)
-7.8 state_piecewise(5)_pw_t24_CC_lambda(2)
-10.8 state_piecewise(5)_pw_t24_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t24_CC_constraint2_:
+1 state(reach2_level_downstream_t24)
-1.72 state_piecewise(5)_pw_t24_CC_lambda(1)
-1.6600000000000001 state_piecewise(5)_pw_t24_CC_lambda(2)
-1.6 state_piecewise(5)_pw_t24_CC_lambda(3)
= 0

c_e_state_piecewise(5)_pw_t24_CC_constraint3_:
+1 state_piecewise(5)_pw_t24_CC_lambda(1)
+1 state_piecewise(5)_pw_t24_CC_lambda(2)
+1 state_piecewise(5)_pw_t24_CC_lambda(3)
= 1

c_u_state_piecewise(5)_pw_t24_CC_constraint4(1)_:
+1 state_piecewise(5)_pw_t24_CC_lambda(1)
-1 state_piecewise(5)_pw_t24_CC_bin_y(1)
<= 0

c_u_state_piecewise(5)_pw_t24_CC_constraint4(2)_:
+1 state_piecewise(5)_pw_t24_CC_lambda(2)
-1 state_piecewise(5)_pw_t24_CC_bin_y(1)
-1 state_piecewise(5)_pw_t24_CC_bin_y(2)
<= 0

c_u_state_piecewise(5)_pw_t24_CC_constraint4(3)_:
+1 state_piecewise(5)_pw_t24_CC_lambda(3)
-1 state_piecewise(5)_pw_t24_CC_bin_y(2)
<= 0

c_e_state_piecewise(5)_pw_t24_CC_constraint5_:
+1 state_piecewise(5)_pw_t24_CC_bin_y(1)
+1 state_piecewise(5)_pw_t24_CC_bin_y(2)
= 1

c_e_state_piecewise(6)_pw_t01_CC_constraint1_:
+1 state(reach2_storage_t01)
-250.0 state_piecewise(6)_pw_t01_CC_lambda(1)
-330.0 state_piecewise(6)_pw_t01_CC_lambda(2)
-410.0 state_piecewise(6)_pw_t01_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t01_CC_constraint2_:
+1 state(reach2_discharge_downstream_t01)
-2.6 state_piecewise(6)_pw_t01_CC_lambda(1)
-7.8 state_piecewise(6)_pw_t01_CC_lambda(2)
-10.8 state_piecewise(6)_pw_t01_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t01_CC_constraint3_:
+1 state_piecewise(6)_pw_t01_CC_lambda(1)
+1 state_piecewise(6)_pw_t01_CC_lambda(2)
+1 state_piecewise(6)_pw_t01_CC_lambda(3)
= 1

c_u_state_piecewise(6)_pw_t01_CC_constraint4(1)_:
+1 state_piecewise(6)_pw_t01_CC_lambda(1)
-1 state_piecewise(6)_pw_t01_CC_bin_y(1)
<= 0

c_u_state_piecewise(6)_pw_t01_CC_constraint4(2)_:
+1 state_piecewise(6)_pw_t01_CC_lambda(2)
-1 state_piecewise(6)_pw_t01_CC_bin_y(1)
-1 state_piecewise(6)_pw_t01_CC_bin_y(2)
<= 0

c_u_state_piecewise(6)_pw_t01_CC_constraint4(3)_:
+1 state_piecewise(6)_pw_t01_CC_lambda(3)
-1 state_piecewise(6)_pw_t01_CC_bin_y(2)
<= 0

c_e_state_piecewise(6)_pw_t01_CC_constraint5_:
+1 state_piecewise(6)_pw_t01_CC_bin_y(1)
+1 state_piecewise(6)_pw_t01_CC_bin_y(2)
= 1

c_e_state_piecewise(6)_pw_t02_CC_constraint1_:
+1 state(reach2_storage_t02)
-250.0 state_piecewise(6)_pw_t02_CC_lambda(1)
-330.0 state_piecewise(6)_pw_t02_CC_lambda(2)
-410.0 state_piecewise(6)_pw_t02_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t02_CC_constraint2_:
+1 state(reach2_discharge_downstream_t02)
-2.6 state_piecewise(6)_pw_t02_CC_lambda(1)
-7.8 state_piecewise(6)_pw_t02_CC_lambda(2)
-10.8 state_piecewise(6)_pw_t02_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t02_CC_constraint3_:
+1 state_piecewise(6)_pw_t02_CC_lambda(1)
+1 state_piecewise(6)_pw_t02_CC_lambda(2)
+1 state_piecewise(6)_pw_t02_CC_lambda(3)
= 1

c_u_state_piecewise(6)_pw_t02_CC_constraint4(1)_:
+1 state_piecewise(6)_pw_t02_CC_lambda(1)
-1 state_piecewise(6)_pw_t02_CC_bin_y(1)
<= 0

c_u_state_piecewise(6)_pw_t02_CC_constraint4(2)_:
+1 state_piecewise(6)_pw_t02_CC_lambda(2)
-1 state_piecewise(6)_pw_t02_CC_bin_y(1)
-1 state_piecewise(6)_pw_t02_CC_bin_y(2)
<= 0

c_u_state_piecewise(6)_pw_t02_CC_constraint4(3)_:
+1 state_piecewise(6)_pw_t02_CC_lambda(3)
-1 state_piecewise(6)_pw_t02_CC_bin_y(2)
<= 0

c_e_state_piecewise(6)_pw_t02_CC_constraint5_:
+1 state_piecewise(6)_pw_t02_CC_bin_y(1)
+1 state_piecewise(6)_pw_t02_CC_bin_y(2)
= 1

c_e_state_piecewise(6)_pw_t03_CC_constraint1_:
+1 state(reach2_storage_t03)
-250.0 state_piecewise(6)_pw_t03_CC_lambda(1)
-330.0 state_piecewise(6)_pw_t03_CC_lambda(2)
-410.0 state_piecewise(6)_pw_t03_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t03_CC_constraint2_:
+1 state(reach2_discharge_downstream_t03)
-2.6 state_piecewise(6)_pw_t03_CC_lambda(1)
-7.8 state_piecewise(6)_pw_t03_CC_lambda(2)
-10.8 state_piecewise(6)_pw_t03_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t03_CC_constraint3_:
+1 state_piecewise(6)_pw_t03_CC_lambda(1)
+1 state_piecewise(6)_pw_t03_CC_lambda(2)
+1 state_piecewise(6)_pw_t03_CC_lambda(3)
= 1

c_u_state_piecewise(6)_pw_t03_CC_constraint4(1)_:
+1 state_piecewise(6)_pw_t03_CC_lambda(1)
-1 state_piecewise(6)_pw_t03_CC_bin_y(1)
<= 0

c_u_state_piecewise(6)_pw_t03_CC_constraint4(2)_:
+1 state_piecewise(6)_pw_t03_CC_lambda(2)
-1 state_piecewise(6)_pw_t03_CC_bin_y(1)
-1 state_piecewise(6)_pw_t03_CC_bin_y(2)
<= 0

c_u_state_piecewise(6)_pw_t03_CC_constraint4(3)_:
+1 state_piecewise(6)_pw_t03_CC_lambda(3)
-1 state_piecewise(6)_pw_t03_CC_bin_y(2)
<= 0

c_e_state_piecewise(6)_pw_t03_CC_constraint5_:
+1 state_piecewise(6)_pw_t03_CC_bin_y(1)
+1 state_piecewise(6)_pw_t03_CC_bin_y(2)
= 1

c_e_state_piecewise(6)_pw_t04_CC_constraint1_:
+1 state(reach2_storage_t04)
-250.0 state_piecewise(6)_pw_t04_CC_lambda(1)
-330.0 state_piecewise(6)_pw_t04_CC_lambda(2)
-410.0 state_piecewise(6)_pw_t04_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t04_CC_constraint2_:
+1 state(reach2_discharge_downstream_t04)
-2.6 state_piecewise(6)_pw_t04_CC_lambda(1)
-7.8 state_piecewise(6)_pw_t04_CC_lambda(2)
-10.8 state_piecewise(6)_pw_t04_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t04_CC_constraint3_:
+1 state_piecewise(6)_pw_t04_CC_lambda(1)
+1 state_piecewise(6)_pw_t04_CC_lambda(2)
+1 state_piecewise(6)_pw_t04_CC_lambda(3)
= 1

c_u_state_piecewise(6)_pw_t04_CC_constraint4(1)_:
+1 state_piecewise(6)_pw_t04_CC_lambda(1)
-1 state_piecewise(6)_pw_t04_CC_bin_y(1)
<= 0

c_u_state_piecewise(6)_pw_t04_CC_constraint4(2)_:
+1 state_piecewise(6)_pw_t04_CC_lambda(2)
-1 state_piecewise(6)_pw_t04_CC_bin_y(1)
-1 state_piecewise(6)_pw_t04_CC_bin_y(2)
<= 0

c_u_state_piecewise(6)_pw_t04_CC_constraint4(3)_:
+1 state_piecewise(6)_pw_t04_CC_lambda(3)
-1 state_piecewise(6)_pw_t04_CC_bin_y(2)
<= 0

c_e_state_piecewise(6)_pw_t04_CC_constraint5_:
+1 state_piecewise(6)_pw_t04_CC_bin_y(1)
+1 state_piecewise(6)_pw_t04_CC_bin_y(2)
= 1

c_e_state_piecewise(6)_pw_t05_CC_constraint1_:
+1 state(reach2_storage_t05)
-250.0 state_piecewise(6)_pw_t05_CC_lambda(1)
-330.0 state_piecewise(6)_pw_t05_CC_lambda(2)
-410.0 state_piecewise(6)_pw_t05_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t05_CC_constraint2_:
+1 state(reach2_discharge_downstream_t05)
-2.6 state_piecewise(6)_pw_t05_CC_lambda(1)
-7.8 state_piecewise(6)_pw_t05_CC_lambda(2)
-10.8 state_piecewise(6)_pw_t05_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t05_CC_constraint3_:
+1 state_piecewise(6)_pw_t05_CC_lambda(1)
+1 state_piecewise(6)_pw_t05_CC_lambda(2)
+1 state_piecewise(6)_pw_t05_CC_lambda(3)
= 1

c_u_state_piecewise(6)_pw_t05_CC_constraint4(1)_:
+1 state_piecewise(6)_pw_t05_CC_lambda(1)
-1 state_piecewise(6)_pw_t05_CC_bin_y(1)
<= 0

c_u_state_piecewise(6)_pw_t05_CC_constraint4(2)_:
+1 state_piecewise(6)_pw_t05_CC_lambda(2)
-1 state_piecewise(6)_pw_t05_CC_bin_y(1)
-1 state_piecewise(6)_pw_t05_CC_bin_y(2)
<= 0

c_u_state_piecewise(6)_pw_t05_CC_constraint4(3)_:
+1 state_piecewise(6)_pw_t05_CC_lambda(3)
-1 state_piecewise(6)_pw_t05_CC_bin_y(2)
<= 0

c_e_state_piecewise(6)_pw_t05_CC_constraint5_:
+1 state_piecewise(6)_pw_t05_CC_bin_y(1)
+1 state_piecewise(6)_pw_t05_CC_bin_y(2)
= 1

c_e_state_piecewise(6)_pw_t06_CC_constraint1_:
+1 state(reach2_storage_t06)
-250.0 state_piecewise(6)_pw_t06_CC_lambda(1)
-330.0 state_piecewise(6)_pw_t06_CC_lambda(2)
-410.0 state_piecewise(6)_pw_t06_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t06_CC_constraint2_:
+1 state(reach2_discharge_downstream_t06)
-2.6 state_piecewise(6)_pw_t06_CC_lambda(1)
-7.8 state_piecewise(6)_pw_t06_CC_lambda(2)
-10.8 state_piecewise(6)_pw_t06_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t06_CC_constraint3_:
+1 state_piecewise(6)_pw_t06_CC_lambda(1)
+1 state_piecewise(6)_pw_t06_CC_lambda(2)
+1 state_piecewise(6)_pw_t06_CC_lambda(3)
= 1

c_u_state_piecewise(6)_pw_t06_CC_constraint4(1)_:
+1 state_piecewise(6)_pw_t06_CC_lambda(1)
-1 state_piecewise(6)_pw_t06_CC_bin_y(1)
<= 0

c_u_state_piecewise(6)_pw_t06_CC_constraint4(2)_:
+1 state_piecewise(6)_pw_t06_CC_lambda(2)
-1 state_piecewise(6)_pw_t06_CC_bin_y(1)
-1 state_piecewise(6)_pw_t06_CC_bin_y(2)
<= 0

c_u_state_piecewise(6)_pw_t06_CC_constraint4(3)_:
+1 state_piecewise(6)_pw_t06_CC_lambda(3)
-1 state_piecewise(6)_pw_t06_CC_bin_y(2)
<= 0

c_e_state_piecewise(6)_pw_t06_CC_constraint5_:
+1 state_piecewise(6)_pw_t06_CC_bin_y(1)
+1 state_piecewise(6)_pw_t06_CC_bin_y(2)
= 1

c_e_state_piecewise(6)_pw_t07_CC_constraint1_:
+1 state(reach2_storage_t07)
-250.0 state_piecewise(6)_pw_t07_CC_lambda(1)
-330.0 state_piecewise(6)_pw_t07_CC_lambda(2)
-410.0 state_piecewise(6)_pw_t07_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t07_CC_constraint2_:
+1 state(reach2_discharge_downstream_t07)
-2.6 state_piecewise(6)_pw_t07_CC_lambda(1)
-7.8 state_piecewise(6)_pw_t07_CC_lambda(2)
-10.8 state_piecewise(6)_pw_t07_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t07_CC_constraint3_:
+1 state_piecewise(6)_pw_t07_CC_lambda(1)
+1 state_piecewise(6)_pw_t07_CC_lambda(2)
+1 state_piecewise(6)_pw_t07_CC_lambda(3)
= 1

c_u_state_piecewise(6)_pw_t07_CC_constraint4(1)_:
+1 state_piecewise(6)_pw_t07_CC_lambda(1)
-1 state_piecewise(6)_pw_t07_CC_bin_y(1)
<= 0

c_u_state_piecewise(6)_pw_t07_CC_constraint4(2)_:
+1 state_piecewise(6)_pw_t07_CC_lambda(2)
-1 state_piecewise(6)_pw_t07_CC_bin_y(1)
-1 state_piecewise(6)_pw_t07_CC_bin_y(2)
<= 0

c_u_state_piecewise(6)_pw_t07_CC_constraint4(3)_:
+1 state_piecewise(6)_pw_t07_CC_lambda(3)
-1 state_piecewise(6)_pw_t07_CC_bin_y(2)
<= 0

c_e_state_piecewise(6)_pw_t07_CC_constraint5_:
+1 state_piecewise(6)_pw_t07_CC_bin_y(1)
+1 state_piecewise(6)_pw_t07_CC_bin_y(2)
= 1

c_e_state_piecewise(6)_pw_t08_CC_constraint1_:
+1 state(reach2_storage_t08)
-250.0 state_piecewise(6)_pw_t08_CC_lambda(1)
-330.0 state_piecewise(6)_pw_t08_CC_lambda(2)
-410.0 state_piecewise(6)_pw_t08_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t08_CC_constraint2_:
+1 state(reach2_discharge_downstream_t08)
-2.6 state_piecewise(6)_pw_t08_CC_lambda(1)
-7.8 state_piecewise(6)_pw_t08_CC_lambda(2)
-10.8 state_piecewise(6)_pw_t08_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t08_CC_constraint3_:
+1 state_piecewise(6)_pw_t08_CC_lambda(1)
+1 state_piecewise(6)_pw_t08_CC_lambda(2)
+1 state_piecewise(6)_pw_t08_CC_lambda(3)
= 1

c_u_state_piecewise(6)_pw_t08_CC_constraint4(1)_:
+1 state_piecewise(6)_pw_t08_CC_lambda(1)
-1 state_piecewise(6)_pw_t08_CC_bin_y(1)
<= 0

c_u_state_piecewise(6)_pw_t08_CC_constraint4(2)_:
+1 state_piecewise(6)_pw_t08_CC_lambda(2)
-1 state_piecewise(6)_pw_t08_CC_bin_y(1)
-1 state_piecewise(6)_pw_t08_CC_bin_y(2)
<= 0

c_u_state_piecewise(6)_pw_t08_CC_constraint4(3)_:
+1 state_piecewise(6)_pw_t08_CC_lambda(3)
-1 state_piecewise(6)_pw_t08_CC_bin_y(2)
<= 0

c_e_state_piecewise(6)_pw_t08_CC_constraint5_:
+1 state_piecewise(6)_pw_t08_CC_bin_y(1)
+1 state_piecewise(6)_pw_t08_CC_bin_y(2)
= 1

c_e_state_piecewise(6)_pw_t09_CC_constraint1_:
+1 state(reach2_storage_t09)
-250.0 state_piecewise(6)_pw_t09_CC_lambda(1)
-330.0 state_piecewise(6)_pw_t09_CC_lambda(2)
-410.0 state_piecewise(6)_pw_t09_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t09_CC_constraint2_:
+1 state(reach2_discharge_downstream_t09)
-2.6 state_piecewise(6)_pw_t09_CC_lambda(1)
-7.8 state_piecewise(6)_pw_t09_CC_lambda(2)
-10.8 state_piecewise(6)_pw_t09_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t09_CC_constraint3_:
+1 state_piecewise(6)_pw_t09_CC_lambda(1)
+1 state_piecewise(6)_pw_t09_CC_lambda(2)
+1 state_piecewise(6)_pw_t09_CC_lambda(3)
= 1

c_u_state_piecewise(6)_pw_t09_CC_constraint4(1)_:
+1 state_piecewise(6)_pw_t09_CC_lambda(1)
-1 state_piecewise(6)_pw_t09_CC_bin_y(1)
<= 0

c_u_state_piecewise(6)_pw_t09_CC_constraint4(2)_:
+1 state_piecewise(6)_pw_t09_CC_lambda(2)
-1 state_piecewise(6)_pw_t09_CC_bin_y(1)
-1 state_piecewise(6)_pw_t09_CC_bin_y(2)
<= 0

c_u_state_piecewise(6)_pw_t09_CC_constraint4(3)_:
+1 state_piecewise(6)_pw_t09_CC_lambda(3)
-1 state_piecewise(6)_pw_t09_CC_bin_y(2)
<= 0

c_e_state_piecewise(6)_pw_t09_CC_constraint5_:
+1 state_piecewise(6)_pw_t09_CC_bin_y(1)
+1 state_piecewise(6)_pw_t09_CC_bin_y(2)
= 1

c_e_state_piecewise(6)_pw_t10_CC_constraint1_:
+1 state(reach2_storage_t10)
-250.0 state_piecewise(6)_pw_t10_CC_lambda(1)
-330.0 state_piecewise(6)_pw_t10_CC_lambda(2)
-410.0 state_piecewise(6)_pw_t10_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t10_CC_constraint2_:
+1 state(reach2_discharge_downstream_t10)
-2.6 state_piecewise(6)_pw_t10_CC_lambda(1)
-7.8 state_piecewise(6)_pw_t10_CC_lambda(2)
-10.8 state_piecewise(6)_pw_t10_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t10_CC_constraint3_:
+1 state_piecewise(6)_pw_t10_CC_lambda(1)
+1 state_piecewise(6)_pw_t10_CC_lambda(2)
+1 state_piecewise(6)_pw_t10_CC_lambda(3)
= 1

c_u_state_piecewise(6)_pw_t10_CC_constraint4(1)_:
+1 state_piecewise(6)_pw_t10_CC_lambda(1)
-1 state_piecewise(6)_pw_t10_CC_bin_y(1)
<= 0

c_u_state_piecewise(6)_pw_t10_CC_constraint4(2)_:
+1 state_piecewise(6)_pw_t10_CC_lambda(2)
-1 state_piecewise(6)_pw_t10_CC_bin_y(1)
-1 state_piecewise(6)_pw_t10_CC_bin_y(2)
<= 0

c_u_state_piecewise(6)_pw_t10_CC_constraint4(3)_:
+1 state_piecewise(6)_pw_t10_CC_lambda(3)
-1 state_piecewise(6)_pw_t10_CC_bin_y(2)
<= 0

c_e_state_piecewise(6)_pw_t10_CC_constraint5_:
+1 state_piecewise(6)_pw_t10_CC_bin_y(1)
+1 state_piecewise(6)_pw_t10_CC_bin_y(2)
= 1

c_e_state_piecewise(6)_pw_t11_CC_constraint1_:
+1 state(reach2_storage_t11)
-250.0 state_piecewise(6)_pw_t11_CC_lambda(1)
-330.0 state_piecewise(6)_pw_t11_CC_lambda(2)
-410.0 state_piecewise(6)_pw_t11_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t11_CC_constraint2_:
+1 state(reach2_discharge_downstream_t11)
-2.6 state_piecewise(6)_pw_t11_CC_lambda(1)
-7.8 state_piecewise(6)_pw_t11_CC_lambda(2)
-10.8 state_piecewise(6)_pw_t11_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t11_CC_constraint3_:
+1 state_piecewise(6)_pw_t11_CC_lambda(1)
+1 state_piecewise(6)_pw_t11_CC_lambda(2)
+1 state_piecewise(6)_pw_t11_CC_lambda(3)
= 1

c_u_state_piecewise(6)_pw_t11_CC_constraint4(1)_:
+1 state_piecewise(6)_pw_t11_CC_lambda(1)
-1 state_piecewise(6)_pw_t11_CC_bin_y(1)
<= 0

c_u_state_piecewise(6)_pw_t11_CC_constraint4(2)_:
+1 state_piecewise(6)_pw_t11_CC_lambda(2)
-1 state_piecewise(6)_pw_t11_CC_bin_y(1)
-1 state_piecewise(6)_pw_t11_CC_bin_y(2)
<= 0

c_u_state_piecewise(6)_pw_t11_CC_constraint4(3)_:
+1 state_piecewise(6)_pw_t11_CC_lambda(3)
-1 state_piecewise(6)_pw_t11_CC_bin_y(2)
<= 0

c_e_state_piecewise(6)_pw_t11_CC_constraint5_:
+1 state_piecewise(6)_pw_t11_CC_bin_y(1)
+1 state_piecewise(6)_pw_t11_CC_bin_y(2)
= 1

c_e_state_piecewise(6)_pw_t12_CC_constraint1_:
+1 state(reach2_storage_t12)
-250.0 state_piecewise(6)_pw_t12_CC_lambda(1)
-330.0 state_piecewise(6)_pw_t12_CC_lambda(2)
-410.0 state_piecewise(6)_pw_t12_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t12_CC_constraint2_:
+1 state(reach2_discharge_downstream_t12)
-2.6 state_piecewise(6)_pw_t12_CC_lambda(1)
-7.8 state_piecewise(6)_pw_t12_CC_lambda(2)
-10.8 state_piecewise(6)_pw_t12_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t12_CC_constraint3_:
+1 state_piecewise(6)_pw_t12_CC_lambda(1)
+1 state_piecewise(6)_pw_t12_CC_lambda(2)
+1 state_piecewise(6)_pw_t12_CC_lambda(3)
= 1

c_u_state_piecewise(6)_pw_t12_CC_constraint4(1)_:
+1 state_piecewise(6)_pw_t12_CC_lambda(1)
-1 state_piecewise(6)_pw_t12_CC_bin_y(1)
<= 0

c_u_state_piecewise(6)_pw_t12_CC_constraint4(2)_:
+1 state_piecewise(6)_pw_t12_CC_lambda(2)
-1 state_piecewise(6)_pw_t12_CC_bin_y(1)
-1 state_piecewise(6)_pw_t12_CC_bin_y(2)
<= 0

c_u_state_piecewise(6)_pw_t12_CC_constraint4(3)_:
+1 state_piecewise(6)_pw_t12_CC_lambda(3)
-1 state_piecewise(6)_pw_t12_CC_bin_y(2)
<= 0

c_e_state_piecewise(6)_pw_t12_CC_constraint5_:
+1 state_piecewise(6)_pw_t12_CC_bin_y(1)
+1 state_piecewise(6)_pw_t12_CC_bin_y(2)
= 1

c_e_state_piecewise(6)_pw_t13_CC_constraint1_:
+1 state(reach2_storage_t13)
-250.0 state_piecewise(6)_pw_t13_CC_lambda(1)
-330.0 state_piecewise(6)_pw_t13_CC_lambda(2)
-410.0 state_piecewise(6)_pw_t13_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t13_CC_constraint2_:
+1 state(reach2_discharge_downstream_t13)
-2.6 state_piecewise(6)_pw_t13_CC_lambda(1)
-7.8 state_piecewise(6)_pw_t13_CC_lambda(2)
-10.8 state_piecewise(6)_pw_t13_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t13_CC_constraint3_:
+1 state_piecewise(6)_pw_t13_CC_lambda(1)
+1 state_piecewise(6)_pw_t13_CC_lambda(2)
+1 state_piecewise(6)_pw_t13_CC_lambda(3)
= 1

c_u_state_piecewise(6)_pw_t13_CC_constraint4(1)_:
+1 state_piecewise(6)_pw_t13_CC_lambda(1)
-1 state_piecewise(6)_pw_t13_CC_bin_y(1)
<= 0

c_u_state_piecewise(6)_pw_t13_CC_constraint4(2)_:
+1 state_piecewise(6)_pw_t13_CC_lambda(2)
-1 state_piecewise(6)_pw_t13_CC_bin_y(1)
-1 state_piecewise(6)_pw_t13_CC_bin_y(2)
<= 0

c_u_state_piecewise(6)_pw_t13_CC_constraint4(3)_:
+1 state_piecewise(6)_pw_t13_CC_lambda(3)
-1 state_piecewise(6)_pw_t13_CC_bin_y(2)
<= 0

c_e_state_piecewise(6)_pw_t13_CC_constraint5_:
+1 state_piecewise(6)_pw_t13_CC_bin_y(1)
+1 state_piecewise(6)_pw_t13_CC_bin_y(2)
= 1

c_e_state_piecewise(6)_pw_t14_CC_constraint1_:
+1 state(reach2_storage_t14)
-250.0 state_piecewise(6)_pw_t14_CC_lambda(1)
-330.0 state_piecewise(6)_pw_t14_CC_lambda(2)
-410.0 state_piecewise(6)_pw_t14_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t14_CC_constraint2_:
+1 state(reach2_discharge_downstream_t14)
-2.6 state_piecewise(6)_pw_t14_CC_lambda(1)
-7.8 state_piecewise(6)_pw_t14_CC_lambda(2)
-10.8 state_piecewise(6)_pw_t14_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t14_CC_constraint3_:
+1 state_piecewise(6)_pw_t14_CC_lambda(1)
+1 state_piecewise(6)_pw_t14_CC_lambda(2)
+1 state_piecewise(6)_pw_t14_CC_lambda(3)
= 1

c_u_state_piecewise(6)_pw_t14_CC_constraint4(1)_:
+1 state_piecewise(6)_pw_t14_CC_lambda(1)
-1 state_piecewise(6)_pw_t14_CC_bin_y(1)
<= 0

c_u_state_piecewise(6)_pw_t14_CC_constraint4(2)_:
+1 state_piecewise(6)_pw_t14_CC_lambda(2)
-1 state_piecewise(6)_pw_t14_CC_bin_y(1)
-1 state_piecewise(6)_pw_t14_CC_bin_y(2)
<= 0

c_u_state_piecewise(6)_pw_t14_CC_constraint4(3)_:
+1 state_piecewise(6)_pw_t14_CC_lambda(3)
-1 state_piecewise(6)_pw_t14_CC_bin_y(2)
<= 0

c_e_state_piecewise(6)_pw_t14_CC_constraint5_:
+1 state_piecewise(6)_pw_t14_CC_bin_y(1)
+1 state_piecewise(6)_pw_t14_CC_bin_y(2)
= 1

c_e_state_piecewise(6)_pw_t15_CC_constraint1_:
+1 state(reach2_storage_t15)
-250.0 state_piecewise(6)_pw_t15_CC_lambda(1)
-330.0 state_piecewise(6)_pw_t15_CC_lambda(2)
-410.0 state_piecewise(6)_pw_t15_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t15_CC_constraint2_:
+1 state(reach2_discharge_downstream_t15)
-2.6 state_piecewise(6)_pw_t15_CC_lambda(1)
-7.8 state_piecewise(6)_pw_t15_CC_lambda(2)
-10.8 state_piecewise(6)_pw_t15_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t15_CC_constraint3_:
+1 state_piecewise(6)_pw_t15_CC_lambda(1)
+1 state_piecewise(6)_pw_t15_CC_lambda(2)
+1 state_piecewise(6)_pw_t15_CC_lambda(3)
= 1

c_u_state_piecewise(6)_pw_t15_CC_constraint4(1)_:
+1 state_piecewise(6)_pw_t15_CC_lambda(1)
-1 state_piecewise(6)_pw_t15_CC_bin_y(1)
<= 0

c_u_state_piecewise(6)_pw_t15_CC_constraint4(2)_:
+1 state_piecewise(6)_pw_t15_CC_lambda(2)
-1 state_piecewise(6)_pw_t15_CC_bin_y(1)
-1 state_piecewise(6)_pw_t15_CC_bin_y(2)
<= 0

c_u_state_piecewise(6)_pw_t15_CC_constraint4(3)_:
+1 state_piecewise(6)_pw_t15_CC_lambda(3)
-1 state_piecewise(6)_pw_t15_CC_bin_y(2)
<= 0

c_e_state_piecewise(6)_pw_t15_CC_constraint5_:
+1 state_piecewise(6)_pw_t15_CC_bin_y(1)
+1 state_piecewise(6)_pw_t15_CC_bin_y(2)
= 1

c_e_state_piecewise(6)_pw_t16_CC_constraint1_:
+1 state(reach2_storage_t16)
-250.0 state_piecewise(6)_pw_t16_CC_lambda(1)
-330.0 state_piecewise(6)_pw_t16_CC_lambda(2)
-410.0 state_piecewise(6)_pw_t16_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t16_CC_constraint2_:
+1 state(reach2_discharge_downstream_t16)
-2.6 state_piecewise(6)_pw_t16_CC_lambda(1)
-7.8 state_piecewise(6)_pw_t16_CC_lambda(2)
-10.8 state_piecewise(6)_pw_t16_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t16_CC_constraint3_:
+1 state_piecewise(6)_pw_t16_CC_lambda(1)
+1 state_piecewise(6)_pw_t16_CC_lambda(2)
+1 state_piecewise(6)_pw_t16_CC_lambda(3)
= 1

c_u_state_piecewise(6)_pw_t16_CC_constraint4(1)_:
+1 state_piecewise(6)_pw_t16_CC_lambda(1)
-1 state_piecewise(6)_pw_t16_CC_bin_y(1)
<= 0

c_u_state_piecewise(6)_pw_t16_CC_constraint4(2)_:
+1 state_piecewise(6)_pw_t16_CC_lambda(2)
-1 state_piecewise(6)_pw_t16_CC_bin_y(1)
-1 state_piecewise(6)_pw_t16_CC_bin_y(2)
<= 0

c_u_state_piecewise(6)_pw_t16_CC_constraint4(3)_:
+1 state_piecewise(6)_pw_t16_CC_lambda(3)
-1 state_piecewise(6)_pw_t16_CC_bin_y(2)
<= 0

c_e_state_piecewise(6)_pw_t16_CC_constraint5_:
+1 state_piecewise(6)_pw_t16_CC_bin_y(1)
+1 state_piecewise(6)_pw_t16_CC_bin_y(2)
= 1

c_e_state_piecewise(6)_pw_t17_CC_constraint1_:
+1 state(reach2_storage_t17)
-250.0 state_piecewise(6)_pw_t17_CC_lambda(1)
-330.0 state_piecewise(6)_pw_t17_CC_lambda(2)
-410.0 state_piecewise(6)_pw_t17_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t17_CC_constraint2_:
+1 state(reach2_discharge_downstream_t17)
-2.6 state_piecewise(6)_pw_t17_CC_lambda(1)
-7.8 state_piecewise(6)_pw_t17_CC_lambda(2)
-10.8 state_piecewise(6)_pw_t17_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t17_CC_constraint3_:
+1 state_piecewise(6)_pw_t17_CC_lambda(1)
+1 state_piecewise(6)_pw_t17_CC_lambda(2)
+1 state_piecewise(6)_pw_t17_CC_lambda(3)
= 1

c_u_state_piecewise(6)_pw_t17_CC_constraint4(1)_:
+1 state_piecewise(6)_pw_t17_CC_lambda(1)
-1 state_piecewise(6)_pw_t17_CC_bin_y(1)
<= 0

c_u_state_piecewise(6)_pw_t17_CC_constraint4(2)_:
+1 state_piecewise(6)_pw_t17_CC_lambda(2)
-1 state_piecewise(6)_pw_t17_CC_bin_y(1)
-1 state_piecewise(6)_pw_t17_CC_bin_y(2)
<= 0

c_u_state_piecewise(6)_pw_t17_CC_constraint4(3)_:
+1 state_piecewise(6)_pw_t17_CC_lambda(3)
-1 state_piecewise(6)_pw_t17_CC_bin_y(2)
<= 0

c_e_state_piecewise(6)_pw_t17_CC_constraint5_:
+1 state_piecewise(6)_pw_t17_CC_bin_y(1)
+1 state_piecewise(6)_pw_t17_CC_bin_y(2)
= 1

c_e_state_piecewise(6)_pw_t18_CC_constraint1_:
+1 state(reach2_storage_t18)
-250.0 state_piecewise(6)_pw_t18_CC_lambda(1)
-330.0 state_piecewise(6)_pw_t18_CC_lambda(2)
-410.0 state_piecewise(6)_pw_t18_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t18_CC_constraint2_:
+1 state(reach2_discharge_downstream_t18)
-2.6 state_piecewise(6)_pw_t18_CC_lambda(1)
-7.8 state_piecewise(6)_pw_t18_CC_lambda(2)
-10.8 state_piecewise(6)_pw_t18_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t18_CC_constraint3_:
+1 state_piecewise(6)_pw_t18_CC_lambda(1)
+1 state_piecewise(6)_pw_t18_CC_lambda(2)
+1 state_piecewise(6)_pw_t18_CC_lambda(3)
= 1

c_u_state_piecewise(6)_pw_t18_CC_constraint4(1)_:
+1 state_piecewise(6)_pw_t18_CC_lambda(1)
-1 state_piecewise(6)_pw_t18_CC_bin_y(1)
<= 0

c_u_state_piecewise(6)_pw_t18_CC_constraint4(2)_:
+1 state_piecewise(6)_pw_t18_CC_lambda(2)
-1 state_piecewise(6)_pw_t18_CC_bin_y(1)
-1 state_piecewise(6)_pw_t18_CC_bin_y(2)
<= 0

c_u_state_piecewise(6)_pw_t18_CC_constraint4(3)_:
+1 state_piecewise(6)_pw_t18_CC_lambda(3)
-1 state_piecewise(6)_pw_t18_CC_bin_y(2)
<= 0

c_e_state_piecewise(6)_pw_t18_CC_constraint5_:
+1 state_piecewise(6)_pw_t18_CC_bin_y(1)
+1 state_piecewise(6)_pw_t18_CC_bin_y(2)
= 1

c_e_state_piecewise(6)_pw_t19_CC_constraint1_:
+1 state(reach2_storage_t19)
-250.0 state_piecewise(6)_pw_t19_CC_lambda(1)
-330.0 state_piecewise(6)_pw_t19_CC_lambda(2)
-410.0 state_piecewise(6)_pw_t19_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t19_CC_constraint2_:
+1 state(reach2_discharge_downstream_t19)
-2.6 state_piecewise(6)_pw_t19_CC_lambda(1)
-7.8 state_piecewise(6)_pw_t19_CC_lambda(2)
-10.8 state_piecewise(6)_pw_t19_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t19_CC_constraint3_:
+1 state_piecewise(6)_pw_t19_CC_lambda(1)
+1 state_piecewise(6)_pw_t19_CC_lambda(2)
+1 state_piecewise(6)_pw_t19_CC_lambda(3)
= 1

c_u_state_piecewise(6)_pw_t19_CC_constraint4(1)_:
+1 state_piecewise(6)_pw_t19_CC_lambda(1)
-1 state_piecewise(6)_pw_t19_CC_bin_y(1)
<= 0

c_u_state_piecewise(6)_pw_t19_CC_constraint4(2)_:
+1 state_piecewise(6)_pw_t19_CC_lambda(2)
-1 state_piecewise(6)_pw_t19_CC_bin_y(1)
-1 state_piecewise(6)_pw_t19_CC_bin_y(2)
<= 0

c_u_state_piecewise(6)_pw_t19_CC_constraint4(3)_:
+1 state_piecewise(6)_pw_t19_CC_lambda(3)
-1 state_piecewise(6)_pw_t19_CC_bin_y(2)
<= 0

c_e_state_piecewise(6)_pw_t19_CC_constraint5_:
+1 state_piecewise(6)_pw_t19_CC_bin_y(1)
+1 state_piecewise(6)_pw_t19_CC_bin_y(2)
= 1

c_e_state_piecewise(6)_pw_t20_CC_constraint1_:
+1 state(reach2_storage_t20)
-250.0 state_piecewise(6)_pw_t20_CC_lambda(1)
-330.0 state_piecewise(6)_pw_t20_CC_lambda(2)
-410.0 state_piecewise(6)_pw_t20_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t20_CC_constraint2_:
+1 state(reach2_discharge_downstream_t20)
-2.6 state_piecewise(6)_pw_t20_CC_lambda(1)
-7.8 state_piecewise(6)_pw_t20_CC_lambda(2)
-10.8 state_piecewise(6)_pw_t20_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t20_CC_constraint3_:
+1 state_piecewise(6)_pw_t20_CC_lambda(1)
+1 state_piecewise(6)_pw_t20_CC_lambda(2)
+1 state_piecewise(6)_pw_t20_CC_lambda(3)
= 1

c_u_state_piecewise(6)_pw_t20_CC_constraint4(1)_:
+1 state_piecewise(6)_pw_t20_CC_lambda(1)
-1 state_piecewise(6)_pw_t20_CC_bin_y(1)
<= 0

c_u_state_piecewise(6)_pw_t20_CC_constraint4(2)_:
+1 state_piecewise(6)_pw_t20_CC_lambda(2)
-1 state_piecewise(6)_pw_t20_CC_bin_y(1)
-1 state_piecewise(6)_pw_t20_CC_bin_y(2)
<= 0

c_u_state_piecewise(6)_pw_t20_CC_constraint4(3)_:
+1 state_piecewise(6)_pw_t20_CC_lambda(3)
-1 state_piecewise(6)_pw_t20_CC_bin_y(2)
<= 0

c_e_state_piecewise(6)_pw_t20_CC_constraint5_:
+1 state_piecewise(6)_pw_t20_CC_bin_y(1)
+1 state_piecewise(6)_pw_t20_CC_bin_y(2)
= 1

c_e_state_piecewise(6)_pw_t21_CC_constraint1_:
+1 state(reach2_storage_t21)
-250.0 state_piecewise(6)_pw_t21_CC_lambda(1)
-330.0 state_piecewise(6)_pw_t21_CC_lambda(2)
-410.0 state_piecewise(6)_pw_t21_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t21_CC_constraint2_:
+1 state(reach2_discharge_downstream_t21)
-2.6 state_piecewise(6)_pw_t21_CC_lambda(1)
-7.8 state_piecewise(6)_pw_t21_CC_lambda(2)
-10.8 state_piecewise(6)_pw_t21_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t21_CC_constraint3_:
+1 state_piecewise(6)_pw_t21_CC_lambda(1)
+1 state_piecewise(6)_pw_t21_CC_lambda(2)
+1 state_piecewise(6)_pw_t21_CC_lambda(3)
= 1

c_u_state_piecewise(6)_pw_t21_CC_constraint4(1)_:
+1 state_piecewise(6)_pw_t21_CC_lambda(1)
-1 state_piecewise(6)_pw_t21_CC_bin_y(1)
<= 0

c_u_state_piecewise(6)_pw_t21_CC_constraint4(2)_:
+1 state_piecewise(6)_pw_t21_CC_lambda(2)
-1 state_piecewise(6)_pw_t21_CC_bin_y(1)
-1 state_piecewise(6)_pw_t21_CC_bin_y(2)
<= 0

c_u_state_piecewise(6)_pw_t21_CC_constraint4(3)_:
+1 state_piecewise(6)_pw_t21_CC_lambda(3)
-1 state_piecewise(6)_pw_t21_CC_bin_y(2)
<= 0

c_e_state_piecewise(6)_pw_t21_CC_constraint5_:
+1 state_piecewise(6)_pw_t21_CC_bin_y(1)
+1 state_piecewise(6)_pw_t21_CC_bin_y(2)
= 1

c_e_state_piecewise(6)_pw_t22_CC_constraint1_:
+1 state(reach2_storage_t22)
-250.0 state_piecewise(6)_pw_t22_CC_lambda(1)
-330.0 state_piecewise(6)_pw_t22_CC_lambda(2)
-410.0 state_piecewise(6)_pw_t22_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t22_CC_constraint2_:
+1 state(reach2_discharge_downstream_t22)
-2.6 state_piecewise(6)_pw_t22_CC_lambda(1)
-7.8 state_piecewise(6)_pw_t22_CC_lambda(2)
-10.8 state_piecewise(6)_pw_t22_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t22_CC_constraint3_:
+1 state_piecewise(6)_pw_t22_CC_lambda(1)
+1 state_piecewise(6)_pw_t22_CC_lambda(2)
+1 state_piecewise(6)_pw_t22_CC_lambda(3)
= 1

c_u_state_piecewise(6)_pw_t22_CC_constraint4(1)_:
+1 state_piecewise(6)_pw_t22_CC_lambda(1)
-1 state_piecewise(6)_pw_t22_CC_bin_y(1)
<= 0

c_u_state_piecewise(6)_pw_t22_CC_constraint4(2)_:
+1 state_piecewise(6)_pw_t22_CC_lambda(2)
-1 state_piecewise(6)_pw_t22_CC_bin_y(1)
-1 state_piecewise(6)_pw_t22_CC_bin_y(2)
<= 0

c_u_state_piecewise(6)_pw_t22_CC_constraint4(3)_:
+1 state_piecewise(6)_pw_t22_CC_lambda(3)
-1 state_piecewise(6)_pw_t22_CC_bin_y(2)
<= 0

c_e_state_piecewise(6)_pw_t22_CC_constraint5_:
+1 state_piecewise(6)_pw_t22_CC_bin_y(1)
+1 state_piecewise(6)_pw_t22_CC_bin_y(2)
= 1

c_e_state_piecewise(6)_pw_t23_CC_constraint1_:
+1 state(reach2_storage_t23)
-250.0 state_piecewise(6)_pw_t23_CC_lambda(1)
-330.0 state_piecewise(6)_pw_t23_CC_lambda(2)
-410.0 state_piecewise(6)_pw_t23_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t23_CC_constraint2_:
+1 state(reach2_discharge_downstream_t23)
-2.6 state_piecewise(6)_pw_t23_CC_lambda(1)
-7.8 state_piecewise(6)_pw_t23_CC_lambda(2)
-10.8 state_piecewise(6)_pw_t23_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t23_CC_constraint3_:
+1 state_piecewise(6)_pw_t23_CC_lambda(1)
+1 state_piecewise(6)_pw_t23_CC_lambda(2)
+1 state_piecewise(6)_pw_t23_CC_lambda(3)
= 1

c_u_state_piecewise(6)_pw_t23_CC_constraint4(1)_:
+1 state_piecewise(6)_pw_t23_CC_lambda(1)
-1 state_piecewise(6)_pw_t23_CC_bin_y(1)
<= 0

c_u_state_piecewise(6)_pw_t23_CC_constraint4(2)_:
+1 state_piecewise(6)_pw_t23_CC_lambda(2)
-1 state_piecewise(6)_pw_t23_CC_bin_y(1)
-1 state_piecewise(6)_pw_t23_CC_bin_y(2)
<= 0

c_u_state_piecewise(6)_pw_t23_CC_constraint4(3)_:
+1 state_piecewise(6)_pw_t23_CC_lambda(3)
-1 state_piecewise(6)_pw_t23_CC_bin_y(2)
<= 0

c_e_state_piecewise(6)_pw_t23_CC_constraint5_:
+1 state_piecewise(6)_pw_t23_CC_bin_y(1)
+1 state_piecewise(6)_pw_t23_CC_bin_y(2)
= 1

c_e_state_piecewise(6)_pw_t24_CC_constraint1_:
+1 state(reach2_storage_t24)
-250.0 state_piecewise(6)_pw_t24_CC_lambda(1)
-330.0 state_piecewise(6)_pw_t24_CC_lambda(2)
-410.0 state_piecewise(6)_pw_t24_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t24_CC_constraint2_:
+1 state(reach2_discharge_downstream_t24)
-2.6 state_piecewise(6)_pw_t24_CC_lambda(1)
-7.8 state_piecewise(6)_pw_t24_CC_lambda(2)
-10.8 state_piecewise(6)_pw_t24_CC_lambda(3)
= 0

c_e_state_piecewise(6)_pw_t24_CC_constraint3_:
+1 state_piecewise(6)_pw_t24_CC_lambda(1)
+1 state_piecewise(6)_pw_t24_CC_lambda(2)
+1 state_piecewise(6)_pw_t24_CC_lambda(3)
= 1

c_u_state_piecewise(6)_pw_t24_CC_constraint4(1)_:
+1 state_piecewise(6)_pw_t24_CC_lambda(1)
-1 state_piecewise(6)_pw_t24_CC_bin_y(1)
<= 0

c_u_state_piecewise(6)_pw_t24_CC_constraint4(2)_:
+1 state_piecewise(6)_pw_t24_CC_lambda(2)
-1 state_piecewise(6)_pw_t24_CC_bin_y(1)
-1 state_piecewise(6)_pw_t24_CC_bin_y(2)
<= 0

c_u_state_piecewise(6)_pw_t24_CC_constraint4(3)_:
+1 state_piecewise(6)_pw_t24_CC_lambda(3)
-1 state_piecewise(6)_pw_t24_CC_bin_y(2)
<= 0

c_e_state_piecewise(6)_pw_t24_CC_constraint5_:
+1 state_piecewise(6)_pw_t24_CC_bin_y(1)
+1 state_piecewise(6)_pw_t24_CC_bin_y(2)
= 1

c_e_gate_piecewise(gate1_t01)_CC_constraint1_:
+1 gate_opening(gate1_t01)
-0.25 gate_piecewise(gate1_t01)_CC_lambda(2)
-0.5 gate_piecewise(gate1_t01)_CC_lambda(3)
-0.75 gate_piecewise(gate1_t01)_CC_lambda(4)
-1 gate_piecewise(gate1_t01)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t01)_CC_constraint2_:
+1 gate_capacity(gate1_t01)
-6.0 gate_piecewise(gate1_t01)_CC_lambda(2)
-10.5 gate_piecewise(gate1_t01)_CC_lambda(3)
-13.5 gate_piecewise(gate1_t01)_CC_lambda(4)
-18.0 gate_piecewise(gate1_t01)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t01)_CC_constraint3_:
+1 gate_piecewise(gate1_t01)_CC_lambda(1)
+1 gate_piecewise(gate1_t01)_CC_lambda(2)
+1 gate_piecewise(gate1_t01)_CC_lambda(3)
+1 gate_piecewise(gate1_t01)_CC_lambda(4)
+1 gate_piecewise(gate1_t01)_CC_lambda(5)
= 1

c_u_gate_piecewise(gate1_t01)_CC_constraint4(1)_:
+1 gate_piecewise(gate1_t01)_CC_lambda(1)
-1 gate_piecewise(gate1_t01)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate1_t01)_CC_constraint4(2)_:
+1 gate_piecewise(gate1_t01)_CC_lambda(2)
-1 gate_piecewise(gate1_t01)_CC_bin_y(1)
-1 gate_piecewise(gate1_t01)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate1_t01)_CC_constraint4(3)_:
+1 gate_piecewise(gate1_t01)_CC_lambda(3)
-1 gate_piecewise(gate1_t01)_CC_bin_y(2)
-1 gate_piecewise(gate1_t01)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate1_t01)_CC_constraint4(4)_:
+1 gate_piecewise(gate1_t01)_CC_lambda(4)
-1 gate_piecewise(gate1_t01)_CC_bin_y(3)
-1 gate_piecewise(gate1_t01)_CC_bin_y(4)
<= 0

c_u_gate_piecewise(gate1_t01)_CC_constraint4(5)_:
+1 gate_piecewise(gate1_t01)_CC_lambda(5)
-1 gate_piecewise(gate1_t01)_CC_bin_y(4)
<= 0

c_e_gate_piecewise(gate1_t01)_CC_constraint5_:
+1 gate_piecewise(gate1_t01)_CC_bin_y(1)
+1 gate_piecewise(gate1_t01)_CC_bin_y(2)
+1 gate_piecewise(gate1_t01)_CC_bin_y(3)
+1 gate_piecewise(gate1_t01)_CC_bin_y(4)
= 1

c_e_gate_piecewise(gate1_t02)_CC_constraint1_:
+1 gate_opening(gate1_t02)
-0.25 gate_piecewise(gate1_t02)_CC_lambda(2)
-0.5 gate_piecewise(gate1_t02)_CC_lambda(3)
-0.75 gate_piecewise(gate1_t02)_CC_lambda(4)
-1 gate_piecewise(gate1_t02)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t02)_CC_constraint2_:
+1 gate_capacity(gate1_t02)
-6.0 gate_piecewise(gate1_t02)_CC_lambda(2)
-10.5 gate_piecewise(gate1_t02)_CC_lambda(3)
-13.5 gate_piecewise(gate1_t02)_CC_lambda(4)
-18.0 gate_piecewise(gate1_t02)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t02)_CC_constraint3_:
+1 gate_piecewise(gate1_t02)_CC_lambda(1)
+1 gate_piecewise(gate1_t02)_CC_lambda(2)
+1 gate_piecewise(gate1_t02)_CC_lambda(3)
+1 gate_piecewise(gate1_t02)_CC_lambda(4)
+1 gate_piecewise(gate1_t02)_CC_lambda(5)
= 1

c_u_gate_piecewise(gate1_t02)_CC_constraint4(1)_:
+1 gate_piecewise(gate1_t02)_CC_lambda(1)
-1 gate_piecewise(gate1_t02)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate1_t02)_CC_constraint4(2)_:
+1 gate_piecewise(gate1_t02)_CC_lambda(2)
-1 gate_piecewise(gate1_t02)_CC_bin_y(1)
-1 gate_piecewise(gate1_t02)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate1_t02)_CC_constraint4(3)_:
+1 gate_piecewise(gate1_t02)_CC_lambda(3)
-1 gate_piecewise(gate1_t02)_CC_bin_y(2)
-1 gate_piecewise(gate1_t02)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate1_t02)_CC_constraint4(4)_:
+1 gate_piecewise(gate1_t02)_CC_lambda(4)
-1 gate_piecewise(gate1_t02)_CC_bin_y(3)
-1 gate_piecewise(gate1_t02)_CC_bin_y(4)
<= 0

c_u_gate_piecewise(gate1_t02)_CC_constraint4(5)_:
+1 gate_piecewise(gate1_t02)_CC_lambda(5)
-1 gate_piecewise(gate1_t02)_CC_bin_y(4)
<= 0

c_e_gate_piecewise(gate1_t02)_CC_constraint5_:
+1 gate_piecewise(gate1_t02)_CC_bin_y(1)
+1 gate_piecewise(gate1_t02)_CC_bin_y(2)
+1 gate_piecewise(gate1_t02)_CC_bin_y(3)
+1 gate_piecewise(gate1_t02)_CC_bin_y(4)
= 1

c_e_gate_piecewise(gate1_t03)_CC_constraint1_:
+1 gate_opening(gate1_t03)
-0.25 gate_piecewise(gate1_t03)_CC_lambda(2)
-0.5 gate_piecewise(gate1_t03)_CC_lambda(3)
-0.75 gate_piecewise(gate1_t03)_CC_lambda(4)
-1 gate_piecewise(gate1_t03)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t03)_CC_constraint2_:
+1 gate_capacity(gate1_t03)
-6.0 gate_piecewise(gate1_t03)_CC_lambda(2)
-10.5 gate_piecewise(gate1_t03)_CC_lambda(3)
-13.5 gate_piecewise(gate1_t03)_CC_lambda(4)
-18.0 gate_piecewise(gate1_t03)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t03)_CC_constraint3_:
+1 gate_piecewise(gate1_t03)_CC_lambda(1)
+1 gate_piecewise(gate1_t03)_CC_lambda(2)
+1 gate_piecewise(gate1_t03)_CC_lambda(3)
+1 gate_piecewise(gate1_t03)_CC_lambda(4)
+1 gate_piecewise(gate1_t03)_CC_lambda(5)
= 1

c_u_gate_piecewise(gate1_t03)_CC_constraint4(1)_:
+1 gate_piecewise(gate1_t03)_CC_lambda(1)
-1 gate_piecewise(gate1_t03)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate1_t03)_CC_constraint4(2)_:
+1 gate_piecewise(gate1_t03)_CC_lambda(2)
-1 gate_piecewise(gate1_t03)_CC_bin_y(1)
-1 gate_piecewise(gate1_t03)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate1_t03)_CC_constraint4(3)_:
+1 gate_piecewise(gate1_t03)_CC_lambda(3)
-1 gate_piecewise(gate1_t03)_CC_bin_y(2)
-1 gate_piecewise(gate1_t03)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate1_t03)_CC_constraint4(4)_:
+1 gate_piecewise(gate1_t03)_CC_lambda(4)
-1 gate_piecewise(gate1_t03)_CC_bin_y(3)
-1 gate_piecewise(gate1_t03)_CC_bin_y(4)
<= 0

c_u_gate_piecewise(gate1_t03)_CC_constraint4(5)_:
+1 gate_piecewise(gate1_t03)_CC_lambda(5)
-1 gate_piecewise(gate1_t03)_CC_bin_y(4)
<= 0

c_e_gate_piecewise(gate1_t03)_CC_constraint5_:
+1 gate_piecewise(gate1_t03)_CC_bin_y(1)
+1 gate_piecewise(gate1_t03)_CC_bin_y(2)
+1 gate_piecewise(gate1_t03)_CC_bin_y(3)
+1 gate_piecewise(gate1_t03)_CC_bin_y(4)
= 1

c_e_gate_piecewise(gate1_t04)_CC_constraint1_:
+1 gate_opening(gate1_t04)
-0.25 gate_piecewise(gate1_t04)_CC_lambda(2)
-0.5 gate_piecewise(gate1_t04)_CC_lambda(3)
-0.75 gate_piecewise(gate1_t04)_CC_lambda(4)
-1 gate_piecewise(gate1_t04)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t04)_CC_constraint2_:
+1 gate_capacity(gate1_t04)
-6.0 gate_piecewise(gate1_t04)_CC_lambda(2)
-10.5 gate_piecewise(gate1_t04)_CC_lambda(3)
-13.5 gate_piecewise(gate1_t04)_CC_lambda(4)
-18.0 gate_piecewise(gate1_t04)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t04)_CC_constraint3_:
+1 gate_piecewise(gate1_t04)_CC_lambda(1)
+1 gate_piecewise(gate1_t04)_CC_lambda(2)
+1 gate_piecewise(gate1_t04)_CC_lambda(3)
+1 gate_piecewise(gate1_t04)_CC_lambda(4)
+1 gate_piecewise(gate1_t04)_CC_lambda(5)
= 1

c_u_gate_piecewise(gate1_t04)_CC_constraint4(1)_:
+1 gate_piecewise(gate1_t04)_CC_lambda(1)
-1 gate_piecewise(gate1_t04)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate1_t04)_CC_constraint4(2)_:
+1 gate_piecewise(gate1_t04)_CC_lambda(2)
-1 gate_piecewise(gate1_t04)_CC_bin_y(1)
-1 gate_piecewise(gate1_t04)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate1_t04)_CC_constraint4(3)_:
+1 gate_piecewise(gate1_t04)_CC_lambda(3)
-1 gate_piecewise(gate1_t04)_CC_bin_y(2)
-1 gate_piecewise(gate1_t04)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate1_t04)_CC_constraint4(4)_:
+1 gate_piecewise(gate1_t04)_CC_lambda(4)
-1 gate_piecewise(gate1_t04)_CC_bin_y(3)
-1 gate_piecewise(gate1_t04)_CC_bin_y(4)
<= 0

c_u_gate_piecewise(gate1_t04)_CC_constraint4(5)_:
+1 gate_piecewise(gate1_t04)_CC_lambda(5)
-1 gate_piecewise(gate1_t04)_CC_bin_y(4)
<= 0

c_e_gate_piecewise(gate1_t04)_CC_constraint5_:
+1 gate_piecewise(gate1_t04)_CC_bin_y(1)
+1 gate_piecewise(gate1_t04)_CC_bin_y(2)
+1 gate_piecewise(gate1_t04)_CC_bin_y(3)
+1 gate_piecewise(gate1_t04)_CC_bin_y(4)
= 1

c_e_gate_piecewise(gate1_t05)_CC_constraint1_:
+1 gate_opening(gate1_t05)
-0.25 gate_piecewise(gate1_t05)_CC_lambda(2)
-0.5 gate_piecewise(gate1_t05)_CC_lambda(3)
-0.75 gate_piecewise(gate1_t05)_CC_lambda(4)
-1 gate_piecewise(gate1_t05)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t05)_CC_constraint2_:
+1 gate_capacity(gate1_t05)
-6.0 gate_piecewise(gate1_t05)_CC_lambda(2)
-10.5 gate_piecewise(gate1_t05)_CC_lambda(3)
-13.5 gate_piecewise(gate1_t05)_CC_lambda(4)
-18.0 gate_piecewise(gate1_t05)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t05)_CC_constraint3_:
+1 gate_piecewise(gate1_t05)_CC_lambda(1)
+1 gate_piecewise(gate1_t05)_CC_lambda(2)
+1 gate_piecewise(gate1_t05)_CC_lambda(3)
+1 gate_piecewise(gate1_t05)_CC_lambda(4)
+1 gate_piecewise(gate1_t05)_CC_lambda(5)
= 1

c_u_gate_piecewise(gate1_t05)_CC_constraint4(1)_:
+1 gate_piecewise(gate1_t05)_CC_lambda(1)
-1 gate_piecewise(gate1_t05)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate1_t05)_CC_constraint4(2)_:
+1 gate_piecewise(gate1_t05)_CC_lambda(2)
-1 gate_piecewise(gate1_t05)_CC_bin_y(1)
-1 gate_piecewise(gate1_t05)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate1_t05)_CC_constraint4(3)_:
+1 gate_piecewise(gate1_t05)_CC_lambda(3)
-1 gate_piecewise(gate1_t05)_CC_bin_y(2)
-1 gate_piecewise(gate1_t05)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate1_t05)_CC_constraint4(4)_:
+1 gate_piecewise(gate1_t05)_CC_lambda(4)
-1 gate_piecewise(gate1_t05)_CC_bin_y(3)
-1 gate_piecewise(gate1_t05)_CC_bin_y(4)
<= 0

c_u_gate_piecewise(gate1_t05)_CC_constraint4(5)_:
+1 gate_piecewise(gate1_t05)_CC_lambda(5)
-1 gate_piecewise(gate1_t05)_CC_bin_y(4)
<= 0

c_e_gate_piecewise(gate1_t05)_CC_constraint5_:
+1 gate_piecewise(gate1_t05)_CC_bin_y(1)
+1 gate_piecewise(gate1_t05)_CC_bin_y(2)
+1 gate_piecewise(gate1_t05)_CC_bin_y(3)
+1 gate_piecewise(gate1_t05)_CC_bin_y(4)
= 1

c_e_gate_piecewise(gate1_t06)_CC_constraint1_:
+1 gate_opening(gate1_t06)
-0.25 gate_piecewise(gate1_t06)_CC_lambda(2)
-0.5 gate_piecewise(gate1_t06)_CC_lambda(3)
-0.75 gate_piecewise(gate1_t06)_CC_lambda(4)
-1 gate_piecewise(gate1_t06)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t06)_CC_constraint2_:
+1 gate_capacity(gate1_t06)
-6.0 gate_piecewise(gate1_t06)_CC_lambda(2)
-10.5 gate_piecewise(gate1_t06)_CC_lambda(3)
-13.5 gate_piecewise(gate1_t06)_CC_lambda(4)
-18.0 gate_piecewise(gate1_t06)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t06)_CC_constraint3_:
+1 gate_piecewise(gate1_t06)_CC_lambda(1)
+1 gate_piecewise(gate1_t06)_CC_lambda(2)
+1 gate_piecewise(gate1_t06)_CC_lambda(3)
+1 gate_piecewise(gate1_t06)_CC_lambda(4)
+1 gate_piecewise(gate1_t06)_CC_lambda(5)
= 1

c_u_gate_piecewise(gate1_t06)_CC_constraint4(1)_:
+1 gate_piecewise(gate1_t06)_CC_lambda(1)
-1 gate_piecewise(gate1_t06)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate1_t06)_CC_constraint4(2)_:
+1 gate_piecewise(gate1_t06)_CC_lambda(2)
-1 gate_piecewise(gate1_t06)_CC_bin_y(1)
-1 gate_piecewise(gate1_t06)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate1_t06)_CC_constraint4(3)_:
+1 gate_piecewise(gate1_t06)_CC_lambda(3)
-1 gate_piecewise(gate1_t06)_CC_bin_y(2)
-1 gate_piecewise(gate1_t06)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate1_t06)_CC_constraint4(4)_:
+1 gate_piecewise(gate1_t06)_CC_lambda(4)
-1 gate_piecewise(gate1_t06)_CC_bin_y(3)
-1 gate_piecewise(gate1_t06)_CC_bin_y(4)
<= 0

c_u_gate_piecewise(gate1_t06)_CC_constraint4(5)_:
+1 gate_piecewise(gate1_t06)_CC_lambda(5)
-1 gate_piecewise(gate1_t06)_CC_bin_y(4)
<= 0

c_e_gate_piecewise(gate1_t06)_CC_constraint5_:
+1 gate_piecewise(gate1_t06)_CC_bin_y(1)
+1 gate_piecewise(gate1_t06)_CC_bin_y(2)
+1 gate_piecewise(gate1_t06)_CC_bin_y(3)
+1 gate_piecewise(gate1_t06)_CC_bin_y(4)
= 1

c_e_gate_piecewise(gate1_t07)_CC_constraint1_:
+1 gate_opening(gate1_t07)
-0.25 gate_piecewise(gate1_t07)_CC_lambda(2)
-0.5 gate_piecewise(gate1_t07)_CC_lambda(3)
-0.75 gate_piecewise(gate1_t07)_CC_lambda(4)
-1 gate_piecewise(gate1_t07)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t07)_CC_constraint2_:
+1 gate_capacity(gate1_t07)
-6.0 gate_piecewise(gate1_t07)_CC_lambda(2)
-10.5 gate_piecewise(gate1_t07)_CC_lambda(3)
-13.5 gate_piecewise(gate1_t07)_CC_lambda(4)
-18.0 gate_piecewise(gate1_t07)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t07)_CC_constraint3_:
+1 gate_piecewise(gate1_t07)_CC_lambda(1)
+1 gate_piecewise(gate1_t07)_CC_lambda(2)
+1 gate_piecewise(gate1_t07)_CC_lambda(3)
+1 gate_piecewise(gate1_t07)_CC_lambda(4)
+1 gate_piecewise(gate1_t07)_CC_lambda(5)
= 1

c_u_gate_piecewise(gate1_t07)_CC_constraint4(1)_:
+1 gate_piecewise(gate1_t07)_CC_lambda(1)
-1 gate_piecewise(gate1_t07)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate1_t07)_CC_constraint4(2)_:
+1 gate_piecewise(gate1_t07)_CC_lambda(2)
-1 gate_piecewise(gate1_t07)_CC_bin_y(1)
-1 gate_piecewise(gate1_t07)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate1_t07)_CC_constraint4(3)_:
+1 gate_piecewise(gate1_t07)_CC_lambda(3)
-1 gate_piecewise(gate1_t07)_CC_bin_y(2)
-1 gate_piecewise(gate1_t07)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate1_t07)_CC_constraint4(4)_:
+1 gate_piecewise(gate1_t07)_CC_lambda(4)
-1 gate_piecewise(gate1_t07)_CC_bin_y(3)
-1 gate_piecewise(gate1_t07)_CC_bin_y(4)
<= 0

c_u_gate_piecewise(gate1_t07)_CC_constraint4(5)_:
+1 gate_piecewise(gate1_t07)_CC_lambda(5)
-1 gate_piecewise(gate1_t07)_CC_bin_y(4)
<= 0

c_e_gate_piecewise(gate1_t07)_CC_constraint5_:
+1 gate_piecewise(gate1_t07)_CC_bin_y(1)
+1 gate_piecewise(gate1_t07)_CC_bin_y(2)
+1 gate_piecewise(gate1_t07)_CC_bin_y(3)
+1 gate_piecewise(gate1_t07)_CC_bin_y(4)
= 1

c_e_gate_piecewise(gate1_t08)_CC_constraint1_:
+1 gate_opening(gate1_t08)
-0.25 gate_piecewise(gate1_t08)_CC_lambda(2)
-0.5 gate_piecewise(gate1_t08)_CC_lambda(3)
-0.75 gate_piecewise(gate1_t08)_CC_lambda(4)
-1 gate_piecewise(gate1_t08)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t08)_CC_constraint2_:
+1 gate_capacity(gate1_t08)
-6.0 gate_piecewise(gate1_t08)_CC_lambda(2)
-10.5 gate_piecewise(gate1_t08)_CC_lambda(3)
-13.5 gate_piecewise(gate1_t08)_CC_lambda(4)
-18.0 gate_piecewise(gate1_t08)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t08)_CC_constraint3_:
+1 gate_piecewise(gate1_t08)_CC_lambda(1)
+1 gate_piecewise(gate1_t08)_CC_lambda(2)
+1 gate_piecewise(gate1_t08)_CC_lambda(3)
+1 gate_piecewise(gate1_t08)_CC_lambda(4)
+1 gate_piecewise(gate1_t08)_CC_lambda(5)
= 1

c_u_gate_piecewise(gate1_t08)_CC_constraint4(1)_:
+1 gate_piecewise(gate1_t08)_CC_lambda(1)
-1 gate_piecewise(gate1_t08)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate1_t08)_CC_constraint4(2)_:
+1 gate_piecewise(gate1_t08)_CC_lambda(2)
-1 gate_piecewise(gate1_t08)_CC_bin_y(1)
-1 gate_piecewise(gate1_t08)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate1_t08)_CC_constraint4(3)_:
+1 gate_piecewise(gate1_t08)_CC_lambda(3)
-1 gate_piecewise(gate1_t08)_CC_bin_y(2)
-1 gate_piecewise(gate1_t08)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate1_t08)_CC_constraint4(4)_:
+1 gate_piecewise(gate1_t08)_CC_lambda(4)
-1 gate_piecewise(gate1_t08)_CC_bin_y(3)
-1 gate_piecewise(gate1_t08)_CC_bin_y(4)
<= 0

c_u_gate_piecewise(gate1_t08)_CC_constraint4(5)_:
+1 gate_piecewise(gate1_t08)_CC_lambda(5)
-1 gate_piecewise(gate1_t08)_CC_bin_y(4)
<= 0

c_e_gate_piecewise(gate1_t08)_CC_constraint5_:
+1 gate_piecewise(gate1_t08)_CC_bin_y(1)
+1 gate_piecewise(gate1_t08)_CC_bin_y(2)
+1 gate_piecewise(gate1_t08)_CC_bin_y(3)
+1 gate_piecewise(gate1_t08)_CC_bin_y(4)
= 1

c_e_gate_piecewise(gate1_t09)_CC_constraint1_:
+1 gate_opening(gate1_t09)
-0.25 gate_piecewise(gate1_t09)_CC_lambda(2)
-0.5 gate_piecewise(gate1_t09)_CC_lambda(3)
-0.75 gate_piecewise(gate1_t09)_CC_lambda(4)
-1 gate_piecewise(gate1_t09)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t09)_CC_constraint2_:
+1 gate_capacity(gate1_t09)
-6.0 gate_piecewise(gate1_t09)_CC_lambda(2)
-10.5 gate_piecewise(gate1_t09)_CC_lambda(3)
-13.5 gate_piecewise(gate1_t09)_CC_lambda(4)
-18.0 gate_piecewise(gate1_t09)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t09)_CC_constraint3_:
+1 gate_piecewise(gate1_t09)_CC_lambda(1)
+1 gate_piecewise(gate1_t09)_CC_lambda(2)
+1 gate_piecewise(gate1_t09)_CC_lambda(3)
+1 gate_piecewise(gate1_t09)_CC_lambda(4)
+1 gate_piecewise(gate1_t09)_CC_lambda(5)
= 1

c_u_gate_piecewise(gate1_t09)_CC_constraint4(1)_:
+1 gate_piecewise(gate1_t09)_CC_lambda(1)
-1 gate_piecewise(gate1_t09)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate1_t09)_CC_constraint4(2)_:
+1 gate_piecewise(gate1_t09)_CC_lambda(2)
-1 gate_piecewise(gate1_t09)_CC_bin_y(1)
-1 gate_piecewise(gate1_t09)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate1_t09)_CC_constraint4(3)_:
+1 gate_piecewise(gate1_t09)_CC_lambda(3)
-1 gate_piecewise(gate1_t09)_CC_bin_y(2)
-1 gate_piecewise(gate1_t09)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate1_t09)_CC_constraint4(4)_:
+1 gate_piecewise(gate1_t09)_CC_lambda(4)
-1 gate_piecewise(gate1_t09)_CC_bin_y(3)
-1 gate_piecewise(gate1_t09)_CC_bin_y(4)
<= 0

c_u_gate_piecewise(gate1_t09)_CC_constraint4(5)_:
+1 gate_piecewise(gate1_t09)_CC_lambda(5)
-1 gate_piecewise(gate1_t09)_CC_bin_y(4)
<= 0

c_e_gate_piecewise(gate1_t09)_CC_constraint5_:
+1 gate_piecewise(gate1_t09)_CC_bin_y(1)
+1 gate_piecewise(gate1_t09)_CC_bin_y(2)
+1 gate_piecewise(gate1_t09)_CC_bin_y(3)
+1 gate_piecewise(gate1_t09)_CC_bin_y(4)
= 1

c_e_gate_piecewise(gate1_t10)_CC_constraint1_:
+1 gate_opening(gate1_t10)
-0.25 gate_piecewise(gate1_t10)_CC_lambda(2)
-0.5 gate_piecewise(gate1_t10)_CC_lambda(3)
-0.75 gate_piecewise(gate1_t10)_CC_lambda(4)
-1 gate_piecewise(gate1_t10)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t10)_CC_constraint2_:
+1 gate_capacity(gate1_t10)
-6.0 gate_piecewise(gate1_t10)_CC_lambda(2)
-10.5 gate_piecewise(gate1_t10)_CC_lambda(3)
-13.5 gate_piecewise(gate1_t10)_CC_lambda(4)
-18.0 gate_piecewise(gate1_t10)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t10)_CC_constraint3_:
+1 gate_piecewise(gate1_t10)_CC_lambda(1)
+1 gate_piecewise(gate1_t10)_CC_lambda(2)
+1 gate_piecewise(gate1_t10)_CC_lambda(3)
+1 gate_piecewise(gate1_t10)_CC_lambda(4)
+1 gate_piecewise(gate1_t10)_CC_lambda(5)
= 1

c_u_gate_piecewise(gate1_t10)_CC_constraint4(1)_:
+1 gate_piecewise(gate1_t10)_CC_lambda(1)
-1 gate_piecewise(gate1_t10)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate1_t10)_CC_constraint4(2)_:
+1 gate_piecewise(gate1_t10)_CC_lambda(2)
-1 gate_piecewise(gate1_t10)_CC_bin_y(1)
-1 gate_piecewise(gate1_t10)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate1_t10)_CC_constraint4(3)_:
+1 gate_piecewise(gate1_t10)_CC_lambda(3)
-1 gate_piecewise(gate1_t10)_CC_bin_y(2)
-1 gate_piecewise(gate1_t10)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate1_t10)_CC_constraint4(4)_:
+1 gate_piecewise(gate1_t10)_CC_lambda(4)
-1 gate_piecewise(gate1_t10)_CC_bin_y(3)
-1 gate_piecewise(gate1_t10)_CC_bin_y(4)
<= 0

c_u_gate_piecewise(gate1_t10)_CC_constraint4(5)_:
+1 gate_piecewise(gate1_t10)_CC_lambda(5)
-1 gate_piecewise(gate1_t10)_CC_bin_y(4)
<= 0

c_e_gate_piecewise(gate1_t10)_CC_constraint5_:
+1 gate_piecewise(gate1_t10)_CC_bin_y(1)
+1 gate_piecewise(gate1_t10)_CC_bin_y(2)
+1 gate_piecewise(gate1_t10)_CC_bin_y(3)
+1 gate_piecewise(gate1_t10)_CC_bin_y(4)
= 1

c_e_gate_piecewise(gate1_t11)_CC_constraint1_:
+1 gate_opening(gate1_t11)
-0.25 gate_piecewise(gate1_t11)_CC_lambda(2)
-0.5 gate_piecewise(gate1_t11)_CC_lambda(3)
-0.75 gate_piecewise(gate1_t11)_CC_lambda(4)
-1 gate_piecewise(gate1_t11)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t11)_CC_constraint2_:
+1 gate_capacity(gate1_t11)
-6.0 gate_piecewise(gate1_t11)_CC_lambda(2)
-10.5 gate_piecewise(gate1_t11)_CC_lambda(3)
-13.5 gate_piecewise(gate1_t11)_CC_lambda(4)
-18.0 gate_piecewise(gate1_t11)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t11)_CC_constraint3_:
+1 gate_piecewise(gate1_t11)_CC_lambda(1)
+1 gate_piecewise(gate1_t11)_CC_lambda(2)
+1 gate_piecewise(gate1_t11)_CC_lambda(3)
+1 gate_piecewise(gate1_t11)_CC_lambda(4)
+1 gate_piecewise(gate1_t11)_CC_lambda(5)
= 1

c_u_gate_piecewise(gate1_t11)_CC_constraint4(1)_:
+1 gate_piecewise(gate1_t11)_CC_lambda(1)
-1 gate_piecewise(gate1_t11)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate1_t11)_CC_constraint4(2)_:
+1 gate_piecewise(gate1_t11)_CC_lambda(2)
-1 gate_piecewise(gate1_t11)_CC_bin_y(1)
-1 gate_piecewise(gate1_t11)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate1_t11)_CC_constraint4(3)_:
+1 gate_piecewise(gate1_t11)_CC_lambda(3)
-1 gate_piecewise(gate1_t11)_CC_bin_y(2)
-1 gate_piecewise(gate1_t11)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate1_t11)_CC_constraint4(4)_:
+1 gate_piecewise(gate1_t11)_CC_lambda(4)
-1 gate_piecewise(gate1_t11)_CC_bin_y(3)
-1 gate_piecewise(gate1_t11)_CC_bin_y(4)
<= 0

c_u_gate_piecewise(gate1_t11)_CC_constraint4(5)_:
+1 gate_piecewise(gate1_t11)_CC_lambda(5)
-1 gate_piecewise(gate1_t11)_CC_bin_y(4)
<= 0

c_e_gate_piecewise(gate1_t11)_CC_constraint5_:
+1 gate_piecewise(gate1_t11)_CC_bin_y(1)
+1 gate_piecewise(gate1_t11)_CC_bin_y(2)
+1 gate_piecewise(gate1_t11)_CC_bin_y(3)
+1 gate_piecewise(gate1_t11)_CC_bin_y(4)
= 1

c_e_gate_piecewise(gate1_t12)_CC_constraint1_:
+1 gate_opening(gate1_t12)
-0.25 gate_piecewise(gate1_t12)_CC_lambda(2)
-0.5 gate_piecewise(gate1_t12)_CC_lambda(3)
-0.75 gate_piecewise(gate1_t12)_CC_lambda(4)
-1 gate_piecewise(gate1_t12)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t12)_CC_constraint2_:
+1 gate_capacity(gate1_t12)
-6.0 gate_piecewise(gate1_t12)_CC_lambda(2)
-10.5 gate_piecewise(gate1_t12)_CC_lambda(3)
-13.5 gate_piecewise(gate1_t12)_CC_lambda(4)
-18.0 gate_piecewise(gate1_t12)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t12)_CC_constraint3_:
+1 gate_piecewise(gate1_t12)_CC_lambda(1)
+1 gate_piecewise(gate1_t12)_CC_lambda(2)
+1 gate_piecewise(gate1_t12)_CC_lambda(3)
+1 gate_piecewise(gate1_t12)_CC_lambda(4)
+1 gate_piecewise(gate1_t12)_CC_lambda(5)
= 1

c_u_gate_piecewise(gate1_t12)_CC_constraint4(1)_:
+1 gate_piecewise(gate1_t12)_CC_lambda(1)
-1 gate_piecewise(gate1_t12)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate1_t12)_CC_constraint4(2)_:
+1 gate_piecewise(gate1_t12)_CC_lambda(2)
-1 gate_piecewise(gate1_t12)_CC_bin_y(1)
-1 gate_piecewise(gate1_t12)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate1_t12)_CC_constraint4(3)_:
+1 gate_piecewise(gate1_t12)_CC_lambda(3)
-1 gate_piecewise(gate1_t12)_CC_bin_y(2)
-1 gate_piecewise(gate1_t12)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate1_t12)_CC_constraint4(4)_:
+1 gate_piecewise(gate1_t12)_CC_lambda(4)
-1 gate_piecewise(gate1_t12)_CC_bin_y(3)
-1 gate_piecewise(gate1_t12)_CC_bin_y(4)
<= 0

c_u_gate_piecewise(gate1_t12)_CC_constraint4(5)_:
+1 gate_piecewise(gate1_t12)_CC_lambda(5)
-1 gate_piecewise(gate1_t12)_CC_bin_y(4)
<= 0

c_e_gate_piecewise(gate1_t12)_CC_constraint5_:
+1 gate_piecewise(gate1_t12)_CC_bin_y(1)
+1 gate_piecewise(gate1_t12)_CC_bin_y(2)
+1 gate_piecewise(gate1_t12)_CC_bin_y(3)
+1 gate_piecewise(gate1_t12)_CC_bin_y(4)
= 1

c_e_gate_piecewise(gate1_t13)_CC_constraint1_:
+1 gate_opening(gate1_t13)
-0.25 gate_piecewise(gate1_t13)_CC_lambda(2)
-0.5 gate_piecewise(gate1_t13)_CC_lambda(3)
-0.75 gate_piecewise(gate1_t13)_CC_lambda(4)
-1 gate_piecewise(gate1_t13)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t13)_CC_constraint2_:
+1 gate_capacity(gate1_t13)
-6.0 gate_piecewise(gate1_t13)_CC_lambda(2)
-10.5 gate_piecewise(gate1_t13)_CC_lambda(3)
-13.5 gate_piecewise(gate1_t13)_CC_lambda(4)
-18.0 gate_piecewise(gate1_t13)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t13)_CC_constraint3_:
+1 gate_piecewise(gate1_t13)_CC_lambda(1)
+1 gate_piecewise(gate1_t13)_CC_lambda(2)
+1 gate_piecewise(gate1_t13)_CC_lambda(3)
+1 gate_piecewise(gate1_t13)_CC_lambda(4)
+1 gate_piecewise(gate1_t13)_CC_lambda(5)
= 1

c_u_gate_piecewise(gate1_t13)_CC_constraint4(1)_:
+1 gate_piecewise(gate1_t13)_CC_lambda(1)
-1 gate_piecewise(gate1_t13)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate1_t13)_CC_constraint4(2)_:
+1 gate_piecewise(gate1_t13)_CC_lambda(2)
-1 gate_piecewise(gate1_t13)_CC_bin_y(1)
-1 gate_piecewise(gate1_t13)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate1_t13)_CC_constraint4(3)_:
+1 gate_piecewise(gate1_t13)_CC_lambda(3)
-1 gate_piecewise(gate1_t13)_CC_bin_y(2)
-1 gate_piecewise(gate1_t13)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate1_t13)_CC_constraint4(4)_:
+1 gate_piecewise(gate1_t13)_CC_lambda(4)
-1 gate_piecewise(gate1_t13)_CC_bin_y(3)
-1 gate_piecewise(gate1_t13)_CC_bin_y(4)
<= 0

c_u_gate_piecewise(gate1_t13)_CC_constraint4(5)_:
+1 gate_piecewise(gate1_t13)_CC_lambda(5)
-1 gate_piecewise(gate1_t13)_CC_bin_y(4)
<= 0

c_e_gate_piecewise(gate1_t13)_CC_constraint5_:
+1 gate_piecewise(gate1_t13)_CC_bin_y(1)
+1 gate_piecewise(gate1_t13)_CC_bin_y(2)
+1 gate_piecewise(gate1_t13)_CC_bin_y(3)
+1 gate_piecewise(gate1_t13)_CC_bin_y(4)
= 1

c_e_gate_piecewise(gate1_t14)_CC_constraint1_:
+1 gate_opening(gate1_t14)
-0.25 gate_piecewise(gate1_t14)_CC_lambda(2)
-0.5 gate_piecewise(gate1_t14)_CC_lambda(3)
-0.75 gate_piecewise(gate1_t14)_CC_lambda(4)
-1 gate_piecewise(gate1_t14)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t14)_CC_constraint2_:
+1 gate_capacity(gate1_t14)
-6.0 gate_piecewise(gate1_t14)_CC_lambda(2)
-10.5 gate_piecewise(gate1_t14)_CC_lambda(3)
-13.5 gate_piecewise(gate1_t14)_CC_lambda(4)
-18.0 gate_piecewise(gate1_t14)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t14)_CC_constraint3_:
+1 gate_piecewise(gate1_t14)_CC_lambda(1)
+1 gate_piecewise(gate1_t14)_CC_lambda(2)
+1 gate_piecewise(gate1_t14)_CC_lambda(3)
+1 gate_piecewise(gate1_t14)_CC_lambda(4)
+1 gate_piecewise(gate1_t14)_CC_lambda(5)
= 1

c_u_gate_piecewise(gate1_t14)_CC_constraint4(1)_:
+1 gate_piecewise(gate1_t14)_CC_lambda(1)
-1 gate_piecewise(gate1_t14)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate1_t14)_CC_constraint4(2)_:
+1 gate_piecewise(gate1_t14)_CC_lambda(2)
-1 gate_piecewise(gate1_t14)_CC_bin_y(1)
-1 gate_piecewise(gate1_t14)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate1_t14)_CC_constraint4(3)_:
+1 gate_piecewise(gate1_t14)_CC_lambda(3)
-1 gate_piecewise(gate1_t14)_CC_bin_y(2)
-1 gate_piecewise(gate1_t14)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate1_t14)_CC_constraint4(4)_:
+1 gate_piecewise(gate1_t14)_CC_lambda(4)
-1 gate_piecewise(gate1_t14)_CC_bin_y(3)
-1 gate_piecewise(gate1_t14)_CC_bin_y(4)
<= 0

c_u_gate_piecewise(gate1_t14)_CC_constraint4(5)_:
+1 gate_piecewise(gate1_t14)_CC_lambda(5)
-1 gate_piecewise(gate1_t14)_CC_bin_y(4)
<= 0

c_e_gate_piecewise(gate1_t14)_CC_constraint5_:
+1 gate_piecewise(gate1_t14)_CC_bin_y(1)
+1 gate_piecewise(gate1_t14)_CC_bin_y(2)
+1 gate_piecewise(gate1_t14)_CC_bin_y(3)
+1 gate_piecewise(gate1_t14)_CC_bin_y(4)
= 1

c_e_gate_piecewise(gate1_t15)_CC_constraint1_:
+1 gate_opening(gate1_t15)
-0.25 gate_piecewise(gate1_t15)_CC_lambda(2)
-0.5 gate_piecewise(gate1_t15)_CC_lambda(3)
-0.75 gate_piecewise(gate1_t15)_CC_lambda(4)
-1 gate_piecewise(gate1_t15)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t15)_CC_constraint2_:
+1 gate_capacity(gate1_t15)
-6.0 gate_piecewise(gate1_t15)_CC_lambda(2)
-10.5 gate_piecewise(gate1_t15)_CC_lambda(3)
-13.5 gate_piecewise(gate1_t15)_CC_lambda(4)
-18.0 gate_piecewise(gate1_t15)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t15)_CC_constraint3_:
+1 gate_piecewise(gate1_t15)_CC_lambda(1)
+1 gate_piecewise(gate1_t15)_CC_lambda(2)
+1 gate_piecewise(gate1_t15)_CC_lambda(3)
+1 gate_piecewise(gate1_t15)_CC_lambda(4)
+1 gate_piecewise(gate1_t15)_CC_lambda(5)
= 1

c_u_gate_piecewise(gate1_t15)_CC_constraint4(1)_:
+1 gate_piecewise(gate1_t15)_CC_lambda(1)
-1 gate_piecewise(gate1_t15)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate1_t15)_CC_constraint4(2)_:
+1 gate_piecewise(gate1_t15)_CC_lambda(2)
-1 gate_piecewise(gate1_t15)_CC_bin_y(1)
-1 gate_piecewise(gate1_t15)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate1_t15)_CC_constraint4(3)_:
+1 gate_piecewise(gate1_t15)_CC_lambda(3)
-1 gate_piecewise(gate1_t15)_CC_bin_y(2)
-1 gate_piecewise(gate1_t15)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate1_t15)_CC_constraint4(4)_:
+1 gate_piecewise(gate1_t15)_CC_lambda(4)
-1 gate_piecewise(gate1_t15)_CC_bin_y(3)
-1 gate_piecewise(gate1_t15)_CC_bin_y(4)
<= 0

c_u_gate_piecewise(gate1_t15)_CC_constraint4(5)_:
+1 gate_piecewise(gate1_t15)_CC_lambda(5)
-1 gate_piecewise(gate1_t15)_CC_bin_y(4)
<= 0

c_e_gate_piecewise(gate1_t15)_CC_constraint5_:
+1 gate_piecewise(gate1_t15)_CC_bin_y(1)
+1 gate_piecewise(gate1_t15)_CC_bin_y(2)
+1 gate_piecewise(gate1_t15)_CC_bin_y(3)
+1 gate_piecewise(gate1_t15)_CC_bin_y(4)
= 1

c_e_gate_piecewise(gate1_t16)_CC_constraint1_:
+1 gate_opening(gate1_t16)
-0.25 gate_piecewise(gate1_t16)_CC_lambda(2)
-0.5 gate_piecewise(gate1_t16)_CC_lambda(3)
-0.75 gate_piecewise(gate1_t16)_CC_lambda(4)
-1 gate_piecewise(gate1_t16)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t16)_CC_constraint2_:
+1 gate_capacity(gate1_t16)
-6.0 gate_piecewise(gate1_t16)_CC_lambda(2)
-10.5 gate_piecewise(gate1_t16)_CC_lambda(3)
-13.5 gate_piecewise(gate1_t16)_CC_lambda(4)
-18.0 gate_piecewise(gate1_t16)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t16)_CC_constraint3_:
+1 gate_piecewise(gate1_t16)_CC_lambda(1)
+1 gate_piecewise(gate1_t16)_CC_lambda(2)
+1 gate_piecewise(gate1_t16)_CC_lambda(3)
+1 gate_piecewise(gate1_t16)_CC_lambda(4)
+1 gate_piecewise(gate1_t16)_CC_lambda(5)
= 1

c_u_gate_piecewise(gate1_t16)_CC_constraint4(1)_:
+1 gate_piecewise(gate1_t16)_CC_lambda(1)
-1 gate_piecewise(gate1_t16)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate1_t16)_CC_constraint4(2)_:
+1 gate_piecewise(gate1_t16)_CC_lambda(2)
-1 gate_piecewise(gate1_t16)_CC_bin_y(1)
-1 gate_piecewise(gate1_t16)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate1_t16)_CC_constraint4(3)_:
+1 gate_piecewise(gate1_t16)_CC_lambda(3)
-1 gate_piecewise(gate1_t16)_CC_bin_y(2)
-1 gate_piecewise(gate1_t16)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate1_t16)_CC_constraint4(4)_:
+1 gate_piecewise(gate1_t16)_CC_lambda(4)
-1 gate_piecewise(gate1_t16)_CC_bin_y(3)
-1 gate_piecewise(gate1_t16)_CC_bin_y(4)
<= 0

c_u_gate_piecewise(gate1_t16)_CC_constraint4(5)_:
+1 gate_piecewise(gate1_t16)_CC_lambda(5)
-1 gate_piecewise(gate1_t16)_CC_bin_y(4)
<= 0

c_e_gate_piecewise(gate1_t16)_CC_constraint5_:
+1 gate_piecewise(gate1_t16)_CC_bin_y(1)
+1 gate_piecewise(gate1_t16)_CC_bin_y(2)
+1 gate_piecewise(gate1_t16)_CC_bin_y(3)
+1 gate_piecewise(gate1_t16)_CC_bin_y(4)
= 1

c_e_gate_piecewise(gate1_t17)_CC_constraint1_:
+1 gate_opening(gate1_t17)
-0.25 gate_piecewise(gate1_t17)_CC_lambda(2)
-0.5 gate_piecewise(gate1_t17)_CC_lambda(3)
-0.75 gate_piecewise(gate1_t17)_CC_lambda(4)
-1 gate_piecewise(gate1_t17)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t17)_CC_constraint2_:
+1 gate_capacity(gate1_t17)
-6.0 gate_piecewise(gate1_t17)_CC_lambda(2)
-10.5 gate_piecewise(gate1_t17)_CC_lambda(3)
-13.5 gate_piecewise(gate1_t17)_CC_lambda(4)
-18.0 gate_piecewise(gate1_t17)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t17)_CC_constraint3_:
+1 gate_piecewise(gate1_t17)_CC_lambda(1)
+1 gate_piecewise(gate1_t17)_CC_lambda(2)
+1 gate_piecewise(gate1_t17)_CC_lambda(3)
+1 gate_piecewise(gate1_t17)_CC_lambda(4)
+1 gate_piecewise(gate1_t17)_CC_lambda(5)
= 1

c_u_gate_piecewise(gate1_t17)_CC_constraint4(1)_:
+1 gate_piecewise(gate1_t17)_CC_lambda(1)
-1 gate_piecewise(gate1_t17)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate1_t17)_CC_constraint4(2)_:
+1 gate_piecewise(gate1_t17)_CC_lambda(2)
-1 gate_piecewise(gate1_t17)_CC_bin_y(1)
-1 gate_piecewise(gate1_t17)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate1_t17)_CC_constraint4(3)_:
+1 gate_piecewise(gate1_t17)_CC_lambda(3)
-1 gate_piecewise(gate1_t17)_CC_bin_y(2)
-1 gate_piecewise(gate1_t17)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate1_t17)_CC_constraint4(4)_:
+1 gate_piecewise(gate1_t17)_CC_lambda(4)
-1 gate_piecewise(gate1_t17)_CC_bin_y(3)
-1 gate_piecewise(gate1_t17)_CC_bin_y(4)
<= 0

c_u_gate_piecewise(gate1_t17)_CC_constraint4(5)_:
+1 gate_piecewise(gate1_t17)_CC_lambda(5)
-1 gate_piecewise(gate1_t17)_CC_bin_y(4)
<= 0

c_e_gate_piecewise(gate1_t17)_CC_constraint5_:
+1 gate_piecewise(gate1_t17)_CC_bin_y(1)
+1 gate_piecewise(gate1_t17)_CC_bin_y(2)
+1 gate_piecewise(gate1_t17)_CC_bin_y(3)
+1 gate_piecewise(gate1_t17)_CC_bin_y(4)
= 1

c_e_gate_piecewise(gate1_t18)_CC_constraint1_:
+1 gate_opening(gate1_t18)
-0.25 gate_piecewise(gate1_t18)_CC_lambda(2)
-0.5 gate_piecewise(gate1_t18)_CC_lambda(3)
-0.75 gate_piecewise(gate1_t18)_CC_lambda(4)
-1 gate_piecewise(gate1_t18)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t18)_CC_constraint2_:
+1 gate_capacity(gate1_t18)
-6.0 gate_piecewise(gate1_t18)_CC_lambda(2)
-10.5 gate_piecewise(gate1_t18)_CC_lambda(3)
-13.5 gate_piecewise(gate1_t18)_CC_lambda(4)
-18.0 gate_piecewise(gate1_t18)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t18)_CC_constraint3_:
+1 gate_piecewise(gate1_t18)_CC_lambda(1)
+1 gate_piecewise(gate1_t18)_CC_lambda(2)
+1 gate_piecewise(gate1_t18)_CC_lambda(3)
+1 gate_piecewise(gate1_t18)_CC_lambda(4)
+1 gate_piecewise(gate1_t18)_CC_lambda(5)
= 1

c_u_gate_piecewise(gate1_t18)_CC_constraint4(1)_:
+1 gate_piecewise(gate1_t18)_CC_lambda(1)
-1 gate_piecewise(gate1_t18)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate1_t18)_CC_constraint4(2)_:
+1 gate_piecewise(gate1_t18)_CC_lambda(2)
-1 gate_piecewise(gate1_t18)_CC_bin_y(1)
-1 gate_piecewise(gate1_t18)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate1_t18)_CC_constraint4(3)_:
+1 gate_piecewise(gate1_t18)_CC_lambda(3)
-1 gate_piecewise(gate1_t18)_CC_bin_y(2)
-1 gate_piecewise(gate1_t18)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate1_t18)_CC_constraint4(4)_:
+1 gate_piecewise(gate1_t18)_CC_lambda(4)
-1 gate_piecewise(gate1_t18)_CC_bin_y(3)
-1 gate_piecewise(gate1_t18)_CC_bin_y(4)
<= 0

c_u_gate_piecewise(gate1_t18)_CC_constraint4(5)_:
+1 gate_piecewise(gate1_t18)_CC_lambda(5)
-1 gate_piecewise(gate1_t18)_CC_bin_y(4)
<= 0

c_e_gate_piecewise(gate1_t18)_CC_constraint5_:
+1 gate_piecewise(gate1_t18)_CC_bin_y(1)
+1 gate_piecewise(gate1_t18)_CC_bin_y(2)
+1 gate_piecewise(gate1_t18)_CC_bin_y(3)
+1 gate_piecewise(gate1_t18)_CC_bin_y(4)
= 1

c_e_gate_piecewise(gate1_t19)_CC_constraint1_:
+1 gate_opening(gate1_t19)
-0.25 gate_piecewise(gate1_t19)_CC_lambda(2)
-0.5 gate_piecewise(gate1_t19)_CC_lambda(3)
-0.75 gate_piecewise(gate1_t19)_CC_lambda(4)
-1 gate_piecewise(gate1_t19)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t19)_CC_constraint2_:
+1 gate_capacity(gate1_t19)
-6.0 gate_piecewise(gate1_t19)_CC_lambda(2)
-10.5 gate_piecewise(gate1_t19)_CC_lambda(3)
-13.5 gate_piecewise(gate1_t19)_CC_lambda(4)
-18.0 gate_piecewise(gate1_t19)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t19)_CC_constraint3_:
+1 gate_piecewise(gate1_t19)_CC_lambda(1)
+1 gate_piecewise(gate1_t19)_CC_lambda(2)
+1 gate_piecewise(gate1_t19)_CC_lambda(3)
+1 gate_piecewise(gate1_t19)_CC_lambda(4)
+1 gate_piecewise(gate1_t19)_CC_lambda(5)
= 1

c_u_gate_piecewise(gate1_t19)_CC_constraint4(1)_:
+1 gate_piecewise(gate1_t19)_CC_lambda(1)
-1 gate_piecewise(gate1_t19)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate1_t19)_CC_constraint4(2)_:
+1 gate_piecewise(gate1_t19)_CC_lambda(2)
-1 gate_piecewise(gate1_t19)_CC_bin_y(1)
-1 gate_piecewise(gate1_t19)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate1_t19)_CC_constraint4(3)_:
+1 gate_piecewise(gate1_t19)_CC_lambda(3)
-1 gate_piecewise(gate1_t19)_CC_bin_y(2)
-1 gate_piecewise(gate1_t19)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate1_t19)_CC_constraint4(4)_:
+1 gate_piecewise(gate1_t19)_CC_lambda(4)
-1 gate_piecewise(gate1_t19)_CC_bin_y(3)
-1 gate_piecewise(gate1_t19)_CC_bin_y(4)
<= 0

c_u_gate_piecewise(gate1_t19)_CC_constraint4(5)_:
+1 gate_piecewise(gate1_t19)_CC_lambda(5)
-1 gate_piecewise(gate1_t19)_CC_bin_y(4)
<= 0

c_e_gate_piecewise(gate1_t19)_CC_constraint5_:
+1 gate_piecewise(gate1_t19)_CC_bin_y(1)
+1 gate_piecewise(gate1_t19)_CC_bin_y(2)
+1 gate_piecewise(gate1_t19)_CC_bin_y(3)
+1 gate_piecewise(gate1_t19)_CC_bin_y(4)
= 1

c_e_gate_piecewise(gate1_t20)_CC_constraint1_:
+1 gate_opening(gate1_t20)
-0.25 gate_piecewise(gate1_t20)_CC_lambda(2)
-0.5 gate_piecewise(gate1_t20)_CC_lambda(3)
-0.75 gate_piecewise(gate1_t20)_CC_lambda(4)
-1 gate_piecewise(gate1_t20)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t20)_CC_constraint2_:
+1 gate_capacity(gate1_t20)
-6.0 gate_piecewise(gate1_t20)_CC_lambda(2)
-10.5 gate_piecewise(gate1_t20)_CC_lambda(3)
-13.5 gate_piecewise(gate1_t20)_CC_lambda(4)
-18.0 gate_piecewise(gate1_t20)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t20)_CC_constraint3_:
+1 gate_piecewise(gate1_t20)_CC_lambda(1)
+1 gate_piecewise(gate1_t20)_CC_lambda(2)
+1 gate_piecewise(gate1_t20)_CC_lambda(3)
+1 gate_piecewise(gate1_t20)_CC_lambda(4)
+1 gate_piecewise(gate1_t20)_CC_lambda(5)
= 1

c_u_gate_piecewise(gate1_t20)_CC_constraint4(1)_:
+1 gate_piecewise(gate1_t20)_CC_lambda(1)
-1 gate_piecewise(gate1_t20)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate1_t20)_CC_constraint4(2)_:
+1 gate_piecewise(gate1_t20)_CC_lambda(2)
-1 gate_piecewise(gate1_t20)_CC_bin_y(1)
-1 gate_piecewise(gate1_t20)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate1_t20)_CC_constraint4(3)_:
+1 gate_piecewise(gate1_t20)_CC_lambda(3)
-1 gate_piecewise(gate1_t20)_CC_bin_y(2)
-1 gate_piecewise(gate1_t20)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate1_t20)_CC_constraint4(4)_:
+1 gate_piecewise(gate1_t20)_CC_lambda(4)
-1 gate_piecewise(gate1_t20)_CC_bin_y(3)
-1 gate_piecewise(gate1_t20)_CC_bin_y(4)
<= 0

c_u_gate_piecewise(gate1_t20)_CC_constraint4(5)_:
+1 gate_piecewise(gate1_t20)_CC_lambda(5)
-1 gate_piecewise(gate1_t20)_CC_bin_y(4)
<= 0

c_e_gate_piecewise(gate1_t20)_CC_constraint5_:
+1 gate_piecewise(gate1_t20)_CC_bin_y(1)
+1 gate_piecewise(gate1_t20)_CC_bin_y(2)
+1 gate_piecewise(gate1_t20)_CC_bin_y(3)
+1 gate_piecewise(gate1_t20)_CC_bin_y(4)
= 1

c_e_gate_piecewise(gate1_t21)_CC_constraint1_:
+1 gate_opening(gate1_t21)
-0.25 gate_piecewise(gate1_t21)_CC_lambda(2)
-0.5 gate_piecewise(gate1_t21)_CC_lambda(3)
-0.75 gate_piecewise(gate1_t21)_CC_lambda(4)
-1 gate_piecewise(gate1_t21)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t21)_CC_constraint2_:
+1 gate_capacity(gate1_t21)
-6.0 gate_piecewise(gate1_t21)_CC_lambda(2)
-10.5 gate_piecewise(gate1_t21)_CC_lambda(3)
-13.5 gate_piecewise(gate1_t21)_CC_lambda(4)
-18.0 gate_piecewise(gate1_t21)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t21)_CC_constraint3_:
+1 gate_piecewise(gate1_t21)_CC_lambda(1)
+1 gate_piecewise(gate1_t21)_CC_lambda(2)
+1 gate_piecewise(gate1_t21)_CC_lambda(3)
+1 gate_piecewise(gate1_t21)_CC_lambda(4)
+1 gate_piecewise(gate1_t21)_CC_lambda(5)
= 1

c_u_gate_piecewise(gate1_t21)_CC_constraint4(1)_:
+1 gate_piecewise(gate1_t21)_CC_lambda(1)
-1 gate_piecewise(gate1_t21)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate1_t21)_CC_constraint4(2)_:
+1 gate_piecewise(gate1_t21)_CC_lambda(2)
-1 gate_piecewise(gate1_t21)_CC_bin_y(1)
-1 gate_piecewise(gate1_t21)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate1_t21)_CC_constraint4(3)_:
+1 gate_piecewise(gate1_t21)_CC_lambda(3)
-1 gate_piecewise(gate1_t21)_CC_bin_y(2)
-1 gate_piecewise(gate1_t21)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate1_t21)_CC_constraint4(4)_:
+1 gate_piecewise(gate1_t21)_CC_lambda(4)
-1 gate_piecewise(gate1_t21)_CC_bin_y(3)
-1 gate_piecewise(gate1_t21)_CC_bin_y(4)
<= 0

c_u_gate_piecewise(gate1_t21)_CC_constraint4(5)_:
+1 gate_piecewise(gate1_t21)_CC_lambda(5)
-1 gate_piecewise(gate1_t21)_CC_bin_y(4)
<= 0

c_e_gate_piecewise(gate1_t21)_CC_constraint5_:
+1 gate_piecewise(gate1_t21)_CC_bin_y(1)
+1 gate_piecewise(gate1_t21)_CC_bin_y(2)
+1 gate_piecewise(gate1_t21)_CC_bin_y(3)
+1 gate_piecewise(gate1_t21)_CC_bin_y(4)
= 1

c_e_gate_piecewise(gate1_t22)_CC_constraint1_:
+1 gate_opening(gate1_t22)
-0.25 gate_piecewise(gate1_t22)_CC_lambda(2)
-0.5 gate_piecewise(gate1_t22)_CC_lambda(3)
-0.75 gate_piecewise(gate1_t22)_CC_lambda(4)
-1 gate_piecewise(gate1_t22)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t22)_CC_constraint2_:
+1 gate_capacity(gate1_t22)
-6.0 gate_piecewise(gate1_t22)_CC_lambda(2)
-10.5 gate_piecewise(gate1_t22)_CC_lambda(3)
-13.5 gate_piecewise(gate1_t22)_CC_lambda(4)
-18.0 gate_piecewise(gate1_t22)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t22)_CC_constraint3_:
+1 gate_piecewise(gate1_t22)_CC_lambda(1)
+1 gate_piecewise(gate1_t22)_CC_lambda(2)
+1 gate_piecewise(gate1_t22)_CC_lambda(3)
+1 gate_piecewise(gate1_t22)_CC_lambda(4)
+1 gate_piecewise(gate1_t22)_CC_lambda(5)
= 1

c_u_gate_piecewise(gate1_t22)_CC_constraint4(1)_:
+1 gate_piecewise(gate1_t22)_CC_lambda(1)
-1 gate_piecewise(gate1_t22)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate1_t22)_CC_constraint4(2)_:
+1 gate_piecewise(gate1_t22)_CC_lambda(2)
-1 gate_piecewise(gate1_t22)_CC_bin_y(1)
-1 gate_piecewise(gate1_t22)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate1_t22)_CC_constraint4(3)_:
+1 gate_piecewise(gate1_t22)_CC_lambda(3)
-1 gate_piecewise(gate1_t22)_CC_bin_y(2)
-1 gate_piecewise(gate1_t22)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate1_t22)_CC_constraint4(4)_:
+1 gate_piecewise(gate1_t22)_CC_lambda(4)
-1 gate_piecewise(gate1_t22)_CC_bin_y(3)
-1 gate_piecewise(gate1_t22)_CC_bin_y(4)
<= 0

c_u_gate_piecewise(gate1_t22)_CC_constraint4(5)_:
+1 gate_piecewise(gate1_t22)_CC_lambda(5)
-1 gate_piecewise(gate1_t22)_CC_bin_y(4)
<= 0

c_e_gate_piecewise(gate1_t22)_CC_constraint5_:
+1 gate_piecewise(gate1_t22)_CC_bin_y(1)
+1 gate_piecewise(gate1_t22)_CC_bin_y(2)
+1 gate_piecewise(gate1_t22)_CC_bin_y(3)
+1 gate_piecewise(gate1_t22)_CC_bin_y(4)
= 1

c_e_gate_piecewise(gate1_t23)_CC_constraint1_:
+1 gate_opening(gate1_t23)
-0.25 gate_piecewise(gate1_t23)_CC_lambda(2)
-0.5 gate_piecewise(gate1_t23)_CC_lambda(3)
-0.75 gate_piecewise(gate1_t23)_CC_lambda(4)
-1 gate_piecewise(gate1_t23)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t23)_CC_constraint2_:
+1 gate_capacity(gate1_t23)
-6.0 gate_piecewise(gate1_t23)_CC_lambda(2)
-10.5 gate_piecewise(gate1_t23)_CC_lambda(3)
-13.5 gate_piecewise(gate1_t23)_CC_lambda(4)
-18.0 gate_piecewise(gate1_t23)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t23)_CC_constraint3_:
+1 gate_piecewise(gate1_t23)_CC_lambda(1)
+1 gate_piecewise(gate1_t23)_CC_lambda(2)
+1 gate_piecewise(gate1_t23)_CC_lambda(3)
+1 gate_piecewise(gate1_t23)_CC_lambda(4)
+1 gate_piecewise(gate1_t23)_CC_lambda(5)
= 1

c_u_gate_piecewise(gate1_t23)_CC_constraint4(1)_:
+1 gate_piecewise(gate1_t23)_CC_lambda(1)
-1 gate_piecewise(gate1_t23)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate1_t23)_CC_constraint4(2)_:
+1 gate_piecewise(gate1_t23)_CC_lambda(2)
-1 gate_piecewise(gate1_t23)_CC_bin_y(1)
-1 gate_piecewise(gate1_t23)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate1_t23)_CC_constraint4(3)_:
+1 gate_piecewise(gate1_t23)_CC_lambda(3)
-1 gate_piecewise(gate1_t23)_CC_bin_y(2)
-1 gate_piecewise(gate1_t23)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate1_t23)_CC_constraint4(4)_:
+1 gate_piecewise(gate1_t23)_CC_lambda(4)
-1 gate_piecewise(gate1_t23)_CC_bin_y(3)
-1 gate_piecewise(gate1_t23)_CC_bin_y(4)
<= 0

c_u_gate_piecewise(gate1_t23)_CC_constraint4(5)_:
+1 gate_piecewise(gate1_t23)_CC_lambda(5)
-1 gate_piecewise(gate1_t23)_CC_bin_y(4)
<= 0

c_e_gate_piecewise(gate1_t23)_CC_constraint5_:
+1 gate_piecewise(gate1_t23)_CC_bin_y(1)
+1 gate_piecewise(gate1_t23)_CC_bin_y(2)
+1 gate_piecewise(gate1_t23)_CC_bin_y(3)
+1 gate_piecewise(gate1_t23)_CC_bin_y(4)
= 1

c_e_gate_piecewise(gate1_t24)_CC_constraint1_:
+1 gate_opening(gate1_t24)
-0.25 gate_piecewise(gate1_t24)_CC_lambda(2)
-0.5 gate_piecewise(gate1_t24)_CC_lambda(3)
-0.75 gate_piecewise(gate1_t24)_CC_lambda(4)
-1 gate_piecewise(gate1_t24)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t24)_CC_constraint2_:
+1 gate_capacity(gate1_t24)
-6.0 gate_piecewise(gate1_t24)_CC_lambda(2)
-10.5 gate_piecewise(gate1_t24)_CC_lambda(3)
-13.5 gate_piecewise(gate1_t24)_CC_lambda(4)
-18.0 gate_piecewise(gate1_t24)_CC_lambda(5)
= 0

c_e_gate_piecewise(gate1_t24)_CC_constraint3_:
+1 gate_piecewise(gate1_t24)_CC_lambda(1)
+1 gate_piecewise(gate1_t24)_CC_lambda(2)
+1 gate_piecewise(gate1_t24)_CC_lambda(3)
+1 gate_piecewise(gate1_t24)_CC_lambda(4)
+1 gate_piecewise(gate1_t24)_CC_lambda(5)
= 1

c_u_gate_piecewise(gate1_t24)_CC_constraint4(1)_:
+1 gate_piecewise(gate1_t24)_CC_lambda(1)
-1 gate_piecewise(gate1_t24)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate1_t24)_CC_constraint4(2)_:
+1 gate_piecewise(gate1_t24)_CC_lambda(2)
-1 gate_piecewise(gate1_t24)_CC_bin_y(1)
-1 gate_piecewise(gate1_t24)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate1_t24)_CC_constraint4(3)_:
+1 gate_piecewise(gate1_t24)_CC_lambda(3)
-1 gate_piecewise(gate1_t24)_CC_bin_y(2)
-1 gate_piecewise(gate1_t24)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate1_t24)_CC_constraint4(4)_:
+1 gate_piecewise(gate1_t24)_CC_lambda(4)
-1 gate_piecewise(gate1_t24)_CC_bin_y(3)
-1 gate_piecewise(gate1_t24)_CC_bin_y(4)
<= 0

c_u_gate_piecewise(gate1_t24)_CC_constraint4(5)_:
+1 gate_piecewise(gate1_t24)_CC_lambda(5)
-1 gate_piecewise(gate1_t24)_CC_bin_y(4)
<= 0

c_e_gate_piecewise(gate1_t24)_CC_constraint5_:
+1 gate_piecewise(gate1_t24)_CC_bin_y(1)
+1 gate_piecewise(gate1_t24)_CC_bin_y(2)
+1 gate_piecewise(gate1_t24)_CC_bin_y(3)
+1 gate_piecewise(gate1_t24)_CC_bin_y(4)
= 1

c_e_gate_piecewise(gate2_t01)_CC_constraint1_:
+1 gate_opening(gate2_t01)
-0.3 gate_piecewise(gate2_t01)_CC_lambda(2)
-0.6 gate_piecewise(gate2_t01)_CC_lambda(3)
-1 gate_piecewise(gate2_t01)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t01)_CC_constraint2_:
+1 gate_capacity(gate2_t01)
-5.5 gate_piecewise(gate2_t01)_CC_lambda(2)
-10.0 gate_piecewise(gate2_t01)_CC_lambda(3)
-14.0 gate_piecewise(gate2_t01)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t01)_CC_constraint3_:
+1 gate_piecewise(gate2_t01)_CC_lambda(1)
+1 gate_piecewise(gate2_t01)_CC_lambda(2)
+1 gate_piecewise(gate2_t01)_CC_lambda(3)
+1 gate_piecewise(gate2_t01)_CC_lambda(4)
= 1

c_u_gate_piecewise(gate2_t01)_CC_constraint4(1)_:
+1 gate_piecewise(gate2_t01)_CC_lambda(1)
-1 gate_piecewise(gate2_t01)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate2_t01)_CC_constraint4(2)_:
+1 gate_piecewise(gate2_t01)_CC_lambda(2)
-1 gate_piecewise(gate2_t01)_CC_bin_y(1)
-1 gate_piecewise(gate2_t01)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate2_t01)_CC_constraint4(3)_:
+1 gate_piecewise(gate2_t01)_CC_lambda(3)
-1 gate_piecewise(gate2_t01)_CC_bin_y(2)
-1 gate_piecewise(gate2_t01)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate2_t01)_CC_constraint4(4)_:
+1 gate_piecewise(gate2_t01)_CC_lambda(4)
-1 gate_piecewise(gate2_t01)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(gate2_t01)_CC_constraint5_:
+1 gate_piecewise(gate2_t01)_CC_bin_y(1)
+1 gate_piecewise(gate2_t01)_CC_bin_y(2)
+1 gate_piecewise(gate2_t01)_CC_bin_y(3)
= 1

c_e_gate_piecewise(gate2_t02)_CC_constraint1_:
+1 gate_opening(gate2_t02)
-0.3 gate_piecewise(gate2_t02)_CC_lambda(2)
-0.6 gate_piecewise(gate2_t02)_CC_lambda(3)
-1 gate_piecewise(gate2_t02)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t02)_CC_constraint2_:
+1 gate_capacity(gate2_t02)
-5.5 gate_piecewise(gate2_t02)_CC_lambda(2)
-10.0 gate_piecewise(gate2_t02)_CC_lambda(3)
-14.0 gate_piecewise(gate2_t02)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t02)_CC_constraint3_:
+1 gate_piecewise(gate2_t02)_CC_lambda(1)
+1 gate_piecewise(gate2_t02)_CC_lambda(2)
+1 gate_piecewise(gate2_t02)_CC_lambda(3)
+1 gate_piecewise(gate2_t02)_CC_lambda(4)
= 1

c_u_gate_piecewise(gate2_t02)_CC_constraint4(1)_:
+1 gate_piecewise(gate2_t02)_CC_lambda(1)
-1 gate_piecewise(gate2_t02)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate2_t02)_CC_constraint4(2)_:
+1 gate_piecewise(gate2_t02)_CC_lambda(2)
-1 gate_piecewise(gate2_t02)_CC_bin_y(1)
-1 gate_piecewise(gate2_t02)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate2_t02)_CC_constraint4(3)_:
+1 gate_piecewise(gate2_t02)_CC_lambda(3)
-1 gate_piecewise(gate2_t02)_CC_bin_y(2)
-1 gate_piecewise(gate2_t02)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate2_t02)_CC_constraint4(4)_:
+1 gate_piecewise(gate2_t02)_CC_lambda(4)
-1 gate_piecewise(gate2_t02)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(gate2_t02)_CC_constraint5_:
+1 gate_piecewise(gate2_t02)_CC_bin_y(1)
+1 gate_piecewise(gate2_t02)_CC_bin_y(2)
+1 gate_piecewise(gate2_t02)_CC_bin_y(3)
= 1

c_e_gate_piecewise(gate2_t03)_CC_constraint1_:
+1 gate_opening(gate2_t03)
-0.3 gate_piecewise(gate2_t03)_CC_lambda(2)
-0.6 gate_piecewise(gate2_t03)_CC_lambda(3)
-1 gate_piecewise(gate2_t03)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t03)_CC_constraint2_:
+1 gate_capacity(gate2_t03)
-5.5 gate_piecewise(gate2_t03)_CC_lambda(2)
-10.0 gate_piecewise(gate2_t03)_CC_lambda(3)
-14.0 gate_piecewise(gate2_t03)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t03)_CC_constraint3_:
+1 gate_piecewise(gate2_t03)_CC_lambda(1)
+1 gate_piecewise(gate2_t03)_CC_lambda(2)
+1 gate_piecewise(gate2_t03)_CC_lambda(3)
+1 gate_piecewise(gate2_t03)_CC_lambda(4)
= 1

c_u_gate_piecewise(gate2_t03)_CC_constraint4(1)_:
+1 gate_piecewise(gate2_t03)_CC_lambda(1)
-1 gate_piecewise(gate2_t03)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate2_t03)_CC_constraint4(2)_:
+1 gate_piecewise(gate2_t03)_CC_lambda(2)
-1 gate_piecewise(gate2_t03)_CC_bin_y(1)
-1 gate_piecewise(gate2_t03)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate2_t03)_CC_constraint4(3)_:
+1 gate_piecewise(gate2_t03)_CC_lambda(3)
-1 gate_piecewise(gate2_t03)_CC_bin_y(2)
-1 gate_piecewise(gate2_t03)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate2_t03)_CC_constraint4(4)_:
+1 gate_piecewise(gate2_t03)_CC_lambda(4)
-1 gate_piecewise(gate2_t03)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(gate2_t03)_CC_constraint5_:
+1 gate_piecewise(gate2_t03)_CC_bin_y(1)
+1 gate_piecewise(gate2_t03)_CC_bin_y(2)
+1 gate_piecewise(gate2_t03)_CC_bin_y(3)
= 1

c_e_gate_piecewise(gate2_t04)_CC_constraint1_:
+1 gate_opening(gate2_t04)
-0.3 gate_piecewise(gate2_t04)_CC_lambda(2)
-0.6 gate_piecewise(gate2_t04)_CC_lambda(3)
-1 gate_piecewise(gate2_t04)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t04)_CC_constraint2_:
+1 gate_capacity(gate2_t04)
-5.5 gate_piecewise(gate2_t04)_CC_lambda(2)
-10.0 gate_piecewise(gate2_t04)_CC_lambda(3)
-14.0 gate_piecewise(gate2_t04)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t04)_CC_constraint3_:
+1 gate_piecewise(gate2_t04)_CC_lambda(1)
+1 gate_piecewise(gate2_t04)_CC_lambda(2)
+1 gate_piecewise(gate2_t04)_CC_lambda(3)
+1 gate_piecewise(gate2_t04)_CC_lambda(4)
= 1

c_u_gate_piecewise(gate2_t04)_CC_constraint4(1)_:
+1 gate_piecewise(gate2_t04)_CC_lambda(1)
-1 gate_piecewise(gate2_t04)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate2_t04)_CC_constraint4(2)_:
+1 gate_piecewise(gate2_t04)_CC_lambda(2)
-1 gate_piecewise(gate2_t04)_CC_bin_y(1)
-1 gate_piecewise(gate2_t04)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate2_t04)_CC_constraint4(3)_:
+1 gate_piecewise(gate2_t04)_CC_lambda(3)
-1 gate_piecewise(gate2_t04)_CC_bin_y(2)
-1 gate_piecewise(gate2_t04)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate2_t04)_CC_constraint4(4)_:
+1 gate_piecewise(gate2_t04)_CC_lambda(4)
-1 gate_piecewise(gate2_t04)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(gate2_t04)_CC_constraint5_:
+1 gate_piecewise(gate2_t04)_CC_bin_y(1)
+1 gate_piecewise(gate2_t04)_CC_bin_y(2)
+1 gate_piecewise(gate2_t04)_CC_bin_y(3)
= 1

c_e_gate_piecewise(gate2_t05)_CC_constraint1_:
+1 gate_opening(gate2_t05)
-0.3 gate_piecewise(gate2_t05)_CC_lambda(2)
-0.6 gate_piecewise(gate2_t05)_CC_lambda(3)
-1 gate_piecewise(gate2_t05)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t05)_CC_constraint2_:
+1 gate_capacity(gate2_t05)
-5.5 gate_piecewise(gate2_t05)_CC_lambda(2)
-10.0 gate_piecewise(gate2_t05)_CC_lambda(3)
-14.0 gate_piecewise(gate2_t05)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t05)_CC_constraint3_:
+1 gate_piecewise(gate2_t05)_CC_lambda(1)
+1 gate_piecewise(gate2_t05)_CC_lambda(2)
+1 gate_piecewise(gate2_t05)_CC_lambda(3)
+1 gate_piecewise(gate2_t05)_CC_lambda(4)
= 1

c_u_gate_piecewise(gate2_t05)_CC_constraint4(1)_:
+1 gate_piecewise(gate2_t05)_CC_lambda(1)
-1 gate_piecewise(gate2_t05)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate2_t05)_CC_constraint4(2)_:
+1 gate_piecewise(gate2_t05)_CC_lambda(2)
-1 gate_piecewise(gate2_t05)_CC_bin_y(1)
-1 gate_piecewise(gate2_t05)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate2_t05)_CC_constraint4(3)_:
+1 gate_piecewise(gate2_t05)_CC_lambda(3)
-1 gate_piecewise(gate2_t05)_CC_bin_y(2)
-1 gate_piecewise(gate2_t05)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate2_t05)_CC_constraint4(4)_:
+1 gate_piecewise(gate2_t05)_CC_lambda(4)
-1 gate_piecewise(gate2_t05)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(gate2_t05)_CC_constraint5_:
+1 gate_piecewise(gate2_t05)_CC_bin_y(1)
+1 gate_piecewise(gate2_t05)_CC_bin_y(2)
+1 gate_piecewise(gate2_t05)_CC_bin_y(3)
= 1

c_e_gate_piecewise(gate2_t06)_CC_constraint1_:
+1 gate_opening(gate2_t06)
-0.3 gate_piecewise(gate2_t06)_CC_lambda(2)
-0.6 gate_piecewise(gate2_t06)_CC_lambda(3)
-1 gate_piecewise(gate2_t06)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t06)_CC_constraint2_:
+1 gate_capacity(gate2_t06)
-5.5 gate_piecewise(gate2_t06)_CC_lambda(2)
-10.0 gate_piecewise(gate2_t06)_CC_lambda(3)
-14.0 gate_piecewise(gate2_t06)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t06)_CC_constraint3_:
+1 gate_piecewise(gate2_t06)_CC_lambda(1)
+1 gate_piecewise(gate2_t06)_CC_lambda(2)
+1 gate_piecewise(gate2_t06)_CC_lambda(3)
+1 gate_piecewise(gate2_t06)_CC_lambda(4)
= 1

c_u_gate_piecewise(gate2_t06)_CC_constraint4(1)_:
+1 gate_piecewise(gate2_t06)_CC_lambda(1)
-1 gate_piecewise(gate2_t06)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate2_t06)_CC_constraint4(2)_:
+1 gate_piecewise(gate2_t06)_CC_lambda(2)
-1 gate_piecewise(gate2_t06)_CC_bin_y(1)
-1 gate_piecewise(gate2_t06)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate2_t06)_CC_constraint4(3)_:
+1 gate_piecewise(gate2_t06)_CC_lambda(3)
-1 gate_piecewise(gate2_t06)_CC_bin_y(2)
-1 gate_piecewise(gate2_t06)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate2_t06)_CC_constraint4(4)_:
+1 gate_piecewise(gate2_t06)_CC_lambda(4)
-1 gate_piecewise(gate2_t06)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(gate2_t06)_CC_constraint5_:
+1 gate_piecewise(gate2_t06)_CC_bin_y(1)
+1 gate_piecewise(gate2_t06)_CC_bin_y(2)
+1 gate_piecewise(gate2_t06)_CC_bin_y(3)
= 1

c_e_gate_piecewise(gate2_t07)_CC_constraint1_:
+1 gate_opening(gate2_t07)
-0.3 gate_piecewise(gate2_t07)_CC_lambda(2)
-0.6 gate_piecewise(gate2_t07)_CC_lambda(3)
-1 gate_piecewise(gate2_t07)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t07)_CC_constraint2_:
+1 gate_capacity(gate2_t07)
-5.5 gate_piecewise(gate2_t07)_CC_lambda(2)
-10.0 gate_piecewise(gate2_t07)_CC_lambda(3)
-14.0 gate_piecewise(gate2_t07)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t07)_CC_constraint3_:
+1 gate_piecewise(gate2_t07)_CC_lambda(1)
+1 gate_piecewise(gate2_t07)_CC_lambda(2)
+1 gate_piecewise(gate2_t07)_CC_lambda(3)
+1 gate_piecewise(gate2_t07)_CC_lambda(4)
= 1

c_u_gate_piecewise(gate2_t07)_CC_constraint4(1)_:
+1 gate_piecewise(gate2_t07)_CC_lambda(1)
-1 gate_piecewise(gate2_t07)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate2_t07)_CC_constraint4(2)_:
+1 gate_piecewise(gate2_t07)_CC_lambda(2)
-1 gate_piecewise(gate2_t07)_CC_bin_y(1)
-1 gate_piecewise(gate2_t07)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate2_t07)_CC_constraint4(3)_:
+1 gate_piecewise(gate2_t07)_CC_lambda(3)
-1 gate_piecewise(gate2_t07)_CC_bin_y(2)
-1 gate_piecewise(gate2_t07)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate2_t07)_CC_constraint4(4)_:
+1 gate_piecewise(gate2_t07)_CC_lambda(4)
-1 gate_piecewise(gate2_t07)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(gate2_t07)_CC_constraint5_:
+1 gate_piecewise(gate2_t07)_CC_bin_y(1)
+1 gate_piecewise(gate2_t07)_CC_bin_y(2)
+1 gate_piecewise(gate2_t07)_CC_bin_y(3)
= 1

c_e_gate_piecewise(gate2_t08)_CC_constraint1_:
+1 gate_opening(gate2_t08)
-0.3 gate_piecewise(gate2_t08)_CC_lambda(2)
-0.6 gate_piecewise(gate2_t08)_CC_lambda(3)
-1 gate_piecewise(gate2_t08)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t08)_CC_constraint2_:
+1 gate_capacity(gate2_t08)
-5.5 gate_piecewise(gate2_t08)_CC_lambda(2)
-10.0 gate_piecewise(gate2_t08)_CC_lambda(3)
-14.0 gate_piecewise(gate2_t08)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t08)_CC_constraint3_:
+1 gate_piecewise(gate2_t08)_CC_lambda(1)
+1 gate_piecewise(gate2_t08)_CC_lambda(2)
+1 gate_piecewise(gate2_t08)_CC_lambda(3)
+1 gate_piecewise(gate2_t08)_CC_lambda(4)
= 1

c_u_gate_piecewise(gate2_t08)_CC_constraint4(1)_:
+1 gate_piecewise(gate2_t08)_CC_lambda(1)
-1 gate_piecewise(gate2_t08)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate2_t08)_CC_constraint4(2)_:
+1 gate_piecewise(gate2_t08)_CC_lambda(2)
-1 gate_piecewise(gate2_t08)_CC_bin_y(1)
-1 gate_piecewise(gate2_t08)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate2_t08)_CC_constraint4(3)_:
+1 gate_piecewise(gate2_t08)_CC_lambda(3)
-1 gate_piecewise(gate2_t08)_CC_bin_y(2)
-1 gate_piecewise(gate2_t08)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate2_t08)_CC_constraint4(4)_:
+1 gate_piecewise(gate2_t08)_CC_lambda(4)
-1 gate_piecewise(gate2_t08)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(gate2_t08)_CC_constraint5_:
+1 gate_piecewise(gate2_t08)_CC_bin_y(1)
+1 gate_piecewise(gate2_t08)_CC_bin_y(2)
+1 gate_piecewise(gate2_t08)_CC_bin_y(3)
= 1

c_e_gate_piecewise(gate2_t09)_CC_constraint1_:
+1 gate_opening(gate2_t09)
-0.3 gate_piecewise(gate2_t09)_CC_lambda(2)
-0.6 gate_piecewise(gate2_t09)_CC_lambda(3)
-1 gate_piecewise(gate2_t09)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t09)_CC_constraint2_:
+1 gate_capacity(gate2_t09)
-5.5 gate_piecewise(gate2_t09)_CC_lambda(2)
-10.0 gate_piecewise(gate2_t09)_CC_lambda(3)
-14.0 gate_piecewise(gate2_t09)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t09)_CC_constraint3_:
+1 gate_piecewise(gate2_t09)_CC_lambda(1)
+1 gate_piecewise(gate2_t09)_CC_lambda(2)
+1 gate_piecewise(gate2_t09)_CC_lambda(3)
+1 gate_piecewise(gate2_t09)_CC_lambda(4)
= 1

c_u_gate_piecewise(gate2_t09)_CC_constraint4(1)_:
+1 gate_piecewise(gate2_t09)_CC_lambda(1)
-1 gate_piecewise(gate2_t09)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate2_t09)_CC_constraint4(2)_:
+1 gate_piecewise(gate2_t09)_CC_lambda(2)
-1 gate_piecewise(gate2_t09)_CC_bin_y(1)
-1 gate_piecewise(gate2_t09)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate2_t09)_CC_constraint4(3)_:
+1 gate_piecewise(gate2_t09)_CC_lambda(3)
-1 gate_piecewise(gate2_t09)_CC_bin_y(2)
-1 gate_piecewise(gate2_t09)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate2_t09)_CC_constraint4(4)_:
+1 gate_piecewise(gate2_t09)_CC_lambda(4)
-1 gate_piecewise(gate2_t09)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(gate2_t09)_CC_constraint5_:
+1 gate_piecewise(gate2_t09)_CC_bin_y(1)
+1 gate_piecewise(gate2_t09)_CC_bin_y(2)
+1 gate_piecewise(gate2_t09)_CC_bin_y(3)
= 1

c_e_gate_piecewise(gate2_t10)_CC_constraint1_:
+1 gate_opening(gate2_t10)
-0.3 gate_piecewise(gate2_t10)_CC_lambda(2)
-0.6 gate_piecewise(gate2_t10)_CC_lambda(3)
-1 gate_piecewise(gate2_t10)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t10)_CC_constraint2_:
+1 gate_capacity(gate2_t10)
-5.5 gate_piecewise(gate2_t10)_CC_lambda(2)
-10.0 gate_piecewise(gate2_t10)_CC_lambda(3)
-14.0 gate_piecewise(gate2_t10)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t10)_CC_constraint3_:
+1 gate_piecewise(gate2_t10)_CC_lambda(1)
+1 gate_piecewise(gate2_t10)_CC_lambda(2)
+1 gate_piecewise(gate2_t10)_CC_lambda(3)
+1 gate_piecewise(gate2_t10)_CC_lambda(4)
= 1

c_u_gate_piecewise(gate2_t10)_CC_constraint4(1)_:
+1 gate_piecewise(gate2_t10)_CC_lambda(1)
-1 gate_piecewise(gate2_t10)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate2_t10)_CC_constraint4(2)_:
+1 gate_piecewise(gate2_t10)_CC_lambda(2)
-1 gate_piecewise(gate2_t10)_CC_bin_y(1)
-1 gate_piecewise(gate2_t10)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate2_t10)_CC_constraint4(3)_:
+1 gate_piecewise(gate2_t10)_CC_lambda(3)
-1 gate_piecewise(gate2_t10)_CC_bin_y(2)
-1 gate_piecewise(gate2_t10)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate2_t10)_CC_constraint4(4)_:
+1 gate_piecewise(gate2_t10)_CC_lambda(4)
-1 gate_piecewise(gate2_t10)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(gate2_t10)_CC_constraint5_:
+1 gate_piecewise(gate2_t10)_CC_bin_y(1)
+1 gate_piecewise(gate2_t10)_CC_bin_y(2)
+1 gate_piecewise(gate2_t10)_CC_bin_y(3)
= 1

c_e_gate_piecewise(gate2_t11)_CC_constraint1_:
+1 gate_opening(gate2_t11)
-0.3 gate_piecewise(gate2_t11)_CC_lambda(2)
-0.6 gate_piecewise(gate2_t11)_CC_lambda(3)
-1 gate_piecewise(gate2_t11)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t11)_CC_constraint2_:
+1 gate_capacity(gate2_t11)
-5.5 gate_piecewise(gate2_t11)_CC_lambda(2)
-10.0 gate_piecewise(gate2_t11)_CC_lambda(3)
-14.0 gate_piecewise(gate2_t11)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t11)_CC_constraint3_:
+1 gate_piecewise(gate2_t11)_CC_lambda(1)
+1 gate_piecewise(gate2_t11)_CC_lambda(2)
+1 gate_piecewise(gate2_t11)_CC_lambda(3)
+1 gate_piecewise(gate2_t11)_CC_lambda(4)
= 1

c_u_gate_piecewise(gate2_t11)_CC_constraint4(1)_:
+1 gate_piecewise(gate2_t11)_CC_lambda(1)
-1 gate_piecewise(gate2_t11)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate2_t11)_CC_constraint4(2)_:
+1 gate_piecewise(gate2_t11)_CC_lambda(2)
-1 gate_piecewise(gate2_t11)_CC_bin_y(1)
-1 gate_piecewise(gate2_t11)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate2_t11)_CC_constraint4(3)_:
+1 gate_piecewise(gate2_t11)_CC_lambda(3)
-1 gate_piecewise(gate2_t11)_CC_bin_y(2)
-1 gate_piecewise(gate2_t11)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate2_t11)_CC_constraint4(4)_:
+1 gate_piecewise(gate2_t11)_CC_lambda(4)
-1 gate_piecewise(gate2_t11)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(gate2_t11)_CC_constraint5_:
+1 gate_piecewise(gate2_t11)_CC_bin_y(1)
+1 gate_piecewise(gate2_t11)_CC_bin_y(2)
+1 gate_piecewise(gate2_t11)_CC_bin_y(3)
= 1

c_e_gate_piecewise(gate2_t12)_CC_constraint1_:
+1 gate_opening(gate2_t12)
-0.3 gate_piecewise(gate2_t12)_CC_lambda(2)
-0.6 gate_piecewise(gate2_t12)_CC_lambda(3)
-1 gate_piecewise(gate2_t12)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t12)_CC_constraint2_:
+1 gate_capacity(gate2_t12)
-5.5 gate_piecewise(gate2_t12)_CC_lambda(2)
-10.0 gate_piecewise(gate2_t12)_CC_lambda(3)
-14.0 gate_piecewise(gate2_t12)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t12)_CC_constraint3_:
+1 gate_piecewise(gate2_t12)_CC_lambda(1)
+1 gate_piecewise(gate2_t12)_CC_lambda(2)
+1 gate_piecewise(gate2_t12)_CC_lambda(3)
+1 gate_piecewise(gate2_t12)_CC_lambda(4)
= 1

c_u_gate_piecewise(gate2_t12)_CC_constraint4(1)_:
+1 gate_piecewise(gate2_t12)_CC_lambda(1)
-1 gate_piecewise(gate2_t12)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate2_t12)_CC_constraint4(2)_:
+1 gate_piecewise(gate2_t12)_CC_lambda(2)
-1 gate_piecewise(gate2_t12)_CC_bin_y(1)
-1 gate_piecewise(gate2_t12)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate2_t12)_CC_constraint4(3)_:
+1 gate_piecewise(gate2_t12)_CC_lambda(3)
-1 gate_piecewise(gate2_t12)_CC_bin_y(2)
-1 gate_piecewise(gate2_t12)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate2_t12)_CC_constraint4(4)_:
+1 gate_piecewise(gate2_t12)_CC_lambda(4)
-1 gate_piecewise(gate2_t12)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(gate2_t12)_CC_constraint5_:
+1 gate_piecewise(gate2_t12)_CC_bin_y(1)
+1 gate_piecewise(gate2_t12)_CC_bin_y(2)
+1 gate_piecewise(gate2_t12)_CC_bin_y(3)
= 1

c_e_gate_piecewise(gate2_t13)_CC_constraint1_:
+1 gate_opening(gate2_t13)
-0.3 gate_piecewise(gate2_t13)_CC_lambda(2)
-0.6 gate_piecewise(gate2_t13)_CC_lambda(3)
-1 gate_piecewise(gate2_t13)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t13)_CC_constraint2_:
+1 gate_capacity(gate2_t13)
-5.5 gate_piecewise(gate2_t13)_CC_lambda(2)
-10.0 gate_piecewise(gate2_t13)_CC_lambda(3)
-14.0 gate_piecewise(gate2_t13)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t13)_CC_constraint3_:
+1 gate_piecewise(gate2_t13)_CC_lambda(1)
+1 gate_piecewise(gate2_t13)_CC_lambda(2)
+1 gate_piecewise(gate2_t13)_CC_lambda(3)
+1 gate_piecewise(gate2_t13)_CC_lambda(4)
= 1

c_u_gate_piecewise(gate2_t13)_CC_constraint4(1)_:
+1 gate_piecewise(gate2_t13)_CC_lambda(1)
-1 gate_piecewise(gate2_t13)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate2_t13)_CC_constraint4(2)_:
+1 gate_piecewise(gate2_t13)_CC_lambda(2)
-1 gate_piecewise(gate2_t13)_CC_bin_y(1)
-1 gate_piecewise(gate2_t13)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate2_t13)_CC_constraint4(3)_:
+1 gate_piecewise(gate2_t13)_CC_lambda(3)
-1 gate_piecewise(gate2_t13)_CC_bin_y(2)
-1 gate_piecewise(gate2_t13)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate2_t13)_CC_constraint4(4)_:
+1 gate_piecewise(gate2_t13)_CC_lambda(4)
-1 gate_piecewise(gate2_t13)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(gate2_t13)_CC_constraint5_:
+1 gate_piecewise(gate2_t13)_CC_bin_y(1)
+1 gate_piecewise(gate2_t13)_CC_bin_y(2)
+1 gate_piecewise(gate2_t13)_CC_bin_y(3)
= 1

c_e_gate_piecewise(gate2_t14)_CC_constraint1_:
+1 gate_opening(gate2_t14)
-0.3 gate_piecewise(gate2_t14)_CC_lambda(2)
-0.6 gate_piecewise(gate2_t14)_CC_lambda(3)
-1 gate_piecewise(gate2_t14)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t14)_CC_constraint2_:
+1 gate_capacity(gate2_t14)
-5.5 gate_piecewise(gate2_t14)_CC_lambda(2)
-10.0 gate_piecewise(gate2_t14)_CC_lambda(3)
-14.0 gate_piecewise(gate2_t14)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t14)_CC_constraint3_:
+1 gate_piecewise(gate2_t14)_CC_lambda(1)
+1 gate_piecewise(gate2_t14)_CC_lambda(2)
+1 gate_piecewise(gate2_t14)_CC_lambda(3)
+1 gate_piecewise(gate2_t14)_CC_lambda(4)
= 1

c_u_gate_piecewise(gate2_t14)_CC_constraint4(1)_:
+1 gate_piecewise(gate2_t14)_CC_lambda(1)
-1 gate_piecewise(gate2_t14)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate2_t14)_CC_constraint4(2)_:
+1 gate_piecewise(gate2_t14)_CC_lambda(2)
-1 gate_piecewise(gate2_t14)_CC_bin_y(1)
-1 gate_piecewise(gate2_t14)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate2_t14)_CC_constraint4(3)_:
+1 gate_piecewise(gate2_t14)_CC_lambda(3)
-1 gate_piecewise(gate2_t14)_CC_bin_y(2)
-1 gate_piecewise(gate2_t14)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate2_t14)_CC_constraint4(4)_:
+1 gate_piecewise(gate2_t14)_CC_lambda(4)
-1 gate_piecewise(gate2_t14)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(gate2_t14)_CC_constraint5_:
+1 gate_piecewise(gate2_t14)_CC_bin_y(1)
+1 gate_piecewise(gate2_t14)_CC_bin_y(2)
+1 gate_piecewise(gate2_t14)_CC_bin_y(3)
= 1

c_e_gate_piecewise(gate2_t15)_CC_constraint1_:
+1 gate_opening(gate2_t15)
-0.3 gate_piecewise(gate2_t15)_CC_lambda(2)
-0.6 gate_piecewise(gate2_t15)_CC_lambda(3)
-1 gate_piecewise(gate2_t15)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t15)_CC_constraint2_:
+1 gate_capacity(gate2_t15)
-5.5 gate_piecewise(gate2_t15)_CC_lambda(2)
-10.0 gate_piecewise(gate2_t15)_CC_lambda(3)
-14.0 gate_piecewise(gate2_t15)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t15)_CC_constraint3_:
+1 gate_piecewise(gate2_t15)_CC_lambda(1)
+1 gate_piecewise(gate2_t15)_CC_lambda(2)
+1 gate_piecewise(gate2_t15)_CC_lambda(3)
+1 gate_piecewise(gate2_t15)_CC_lambda(4)
= 1

c_u_gate_piecewise(gate2_t15)_CC_constraint4(1)_:
+1 gate_piecewise(gate2_t15)_CC_lambda(1)
-1 gate_piecewise(gate2_t15)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate2_t15)_CC_constraint4(2)_:
+1 gate_piecewise(gate2_t15)_CC_lambda(2)
-1 gate_piecewise(gate2_t15)_CC_bin_y(1)
-1 gate_piecewise(gate2_t15)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate2_t15)_CC_constraint4(3)_:
+1 gate_piecewise(gate2_t15)_CC_lambda(3)
-1 gate_piecewise(gate2_t15)_CC_bin_y(2)
-1 gate_piecewise(gate2_t15)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate2_t15)_CC_constraint4(4)_:
+1 gate_piecewise(gate2_t15)_CC_lambda(4)
-1 gate_piecewise(gate2_t15)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(gate2_t15)_CC_constraint5_:
+1 gate_piecewise(gate2_t15)_CC_bin_y(1)
+1 gate_piecewise(gate2_t15)_CC_bin_y(2)
+1 gate_piecewise(gate2_t15)_CC_bin_y(3)
= 1

c_e_gate_piecewise(gate2_t16)_CC_constraint1_:
+1 gate_opening(gate2_t16)
-0.3 gate_piecewise(gate2_t16)_CC_lambda(2)
-0.6 gate_piecewise(gate2_t16)_CC_lambda(3)
-1 gate_piecewise(gate2_t16)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t16)_CC_constraint2_:
+1 gate_capacity(gate2_t16)
-5.5 gate_piecewise(gate2_t16)_CC_lambda(2)
-10.0 gate_piecewise(gate2_t16)_CC_lambda(3)
-14.0 gate_piecewise(gate2_t16)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t16)_CC_constraint3_:
+1 gate_piecewise(gate2_t16)_CC_lambda(1)
+1 gate_piecewise(gate2_t16)_CC_lambda(2)
+1 gate_piecewise(gate2_t16)_CC_lambda(3)
+1 gate_piecewise(gate2_t16)_CC_lambda(4)
= 1

c_u_gate_piecewise(gate2_t16)_CC_constraint4(1)_:
+1 gate_piecewise(gate2_t16)_CC_lambda(1)
-1 gate_piecewise(gate2_t16)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate2_t16)_CC_constraint4(2)_:
+1 gate_piecewise(gate2_t16)_CC_lambda(2)
-1 gate_piecewise(gate2_t16)_CC_bin_y(1)
-1 gate_piecewise(gate2_t16)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate2_t16)_CC_constraint4(3)_:
+1 gate_piecewise(gate2_t16)_CC_lambda(3)
-1 gate_piecewise(gate2_t16)_CC_bin_y(2)
-1 gate_piecewise(gate2_t16)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate2_t16)_CC_constraint4(4)_:
+1 gate_piecewise(gate2_t16)_CC_lambda(4)
-1 gate_piecewise(gate2_t16)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(gate2_t16)_CC_constraint5_:
+1 gate_piecewise(gate2_t16)_CC_bin_y(1)
+1 gate_piecewise(gate2_t16)_CC_bin_y(2)
+1 gate_piecewise(gate2_t16)_CC_bin_y(3)
= 1

c_e_gate_piecewise(gate2_t17)_CC_constraint1_:
+1 gate_opening(gate2_t17)
-0.3 gate_piecewise(gate2_t17)_CC_lambda(2)
-0.6 gate_piecewise(gate2_t17)_CC_lambda(3)
-1 gate_piecewise(gate2_t17)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t17)_CC_constraint2_:
+1 gate_capacity(gate2_t17)
-5.5 gate_piecewise(gate2_t17)_CC_lambda(2)
-10.0 gate_piecewise(gate2_t17)_CC_lambda(3)
-14.0 gate_piecewise(gate2_t17)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t17)_CC_constraint3_:
+1 gate_piecewise(gate2_t17)_CC_lambda(1)
+1 gate_piecewise(gate2_t17)_CC_lambda(2)
+1 gate_piecewise(gate2_t17)_CC_lambda(3)
+1 gate_piecewise(gate2_t17)_CC_lambda(4)
= 1

c_u_gate_piecewise(gate2_t17)_CC_constraint4(1)_:
+1 gate_piecewise(gate2_t17)_CC_lambda(1)
-1 gate_piecewise(gate2_t17)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate2_t17)_CC_constraint4(2)_:
+1 gate_piecewise(gate2_t17)_CC_lambda(2)
-1 gate_piecewise(gate2_t17)_CC_bin_y(1)
-1 gate_piecewise(gate2_t17)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate2_t17)_CC_constraint4(3)_:
+1 gate_piecewise(gate2_t17)_CC_lambda(3)
-1 gate_piecewise(gate2_t17)_CC_bin_y(2)
-1 gate_piecewise(gate2_t17)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate2_t17)_CC_constraint4(4)_:
+1 gate_piecewise(gate2_t17)_CC_lambda(4)
-1 gate_piecewise(gate2_t17)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(gate2_t17)_CC_constraint5_:
+1 gate_piecewise(gate2_t17)_CC_bin_y(1)
+1 gate_piecewise(gate2_t17)_CC_bin_y(2)
+1 gate_piecewise(gate2_t17)_CC_bin_y(3)
= 1

c_e_gate_piecewise(gate2_t18)_CC_constraint1_:
+1 gate_opening(gate2_t18)
-0.3 gate_piecewise(gate2_t18)_CC_lambda(2)
-0.6 gate_piecewise(gate2_t18)_CC_lambda(3)
-1 gate_piecewise(gate2_t18)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t18)_CC_constraint2_:
+1 gate_capacity(gate2_t18)
-5.5 gate_piecewise(gate2_t18)_CC_lambda(2)
-10.0 gate_piecewise(gate2_t18)_CC_lambda(3)
-14.0 gate_piecewise(gate2_t18)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t18)_CC_constraint3_:
+1 gate_piecewise(gate2_t18)_CC_lambda(1)
+1 gate_piecewise(gate2_t18)_CC_lambda(2)
+1 gate_piecewise(gate2_t18)_CC_lambda(3)
+1 gate_piecewise(gate2_t18)_CC_lambda(4)
= 1

c_u_gate_piecewise(gate2_t18)_CC_constraint4(1)_:
+1 gate_piecewise(gate2_t18)_CC_lambda(1)
-1 gate_piecewise(gate2_t18)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate2_t18)_CC_constraint4(2)_:
+1 gate_piecewise(gate2_t18)_CC_lambda(2)
-1 gate_piecewise(gate2_t18)_CC_bin_y(1)
-1 gate_piecewise(gate2_t18)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate2_t18)_CC_constraint4(3)_:
+1 gate_piecewise(gate2_t18)_CC_lambda(3)
-1 gate_piecewise(gate2_t18)_CC_bin_y(2)
-1 gate_piecewise(gate2_t18)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate2_t18)_CC_constraint4(4)_:
+1 gate_piecewise(gate2_t18)_CC_lambda(4)
-1 gate_piecewise(gate2_t18)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(gate2_t18)_CC_constraint5_:
+1 gate_piecewise(gate2_t18)_CC_bin_y(1)
+1 gate_piecewise(gate2_t18)_CC_bin_y(2)
+1 gate_piecewise(gate2_t18)_CC_bin_y(3)
= 1

c_e_gate_piecewise(gate2_t19)_CC_constraint1_:
+1 gate_opening(gate2_t19)
-0.3 gate_piecewise(gate2_t19)_CC_lambda(2)
-0.6 gate_piecewise(gate2_t19)_CC_lambda(3)
-1 gate_piecewise(gate2_t19)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t19)_CC_constraint2_:
+1 gate_capacity(gate2_t19)
-5.5 gate_piecewise(gate2_t19)_CC_lambda(2)
-10.0 gate_piecewise(gate2_t19)_CC_lambda(3)
-14.0 gate_piecewise(gate2_t19)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t19)_CC_constraint3_:
+1 gate_piecewise(gate2_t19)_CC_lambda(1)
+1 gate_piecewise(gate2_t19)_CC_lambda(2)
+1 gate_piecewise(gate2_t19)_CC_lambda(3)
+1 gate_piecewise(gate2_t19)_CC_lambda(4)
= 1

c_u_gate_piecewise(gate2_t19)_CC_constraint4(1)_:
+1 gate_piecewise(gate2_t19)_CC_lambda(1)
-1 gate_piecewise(gate2_t19)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate2_t19)_CC_constraint4(2)_:
+1 gate_piecewise(gate2_t19)_CC_lambda(2)
-1 gate_piecewise(gate2_t19)_CC_bin_y(1)
-1 gate_piecewise(gate2_t19)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate2_t19)_CC_constraint4(3)_:
+1 gate_piecewise(gate2_t19)_CC_lambda(3)
-1 gate_piecewise(gate2_t19)_CC_bin_y(2)
-1 gate_piecewise(gate2_t19)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate2_t19)_CC_constraint4(4)_:
+1 gate_piecewise(gate2_t19)_CC_lambda(4)
-1 gate_piecewise(gate2_t19)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(gate2_t19)_CC_constraint5_:
+1 gate_piecewise(gate2_t19)_CC_bin_y(1)
+1 gate_piecewise(gate2_t19)_CC_bin_y(2)
+1 gate_piecewise(gate2_t19)_CC_bin_y(3)
= 1

c_e_gate_piecewise(gate2_t20)_CC_constraint1_:
+1 gate_opening(gate2_t20)
-0.3 gate_piecewise(gate2_t20)_CC_lambda(2)
-0.6 gate_piecewise(gate2_t20)_CC_lambda(3)
-1 gate_piecewise(gate2_t20)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t20)_CC_constraint2_:
+1 gate_capacity(gate2_t20)
-5.5 gate_piecewise(gate2_t20)_CC_lambda(2)
-10.0 gate_piecewise(gate2_t20)_CC_lambda(3)
-14.0 gate_piecewise(gate2_t20)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t20)_CC_constraint3_:
+1 gate_piecewise(gate2_t20)_CC_lambda(1)
+1 gate_piecewise(gate2_t20)_CC_lambda(2)
+1 gate_piecewise(gate2_t20)_CC_lambda(3)
+1 gate_piecewise(gate2_t20)_CC_lambda(4)
= 1

c_u_gate_piecewise(gate2_t20)_CC_constraint4(1)_:
+1 gate_piecewise(gate2_t20)_CC_lambda(1)
-1 gate_piecewise(gate2_t20)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate2_t20)_CC_constraint4(2)_:
+1 gate_piecewise(gate2_t20)_CC_lambda(2)
-1 gate_piecewise(gate2_t20)_CC_bin_y(1)
-1 gate_piecewise(gate2_t20)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate2_t20)_CC_constraint4(3)_:
+1 gate_piecewise(gate2_t20)_CC_lambda(3)
-1 gate_piecewise(gate2_t20)_CC_bin_y(2)
-1 gate_piecewise(gate2_t20)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate2_t20)_CC_constraint4(4)_:
+1 gate_piecewise(gate2_t20)_CC_lambda(4)
-1 gate_piecewise(gate2_t20)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(gate2_t20)_CC_constraint5_:
+1 gate_piecewise(gate2_t20)_CC_bin_y(1)
+1 gate_piecewise(gate2_t20)_CC_bin_y(2)
+1 gate_piecewise(gate2_t20)_CC_bin_y(3)
= 1

c_e_gate_piecewise(gate2_t21)_CC_constraint1_:
+1 gate_opening(gate2_t21)
-0.3 gate_piecewise(gate2_t21)_CC_lambda(2)
-0.6 gate_piecewise(gate2_t21)_CC_lambda(3)
-1 gate_piecewise(gate2_t21)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t21)_CC_constraint2_:
+1 gate_capacity(gate2_t21)
-5.5 gate_piecewise(gate2_t21)_CC_lambda(2)
-10.0 gate_piecewise(gate2_t21)_CC_lambda(3)
-14.0 gate_piecewise(gate2_t21)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t21)_CC_constraint3_:
+1 gate_piecewise(gate2_t21)_CC_lambda(1)
+1 gate_piecewise(gate2_t21)_CC_lambda(2)
+1 gate_piecewise(gate2_t21)_CC_lambda(3)
+1 gate_piecewise(gate2_t21)_CC_lambda(4)
= 1

c_u_gate_piecewise(gate2_t21)_CC_constraint4(1)_:
+1 gate_piecewise(gate2_t21)_CC_lambda(1)
-1 gate_piecewise(gate2_t21)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate2_t21)_CC_constraint4(2)_:
+1 gate_piecewise(gate2_t21)_CC_lambda(2)
-1 gate_piecewise(gate2_t21)_CC_bin_y(1)
-1 gate_piecewise(gate2_t21)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate2_t21)_CC_constraint4(3)_:
+1 gate_piecewise(gate2_t21)_CC_lambda(3)
-1 gate_piecewise(gate2_t21)_CC_bin_y(2)
-1 gate_piecewise(gate2_t21)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate2_t21)_CC_constraint4(4)_:
+1 gate_piecewise(gate2_t21)_CC_lambda(4)
-1 gate_piecewise(gate2_t21)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(gate2_t21)_CC_constraint5_:
+1 gate_piecewise(gate2_t21)_CC_bin_y(1)
+1 gate_piecewise(gate2_t21)_CC_bin_y(2)
+1 gate_piecewise(gate2_t21)_CC_bin_y(3)
= 1

c_e_gate_piecewise(gate2_t22)_CC_constraint1_:
+1 gate_opening(gate2_t22)
-0.3 gate_piecewise(gate2_t22)_CC_lambda(2)
-0.6 gate_piecewise(gate2_t22)_CC_lambda(3)
-1 gate_piecewise(gate2_t22)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t22)_CC_constraint2_:
+1 gate_capacity(gate2_t22)
-5.5 gate_piecewise(gate2_t22)_CC_lambda(2)
-10.0 gate_piecewise(gate2_t22)_CC_lambda(3)
-14.0 gate_piecewise(gate2_t22)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t22)_CC_constraint3_:
+1 gate_piecewise(gate2_t22)_CC_lambda(1)
+1 gate_piecewise(gate2_t22)_CC_lambda(2)
+1 gate_piecewise(gate2_t22)_CC_lambda(3)
+1 gate_piecewise(gate2_t22)_CC_lambda(4)
= 1

c_u_gate_piecewise(gate2_t22)_CC_constraint4(1)_:
+1 gate_piecewise(gate2_t22)_CC_lambda(1)
-1 gate_piecewise(gate2_t22)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate2_t22)_CC_constraint4(2)_:
+1 gate_piecewise(gate2_t22)_CC_lambda(2)
-1 gate_piecewise(gate2_t22)_CC_bin_y(1)
-1 gate_piecewise(gate2_t22)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate2_t22)_CC_constraint4(3)_:
+1 gate_piecewise(gate2_t22)_CC_lambda(3)
-1 gate_piecewise(gate2_t22)_CC_bin_y(2)
-1 gate_piecewise(gate2_t22)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate2_t22)_CC_constraint4(4)_:
+1 gate_piecewise(gate2_t22)_CC_lambda(4)
-1 gate_piecewise(gate2_t22)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(gate2_t22)_CC_constraint5_:
+1 gate_piecewise(gate2_t22)_CC_bin_y(1)
+1 gate_piecewise(gate2_t22)_CC_bin_y(2)
+1 gate_piecewise(gate2_t22)_CC_bin_y(3)
= 1

c_e_gate_piecewise(gate2_t23)_CC_constraint1_:
+1 gate_opening(gate2_t23)
-0.3 gate_piecewise(gate2_t23)_CC_lambda(2)
-0.6 gate_piecewise(gate2_t23)_CC_lambda(3)
-1 gate_piecewise(gate2_t23)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t23)_CC_constraint2_:
+1 gate_capacity(gate2_t23)
-5.5 gate_piecewise(gate2_t23)_CC_lambda(2)
-10.0 gate_piecewise(gate2_t23)_CC_lambda(3)
-14.0 gate_piecewise(gate2_t23)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t23)_CC_constraint3_:
+1 gate_piecewise(gate2_t23)_CC_lambda(1)
+1 gate_piecewise(gate2_t23)_CC_lambda(2)
+1 gate_piecewise(gate2_t23)_CC_lambda(3)
+1 gate_piecewise(gate2_t23)_CC_lambda(4)
= 1

c_u_gate_piecewise(gate2_t23)_CC_constraint4(1)_:
+1 gate_piecewise(gate2_t23)_CC_lambda(1)
-1 gate_piecewise(gate2_t23)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate2_t23)_CC_constraint4(2)_:
+1 gate_piecewise(gate2_t23)_CC_lambda(2)
-1 gate_piecewise(gate2_t23)_CC_bin_y(1)
-1 gate_piecewise(gate2_t23)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate2_t23)_CC_constraint4(3)_:
+1 gate_piecewise(gate2_t23)_CC_lambda(3)
-1 gate_piecewise(gate2_t23)_CC_bin_y(2)
-1 gate_piecewise(gate2_t23)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate2_t23)_CC_constraint4(4)_:
+1 gate_piecewise(gate2_t23)_CC_lambda(4)
-1 gate_piecewise(gate2_t23)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(gate2_t23)_CC_constraint5_:
+1 gate_piecewise(gate2_t23)_CC_bin_y(1)
+1 gate_piecewise(gate2_t23)_CC_bin_y(2)
+1 gate_piecewise(gate2_t23)_CC_bin_y(3)
= 1

c_e_gate_piecewise(gate2_t24)_CC_constraint1_:
+1 gate_opening(gate2_t24)
-0.3 gate_piecewise(gate2_t24)_CC_lambda(2)
-0.6 gate_piecewise(gate2_t24)_CC_lambda(3)
-1 gate_piecewise(gate2_t24)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t24)_CC_constraint2_:
+1 gate_capacity(gate2_t24)
-5.5 gate_piecewise(gate2_t24)_CC_lambda(2)
-10.0 gate_piecewise(gate2_t24)_CC_lambda(3)
-14.0 gate_piecewise(gate2_t24)_CC_lambda(4)
= 0

c_e_gate_piecewise(gate2_t24)_CC_constraint3_:
+1 gate_piecewise(gate2_t24)_CC_lambda(1)
+1 gate_piecewise(gate2_t24)_CC_lambda(2)
+1 gate_piecewise(gate2_t24)_CC_lambda(3)
+1 gate_piecewise(gate2_t24)_CC_lambda(4)
= 1

c_u_gate_piecewise(gate2_t24)_CC_constraint4(1)_:
+1 gate_piecewise(gate2_t24)_CC_lambda(1)
-1 gate_piecewise(gate2_t24)_CC_bin_y(1)
<= 0

c_u_gate_piecewise(gate2_t24)_CC_constraint4(2)_:
+1 gate_piecewise(gate2_t24)_CC_lambda(2)
-1 gate_piecewise(gate2_t24)_CC_bin_y(1)
-1 gate_piecewise(gate2_t24)_CC_bin_y(2)
<= 0

c_u_gate_piecewise(gate2_t24)_CC_constraint4(3)_:
+1 gate_piecewise(gate2_t24)_CC_lambda(3)
-1 gate_piecewise(gate2_t24)_CC_bin_y(2)
-1 gate_piecewise(gate2_t24)_CC_bin_y(3)
<= 0

c_u_gate_piecewise(gate2_t24)_CC_constraint4(4)_:
+1 gate_piecewise(gate2_t24)_CC_lambda(4)
-1 gate_piecewise(gate2_t24)_CC_bin_y(3)
<= 0

c_e_gate_piecewise(gate2_t24)_CC_constraint5_:
+1 gate_piecewise(gate2_t24)_CC_bin_y(1)
+1 gate_piecewise(gate2_t24)_CC_bin_y(2)
+1 gate_piecewise(gate2_t24)_CC_bin_y(3)
= 1

bounds
   0 <= shortage(outlet_t01) <= +inf
   0 <= shortage(outlet_t02) <= +inf
   0 <= shortage(outlet_t03) <= +inf
   0 <= shortage(outlet_t04) <= +inf
   0 <= shortage(outlet_t05) <= +inf
   0 <= shortage(outlet_t06) <= +inf
   0 <= shortage(outlet_t07) <= +inf
   0 <= shortage(outlet_t08) <= +inf
   0 <= shortage(outlet_t09) <= +inf
   0 <= shortage(outlet_t10) <= +inf
   0 <= shortage(outlet_t11) <= +inf
   0 <= shortage(outlet_t12) <= +inf
   0 <= shortage(outlet_t13) <= +inf
   0 <= shortage(outlet_t14) <= +inf
   0 <= shortage(outlet_t15) <= +inf
   0 <= shortage(outlet_t16) <= +inf
   0 <= shortage(outlet_t17) <= +inf
   0 <= shortage(outlet_t18) <= +inf
   0 <= shortage(outlet_t19) <= +inf
   0 <= shortage(outlet_t20) <= +inf
   0 <= shortage(outlet_t21) <= +inf
   0 <= shortage(outlet_t22) <= +inf
   0 <= shortage(outlet_t23) <= +inf
   0 <= shortage(outlet_t24) <= +inf
   0.0 <= flow(gate1_t01) <= 18.0
   0.0 <= flow(gate1_t02) <= 18.0
   0.0 <= flow(gate1_t03) <= 18.0
   0.0 <= flow(gate1_t04) <= 18.0
   0.0 <= flow(gate1_t05) <= 18.0
   0.0 <= flow(gate1_t06) <= 18.0
   0.0 <= flow(gate1_t07) <= 18.0
   0.0 <= flow(gate1_t08) <= 18.0
   0.0 <= flow(gate1_t09) <= 18.0
   0.0 <= flow(gate1_t10) <= 18.0
   0.0 <= flow(gate1_t11) <= 18.0
   0.0 <= flow(gate1_t12) <= 18.0
   0.0 <= flow(gate1_t13) <= 18.0
   0.0 <= flow(gate1_t14) <= 18.0
   0.0 <= flow(gate1_t15) <= 18.0
   0.0 <= flow(gate1_t16) <= 18.0
   0.0 <= flow(gate1_t17) <= 18.0
   0.0 <= flow(gate1_t18) <= 18.0
   0.0 <= flow(gate1_t19) <= 18.0
   0.0 <= flow(gate1_t20) <= 18.0
   0.0 <= flow(gate1_t21) <= 18.0
   0.0 <= flow(gate1_t22) <= 18.0
   0.0 <= flow(gate1_t23) <= 18.0
   0.0 <= flow(gate1_t24) <= 18.0
   0.0 <= flow(gate2_t01) <= 15.0
   0.0 <= flow(gate2_t02) <= 15.0
   0.0 <= flow(gate2_t03) <= 15.0
   0.0 <= flow(gate2_t04) <= 15.0
   0.0 <= flow(gate2_t05) <= 15.0
   0.0 <= flow(gate2_t06) <= 15.0
   0.0 <= flow(gate2_t07) <= 15.0
   0.0 <= flow(gate2_t08) <= 15.0
   0.0 <= flow(gate2_t09) <= 15.0
   0.0 <= flow(gate2_t10) <= 15.0
   0.0 <= flow(gate2_t11) <= 15.0
   0.0 <= flow(gate2_t12) <= 15.0
   0.0 <= flow(gate2_t13) <= 15.0
   0.0 <= flow(gate2_t14) <= 15.0
   0.0 <= flow(gate2_t15) <= 15.0
   0.0 <= flow(gate2_t16) <= 15.0
   0.0 <= flow(gate2_t17) <= 15.0
   0.0 <= flow(gate2_t18) <= 15.0
   0.0 <= flow(gate2_t19) <= 15.0
   0.0 <= flow(gate2_t20) <= 15.0
   0.0 <= flow(gate2_t21) <= 15.0
   0.0 <= flow(gate2_t22) <= 15.0
   0.0 <= flow(gate2_t23) <= 15.0
   0.0 <= flow(gate2_t24) <= 15.0
   0.0 <= flow(channel23_t01) <= 15.0
   0.0 <= flow(channel23_t02) <= 15.0
   0.0 <= flow(channel23_t03) <= 15.0
   0.0 <= flow(channel23_t04) <= 15.0
   0.0 <= flow(channel23_t05) <= 15.0
   0.0 <= flow(channel23_t06) <= 15.0
   0.0 <= flow(channel23_t07) <= 15.0
   0.0 <= flow(channel23_t08) <= 15.0
   0.0 <= flow(channel23_t09) <= 15.0
   0.0 <= flow(channel23_t10) <= 15.0
   0.0 <= flow(channel23_t11) <= 15.0
   0.0 <= flow(channel23_t12) <= 15.0
   0.0 <= flow(channel23_t13) <= 15.0
   0.0 <= flow(channel23_t14) <= 15.0
   0.0 <= flow(channel23_t15) <= 15.0
   0.0 <= flow(channel23_t16) <= 15.0
   0.0 <= flow(channel23_t17) <= 15.0
   0.0 <= flow(channel23_t18) <= 15.0
   0.0 <= flow(channel23_t19) <= 15.0
   0.0 <= flow(channel23_t20) <= 15.0
   0.0 <= flow(channel23_t21) <= 15.0
   0.0 <= flow(channel23_t22) <= 15.0
   0.0 <= flow(channel23_t23) <= 15.0
   0.0 <= flow(channel23_t24) <= 15.0
   0 <= gate_capacity(gate1_t01) <= +inf
   0 <= gate_capacity(gate1_t02) <= +inf
   0 <= gate_capacity(gate1_t03) <= +inf
   0 <= gate_capacity(gate1_t04) <= +inf
   0 <= gate_capacity(gate1_t05) <= +inf
   0 <= gate_capacity(gate1_t06) <= +inf
   0 <= gate_capacity(gate1_t07) <= +inf
   0 <= gate_capacity(gate1_t08) <= +inf
   0 <= gate_capacity(gate1_t09) <= +inf
   0 <= gate_capacity(gate1_t10) <= +inf
   0 <= gate_capacity(gate1_t11) <= +inf
   0 <= gate_capacity(gate1_t12) <= +inf
   0 <= gate_capacity(gate1_t13) <= +inf
   0 <= gate_capacity(gate1_t14) <= +inf
   0 <= gate_capacity(gate1_t15) <= +inf
   0 <= gate_capacity(gate1_t16) <= +inf
   0 <= gate_capacity(gate1_t17) <= +inf
   0 <= gate_capacity(gate1_t18) <= +inf
   0 <= gate_capacity(gate1_t19) <= +inf
   0 <= gate_capacity(gate1_t20) <= +inf
   0 <= gate_capacity(gate1_t21) <= +inf
   0 <= gate_capacity(gate1_t22) <= +inf
   0 <= gate_capacity(gate1_t23) <= +inf
   0 <= gate_capacity(gate1_t24) <= +inf
   0 <= gate_capacity(gate2_t01) <= +inf
   0 <= gate_capacity(gate2_t02) <= +inf
   0 <= gate_capacity(gate2_t03) <= +inf
   0 <= gate_capacity(gate2_t04) <= +inf
   0 <= gate_capacity(gate2_t05) <= +inf
   0 <= gate_capacity(gate2_t06) <= +inf
   0 <= gate_capacity(gate2_t07) <= +inf
   0 <= gate_capacity(gate2_t08) <= +inf
   0 <= gate_capacity(gate2_t09) <= +inf
   0 <= gate_capacity(gate2_t10) <= +inf
   0 <= gate_capacity(gate2_t11) <= +inf
   0 <= gate_capacity(gate2_t12) <= +inf
   0 <= gate_capacity(gate2_t13) <= +inf
   0 <= gate_capacity(gate2_t14) <= +inf
   0 <= gate_capacity(gate2_t15) <= +inf
   0 <= gate_capacity(gate2_t16) <= +inf
   0 <= gate_capacity(gate2_t17) <= +inf
   0 <= gate_capacity(gate2_t18) <= +inf
   0 <= gate_capacity(gate2_t19) <= +inf
   0 <= gate_capacity(gate2_t20) <= +inf
   0 <= gate_capacity(gate2_t21) <= +inf
   0 <= gate_capacity(gate2_t22) <= +inf
   0 <= gate_capacity(gate2_t23) <= +inf
   0 <= gate_capacity(gate2_t24) <= +inf
   0.0 <= gate_opening(gate1_t01) <= 1.0
   0.0 <= gate_opening(gate1_t02) <= 1.0
   0.0 <= gate_opening(gate1_t03) <= 1.0
   0.0 <= gate_opening(gate1_t04) <= 1.0
   0.0 <= gate_opening(gate1_t05) <= 1.0
   0.0 <= gate_opening(gate1_t06) <= 1.0
   0.0 <= gate_opening(gate1_t07) <= 1.0
   0.0 <= gate_opening(gate1_t08) <= 1.0
   0.0 <= gate_opening(gate1_t09) <= 1.0
   0.0 <= gate_opening(gate1_t10) <= 1.0
   0.0 <= gate_opening(gate1_t11) <= 1.0
   0.0 <= gate_opening(gate1_t12) <= 1.0
   0.0 <= gate_opening(gate1_t13) <= 1.0
   0.0 <= gate_opening(gate1_t14) <= 1.0
   0.0 <= gate_opening(gate1_t15) <= 1.0
   0.0 <= gate_opening(gate1_t16) <= 1.0
   0.0 <= gate_opening(gate1_t17) <= 1.0
   0.0 <= gate_opening(gate1_t18) <= 1.0
   0.0 <= gate_opening(gate1_t19) <= 1.0
   0.0 <= gate_opening(gate1_t20) <= 1.0
   0.0 <= gate_opening(gate1_t21) <= 1.0
   0.0 <= gate_opening(gate1_t22) <= 1.0
   0.0 <= gate_opening(gate1_t23) <= 1.0
   0.0 <= gate_opening(gate1_t24) <= 1.0
   0.0 <= gate_opening(gate2_t01) <= 1.0
   0.0 <= gate_opening(gate2_t02) <= 1.0
   0.0 <= gate_opening(gate2_t03) <= 1.0
   0.0 <= gate_opening(gate2_t04) <= 1.0
   0.0 <= gate_opening(gate2_t05) <= 1.0
   0.0 <= gate_opening(gate2_t06) <= 1.0
   0.0 <= gate_opening(gate2_t07) <= 1.0
   0.0 <= gate_opening(gate2_t08) <= 1.0
   0.0 <= gate_opening(gate2_t09) <= 1.0
   0.0 <= gate_opening(gate2_t10) <= 1.0
   0.0 <= gate_opening(gate2_t11) <= 1.0
   0.0 <= gate_opening(gate2_t12) <= 1.0
   0.0 <= gate_opening(gate2_t13) <= 1.0
   0.0 <= gate_opening(gate2_t14) <= 1.0
   0.0 <= gate_opening(gate2_t15) <= 1.0
   0.0 <= gate_opening(gate2_t16) <= 1.0
   0.0 <= gate_opening(gate2_t17) <= 1.0
   0.0 <= gate_opening(gate2_t18) <= 1.0
   0.0 <= gate_opening(gate2_t19) <= 1.0
   0.0 <= gate_opening(gate2_t20) <= 1.0
   0.0 <= gate_opening(gate2_t21) <= 1.0
   0.0 <= gate_opening(gate2_t22) <= 1.0
   0.0 <= gate_opening(gate2_t23) <= 1.0
   0.0 <= gate_opening(gate2_t24) <= 1.0
   300.0 <= state(reservoir_storage_t01) <= 900.0
   300.0 <= state(reservoir_storage_t02) <= 900.0
   300.0 <= state(reservoir_storage_t03) <= 900.0
   300.0 <= state(reservoir_storage_t04) <= 900.0
   300.0 <= state(reservoir_storage_t05) <= 900.0
   300.0 <= state(reservoir_storage_t06) <= 900.0
   300.0 <= state(reservoir_storage_t07) <= 900.0
   300.0 <= state(reservoir_storage_t08) <= 900.0
   300.0 <= state(reservoir_storage_t09) <= 900.0
   300.0 <= state(reservoir_storage_t10) <= 900.0
   300.0 <= state(reservoir_storage_t11) <= 900.0
   300.0 <= state(reservoir_storage_t12) <= 900.0
   300.0 <= state(reservoir_storage_t13) <= 900.0
   300.0 <= state(reservoir_storage_t14) <= 900.0
   300.0 <= state(reservoir_storage_t15) <= 900.0
   300.0 <= state(reservoir_storage_t16) <= 900.0
   300.0 <= state(reservoir_storage_t17) <= 900.0
   300.0 <= state(reservoir_storage_t18) <= 900.0
   300.0 <= state(reservoir_storage_t19) <= 900.0
   300.0 <= state(reservoir_storage_t20) <= 900.0
   300.0 <= state(reservoir_storage_t21) <= 900.0
   300.0 <= state(reservoir_storage_t22) <= 900.0
   300.0 <= state(reservoir_storage_t23) <= 900.0
   300.0 <= state(reservoir_storage_t24) <= 900.0
   1.8 <= state(reservoir_level_t01) <= 2.5
   1.8 <= state(reservoir_level_t02) <= 2.5
   1.8 <= state(reservoir_level_t03) <= 2.5
   1.8 <= state(reservoir_level_t04) <= 2.5
   1.8 <= state(reservoir_level_t05) <= 2.5
   1.8 <= state(reservoir_level_t06) <= 2.5
   1.8 <= state(reservoir_level_t07) <= 2.5
   1.8 <= state(reservoir_level_t08) <= 2.5
   1.8 <= state(reservoir_level_t09) <= 2.5
   1.8 <= state(reservoir_level_t10) <= 2.5
   1.8 <= state(reservoir_level_t11) <= 2.5
   1.8 <= state(reservoir_level_t12) <= 2.5
   1.8 <= state(reservoir_level_t13) <= 2.5
   1.8 <= state(reservoir_level_t14) <= 2.5
   1.8 <= state(reservoir_level_t15) <= 2.5
   1.8 <= state(reservoir_level_t16) <= 2.5
   1.8 <= state(reservoir_level_t17) <= 2.5
   1.8 <= state(reservoir_level_t18) <= 2.5
   1.8 <= state(reservoir_level_t19) <= 2.5
   1.8 <= state(reservoir_level_t20) <= 2.5
   1.8 <= state(reservoir_level_t21) <= 2.5
   1.8 <= state(reservoir_level_t22) <= 2.5
   1.8 <= state(reservoir_level_t23) <= 2.5
   1.8 <= state(reservoir_level_t24) <= 2.5
   200.0 <= state(reach1_storage_t01) <= 320.0
   200.0 <= state(reach1_storage_t02) <= 320.0
   200.0 <= state(reach1_storage_t03) <= 320.0
   200.0 <= state(reach1_storage_t04) <= 320.0
   200.0 <= state(reach1_storage_t05) <= 320.0
   200.0 <= state(reach1_storage_t06) <= 320.0
   200.0 <= state(reach1_storage_t07) <= 320.0
   200.0 <= state(reach1_storage_t08) <= 320.0
   200.0 <= state(reach1_storage_t09) <= 320.0
   200.0 <= state(reach1_storage_t10) <= 320.0
   200.0 <= state(reach1_storage_t11) <= 320.0
   200.0 <= state(reach1_storage_t12) <= 320.0
   200.0 <= state(reach1_storage_t13) <= 320.0
   200.0 <= state(reach1_storage_t14) <= 320.0
   200.0 <= state(reach1_storage_t15) <= 320.0
   200.0 <= state(reach1_storage_t16) <= 320.0
   200.0 <= state(reach1_storage_t17) <= 320.0
   200.0 <= state(reach1_storage_t18) <= 320.0
   200.0 <= state(reach1_storage_t19) <= 320.0
   200.0 <= state(reach1_storage_t20) <= 320.0
   200.0 <= state(reach1_storage_t21) <= 320.0
   200.0 <= state(reach1_storage_t22) <= 320.0
   200.0 <= state(reach1_storage_t23) <= 320.0
   200.0 <= state(reach1_storage_t24) <= 320.0
   1.6199999999999999 <= state(reach1_level_upstream_t01) <= 2.3499999999999996
   1.6199999999999999 <= state(reach1_level_upstream_t02) <= 2.3499999999999996
   1.6199999999999999 <= state(reach1_level_upstream_t03) <= 2.3499999999999996
   1.6199999999999999 <= state(reach1_level_upstream_t04) <= 2.3499999999999996
   1.6199999999999999 <= state(reach1_level_upstream_t05) <= 2.3499999999999996
   1.6199999999999999 <= state(reach1_level_upstream_t06) <= 2.3499999999999996
   1.6199999999999999 <= state(reach1_level_upstream_t07) <= 2.3499999999999996
   1.6199999999999999 <= state(reach1_level_upstream_t08) <= 2.3499999999999996
   1.6199999999999999 <= state(reach1_level_upstream_t09) <= 2.3499999999999996
   1.6199999999999999 <= state(reach1_level_upstream_t10) <= 2.3499999999999996
   1.6199999999999999 <= state(reach1_level_upstream_t11) <= 2.3499999999999996
   1.6199999999999999 <= state(reach1_level_upstream_t12) <= 2.3499999999999996
   1.6199999999999999 <= state(reach1_level_upstream_t13) <= 2.3499999999999996
   1.6199999999999999 <= state(reach1_level_upstream_t14) <= 2.3499999999999996
   1.6199999999999999 <= state(reach1_level_upstream_t15) <= 2.3499999999999996
   1.6199999999999999 <= state(reach1_level_upstream_t16) <= 2.3499999999999996
   1.6199999999999999 <= state(reach1_level_upstream_t17) <= 2.3499999999999996
   1.6199999999999999 <= state(reach1_level_upstream_t18) <= 2.3499999999999996
   1.6199999999999999 <= state(reach1_level_upstream_t19) <= 2.3499999999999996
   1.6199999999999999 <= state(reach1_level_upstream_t20) <= 2.3499999999999996
   1.6199999999999999 <= state(reach1_level_upstream_t21) <= 2.3499999999999996
   1.6199999999999999 <= state(reach1_level_upstream_t22) <= 2.3499999999999996
   1.6199999999999999 <= state(reach1_level_upstream_t23) <= 2.3499999999999996
   1.6199999999999999 <= state(reach1_level_upstream_t24) <= 2.3499999999999996
   1.5 <= state(reach1_level_downstream_t01) <= 2.2199999999999998
   1.5 <= state(reach1_level_downstream_t02) <= 2.2199999999999998
   1.5 <= state(reach1_level_downstream_t03) <= 2.2199999999999998
   1.5 <= state(reach1_level_downstream_t04) <= 2.2199999999999998
   1.5 <= state(reach1_level_downstream_t05) <= 2.2199999999999998
   1.5 <= state(reach1_level_downstream_t06) <= 2.2199999999999998
   1.5 <= state(reach1_level_downstream_t07) <= 2.2199999999999998
   1.5 <= state(reach1_level_downstream_t08) <= 2.2199999999999998
   1.5 <= state(reach1_level_downstream_t09) <= 2.2199999999999998
   1.5 <= state(reach1_level_downstream_t10) <= 2.2199999999999998
   1.5 <= state(reach1_level_downstream_t11) <= 2.2199999999999998
   1.5 <= state(reach1_level_downstream_t12) <= 2.2199999999999998
   1.5 <= state(reach1_level_downstream_t13) <= 2.2199999999999998
   1.5 <= state(reach1_level_downstream_t14) <= 2.2199999999999998
   1.5 <= state(reach1_level_downstream_t15) <= 2.2199999999999998
   1.5 <= state(reach1_level_downstream_t16) <= 2.2199999999999998
   1.5 <= state(reach1_level_downstream_t17) <= 2.2199999999999998
   1.5 <= state(reach1_level_downstream_t18) <= 2.2199999999999998
   1.5 <= state(reach1_level_downstream_t19) <= 2.2199999999999998
   1.5 <= state(reach1_level_downstream_t20) <= 2.2199999999999998
   1.5 <= state(reach1_level_downstream_t21) <= 2.2199999999999998
   1.5 <= state(reach1_level_downstream_t22) <= 2.2199999999999998
   1.5 <= state(reach1_level_downstream_t23) <= 2.2199999999999998
   1.5 <= state(reach1_level_downstream_t24) <= 2.2199999999999998
   3.04 <= state(reach1_discharge_upstream_t01) <= 16.900000000000002
   3.04 <= state(reach1_discharge_upstream_t02) <= 16.900000000000002
   3.04 <= state(reach1_discharge_upstream_t03) <= 16.900000000000002
   3.04 <= state(reach1_discharge_upstream_t04) <= 16.900000000000002
   3.04 <= state(reach1_discharge_upstream_t05) <= 16.900000000000002
   3.04 <= state(reach1_discharge_upstream_t06) <= 16.900000000000002
   3.04 <= state(reach1_discharge_upstream_t07) <= 16.900000000000002
   3.04 <= state(reach1_discharge_upstream_t08) <= 16.900000000000002
   3.04 <= state(reach1_discharge_upstream_t09) <= 16.900000000000002
   3.04 <= state(reach1_discharge_upstream_t10) <= 16.900000000000002
   3.04 <= state(reach1_discharge_upstream_t11) <= 16.900000000000002
   3.04 <= state(reach1_discharge_upstream_t12) <= 16.900000000000002
   3.04 <= state(reach1_discharge_upstream_t13) <= 16.900000000000002
   3.04 <= state(reach1_discharge_upstream_t14) <= 16.900000000000002
   3.04 <= state(reach1_discharge_upstream_t15) <= 16.900000000000002
   3.04 <= state(reach1_discharge_upstream_t16) <= 16.900000000000002
   3.04 <= state(reach1_discharge_upstream_t17) <= 16.900000000000002
   3.04 <= state(reach1_discharge_upstream_t18) <= 16.900000000000002
   3.04 <= state(reach1_discharge_upstream_t19) <= 16.900000000000002
   3.04 <= state(reach1_discharge_upstream_t20) <= 16.900000000000002
   3.04 <= state(reach1_discharge_upstream_t21) <= 16.900000000000002
   3.04 <= state(reach1_discharge_upstream_t22) <= 16.900000000000002
   3.04 <= state(reach1_discharge_upstream_t23) <= 16.900000000000002
   3.04 <= state(reach1_discharge_upstream_t24) <= 16.900000000000002
   3.04 <= state(reach1_discharge_downstream_t01) <= 16.25
   3.04 <= state(reach1_discharge_downstream_t02) <= 16.25
   3.04 <= state(reach1_discharge_downstream_t03) <= 16.25
   3.04 <= state(reach1_discharge_downstream_t04) <= 16.25
   3.04 <= state(reach1_discharge_downstream_t05) <= 16.25
   3.04 <= state(reach1_discharge_downstream_t06) <= 16.25
   3.04 <= state(reach1_discharge_downstream_t07) <= 16.25
   3.04 <= state(reach1_discharge_downstream_t08) <= 16.25
   3.04 <= state(reach1_discharge_downstream_t09) <= 16.25
   3.04 <= state(reach1_discharge_downstream_t10) <= 16.25
   3.04 <= state(reach1_discharge_downstream_t11) <= 16.25
   3.04 <= state(reach1_discharge_downstream_t12) <= 16.25
   3.04 <= state(reach1_discharge_downstream_t13) <= 16.25
   3.04 <= state(reach1_discharge_downstream_t14) <= 16.25
   3.04 <= state(reach1_discharge_downstream_t15) <= 16.25
   3.04 <= state(reach1_discharge_downstream_t16) <= 16.25
   3.04 <= state(reach1_discharge_downstream_t17) <= 16.25
   3.04 <= state(reach1_discharge_downstream_t18) <= 16.25
   3.04 <= state(reach1_discharge_downstream_t19) <= 16.25
   3.04 <= state(reach1_discharge_downstream_t20) <= 16.25
   3.04 <= state(reach1_discharge_downstream_t21) <= 16.25
   3.04 <= state(reach1_discharge_downstream_t22) <= 16.25
   3.04 <= state(reach1_discharge_downstream_t23) <= 16.25
   3.04 <= state(reach1_discharge_downstream_t24) <= 16.25
   250.0 <= state(reach2_storage_t01) <= 410.0
   250.0 <= state(reach2_storage_t02) <= 410.0
   250.0 <= state(reach2_storage_t03) <= 410.0
   250.0 <= state(reach2_storage_t04) <= 410.0
   250.0 <= state(reach2_storage_t05) <= 410.0
   250.0 <= state(reach2_storage_t06) <= 410.0
   250.0 <= state(reach2_storage_t07) <= 410.0
   250.0 <= state(reach2_storage_t08) <= 410.0
   250.0 <= state(reach2_storage_t09) <= 410.0
   250.0 <= state(reach2_storage_t10) <= 410.0
   250.0 <= state(reach2_storage_t11) <= 410.0
   250.0 <= state(reach2_storage_t12) <= 410.0
   250.0 <= state(reach2_storage_t13) <= 410.0
   250.0 <= state(reach2_storage_t14) <= 410.0
   250.0 <= state(reach2_storage_t15) <= 410.0
   250.0 <= state(reach2_storage_t16) <= 410.0
   250.0 <= state(reach2_storage_t17) <= 410.0
   250.0 <= state(reach2_storage_t18) <= 410.0
   250.0 <= state(reach2_storage_t19) <= 410.0
   250.0 <= state(reach2_storage_t20) <= 410.0
   250.0 <= state(reach2_storage_t21) <= 410.0
   250.0 <= state(reach2_storage_t22) <= 410.0
   250.0 <= state(reach2_storage_t23) <= 410.0
   250.0 <= state(reach2_storage_t24) <= 410.0
   1.42 <= state(reach2_level_upstream_t01) <= 2.15
   1.42 <= state(reach2_level_upstream_t02) <= 2.15
   1.42 <= state(reach2_level_upstream_t03) <= 2.15
   1.42 <= state(reach2_level_upstream_t04) <= 2.15
   1.42 <= state(reach2_level_upstream_t05) <= 2.15
   1.42 <= state(reach2_level_upstream_t06) <= 2.15
   1.42 <= state(reach2_level_upstream_t07) <= 2.15
   1.42 <= state(reach2_level_upstream_t08) <= 2.15
   1.42 <= state(reach2_level_upstream_t09) <= 2.15
   1.42 <= state(reach2_level_upstream_t10) <= 2.15
   1.42 <= state(reach2_level_upstream_t11) <= 2.15
   1.42 <= state(reach2_level_upstream_t12) <= 2.15
   1.42 <= state(reach2_level_upstream_t13) <= 2.15
   1.42 <= state(reach2_level_upstream_t14) <= 2.15
   1.42 <= state(reach2_level_upstream_t15) <= 2.15
   1.42 <= state(reach2_level_upstream_t16) <= 2.15
   1.42 <= state(reach2_level_upstream_t17) <= 2.15
   1.42 <= state(reach2_level_upstream_t18) <= 2.15
   1.42 <= state(reach2_level_upstream_t19) <= 2.15
   1.42 <= state(reach2_level_upstream_t20) <= 2.15
   1.42 <= state(reach2_level_upstream_t21) <= 2.15
   1.42 <= state(reach2_level_upstream_t22) <= 2.15
   1.42 <= state(reach2_level_upstream_t23) <= 2.15
   1.42 <= state(reach2_level_upstream_t24) <= 2.15
   1.3 <= state(reach2_level_downstream_t01) <= 2.02
   1.3 <= state(reach2_level_downstream_t02) <= 2.02
   1.3 <= state(reach2_level_downstream_t03) <= 2.02
   1.3 <= state(reach2_level_downstream_t04) <= 2.02
   1.3 <= state(reach2_level_downstream_t05) <= 2.02
   1.3 <= state(reach2_level_downstream_t06) <= 2.02
   1.3 <= state(reach2_level_downstream_t07) <= 2.02
   1.3 <= state(reach2_level_downstream_t08) <= 2.02
   1.3 <= state(reach2_level_downstream_t09) <= 2.02
   1.3 <= state(reach2_level_downstream_t10) <= 2.02
   1.3 <= state(reach2_level_downstream_t11) <= 2.02
   1.3 <= state(reach2_level_downstream_t12) <= 2.02
   1.3 <= state(reach2_level_downstream_t13) <= 2.02
   1.3 <= state(reach2_level_downstream_t14) <= 2.02
   1.3 <= state(reach2_level_downstream_t15) <= 2.02
   1.3 <= state(reach2_level_downstream_t16) <= 2.02
   1.3 <= state(reach2_level_downstream_t17) <= 2.02
   1.3 <= state(reach2_level_downstream_t18) <= 2.02
   1.3 <= state(reach2_level_downstream_t19) <= 2.02
   1.3 <= state(reach2_level_downstream_t20) <= 2.02
   1.3 <= state(reach2_level_downstream_t21) <= 2.02
   1.3 <= state(reach2_level_downstream_t22) <= 2.02
   1.3 <= state(reach2_level_downstream_t23) <= 2.02
   1.3 <= state(reach2_level_downstream_t24) <= 2.02
   2.08 <= state(reach2_discharge_upstream_t01) <= 14.040000000000001
   2.08 <= state(reach2_discharge_upstream_t02) <= 14.040000000000001
   2.08 <= state(reach2_discharge_upstream_t03) <= 14.040000000000001
   2.08 <= state(reach2_discharge_upstream_t04) <= 14.040000000000001
   2.08 <= state(reach2_discharge_upstream_t05) <= 14.040000000000001
   2.08 <= state(reach2_discharge_upstream_t06) <= 14.040000000000001
   2.08 <= state(reach2_discharge_upstream_t07) <= 14.040000000000001
   2.08 <= state(reach2_discharge_upstream_t08) <= 14.040000000000001
   2.08 <= state(reach2_discharge_upstream_t09) <= 14.040000000000001
   2.08 <= state(reach2_discharge_upstream_t10) <= 14.040000000000001
   2.08 <= state(reach2_discharge_upstream_t11) <= 14.040000000000001
   2.08 <= state(reach2_discharge_upstream_t12) <= 14.040000000000001
   2.08 <= state(reach2_discharge_upstream_t13) <= 14.040000000000001
   2.08 <= state(reach2_discharge_upstream_t14) <= 14.040000000000001
   2.08 <= state(reach2_discharge_upstream_t15) <= 14.040000000000001
   2.08 <= state(reach2_discharge_upstream_t16) <= 14.040000000000001
   2.08 <= state(reach2_discharge_upstream_t17) <= 14.040000000000001
   2.08 <= state(reach2_discharge_upstream_t18) <= 14.040000000000001
   2.08 <= state(reach2_discharge_upstream_t19) <= 14.040000000000001
   2.08 <= state(reach2_discharge_upstream_t20) <= 14.040000000000001
   2.08 <= state(reach2_discharge_upstream_t21) <= 14.040000000000001
   2.08 <= state(reach2_discharge_upstream_t22) <= 14.040000000000001
   2.08 <= state(reach2_discharge_upstream_t23) <= 14.040000000000001
   2.08 <= state(reach2_discharge_upstream_t24) <= 14.040000000000001
   2.08 <= state(reach2_discharge_downstream_t01) <= 13.5
   2.08 <= state(reach2_discharge_downstream_t02) <= 13.5
   2.08 <= state(reach2_discharge_downstream_t03) <= 13.5
   2.08 <= state(reach2_discharge_downstream_t04) <= 13.5
   2.08 <= state(reach2_discharge_downstream_t05) <= 13.5
   2.08 <= state(reach2_discharge_downstream_t06) <= 13.5
   2.08 <= state(reach2_discharge_downstream_t07) <= 13.5
   2.08 <= state(reach2_discharge_downstream_t08) <= 13.5
   2.08 <= state(reach2_discharge_downstream_t09) <= 13.5
   2.08 <= state(reach2_discharge_downstream_t10) <= 13.5
   2.08 <= state(reach2_discharge_downstream_t11) <= 13.5
   2.08 <= state(reach2_discharge_downstream_t12) <= 13.5
   2.08 <= state(reach2_discharge_downstream_t13) <= 13.5
   2.08 <= state(reach2_discharge_downstream_t14) <= 13.5
   2.08 <= state(reach2_discharge_downstream_t15) <= 13.5
   2.08 <= state(reach2_discharge_downstream_t16) <= 13.5
   2.08 <= state(reach2_discharge_downstream_t17) <= 13.5
   2.08 <= state(reach2_discharge_downstream_t18) <= 13.5
   2.08 <= state(reach2_discharge_downstream_t19) <= 13.5
   2.08 <= state(reach2_discharge_downstream_t20) <= 13.5
   2.08 <= state(reach2_discharge_downstream_t21) <= 13.5
   2.08 <= state(reach2_discharge_downstream_t22) <= 13.5
   2.08 <= state(reach2_discharge_downstream_t23) <= 13.5
   2.08 <= state(reach2_discharge_downstream_t24) <= 13.5
   0 <= state_piecewise(0)_pw_t01_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t01_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t01_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t01_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t01_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t02_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t02_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t02_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t02_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t02_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t03_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t03_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t03_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t03_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t03_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t04_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t04_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t04_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t04_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t04_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t05_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t05_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t05_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t05_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t05_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t06_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t06_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t06_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t06_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t06_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t07_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t07_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t07_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t07_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t07_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t08_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t08_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t08_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t08_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t08_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t09_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t09_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t09_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t09_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t09_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t10_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t10_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t10_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t10_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t10_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t11_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t11_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t11_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t11_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t11_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t12_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t12_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t12_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t12_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t12_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t13_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t13_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t13_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t13_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t13_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t14_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t14_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t14_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t14_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t14_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t15_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t15_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t15_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t15_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t15_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t16_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t16_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t16_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t16_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t16_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t17_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t17_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t17_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t17_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t17_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t18_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t18_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t18_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t18_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t18_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t19_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t19_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t19_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t19_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t19_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t20_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t20_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t20_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t20_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t20_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t21_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t21_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t21_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t21_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t21_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t22_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t22_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t22_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t22_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t22_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t23_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t23_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t23_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t23_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t23_CC_bin_y(2) <= 1
   0 <= state_piecewise(0)_pw_t24_CC_lambda(1) <= +inf
   0 <= state_piecewise(0)_pw_t24_CC_lambda(2) <= +inf
   0 <= state_piecewise(0)_pw_t24_CC_lambda(3) <= +inf
   0 <= state_piecewise(0)_pw_t24_CC_bin_y(1) <= 1
   0 <= state_piecewise(0)_pw_t24_CC_bin_y(2) <= 1
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
   0 <= state_piecewise(1)_pw_t13_CC_lambda(1) <= +inf
   0 <= state_piecewise(1)_pw_t13_CC_lambda(2) <= +inf
   0 <= state_piecewise(1)_pw_t13_CC_lambda(3) <= +inf
   0 <= state_piecewise(1)_pw_t13_CC_bin_y(1) <= 1
   0 <= state_piecewise(1)_pw_t13_CC_bin_y(2) <= 1
   0 <= state_piecewise(1)_pw_t14_CC_lambda(1) <= +inf
   0 <= state_piecewise(1)_pw_t14_CC_lambda(2) <= +inf
   0 <= state_piecewise(1)_pw_t14_CC_lambda(3) <= +inf
   0 <= state_piecewise(1)_pw_t14_CC_bin_y(1) <= 1
   0 <= state_piecewise(1)_pw_t14_CC_bin_y(2) <= 1
   0 <= state_piecewise(1)_pw_t15_CC_lambda(1) <= +inf
   0 <= state_piecewise(1)_pw_t15_CC_lambda(2) <= +inf
   0 <= state_piecewise(1)_pw_t15_CC_lambda(3) <= +inf
   0 <= state_piecewise(1)_pw_t15_CC_bin_y(1) <= 1
   0 <= state_piecewise(1)_pw_t15_CC_bin_y(2) <= 1
   0 <= state_piecewise(1)_pw_t16_CC_lambda(1) <= +inf
   0 <= state_piecewise(1)_pw_t16_CC_lambda(2) <= +inf
   0 <= state_piecewise(1)_pw_t16_CC_lambda(3) <= +inf
   0 <= state_piecewise(1)_pw_t16_CC_bin_y(1) <= 1
   0 <= state_piecewise(1)_pw_t16_CC_bin_y(2) <= 1
   0 <= state_piecewise(1)_pw_t17_CC_lambda(1) <= +inf
   0 <= state_piecewise(1)_pw_t17_CC_lambda(2) <= +inf
   0 <= state_piecewise(1)_pw_t17_CC_lambda(3) <= +inf
   0 <= state_piecewise(1)_pw_t17_CC_bin_y(1) <= 1
   0 <= state_piecewise(1)_pw_t17_CC_bin_y(2) <= 1
   0 <= state_piecewise(1)_pw_t18_CC_lambda(1) <= +inf
   0 <= state_piecewise(1)_pw_t18_CC_lambda(2) <= +inf
   0 <= state_piecewise(1)_pw_t18_CC_lambda(3) <= +inf
   0 <= state_piecewise(1)_pw_t18_CC_bin_y(1) <= 1
   0 <= state_piecewise(1)_pw_t18_CC_bin_y(2) <= 1
   0 <= state_piecewise(1)_pw_t19_CC_lambda(1) <= +inf
   0 <= state_piecewise(1)_pw_t19_CC_lambda(2) <= +inf
   0 <= state_piecewise(1)_pw_t19_CC_lambda(3) <= +inf
   0 <= state_piecewise(1)_pw_t19_CC_bin_y(1) <= 1
   0 <= state_piecewise(1)_pw_t19_CC_bin_y(2) <= 1
   0 <= state_piecewise(1)_pw_t20_CC_lambda(1) <= +inf
   0 <= state_piecewise(1)_pw_t20_CC_lambda(2) <= +inf
   0 <= state_piecewise(1)_pw_t20_CC_lambda(3) <= +inf
   0 <= state_piecewise(1)_pw_t20_CC_bin_y(1) <= 1
   0 <= state_piecewise(1)_pw_t20_CC_bin_y(2) <= 1
   0 <= state_piecewise(1)_pw_t21_CC_lambda(1) <= +inf
   0 <= state_piecewise(1)_pw_t21_CC_lambda(2) <= +inf
   0 <= state_piecewise(1)_pw_t21_CC_lambda(3) <= +inf
   0 <= state_piecewise(1)_pw_t21_CC_bin_y(1) <= 1
   0 <= state_piecewise(1)_pw_t21_CC_bin_y(2) <= 1
   0 <= state_piecewise(1)_pw_t22_CC_lambda(1) <= +inf
   0 <= state_piecewise(1)_pw_t22_CC_lambda(2) <= +inf
   0 <= state_piecewise(1)_pw_t22_CC_lambda(3) <= +inf
   0 <= state_piecewise(1)_pw_t22_CC_bin_y(1) <= 1
   0 <= state_piecewise(1)_pw_t22_CC_bin_y(2) <= 1
   0 <= state_piecewise(1)_pw_t23_CC_lambda(1) <= +inf
   0 <= state_piecewise(1)_pw_t23_CC_lambda(2) <= +inf
   0 <= state_piecewise(1)_pw_t23_CC_lambda(3) <= +inf
   0 <= state_piecewise(1)_pw_t23_CC_bin_y(1) <= 1
   0 <= state_piecewise(1)_pw_t23_CC_bin_y(2) <= 1
   0 <= state_piecewise(1)_pw_t24_CC_lambda(1) <= +inf
   0 <= state_piecewise(1)_pw_t24_CC_lambda(2) <= +inf
   0 <= state_piecewise(1)_pw_t24_CC_lambda(3) <= +inf
   0 <= state_piecewise(1)_pw_t24_CC_bin_y(1) <= 1
   0 <= state_piecewise(1)_pw_t24_CC_bin_y(2) <= 1
   0 <= state_piecewise(2)_pw_t01_CC_lambda(1) <= +inf
   0 <= state_piecewise(2)_pw_t01_CC_lambda(2) <= +inf
   0 <= state_piecewise(2)_pw_t01_CC_lambda(3) <= +inf
   0 <= state_piecewise(2)_pw_t01_CC_bin_y(1) <= 1
   0 <= state_piecewise(2)_pw_t01_CC_bin_y(2) <= 1
   0 <= state_piecewise(2)_pw_t02_CC_lambda(1) <= +inf
   0 <= state_piecewise(2)_pw_t02_CC_lambda(2) <= +inf
   0 <= state_piecewise(2)_pw_t02_CC_lambda(3) <= +inf
   0 <= state_piecewise(2)_pw_t02_CC_bin_y(1) <= 1
   0 <= state_piecewise(2)_pw_t02_CC_bin_y(2) <= 1
   0 <= state_piecewise(2)_pw_t03_CC_lambda(1) <= +inf
   0 <= state_piecewise(2)_pw_t03_CC_lambda(2) <= +inf
   0 <= state_piecewise(2)_pw_t03_CC_lambda(3) <= +inf
   0 <= state_piecewise(2)_pw_t03_CC_bin_y(1) <= 1
   0 <= state_piecewise(2)_pw_t03_CC_bin_y(2) <= 1
   0 <= state_piecewise(2)_pw_t04_CC_lambda(1) <= +inf
   0 <= state_piecewise(2)_pw_t04_CC_lambda(2) <= +inf
   0 <= state_piecewise(2)_pw_t04_CC_lambda(3) <= +inf
   0 <= state_piecewise(2)_pw_t04_CC_bin_y(1) <= 1
   0 <= state_piecewise(2)_pw_t04_CC_bin_y(2) <= 1
   0 <= state_piecewise(2)_pw_t05_CC_lambda(1) <= +inf
   0 <= state_piecewise(2)_pw_t05_CC_lambda(2) <= +inf
   0 <= state_piecewise(2)_pw_t05_CC_lambda(3) <= +inf
   0 <= state_piecewise(2)_pw_t05_CC_bin_y(1) <= 1
   0 <= state_piecewise(2)_pw_t05_CC_bin_y(2) <= 1
   0 <= state_piecewise(2)_pw_t06_CC_lambda(1) <= +inf
   0 <= state_piecewise(2)_pw_t06_CC_lambda(2) <= +inf
   0 <= state_piecewise(2)_pw_t06_CC_lambda(3) <= +inf
   0 <= state_piecewise(2)_pw_t06_CC_bin_y(1) <= 1
   0 <= state_piecewise(2)_pw_t06_CC_bin_y(2) <= 1
   0 <= state_piecewise(2)_pw_t07_CC_lambda(1) <= +inf
   0 <= state_piecewise(2)_pw_t07_CC_lambda(2) <= +inf
   0 <= state_piecewise(2)_pw_t07_CC_lambda(3) <= +inf
   0 <= state_piecewise(2)_pw_t07_CC_bin_y(1) <= 1
   0 <= state_piecewise(2)_pw_t07_CC_bin_y(2) <= 1
   0 <= state_piecewise(2)_pw_t08_CC_lambda(1) <= +inf
   0 <= state_piecewise(2)_pw_t08_CC_lambda(2) <= +inf
   0 <= state_piecewise(2)_pw_t08_CC_lambda(3) <= +inf
   0 <= state_piecewise(2)_pw_t08_CC_bin_y(1) <= 1
   0 <= state_piecewise(2)_pw_t08_CC_bin_y(2) <= 1
   0 <= state_piecewise(2)_pw_t09_CC_lambda(1) <= +inf
   0 <= state_piecewise(2)_pw_t09_CC_lambda(2) <= +inf
   0 <= state_piecewise(2)_pw_t09_CC_lambda(3) <= +inf
   0 <= state_piecewise(2)_pw_t09_CC_bin_y(1) <= 1
   0 <= state_piecewise(2)_pw_t09_CC_bin_y(2) <= 1
   0 <= state_piecewise(2)_pw_t10_CC_lambda(1) <= +inf
   0 <= state_piecewise(2)_pw_t10_CC_lambda(2) <= +inf
   0 <= state_piecewise(2)_pw_t10_CC_lambda(3) <= +inf
   0 <= state_piecewise(2)_pw_t10_CC_bin_y(1) <= 1
   0 <= state_piecewise(2)_pw_t10_CC_bin_y(2) <= 1
   0 <= state_piecewise(2)_pw_t11_CC_lambda(1) <= +inf
   0 <= state_piecewise(2)_pw_t11_CC_lambda(2) <= +inf
   0 <= state_piecewise(2)_pw_t11_CC_lambda(3) <= +inf
   0 <= state_piecewise(2)_pw_t11_CC_bin_y(1) <= 1
   0 <= state_piecewise(2)_pw_t11_CC_bin_y(2) <= 1
   0 <= state_piecewise(2)_pw_t12_CC_lambda(1) <= +inf
   0 <= state_piecewise(2)_pw_t12_CC_lambda(2) <= +inf
   0 <= state_piecewise(2)_pw_t12_CC_lambda(3) <= +inf
   0 <= state_piecewise(2)_pw_t12_CC_bin_y(1) <= 1
   0 <= state_piecewise(2)_pw_t12_CC_bin_y(2) <= 1
   0 <= state_piecewise(2)_pw_t13_CC_lambda(1) <= +inf
   0 <= state_piecewise(2)_pw_t13_CC_lambda(2) <= +inf
   0 <= state_piecewise(2)_pw_t13_CC_lambda(3) <= +inf
   0 <= state_piecewise(2)_pw_t13_CC_bin_y(1) <= 1
   0 <= state_piecewise(2)_pw_t13_CC_bin_y(2) <= 1
   0 <= state_piecewise(2)_pw_t14_CC_lambda(1) <= +inf
   0 <= state_piecewise(2)_pw_t14_CC_lambda(2) <= +inf
   0 <= state_piecewise(2)_pw_t14_CC_lambda(3) <= +inf
   0 <= state_piecewise(2)_pw_t14_CC_bin_y(1) <= 1
   0 <= state_piecewise(2)_pw_t14_CC_bin_y(2) <= 1
   0 <= state_piecewise(2)_pw_t15_CC_lambda(1) <= +inf
   0 <= state_piecewise(2)_pw_t15_CC_lambda(2) <= +inf
   0 <= state_piecewise(2)_pw_t15_CC_lambda(3) <= +inf
   0 <= state_piecewise(2)_pw_t15_CC_bin_y(1) <= 1
   0 <= state_piecewise(2)_pw_t15_CC_bin_y(2) <= 1
   0 <= state_piecewise(2)_pw_t16_CC_lambda(1) <= +inf
   0 <= state_piecewise(2)_pw_t16_CC_lambda(2) <= +inf
   0 <= state_piecewise(2)_pw_t16_CC_lambda(3) <= +inf
   0 <= state_piecewise(2)_pw_t16_CC_bin_y(1) <= 1
   0 <= state_piecewise(2)_pw_t16_CC_bin_y(2) <= 1
   0 <= state_piecewise(2)_pw_t17_CC_lambda(1) <= +inf
   0 <= state_piecewise(2)_pw_t17_CC_lambda(2) <= +inf
   0 <= state_piecewise(2)_pw_t17_CC_lambda(3) <= +inf
   0 <= state_piecewise(2)_pw_t17_CC_bin_y(1) <= 1
   0 <= state_piecewise(2)_pw_t17_CC_bin_y(2) <= 1
   0 <= state_piecewise(2)_pw_t18_CC_lambda(1) <= +inf
   0 <= state_piecewise(2)_pw_t18_CC_lambda(2) <= +inf
   0 <= state_piecewise(2)_pw_t18_CC_lambda(3) <= +inf
   0 <= state_piecewise(2)_pw_t18_CC_bin_y(1) <= 1
   0 <= state_piecewise(2)_pw_t18_CC_bin_y(2) <= 1
   0 <= state_piecewise(2)_pw_t19_CC_lambda(1) <= +inf
   0 <= state_piecewise(2)_pw_t19_CC_lambda(2) <= +inf
   0 <= state_piecewise(2)_pw_t19_CC_lambda(3) <= +inf
   0 <= state_piecewise(2)_pw_t19_CC_bin_y(1) <= 1
   0 <= state_piecewise(2)_pw_t19_CC_bin_y(2) <= 1
   0 <= state_piecewise(2)_pw_t20_CC_lambda(1) <= +inf
   0 <= state_piecewise(2)_pw_t20_CC_lambda(2) <= +inf
   0 <= state_piecewise(2)_pw_t20_CC_lambda(3) <= +inf
   0 <= state_piecewise(2)_pw_t20_CC_bin_y(1) <= 1
   0 <= state_piecewise(2)_pw_t20_CC_bin_y(2) <= 1
   0 <= state_piecewise(2)_pw_t21_CC_lambda(1) <= +inf
   0 <= state_piecewise(2)_pw_t21_CC_lambda(2) <= +inf
   0 <= state_piecewise(2)_pw_t21_CC_lambda(3) <= +inf
   0 <= state_piecewise(2)_pw_t21_CC_bin_y(1) <= 1
   0 <= state_piecewise(2)_pw_t21_CC_bin_y(2) <= 1
   0 <= state_piecewise(2)_pw_t22_CC_lambda(1) <= +inf
   0 <= state_piecewise(2)_pw_t22_CC_lambda(2) <= +inf
   0 <= state_piecewise(2)_pw_t22_CC_lambda(3) <= +inf
   0 <= state_piecewise(2)_pw_t22_CC_bin_y(1) <= 1
   0 <= state_piecewise(2)_pw_t22_CC_bin_y(2) <= 1
   0 <= state_piecewise(2)_pw_t23_CC_lambda(1) <= +inf
   0 <= state_piecewise(2)_pw_t23_CC_lambda(2) <= +inf
   0 <= state_piecewise(2)_pw_t23_CC_lambda(3) <= +inf
   0 <= state_piecewise(2)_pw_t23_CC_bin_y(1) <= 1
   0 <= state_piecewise(2)_pw_t23_CC_bin_y(2) <= 1
   0 <= state_piecewise(2)_pw_t24_CC_lambda(1) <= +inf
   0 <= state_piecewise(2)_pw_t24_CC_lambda(2) <= +inf
   0 <= state_piecewise(2)_pw_t24_CC_lambda(3) <= +inf
   0 <= state_piecewise(2)_pw_t24_CC_bin_y(1) <= 1
   0 <= state_piecewise(2)_pw_t24_CC_bin_y(2) <= 1
   0 <= state_piecewise(3)_pw_t01_CC_lambda(1) <= +inf
   0 <= state_piecewise(3)_pw_t01_CC_lambda(2) <= +inf
   0 <= state_piecewise(3)_pw_t01_CC_lambda(3) <= +inf
   0 <= state_piecewise(3)_pw_t01_CC_bin_y(1) <= 1
   0 <= state_piecewise(3)_pw_t01_CC_bin_y(2) <= 1
   0 <= state_piecewise(3)_pw_t02_CC_lambda(1) <= +inf
   0 <= state_piecewise(3)_pw_t02_CC_lambda(2) <= +inf
   0 <= state_piecewise(3)_pw_t02_CC_lambda(3) <= +inf
   0 <= state_piecewise(3)_pw_t02_CC_bin_y(1) <= 1
   0 <= state_piecewise(3)_pw_t02_CC_bin_y(2) <= 1
   0 <= state_piecewise(3)_pw_t03_CC_lambda(1) <= +inf
   0 <= state_piecewise(3)_pw_t03_CC_lambda(2) <= +inf
   0 <= state_piecewise(3)_pw_t03_CC_lambda(3) <= +inf
   0 <= state_piecewise(3)_pw_t03_CC_bin_y(1) <= 1
   0 <= state_piecewise(3)_pw_t03_CC_bin_y(2) <= 1
   0 <= state_piecewise(3)_pw_t04_CC_lambda(1) <= +inf
   0 <= state_piecewise(3)_pw_t04_CC_lambda(2) <= +inf
   0 <= state_piecewise(3)_pw_t04_CC_lambda(3) <= +inf
   0 <= state_piecewise(3)_pw_t04_CC_bin_y(1) <= 1
   0 <= state_piecewise(3)_pw_t04_CC_bin_y(2) <= 1
   0 <= state_piecewise(3)_pw_t05_CC_lambda(1) <= +inf
   0 <= state_piecewise(3)_pw_t05_CC_lambda(2) <= +inf
   0 <= state_piecewise(3)_pw_t05_CC_lambda(3) <= +inf
   0 <= state_piecewise(3)_pw_t05_CC_bin_y(1) <= 1
   0 <= state_piecewise(3)_pw_t05_CC_bin_y(2) <= 1
   0 <= state_piecewise(3)_pw_t06_CC_lambda(1) <= +inf
   0 <= state_piecewise(3)_pw_t06_CC_lambda(2) <= +inf
   0 <= state_piecewise(3)_pw_t06_CC_lambda(3) <= +inf
   0 <= state_piecewise(3)_pw_t06_CC_bin_y(1) <= 1
   0 <= state_piecewise(3)_pw_t06_CC_bin_y(2) <= 1
   0 <= state_piecewise(3)_pw_t07_CC_lambda(1) <= +inf
   0 <= state_piecewise(3)_pw_t07_CC_lambda(2) <= +inf
   0 <= state_piecewise(3)_pw_t07_CC_lambda(3) <= +inf
   0 <= state_piecewise(3)_pw_t07_CC_bin_y(1) <= 1
   0 <= state_piecewise(3)_pw_t07_CC_bin_y(2) <= 1
   0 <= state_piecewise(3)_pw_t08_CC_lambda(1) <= +inf
   0 <= state_piecewise(3)_pw_t08_CC_lambda(2) <= +inf
   0 <= state_piecewise(3)_pw_t08_CC_lambda(3) <= +inf
   0 <= state_piecewise(3)_pw_t08_CC_bin_y(1) <= 1
   0 <= state_piecewise(3)_pw_t08_CC_bin_y(2) <= 1
   0 <= state_piecewise(3)_pw_t09_CC_lambda(1) <= +inf
   0 <= state_piecewise(3)_pw_t09_CC_lambda(2) <= +inf
   0 <= state_piecewise(3)_pw_t09_CC_lambda(3) <= +inf
   0 <= state_piecewise(3)_pw_t09_CC_bin_y(1) <= 1
   0 <= state_piecewise(3)_pw_t09_CC_bin_y(2) <= 1
   0 <= state_piecewise(3)_pw_t10_CC_lambda(1) <= +inf
   0 <= state_piecewise(3)_pw_t10_CC_lambda(2) <= +inf
   0 <= state_piecewise(3)_pw_t10_CC_lambda(3) <= +inf
   0 <= state_piecewise(3)_pw_t10_CC_bin_y(1) <= 1
   0 <= state_piecewise(3)_pw_t10_CC_bin_y(2) <= 1
   0 <= state_piecewise(3)_pw_t11_CC_lambda(1) <= +inf
   0 <= state_piecewise(3)_pw_t11_CC_lambda(2) <= +inf
   0 <= state_piecewise(3)_pw_t11_CC_lambda(3) <= +inf
   0 <= state_piecewise(3)_pw_t11_CC_bin_y(1) <= 1
   0 <= state_piecewise(3)_pw_t11_CC_bin_y(2) <= 1
   0 <= state_piecewise(3)_pw_t12_CC_lambda(1) <= +inf
   0 <= state_piecewise(3)_pw_t12_CC_lambda(2) <= +inf
   0 <= state_piecewise(3)_pw_t12_CC_lambda(3) <= +inf
   0 <= state_piecewise(3)_pw_t12_CC_bin_y(1) <= 1
   0 <= state_piecewise(3)_pw_t12_CC_bin_y(2) <= 1
   0 <= state_piecewise(3)_pw_t13_CC_lambda(1) <= +inf
   0 <= state_piecewise(3)_pw_t13_CC_lambda(2) <= +inf
   0 <= state_piecewise(3)_pw_t13_CC_lambda(3) <= +inf
   0 <= state_piecewise(3)_pw_t13_CC_bin_y(1) <= 1
   0 <= state_piecewise(3)_pw_t13_CC_bin_y(2) <= 1
   0 <= state_piecewise(3)_pw_t14_CC_lambda(1) <= +inf
   0 <= state_piecewise(3)_pw_t14_CC_lambda(2) <= +inf
   0 <= state_piecewise(3)_pw_t14_CC_lambda(3) <= +inf
   0 <= state_piecewise(3)_pw_t14_CC_bin_y(1) <= 1
   0 <= state_piecewise(3)_pw_t14_CC_bin_y(2) <= 1
   0 <= state_piecewise(3)_pw_t15_CC_lambda(1) <= +inf
   0 <= state_piecewise(3)_pw_t15_CC_lambda(2) <= +inf
   0 <= state_piecewise(3)_pw_t15_CC_lambda(3) <= +inf
   0 <= state_piecewise(3)_pw_t15_CC_bin_y(1) <= 1
   0 <= state_piecewise(3)_pw_t15_CC_bin_y(2) <= 1
   0 <= state_piecewise(3)_pw_t16_CC_lambda(1) <= +inf
   0 <= state_piecewise(3)_pw_t16_CC_lambda(2) <= +inf
   0 <= state_piecewise(3)_pw_t16_CC_lambda(3) <= +inf
   0 <= state_piecewise(3)_pw_t16_CC_bin_y(1) <= 1
   0 <= state_piecewise(3)_pw_t16_CC_bin_y(2) <= 1
   0 <= state_piecewise(3)_pw_t17_CC_lambda(1) <= +inf
   0 <= state_piecewise(3)_pw_t17_CC_lambda(2) <= +inf
   0 <= state_piecewise(3)_pw_t17_CC_lambda(3) <= +inf
   0 <= state_piecewise(3)_pw_t17_CC_bin_y(1) <= 1
   0 <= state_piecewise(3)_pw_t17_CC_bin_y(2) <= 1
   0 <= state_piecewise(3)_pw_t18_CC_lambda(1) <= +inf
   0 <= state_piecewise(3)_pw_t18_CC_lambda(2) <= +inf
   0 <= state_piecewise(3)_pw_t18_CC_lambda(3) <= +inf
   0 <= state_piecewise(3)_pw_t18_CC_bin_y(1) <= 1
   0 <= state_piecewise(3)_pw_t18_CC_bin_y(2) <= 1
   0 <= state_piecewise(3)_pw_t19_CC_lambda(1) <= +inf
   0 <= state_piecewise(3)_pw_t19_CC_lambda(2) <= +inf
   0 <= state_piecewise(3)_pw_t19_CC_lambda(3) <= +inf
   0 <= state_piecewise(3)_pw_t19_CC_bin_y(1) <= 1
   0 <= state_piecewise(3)_pw_t19_CC_bin_y(2) <= 1
   0 <= state_piecewise(3)_pw_t20_CC_lambda(1) <= +inf
   0 <= state_piecewise(3)_pw_t20_CC_lambda(2) <= +inf
   0 <= state_piecewise(3)_pw_t20_CC_lambda(3) <= +inf
   0 <= state_piecewise(3)_pw_t20_CC_bin_y(1) <= 1
   0 <= state_piecewise(3)_pw_t20_CC_bin_y(2) <= 1
   0 <= state_piecewise(3)_pw_t21_CC_lambda(1) <= +inf
   0 <= state_piecewise(3)_pw_t21_CC_lambda(2) <= +inf
   0 <= state_piecewise(3)_pw_t21_CC_lambda(3) <= +inf
   0 <= state_piecewise(3)_pw_t21_CC_bin_y(1) <= 1
   0 <= state_piecewise(3)_pw_t21_CC_bin_y(2) <= 1
   0 <= state_piecewise(3)_pw_t22_CC_lambda(1) <= +inf
   0 <= state_piecewise(3)_pw_t22_CC_lambda(2) <= +inf
   0 <= state_piecewise(3)_pw_t22_CC_lambda(3) <= +inf
   0 <= state_piecewise(3)_pw_t22_CC_bin_y(1) <= 1
   0 <= state_piecewise(3)_pw_t22_CC_bin_y(2) <= 1
   0 <= state_piecewise(3)_pw_t23_CC_lambda(1) <= +inf
   0 <= state_piecewise(3)_pw_t23_CC_lambda(2) <= +inf
   0 <= state_piecewise(3)_pw_t23_CC_lambda(3) <= +inf
   0 <= state_piecewise(3)_pw_t23_CC_bin_y(1) <= 1
   0 <= state_piecewise(3)_pw_t23_CC_bin_y(2) <= 1
   0 <= state_piecewise(3)_pw_t24_CC_lambda(1) <= +inf
   0 <= state_piecewise(3)_pw_t24_CC_lambda(2) <= +inf
   0 <= state_piecewise(3)_pw_t24_CC_lambda(3) <= +inf
   0 <= state_piecewise(3)_pw_t24_CC_bin_y(1) <= 1
   0 <= state_piecewise(3)_pw_t24_CC_bin_y(2) <= 1
   0 <= state_piecewise(4)_pw_t01_CC_lambda(1) <= +inf
   0 <= state_piecewise(4)_pw_t01_CC_lambda(2) <= +inf
   0 <= state_piecewise(4)_pw_t01_CC_lambda(3) <= +inf
   0 <= state_piecewise(4)_pw_t01_CC_bin_y(1) <= 1
   0 <= state_piecewise(4)_pw_t01_CC_bin_y(2) <= 1
   0 <= state_piecewise(4)_pw_t02_CC_lambda(1) <= +inf
   0 <= state_piecewise(4)_pw_t02_CC_lambda(2) <= +inf
   0 <= state_piecewise(4)_pw_t02_CC_lambda(3) <= +inf
   0 <= state_piecewise(4)_pw_t02_CC_bin_y(1) <= 1
   0 <= state_piecewise(4)_pw_t02_CC_bin_y(2) <= 1
   0 <= state_piecewise(4)_pw_t03_CC_lambda(1) <= +inf
   0 <= state_piecewise(4)_pw_t03_CC_lambda(2) <= +inf
   0 <= state_piecewise(4)_pw_t03_CC_lambda(3) <= +inf
   0 <= state_piecewise(4)_pw_t03_CC_bin_y(1) <= 1
   0 <= state_piecewise(4)_pw_t03_CC_bin_y(2) <= 1
   0 <= state_piecewise(4)_pw_t04_CC_lambda(1) <= +inf
   0 <= state_piecewise(4)_pw_t04_CC_lambda(2) <= +inf
   0 <= state_piecewise(4)_pw_t04_CC_lambda(3) <= +inf
   0 <= state_piecewise(4)_pw_t04_CC_bin_y(1) <= 1
   0 <= state_piecewise(4)_pw_t04_CC_bin_y(2) <= 1
   0 <= state_piecewise(4)_pw_t05_CC_lambda(1) <= +inf
   0 <= state_piecewise(4)_pw_t05_CC_lambda(2) <= +inf
   0 <= state_piecewise(4)_pw_t05_CC_lambda(3) <= +inf
   0 <= state_piecewise(4)_pw_t05_CC_bin_y(1) <= 1
   0 <= state_piecewise(4)_pw_t05_CC_bin_y(2) <= 1
   0 <= state_piecewise(4)_pw_t06_CC_lambda(1) <= +inf
   0 <= state_piecewise(4)_pw_t06_CC_lambda(2) <= +inf
   0 <= state_piecewise(4)_pw_t06_CC_lambda(3) <= +inf
   0 <= state_piecewise(4)_pw_t06_CC_bin_y(1) <= 1
   0 <= state_piecewise(4)_pw_t06_CC_bin_y(2) <= 1
   0 <= state_piecewise(4)_pw_t07_CC_lambda(1) <= +inf
   0 <= state_piecewise(4)_pw_t07_CC_lambda(2) <= +inf
   0 <= state_piecewise(4)_pw_t07_CC_lambda(3) <= +inf
   0 <= state_piecewise(4)_pw_t07_CC_bin_y(1) <= 1
   0 <= state_piecewise(4)_pw_t07_CC_bin_y(2) <= 1
   0 <= state_piecewise(4)_pw_t08_CC_lambda(1) <= +inf
   0 <= state_piecewise(4)_pw_t08_CC_lambda(2) <= +inf
   0 <= state_piecewise(4)_pw_t08_CC_lambda(3) <= +inf
   0 <= state_piecewise(4)_pw_t08_CC_bin_y(1) <= 1
   0 <= state_piecewise(4)_pw_t08_CC_bin_y(2) <= 1
   0 <= state_piecewise(4)_pw_t09_CC_lambda(1) <= +inf
   0 <= state_piecewise(4)_pw_t09_CC_lambda(2) <= +inf
   0 <= state_piecewise(4)_pw_t09_CC_lambda(3) <= +inf
   0 <= state_piecewise(4)_pw_t09_CC_bin_y(1) <= 1
   0 <= state_piecewise(4)_pw_t09_CC_bin_y(2) <= 1
   0 <= state_piecewise(4)_pw_t10_CC_lambda(1) <= +inf
   0 <= state_piecewise(4)_pw_t10_CC_lambda(2) <= +inf
   0 <= state_piecewise(4)_pw_t10_CC_lambda(3) <= +inf
   0 <= state_piecewise(4)_pw_t10_CC_bin_y(1) <= 1
   0 <= state_piecewise(4)_pw_t10_CC_bin_y(2) <= 1
   0 <= state_piecewise(4)_pw_t11_CC_lambda(1) <= +inf
   0 <= state_piecewise(4)_pw_t11_CC_lambda(2) <= +inf
   0 <= state_piecewise(4)_pw_t11_CC_lambda(3) <= +inf
   0 <= state_piecewise(4)_pw_t11_CC_bin_y(1) <= 1
   0 <= state_piecewise(4)_pw_t11_CC_bin_y(2) <= 1
   0 <= state_piecewise(4)_pw_t12_CC_lambda(1) <= +inf
   0 <= state_piecewise(4)_pw_t12_CC_lambda(2) <= +inf
   0 <= state_piecewise(4)_pw_t12_CC_lambda(3) <= +inf
   0 <= state_piecewise(4)_pw_t12_CC_bin_y(1) <= 1
   0 <= state_piecewise(4)_pw_t12_CC_bin_y(2) <= 1
   0 <= state_piecewise(4)_pw_t13_CC_lambda(1) <= +inf
   0 <= state_piecewise(4)_pw_t13_CC_lambda(2) <= +inf
   0 <= state_piecewise(4)_pw_t13_CC_lambda(3) <= +inf
   0 <= state_piecewise(4)_pw_t13_CC_bin_y(1) <= 1
   0 <= state_piecewise(4)_pw_t13_CC_bin_y(2) <= 1
   0 <= state_piecewise(4)_pw_t14_CC_lambda(1) <= +inf
   0 <= state_piecewise(4)_pw_t14_CC_lambda(2) <= +inf
   0 <= state_piecewise(4)_pw_t14_CC_lambda(3) <= +inf
   0 <= state_piecewise(4)_pw_t14_CC_bin_y(1) <= 1
   0 <= state_piecewise(4)_pw_t14_CC_bin_y(2) <= 1
   0 <= state_piecewise(4)_pw_t15_CC_lambda(1) <= +inf
   0 <= state_piecewise(4)_pw_t15_CC_lambda(2) <= +inf
   0 <= state_piecewise(4)_pw_t15_CC_lambda(3) <= +inf
   0 <= state_piecewise(4)_pw_t15_CC_bin_y(1) <= 1
   0 <= state_piecewise(4)_pw_t15_CC_bin_y(2) <= 1
   0 <= state_piecewise(4)_pw_t16_CC_lambda(1) <= +inf
   0 <= state_piecewise(4)_pw_t16_CC_lambda(2) <= +inf
   0 <= state_piecewise(4)_pw_t16_CC_lambda(3) <= +inf
   0 <= state_piecewise(4)_pw_t16_CC_bin_y(1) <= 1
   0 <= state_piecewise(4)_pw_t16_CC_bin_y(2) <= 1
   0 <= state_piecewise(4)_pw_t17_CC_lambda(1) <= +inf
   0 <= state_piecewise(4)_pw_t17_CC_lambda(2) <= +inf
   0 <= state_piecewise(4)_pw_t17_CC_lambda(3) <= +inf
   0 <= state_piecewise(4)_pw_t17_CC_bin_y(1) <= 1
   0 <= state_piecewise(4)_pw_t17_CC_bin_y(2) <= 1
   0 <= state_piecewise(4)_pw_t18_CC_lambda(1) <= +inf
   0 <= state_piecewise(4)_pw_t18_CC_lambda(2) <= +inf
   0 <= state_piecewise(4)_pw_t18_CC_lambda(3) <= +inf
   0 <= state_piecewise(4)_pw_t18_CC_bin_y(1) <= 1
   0 <= state_piecewise(4)_pw_t18_CC_bin_y(2) <= 1
   0 <= state_piecewise(4)_pw_t19_CC_lambda(1) <= +inf
   0 <= state_piecewise(4)_pw_t19_CC_lambda(2) <= +inf
   0 <= state_piecewise(4)_pw_t19_CC_lambda(3) <= +inf
   0 <= state_piecewise(4)_pw_t19_CC_bin_y(1) <= 1
   0 <= state_piecewise(4)_pw_t19_CC_bin_y(2) <= 1
   0 <= state_piecewise(4)_pw_t20_CC_lambda(1) <= +inf
   0 <= state_piecewise(4)_pw_t20_CC_lambda(2) <= +inf
   0 <= state_piecewise(4)_pw_t20_CC_lambda(3) <= +inf
   0 <= state_piecewise(4)_pw_t20_CC_bin_y(1) <= 1
   0 <= state_piecewise(4)_pw_t20_CC_bin_y(2) <= 1
   0 <= state_piecewise(4)_pw_t21_CC_lambda(1) <= +inf
   0 <= state_piecewise(4)_pw_t21_CC_lambda(2) <= +inf
   0 <= state_piecewise(4)_pw_t21_CC_lambda(3) <= +inf
   0 <= state_piecewise(4)_pw_t21_CC_bin_y(1) <= 1
   0 <= state_piecewise(4)_pw_t21_CC_bin_y(2) <= 1
   0 <= state_piecewise(4)_pw_t22_CC_lambda(1) <= +inf
   0 <= state_piecewise(4)_pw_t22_CC_lambda(2) <= +inf
   0 <= state_piecewise(4)_pw_t22_CC_lambda(3) <= +inf
   0 <= state_piecewise(4)_pw_t22_CC_bin_y(1) <= 1
   0 <= state_piecewise(4)_pw_t22_CC_bin_y(2) <= 1
   0 <= state_piecewise(4)_pw_t23_CC_lambda(1) <= +inf
   0 <= state_piecewise(4)_pw_t23_CC_lambda(2) <= +inf
   0 <= state_piecewise(4)_pw_t23_CC_lambda(3) <= +inf
   0 <= state_piecewise(4)_pw_t23_CC_bin_y(1) <= 1
   0 <= state_piecewise(4)_pw_t23_CC_bin_y(2) <= 1
   0 <= state_piecewise(4)_pw_t24_CC_lambda(1) <= +inf
   0 <= state_piecewise(4)_pw_t24_CC_lambda(2) <= +inf
   0 <= state_piecewise(4)_pw_t24_CC_lambda(3) <= +inf
   0 <= state_piecewise(4)_pw_t24_CC_bin_y(1) <= 1
   0 <= state_piecewise(4)_pw_t24_CC_bin_y(2) <= 1
   0 <= state_piecewise(5)_pw_t01_CC_lambda(1) <= +inf
   0 <= state_piecewise(5)_pw_t01_CC_lambda(2) <= +inf
   0 <= state_piecewise(5)_pw_t01_CC_lambda(3) <= +inf
   0 <= state_piecewise(5)_pw_t01_CC_bin_y(1) <= 1
   0 <= state_piecewise(5)_pw_t01_CC_bin_y(2) <= 1
   0 <= state_piecewise(5)_pw_t02_CC_lambda(1) <= +inf
   0 <= state_piecewise(5)_pw_t02_CC_lambda(2) <= +inf
   0 <= state_piecewise(5)_pw_t02_CC_lambda(3) <= +inf
   0 <= state_piecewise(5)_pw_t02_CC_bin_y(1) <= 1
   0 <= state_piecewise(5)_pw_t02_CC_bin_y(2) <= 1
   0 <= state_piecewise(5)_pw_t03_CC_lambda(1) <= +inf
   0 <= state_piecewise(5)_pw_t03_CC_lambda(2) <= +inf
   0 <= state_piecewise(5)_pw_t03_CC_lambda(3) <= +inf
   0 <= state_piecewise(5)_pw_t03_CC_bin_y(1) <= 1
   0 <= state_piecewise(5)_pw_t03_CC_bin_y(2) <= 1
   0 <= state_piecewise(5)_pw_t04_CC_lambda(1) <= +inf
   0 <= state_piecewise(5)_pw_t04_CC_lambda(2) <= +inf
   0 <= state_piecewise(5)_pw_t04_CC_lambda(3) <= +inf
   0 <= state_piecewise(5)_pw_t04_CC_bin_y(1) <= 1
   0 <= state_piecewise(5)_pw_t04_CC_bin_y(2) <= 1
   0 <= state_piecewise(5)_pw_t05_CC_lambda(1) <= +inf
   0 <= state_piecewise(5)_pw_t05_CC_lambda(2) <= +inf
   0 <= state_piecewise(5)_pw_t05_CC_lambda(3) <= +inf
   0 <= state_piecewise(5)_pw_t05_CC_bin_y(1) <= 1
   0 <= state_piecewise(5)_pw_t05_CC_bin_y(2) <= 1
   0 <= state_piecewise(5)_pw_t06_CC_lambda(1) <= +inf
   0 <= state_piecewise(5)_pw_t06_CC_lambda(2) <= +inf
   0 <= state_piecewise(5)_pw_t06_CC_lambda(3) <= +inf
   0 <= state_piecewise(5)_pw_t06_CC_bin_y(1) <= 1
   0 <= state_piecewise(5)_pw_t06_CC_bin_y(2) <= 1
   0 <= state_piecewise(5)_pw_t07_CC_lambda(1) <= +inf
   0 <= state_piecewise(5)_pw_t07_CC_lambda(2) <= +inf
   0 <= state_piecewise(5)_pw_t07_CC_lambda(3) <= +inf
   0 <= state_piecewise(5)_pw_t07_CC_bin_y(1) <= 1
   0 <= state_piecewise(5)_pw_t07_CC_bin_y(2) <= 1
   0 <= state_piecewise(5)_pw_t08_CC_lambda(1) <= +inf
   0 <= state_piecewise(5)_pw_t08_CC_lambda(2) <= +inf
   0 <= state_piecewise(5)_pw_t08_CC_lambda(3) <= +inf
   0 <= state_piecewise(5)_pw_t08_CC_bin_y(1) <= 1
   0 <= state_piecewise(5)_pw_t08_CC_bin_y(2) <= 1
   0 <= state_piecewise(5)_pw_t09_CC_lambda(1) <= +inf
   0 <= state_piecewise(5)_pw_t09_CC_lambda(2) <= +inf
   0 <= state_piecewise(5)_pw_t09_CC_lambda(3) <= +inf
   0 <= state_piecewise(5)_pw_t09_CC_bin_y(1) <= 1
   0 <= state_piecewise(5)_pw_t09_CC_bin_y(2) <= 1
   0 <= state_piecewise(5)_pw_t10_CC_lambda(1) <= +inf
   0 <= state_piecewise(5)_pw_t10_CC_lambda(2) <= +inf
   0 <= state_piecewise(5)_pw_t10_CC_lambda(3) <= +inf
   0 <= state_piecewise(5)_pw_t10_CC_bin_y(1) <= 1
   0 <= state_piecewise(5)_pw_t10_CC_bin_y(2) <= 1
   0 <= state_piecewise(5)_pw_t11_CC_lambda(1) <= +inf
   0 <= state_piecewise(5)_pw_t11_CC_lambda(2) <= +inf
   0 <= state_piecewise(5)_pw_t11_CC_lambda(3) <= +inf
   0 <= state_piecewise(5)_pw_t11_CC_bin_y(1) <= 1
   0 <= state_piecewise(5)_pw_t11_CC_bin_y(2) <= 1
   0 <= state_piecewise(5)_pw_t12_CC_lambda(1) <= +inf
   0 <= state_piecewise(5)_pw_t12_CC_lambda(2) <= +inf
   0 <= state_piecewise(5)_pw_t12_CC_lambda(3) <= +inf
   0 <= state_piecewise(5)_pw_t12_CC_bin_y(1) <= 1
   0 <= state_piecewise(5)_pw_t12_CC_bin_y(2) <= 1
   0 <= state_piecewise(5)_pw_t13_CC_lambda(1) <= +inf
   0 <= state_piecewise(5)_pw_t13_CC_lambda(2) <= +inf
   0 <= state_piecewise(5)_pw_t13_CC_lambda(3) <= +inf
   0 <= state_piecewise(5)_pw_t13_CC_bin_y(1) <= 1
   0 <= state_piecewise(5)_pw_t13_CC_bin_y(2) <= 1
   0 <= state_piecewise(5)_pw_t14_CC_lambda(1) <= +inf
   0 <= state_piecewise(5)_pw_t14_CC_lambda(2) <= +inf
   0 <= state_piecewise(5)_pw_t14_CC_lambda(3) <= +inf
   0 <= state_piecewise(5)_pw_t14_CC_bin_y(1) <= 1
   0 <= state_piecewise(5)_pw_t14_CC_bin_y(2) <= 1
   0 <= state_piecewise(5)_pw_t15_CC_lambda(1) <= +inf
   0 <= state_piecewise(5)_pw_t15_CC_lambda(2) <= +inf
   0 <= state_piecewise(5)_pw_t15_CC_lambda(3) <= +inf
   0 <= state_piecewise(5)_pw_t15_CC_bin_y(1) <= 1
   0 <= state_piecewise(5)_pw_t15_CC_bin_y(2) <= 1
   0 <= state_piecewise(5)_pw_t16_CC_lambda(1) <= +inf
   0 <= state_piecewise(5)_pw_t16_CC_lambda(2) <= +inf
   0 <= state_piecewise(5)_pw_t16_CC_lambda(3) <= +inf
   0 <= state_piecewise(5)_pw_t16_CC_bin_y(1) <= 1
   0 <= state_piecewise(5)_pw_t16_CC_bin_y(2) <= 1
   0 <= state_piecewise(5)_pw_t17_CC_lambda(1) <= +inf
   0 <= state_piecewise(5)_pw_t17_CC_lambda(2) <= +inf
   0 <= state_piecewise(5)_pw_t17_CC_lambda(3) <= +inf
   0 <= state_piecewise(5)_pw_t17_CC_bin_y(1) <= 1
   0 <= state_piecewise(5)_pw_t17_CC_bin_y(2) <= 1
   0 <= state_piecewise(5)_pw_t18_CC_lambda(1) <= +inf
   0 <= state_piecewise(5)_pw_t18_CC_lambda(2) <= +inf
   0 <= state_piecewise(5)_pw_t18_CC_lambda(3) <= +inf
   0 <= state_piecewise(5)_pw_t18_CC_bin_y(1) <= 1
   0 <= state_piecewise(5)_pw_t18_CC_bin_y(2) <= 1
   0 <= state_piecewise(5)_pw_t19_CC_lambda(1) <= +inf
   0 <= state_piecewise(5)_pw_t19_CC_lambda(2) <= +inf
   0 <= state_piecewise(5)_pw_t19_CC_lambda(3) <= +inf
   0 <= state_piecewise(5)_pw_t19_CC_bin_y(1) <= 1
   0 <= state_piecewise(5)_pw_t19_CC_bin_y(2) <= 1
   0 <= state_piecewise(5)_pw_t20_CC_lambda(1) <= +inf
   0 <= state_piecewise(5)_pw_t20_CC_lambda(2) <= +inf
   0 <= state_piecewise(5)_pw_t20_CC_lambda(3) <= +inf
   0 <= state_piecewise(5)_pw_t20_CC_bin_y(1) <= 1
   0 <= state_piecewise(5)_pw_t20_CC_bin_y(2) <= 1
   0 <= state_piecewise(5)_pw_t21_CC_lambda(1) <= +inf
   0 <= state_piecewise(5)_pw_t21_CC_lambda(2) <= +inf
   0 <= state_piecewise(5)_pw_t21_CC_lambda(3) <= +inf
   0 <= state_piecewise(5)_pw_t21_CC_bin_y(1) <= 1
   0 <= state_piecewise(5)_pw_t21_CC_bin_y(2) <= 1
   0 <= state_piecewise(5)_pw_t22_CC_lambda(1) <= +inf
   0 <= state_piecewise(5)_pw_t22_CC_lambda(2) <= +inf
   0 <= state_piecewise(5)_pw_t22_CC_lambda(3) <= +inf
   0 <= state_piecewise(5)_pw_t22_CC_bin_y(1) <= 1
   0 <= state_piecewise(5)_pw_t22_CC_bin_y(2) <= 1
   0 <= state_piecewise(5)_pw_t23_CC_lambda(1) <= +inf
   0 <= state_piecewise(5)_pw_t23_CC_lambda(2) <= +inf
   0 <= state_piecewise(5)_pw_t23_CC_lambda(3) <= +inf
   0 <= state_piecewise(5)_pw_t23_CC_bin_y(1) <= 1
   0 <= state_piecewise(5)_pw_t23_CC_bin_y(2) <= 1
   0 <= state_piecewise(5)_pw_t24_CC_lambda(1) <= +inf
   0 <= state_piecewise(5)_pw_t24_CC_lambda(2) <= +inf
   0 <= state_piecewise(5)_pw_t24_CC_lambda(3) <= +inf
   0 <= state_piecewise(5)_pw_t24_CC_bin_y(1) <= 1
   0 <= state_piecewise(5)_pw_t24_CC_bin_y(2) <= 1
   0 <= state_piecewise(6)_pw_t01_CC_lambda(1) <= +inf
   0 <= state_piecewise(6)_pw_t01_CC_lambda(2) <= +inf
   0 <= state_piecewise(6)_pw_t01_CC_lambda(3) <= +inf
   0 <= state_piecewise(6)_pw_t01_CC_bin_y(1) <= 1
   0 <= state_piecewise(6)_pw_t01_CC_bin_y(2) <= 1
   0 <= state_piecewise(6)_pw_t02_CC_lambda(1) <= +inf
   0 <= state_piecewise(6)_pw_t02_CC_lambda(2) <= +inf
   0 <= state_piecewise(6)_pw_t02_CC_lambda(3) <= +inf
   0 <= state_piecewise(6)_pw_t02_CC_bin_y(1) <= 1
   0 <= state_piecewise(6)_pw_t02_CC_bin_y(2) <= 1
   0 <= state_piecewise(6)_pw_t03_CC_lambda(1) <= +inf
   0 <= state_piecewise(6)_pw_t03_CC_lambda(2) <= +inf
   0 <= state_piecewise(6)_pw_t03_CC_lambda(3) <= +inf
   0 <= state_piecewise(6)_pw_t03_CC_bin_y(1) <= 1
   0 <= state_piecewise(6)_pw_t03_CC_bin_y(2) <= 1
   0 <= state_piecewise(6)_pw_t04_CC_lambda(1) <= +inf
   0 <= state_piecewise(6)_pw_t04_CC_lambda(2) <= +inf
   0 <= state_piecewise(6)_pw_t04_CC_lambda(3) <= +inf
   0 <= state_piecewise(6)_pw_t04_CC_bin_y(1) <= 1
   0 <= state_piecewise(6)_pw_t04_CC_bin_y(2) <= 1
   0 <= state_piecewise(6)_pw_t05_CC_lambda(1) <= +inf
   0 <= state_piecewise(6)_pw_t05_CC_lambda(2) <= +inf
   0 <= state_piecewise(6)_pw_t05_CC_lambda(3) <= +inf
   0 <= state_piecewise(6)_pw_t05_CC_bin_y(1) <= 1
   0 <= state_piecewise(6)_pw_t05_CC_bin_y(2) <= 1
   0 <= state_piecewise(6)_pw_t06_CC_lambda(1) <= +inf
   0 <= state_piecewise(6)_pw_t06_CC_lambda(2) <= +inf
   0 <= state_piecewise(6)_pw_t06_CC_lambda(3) <= +inf
   0 <= state_piecewise(6)_pw_t06_CC_bin_y(1) <= 1
   0 <= state_piecewise(6)_pw_t06_CC_bin_y(2) <= 1
   0 <= state_piecewise(6)_pw_t07_CC_lambda(1) <= +inf
   0 <= state_piecewise(6)_pw_t07_CC_lambda(2) <= +inf
   0 <= state_piecewise(6)_pw_t07_CC_lambda(3) <= +inf
   0 <= state_piecewise(6)_pw_t07_CC_bin_y(1) <= 1
   0 <= state_piecewise(6)_pw_t07_CC_bin_y(2) <= 1
   0 <= state_piecewise(6)_pw_t08_CC_lambda(1) <= +inf
   0 <= state_piecewise(6)_pw_t08_CC_lambda(2) <= +inf
   0 <= state_piecewise(6)_pw_t08_CC_lambda(3) <= +inf
   0 <= state_piecewise(6)_pw_t08_CC_bin_y(1) <= 1
   0 <= state_piecewise(6)_pw_t08_CC_bin_y(2) <= 1
   0 <= state_piecewise(6)_pw_t09_CC_lambda(1) <= +inf
   0 <= state_piecewise(6)_pw_t09_CC_lambda(2) <= +inf
   0 <= state_piecewise(6)_pw_t09_CC_lambda(3) <= +inf
   0 <= state_piecewise(6)_pw_t09_CC_bin_y(1) <= 1
   0 <= state_piecewise(6)_pw_t09_CC_bin_y(2) <= 1
   0 <= state_piecewise(6)_pw_t10_CC_lambda(1) <= +inf
   0 <= state_piecewise(6)_pw_t10_CC_lambda(2) <= +inf
   0 <= state_piecewise(6)_pw_t10_CC_lambda(3) <= +inf
   0 <= state_piecewise(6)_pw_t10_CC_bin_y(1) <= 1
   0 <= state_piecewise(6)_pw_t10_CC_bin_y(2) <= 1
   0 <= state_piecewise(6)_pw_t11_CC_lambda(1) <= +inf
   0 <= state_piecewise(6)_pw_t11_CC_lambda(2) <= +inf
   0 <= state_piecewise(6)_pw_t11_CC_lambda(3) <= +inf
   0 <= state_piecewise(6)_pw_t11_CC_bin_y(1) <= 1
   0 <= state_piecewise(6)_pw_t11_CC_bin_y(2) <= 1
   0 <= state_piecewise(6)_pw_t12_CC_lambda(1) <= +inf
   0 <= state_piecewise(6)_pw_t12_CC_lambda(2) <= +inf
   0 <= state_piecewise(6)_pw_t12_CC_lambda(3) <= +inf
   0 <= state_piecewise(6)_pw_t12_CC_bin_y(1) <= 1
   0 <= state_piecewise(6)_pw_t12_CC_bin_y(2) <= 1
   0 <= state_piecewise(6)_pw_t13_CC_lambda(1) <= +inf
   0 <= state_piecewise(6)_pw_t13_CC_lambda(2) <= +inf
   0 <= state_piecewise(6)_pw_t13_CC_lambda(3) <= +inf
   0 <= state_piecewise(6)_pw_t13_CC_bin_y(1) <= 1
   0 <= state_piecewise(6)_pw_t13_CC_bin_y(2) <= 1
   0 <= state_piecewise(6)_pw_t14_CC_lambda(1) <= +inf
   0 <= state_piecewise(6)_pw_t14_CC_lambda(2) <= +inf
   0 <= state_piecewise(6)_pw_t14_CC_lambda(3) <= +inf
   0 <= state_piecewise(6)_pw_t14_CC_bin_y(1) <= 1
   0 <= state_piecewise(6)_pw_t14_CC_bin_y(2) <= 1
   0 <= state_piecewise(6)_pw_t15_CC_lambda(1) <= +inf
   0 <= state_piecewise(6)_pw_t15_CC_lambda(2) <= +inf
   0 <= state_piecewise(6)_pw_t15_CC_lambda(3) <= +inf
   0 <= state_piecewise(6)_pw_t15_CC_bin_y(1) <= 1
   0 <= state_piecewise(6)_pw_t15_CC_bin_y(2) <= 1
   0 <= state_piecewise(6)_pw_t16_CC_lambda(1) <= +inf
   0 <= state_piecewise(6)_pw_t16_CC_lambda(2) <= +inf
   0 <= state_piecewise(6)_pw_t16_CC_lambda(3) <= +inf
   0 <= state_piecewise(6)_pw_t16_CC_bin_y(1) <= 1
   0 <= state_piecewise(6)_pw_t16_CC_bin_y(2) <= 1
   0 <= state_piecewise(6)_pw_t17_CC_lambda(1) <= +inf
   0 <= state_piecewise(6)_pw_t17_CC_lambda(2) <= +inf
   0 <= state_piecewise(6)_pw_t17_CC_lambda(3) <= +inf
   0 <= state_piecewise(6)_pw_t17_CC_bin_y(1) <= 1
   0 <= state_piecewise(6)_pw_t17_CC_bin_y(2) <= 1
   0 <= state_piecewise(6)_pw_t18_CC_lambda(1) <= +inf
   0 <= state_piecewise(6)_pw_t18_CC_lambda(2) <= +inf
   0 <= state_piecewise(6)_pw_t18_CC_lambda(3) <= +inf
   0 <= state_piecewise(6)_pw_t18_CC_bin_y(1) <= 1
   0 <= state_piecewise(6)_pw_t18_CC_bin_y(2) <= 1
   0 <= state_piecewise(6)_pw_t19_CC_lambda(1) <= +inf
   0 <= state_piecewise(6)_pw_t19_CC_lambda(2) <= +inf
   0 <= state_piecewise(6)_pw_t19_CC_lambda(3) <= +inf
   0 <= state_piecewise(6)_pw_t19_CC_bin_y(1) <= 1
   0 <= state_piecewise(6)_pw_t19_CC_bin_y(2) <= 1
   0 <= state_piecewise(6)_pw_t20_CC_lambda(1) <= +inf
   0 <= state_piecewise(6)_pw_t20_CC_lambda(2) <= +inf
   0 <= state_piecewise(6)_pw_t20_CC_lambda(3) <= +inf
   0 <= state_piecewise(6)_pw_t20_CC_bin_y(1) <= 1
   0 <= state_piecewise(6)_pw_t20_CC_bin_y(2) <= 1
   0 <= state_piecewise(6)_pw_t21_CC_lambda(1) <= +inf
   0 <= state_piecewise(6)_pw_t21_CC_lambda(2) <= +inf
   0 <= state_piecewise(6)_pw_t21_CC_lambda(3) <= +inf
   0 <= state_piecewise(6)_pw_t21_CC_bin_y(1) <= 1
   0 <= state_piecewise(6)_pw_t21_CC_bin_y(2) <= 1
   0 <= state_piecewise(6)_pw_t22_CC_lambda(1) <= +inf
   0 <= state_piecewise(6)_pw_t22_CC_lambda(2) <= +inf
   0 <= state_piecewise(6)_pw_t22_CC_lambda(3) <= +inf
   0 <= state_piecewise(6)_pw_t22_CC_bin_y(1) <= 1
   0 <= state_piecewise(6)_pw_t22_CC_bin_y(2) <= 1
   0 <= state_piecewise(6)_pw_t23_CC_lambda(1) <= +inf
   0 <= state_piecewise(6)_pw_t23_CC_lambda(2) <= +inf
   0 <= state_piecewise(6)_pw_t23_CC_lambda(3) <= +inf
   0 <= state_piecewise(6)_pw_t23_CC_bin_y(1) <= 1
   0 <= state_piecewise(6)_pw_t23_CC_bin_y(2) <= 1
   0 <= state_piecewise(6)_pw_t24_CC_lambda(1) <= +inf
   0 <= state_piecewise(6)_pw_t24_CC_lambda(2) <= +inf
   0 <= state_piecewise(6)_pw_t24_CC_lambda(3) <= +inf
   0 <= state_piecewise(6)_pw_t24_CC_bin_y(1) <= 1
   0 <= state_piecewise(6)_pw_t24_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t01)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate1_t01)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate1_t01)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate1_t01)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate1_t01)_CC_lambda(5) <= +inf
   0 <= gate_piecewise(gate1_t01)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate1_t01)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t01)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate1_t01)_CC_bin_y(4) <= 1
   0 <= gate_piecewise(gate1_t02)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate1_t02)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate1_t02)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate1_t02)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate1_t02)_CC_lambda(5) <= +inf
   0 <= gate_piecewise(gate1_t02)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate1_t02)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t02)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate1_t02)_CC_bin_y(4) <= 1
   0 <= gate_piecewise(gate1_t03)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate1_t03)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate1_t03)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate1_t03)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate1_t03)_CC_lambda(5) <= +inf
   0 <= gate_piecewise(gate1_t03)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate1_t03)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t03)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate1_t03)_CC_bin_y(4) <= 1
   0 <= gate_piecewise(gate1_t04)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate1_t04)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate1_t04)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate1_t04)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate1_t04)_CC_lambda(5) <= +inf
   0 <= gate_piecewise(gate1_t04)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate1_t04)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t04)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate1_t04)_CC_bin_y(4) <= 1
   0 <= gate_piecewise(gate1_t05)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate1_t05)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate1_t05)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate1_t05)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate1_t05)_CC_lambda(5) <= +inf
   0 <= gate_piecewise(gate1_t05)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate1_t05)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t05)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate1_t05)_CC_bin_y(4) <= 1
   0 <= gate_piecewise(gate1_t06)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate1_t06)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate1_t06)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate1_t06)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate1_t06)_CC_lambda(5) <= +inf
   0 <= gate_piecewise(gate1_t06)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate1_t06)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t06)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate1_t06)_CC_bin_y(4) <= 1
   0 <= gate_piecewise(gate1_t07)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate1_t07)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate1_t07)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate1_t07)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate1_t07)_CC_lambda(5) <= +inf
   0 <= gate_piecewise(gate1_t07)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate1_t07)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t07)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate1_t07)_CC_bin_y(4) <= 1
   0 <= gate_piecewise(gate1_t08)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate1_t08)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate1_t08)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate1_t08)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate1_t08)_CC_lambda(5) <= +inf
   0 <= gate_piecewise(gate1_t08)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate1_t08)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t08)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate1_t08)_CC_bin_y(4) <= 1
   0 <= gate_piecewise(gate1_t09)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate1_t09)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate1_t09)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate1_t09)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate1_t09)_CC_lambda(5) <= +inf
   0 <= gate_piecewise(gate1_t09)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate1_t09)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t09)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate1_t09)_CC_bin_y(4) <= 1
   0 <= gate_piecewise(gate1_t10)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate1_t10)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate1_t10)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate1_t10)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate1_t10)_CC_lambda(5) <= +inf
   0 <= gate_piecewise(gate1_t10)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate1_t10)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t10)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate1_t10)_CC_bin_y(4) <= 1
   0 <= gate_piecewise(gate1_t11)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate1_t11)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate1_t11)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate1_t11)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate1_t11)_CC_lambda(5) <= +inf
   0 <= gate_piecewise(gate1_t11)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate1_t11)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t11)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate1_t11)_CC_bin_y(4) <= 1
   0 <= gate_piecewise(gate1_t12)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate1_t12)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate1_t12)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate1_t12)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate1_t12)_CC_lambda(5) <= +inf
   0 <= gate_piecewise(gate1_t12)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate1_t12)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t12)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate1_t12)_CC_bin_y(4) <= 1
   0 <= gate_piecewise(gate1_t13)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate1_t13)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate1_t13)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate1_t13)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate1_t13)_CC_lambda(5) <= +inf
   0 <= gate_piecewise(gate1_t13)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate1_t13)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t13)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate1_t13)_CC_bin_y(4) <= 1
   0 <= gate_piecewise(gate1_t14)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate1_t14)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate1_t14)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate1_t14)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate1_t14)_CC_lambda(5) <= +inf
   0 <= gate_piecewise(gate1_t14)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate1_t14)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t14)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate1_t14)_CC_bin_y(4) <= 1
   0 <= gate_piecewise(gate1_t15)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate1_t15)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate1_t15)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate1_t15)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate1_t15)_CC_lambda(5) <= +inf
   0 <= gate_piecewise(gate1_t15)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate1_t15)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t15)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate1_t15)_CC_bin_y(4) <= 1
   0 <= gate_piecewise(gate1_t16)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate1_t16)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate1_t16)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate1_t16)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate1_t16)_CC_lambda(5) <= +inf
   0 <= gate_piecewise(gate1_t16)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate1_t16)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t16)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate1_t16)_CC_bin_y(4) <= 1
   0 <= gate_piecewise(gate1_t17)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate1_t17)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate1_t17)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate1_t17)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate1_t17)_CC_lambda(5) <= +inf
   0 <= gate_piecewise(gate1_t17)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate1_t17)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t17)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate1_t17)_CC_bin_y(4) <= 1
   0 <= gate_piecewise(gate1_t18)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate1_t18)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate1_t18)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate1_t18)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate1_t18)_CC_lambda(5) <= +inf
   0 <= gate_piecewise(gate1_t18)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate1_t18)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t18)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate1_t18)_CC_bin_y(4) <= 1
   0 <= gate_piecewise(gate1_t19)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate1_t19)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate1_t19)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate1_t19)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate1_t19)_CC_lambda(5) <= +inf
   0 <= gate_piecewise(gate1_t19)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate1_t19)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t19)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate1_t19)_CC_bin_y(4) <= 1
   0 <= gate_piecewise(gate1_t20)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate1_t20)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate1_t20)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate1_t20)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate1_t20)_CC_lambda(5) <= +inf
   0 <= gate_piecewise(gate1_t20)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate1_t20)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t20)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate1_t20)_CC_bin_y(4) <= 1
   0 <= gate_piecewise(gate1_t21)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate1_t21)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate1_t21)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate1_t21)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate1_t21)_CC_lambda(5) <= +inf
   0 <= gate_piecewise(gate1_t21)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate1_t21)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t21)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate1_t21)_CC_bin_y(4) <= 1
   0 <= gate_piecewise(gate1_t22)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate1_t22)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate1_t22)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate1_t22)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate1_t22)_CC_lambda(5) <= +inf
   0 <= gate_piecewise(gate1_t22)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate1_t22)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t22)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate1_t22)_CC_bin_y(4) <= 1
   0 <= gate_piecewise(gate1_t23)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate1_t23)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate1_t23)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate1_t23)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate1_t23)_CC_lambda(5) <= +inf
   0 <= gate_piecewise(gate1_t23)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate1_t23)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t23)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate1_t23)_CC_bin_y(4) <= 1
   0 <= gate_piecewise(gate1_t24)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate1_t24)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate1_t24)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate1_t24)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate1_t24)_CC_lambda(5) <= +inf
   0 <= gate_piecewise(gate1_t24)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate1_t24)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate1_t24)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate1_t24)_CC_bin_y(4) <= 1
   0 <= gate_piecewise(gate2_t01)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate2_t01)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate2_t01)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate2_t01)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate2_t01)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate2_t01)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate2_t01)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate2_t02)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate2_t02)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate2_t02)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate2_t02)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate2_t02)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate2_t02)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate2_t02)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate2_t03)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate2_t03)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate2_t03)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate2_t03)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate2_t03)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate2_t03)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate2_t03)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate2_t04)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate2_t04)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate2_t04)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate2_t04)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate2_t04)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate2_t04)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate2_t04)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate2_t05)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate2_t05)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate2_t05)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate2_t05)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate2_t05)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate2_t05)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate2_t05)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate2_t06)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate2_t06)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate2_t06)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate2_t06)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate2_t06)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate2_t06)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate2_t06)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate2_t07)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate2_t07)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate2_t07)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate2_t07)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate2_t07)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate2_t07)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate2_t07)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate2_t08)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate2_t08)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate2_t08)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate2_t08)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate2_t08)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate2_t08)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate2_t08)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate2_t09)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate2_t09)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate2_t09)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate2_t09)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate2_t09)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate2_t09)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate2_t09)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate2_t10)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate2_t10)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate2_t10)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate2_t10)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate2_t10)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate2_t10)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate2_t10)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate2_t11)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate2_t11)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate2_t11)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate2_t11)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate2_t11)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate2_t11)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate2_t11)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate2_t12)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate2_t12)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate2_t12)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate2_t12)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate2_t12)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate2_t12)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate2_t12)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate2_t13)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate2_t13)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate2_t13)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate2_t13)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate2_t13)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate2_t13)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate2_t13)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate2_t14)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate2_t14)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate2_t14)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate2_t14)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate2_t14)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate2_t14)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate2_t14)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate2_t15)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate2_t15)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate2_t15)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate2_t15)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate2_t15)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate2_t15)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate2_t15)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate2_t16)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate2_t16)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate2_t16)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate2_t16)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate2_t16)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate2_t16)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate2_t16)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate2_t17)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate2_t17)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate2_t17)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate2_t17)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate2_t17)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate2_t17)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate2_t17)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate2_t18)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate2_t18)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate2_t18)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate2_t18)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate2_t18)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate2_t18)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate2_t18)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate2_t19)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate2_t19)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate2_t19)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate2_t19)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate2_t19)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate2_t19)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate2_t19)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate2_t20)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate2_t20)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate2_t20)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate2_t20)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate2_t20)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate2_t20)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate2_t20)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate2_t21)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate2_t21)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate2_t21)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate2_t21)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate2_t21)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate2_t21)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate2_t21)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate2_t22)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate2_t22)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate2_t22)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate2_t22)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate2_t22)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate2_t22)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate2_t22)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate2_t23)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate2_t23)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate2_t23)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate2_t23)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate2_t23)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate2_t23)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate2_t23)_CC_bin_y(3) <= 1
   0 <= gate_piecewise(gate2_t24)_CC_lambda(1) <= +inf
   0 <= gate_piecewise(gate2_t24)_CC_lambda(2) <= +inf
   0 <= gate_piecewise(gate2_t24)_CC_lambda(3) <= +inf
   0 <= gate_piecewise(gate2_t24)_CC_lambda(4) <= +inf
   0 <= gate_piecewise(gate2_t24)_CC_bin_y(1) <= 1
   0 <= gate_piecewise(gate2_t24)_CC_bin_y(2) <= 1
   0 <= gate_piecewise(gate2_t24)_CC_bin_y(3) <= 1
binary
  state_piecewise(0)_pw_t01_CC_bin_y(1)
  state_piecewise(0)_pw_t01_CC_bin_y(2)
  state_piecewise(0)_pw_t02_CC_bin_y(1)
  state_piecewise(0)_pw_t02_CC_bin_y(2)
  state_piecewise(0)_pw_t03_CC_bin_y(1)
  state_piecewise(0)_pw_t03_CC_bin_y(2)
  state_piecewise(0)_pw_t04_CC_bin_y(1)
  state_piecewise(0)_pw_t04_CC_bin_y(2)
  state_piecewise(0)_pw_t05_CC_bin_y(1)
  state_piecewise(0)_pw_t05_CC_bin_y(2)
  state_piecewise(0)_pw_t06_CC_bin_y(1)
  state_piecewise(0)_pw_t06_CC_bin_y(2)
  state_piecewise(0)_pw_t07_CC_bin_y(1)
  state_piecewise(0)_pw_t07_CC_bin_y(2)
  state_piecewise(0)_pw_t08_CC_bin_y(1)
  state_piecewise(0)_pw_t08_CC_bin_y(2)
  state_piecewise(0)_pw_t09_CC_bin_y(1)
  state_piecewise(0)_pw_t09_CC_bin_y(2)
  state_piecewise(0)_pw_t10_CC_bin_y(1)
  state_piecewise(0)_pw_t10_CC_bin_y(2)
  state_piecewise(0)_pw_t11_CC_bin_y(1)
  state_piecewise(0)_pw_t11_CC_bin_y(2)
  state_piecewise(0)_pw_t12_CC_bin_y(1)
  state_piecewise(0)_pw_t12_CC_bin_y(2)
  state_piecewise(0)_pw_t13_CC_bin_y(1)
  state_piecewise(0)_pw_t13_CC_bin_y(2)
  state_piecewise(0)_pw_t14_CC_bin_y(1)
  state_piecewise(0)_pw_t14_CC_bin_y(2)
  state_piecewise(0)_pw_t15_CC_bin_y(1)
  state_piecewise(0)_pw_t15_CC_bin_y(2)
  state_piecewise(0)_pw_t16_CC_bin_y(1)
  state_piecewise(0)_pw_t16_CC_bin_y(2)
  state_piecewise(0)_pw_t17_CC_bin_y(1)
  state_piecewise(0)_pw_t17_CC_bin_y(2)
  state_piecewise(0)_pw_t18_CC_bin_y(1)
  state_piecewise(0)_pw_t18_CC_bin_y(2)
  state_piecewise(0)_pw_t19_CC_bin_y(1)
  state_piecewise(0)_pw_t19_CC_bin_y(2)
  state_piecewise(0)_pw_t20_CC_bin_y(1)
  state_piecewise(0)_pw_t20_CC_bin_y(2)
  state_piecewise(0)_pw_t21_CC_bin_y(1)
  state_piecewise(0)_pw_t21_CC_bin_y(2)
  state_piecewise(0)_pw_t22_CC_bin_y(1)
  state_piecewise(0)_pw_t22_CC_bin_y(2)
  state_piecewise(0)_pw_t23_CC_bin_y(1)
  state_piecewise(0)_pw_t23_CC_bin_y(2)
  state_piecewise(0)_pw_t24_CC_bin_y(1)
  state_piecewise(0)_pw_t24_CC_bin_y(2)
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
  state_piecewise(1)_pw_t13_CC_bin_y(1)
  state_piecewise(1)_pw_t13_CC_bin_y(2)
  state_piecewise(1)_pw_t14_CC_bin_y(1)
  state_piecewise(1)_pw_t14_CC_bin_y(2)
  state_piecewise(1)_pw_t15_CC_bin_y(1)
  state_piecewise(1)_pw_t15_CC_bin_y(2)
  state_piecewise(1)_pw_t16_CC_bin_y(1)
  state_piecewise(1)_pw_t16_CC_bin_y(2)
  state_piecewise(1)_pw_t17_CC_bin_y(1)
  state_piecewise(1)_pw_t17_CC_bin_y(2)
  state_piecewise(1)_pw_t18_CC_bin_y(1)
  state_piecewise(1)_pw_t18_CC_bin_y(2)
  state_piecewise(1)_pw_t19_CC_bin_y(1)
  state_piecewise(1)_pw_t19_CC_bin_y(2)
  state_piecewise(1)_pw_t20_CC_bin_y(1)
  state_piecewise(1)_pw_t20_CC_bin_y(2)
  state_piecewise(1)_pw_t21_CC_bin_y(1)
  state_piecewise(1)_pw_t21_CC_bin_y(2)
  state_piecewise(1)_pw_t22_CC_bin_y(1)
  state_piecewise(1)_pw_t22_CC_bin_y(2)
  state_piecewise(1)_pw_t23_CC_bin_y(1)
  state_piecewise(1)_pw_t23_CC_bin_y(2)
  state_piecewise(1)_pw_t24_CC_bin_y(1)
  state_piecewise(1)_pw_t24_CC_bin_y(2)
  state_piecewise(2)_pw_t01_CC_bin_y(1)
  state_piecewise(2)_pw_t01_CC_bin_y(2)
  state_piecewise(2)_pw_t02_CC_bin_y(1)
  state_piecewise(2)_pw_t02_CC_bin_y(2)
  state_piecewise(2)_pw_t03_CC_bin_y(1)
  state_piecewise(2)_pw_t03_CC_bin_y(2)
  state_piecewise(2)_pw_t04_CC_bin_y(1)
  state_piecewise(2)_pw_t04_CC_bin_y(2)
  state_piecewise(2)_pw_t05_CC_bin_y(1)
  state_piecewise(2)_pw_t05_CC_bin_y(2)
  state_piecewise(2)_pw_t06_CC_bin_y(1)
  state_piecewise(2)_pw_t06_CC_bin_y(2)
  state_piecewise(2)_pw_t07_CC_bin_y(1)
  state_piecewise(2)_pw_t07_CC_bin_y(2)
  state_piecewise(2)_pw_t08_CC_bin_y(1)
  state_piecewise(2)_pw_t08_CC_bin_y(2)
  state_piecewise(2)_pw_t09_CC_bin_y(1)
  state_piecewise(2)_pw_t09_CC_bin_y(2)
  state_piecewise(2)_pw_t10_CC_bin_y(1)
  state_piecewise(2)_pw_t10_CC_bin_y(2)
  state_piecewise(2)_pw_t11_CC_bin_y(1)
  state_piecewise(2)_pw_t11_CC_bin_y(2)
  state_piecewise(2)_pw_t12_CC_bin_y(1)
  state_piecewise(2)_pw_t12_CC_bin_y(2)
  state_piecewise(2)_pw_t13_CC_bin_y(1)
  state_piecewise(2)_pw_t13_CC_bin_y(2)
  state_piecewise(2)_pw_t14_CC_bin_y(1)
  state_piecewise(2)_pw_t14_CC_bin_y(2)
  state_piecewise(2)_pw_t15_CC_bin_y(1)
  state_piecewise(2)_pw_t15_CC_bin_y(2)
  state_piecewise(2)_pw_t16_CC_bin_y(1)
  state_piecewise(2)_pw_t16_CC_bin_y(2)
  state_piecewise(2)_pw_t17_CC_bin_y(1)
  state_piecewise(2)_pw_t17_CC_bin_y(2)
  state_piecewise(2)_pw_t18_CC_bin_y(1)
  state_piecewise(2)_pw_t18_CC_bin_y(2)
  state_piecewise(2)_pw_t19_CC_bin_y(1)
  state_piecewise(2)_pw_t19_CC_bin_y(2)
  state_piecewise(2)_pw_t20_CC_bin_y(1)
  state_piecewise(2)_pw_t20_CC_bin_y(2)
  state_piecewise(2)_pw_t21_CC_bin_y(1)
  state_piecewise(2)_pw_t21_CC_bin_y(2)
  state_piecewise(2)_pw_t22_CC_bin_y(1)
  state_piecewise(2)_pw_t22_CC_bin_y(2)
  state_piecewise(2)_pw_t23_CC_bin_y(1)
  state_piecewise(2)_pw_t23_CC_bin_y(2)
  state_piecewise(2)_pw_t24_CC_bin_y(1)
  state_piecewise(2)_pw_t24_CC_bin_y(2)
  state_piecewise(3)_pw_t01_CC_bin_y(1)
  state_piecewise(3)_pw_t01_CC_bin_y(2)
  state_piecewise(3)_pw_t02_CC_bin_y(1)
  state_piecewise(3)_pw_t02_CC_bin_y(2)
  state_piecewise(3)_pw_t03_CC_bin_y(1)
  state_piecewise(3)_pw_t03_CC_bin_y(2)
  state_piecewise(3)_pw_t04_CC_bin_y(1)
  state_piecewise(3)_pw_t04_CC_bin_y(2)
  state_piecewise(3)_pw_t05_CC_bin_y(1)
  state_piecewise(3)_pw_t05_CC_bin_y(2)
  state_piecewise(3)_pw_t06_CC_bin_y(1)
  state_piecewise(3)_pw_t06_CC_bin_y(2)
  state_piecewise(3)_pw_t07_CC_bin_y(1)
  state_piecewise(3)_pw_t07_CC_bin_y(2)
  state_piecewise(3)_pw_t08_CC_bin_y(1)
  state_piecewise(3)_pw_t08_CC_bin_y(2)
  state_piecewise(3)_pw_t09_CC_bin_y(1)
  state_piecewise(3)_pw_t09_CC_bin_y(2)
  state_piecewise(3)_pw_t10_CC_bin_y(1)
  state_piecewise(3)_pw_t10_CC_bin_y(2)
  state_piecewise(3)_pw_t11_CC_bin_y(1)
  state_piecewise(3)_pw_t11_CC_bin_y(2)
  state_piecewise(3)_pw_t12_CC_bin_y(1)
  state_piecewise(3)_pw_t12_CC_bin_y(2)
  state_piecewise(3)_pw_t13_CC_bin_y(1)
  state_piecewise(3)_pw_t13_CC_bin_y(2)
  state_piecewise(3)_pw_t14_CC_bin_y(1)
  state_piecewise(3)_pw_t14_CC_bin_y(2)
  state_piecewise(3)_pw_t15_CC_bin_y(1)
  state_piecewise(3)_pw_t15_CC_bin_y(2)
  state_piecewise(3)_pw_t16_CC_bin_y(1)
  state_piecewise(3)_pw_t16_CC_bin_y(2)
  state_piecewise(3)_pw_t17_CC_bin_y(1)
  state_piecewise(3)_pw_t17_CC_bin_y(2)
  state_piecewise(3)_pw_t18_CC_bin_y(1)
  state_piecewise(3)_pw_t18_CC_bin_y(2)
  state_piecewise(3)_pw_t19_CC_bin_y(1)
  state_piecewise(3)_pw_t19_CC_bin_y(2)
  state_piecewise(3)_pw_t20_CC_bin_y(1)
  state_piecewise(3)_pw_t20_CC_bin_y(2)
  state_piecewise(3)_pw_t21_CC_bin_y(1)
  state_piecewise(3)_pw_t21_CC_bin_y(2)
  state_piecewise(3)_pw_t22_CC_bin_y(1)
  state_piecewise(3)_pw_t22_CC_bin_y(2)
  state_piecewise(3)_pw_t23_CC_bin_y(1)
  state_piecewise(3)_pw_t23_CC_bin_y(2)
  state_piecewise(3)_pw_t24_CC_bin_y(1)
  state_piecewise(3)_pw_t24_CC_bin_y(2)
  state_piecewise(4)_pw_t01_CC_bin_y(1)
  state_piecewise(4)_pw_t01_CC_bin_y(2)
  state_piecewise(4)_pw_t02_CC_bin_y(1)
  state_piecewise(4)_pw_t02_CC_bin_y(2)
  state_piecewise(4)_pw_t03_CC_bin_y(1)
  state_piecewise(4)_pw_t03_CC_bin_y(2)
  state_piecewise(4)_pw_t04_CC_bin_y(1)
  state_piecewise(4)_pw_t04_CC_bin_y(2)
  state_piecewise(4)_pw_t05_CC_bin_y(1)
  state_piecewise(4)_pw_t05_CC_bin_y(2)
  state_piecewise(4)_pw_t06_CC_bin_y(1)
  state_piecewise(4)_pw_t06_CC_bin_y(2)
  state_piecewise(4)_pw_t07_CC_bin_y(1)
  state_piecewise(4)_pw_t07_CC_bin_y(2)
  state_piecewise(4)_pw_t08_CC_bin_y(1)
  state_piecewise(4)_pw_t08_CC_bin_y(2)
  state_piecewise(4)_pw_t09_CC_bin_y(1)
  state_piecewise(4)_pw_t09_CC_bin_y(2)
  state_piecewise(4)_pw_t10_CC_bin_y(1)
  state_piecewise(4)_pw_t10_CC_bin_y(2)
  state_piecewise(4)_pw_t11_CC_bin_y(1)
  state_piecewise(4)_pw_t11_CC_bin_y(2)
  state_piecewise(4)_pw_t12_CC_bin_y(1)
  state_piecewise(4)_pw_t12_CC_bin_y(2)
  state_piecewise(4)_pw_t13_CC_bin_y(1)
  state_piecewise(4)_pw_t13_CC_bin_y(2)
  state_piecewise(4)_pw_t14_CC_bin_y(1)
  state_piecewise(4)_pw_t14_CC_bin_y(2)
  state_piecewise(4)_pw_t15_CC_bin_y(1)
  state_piecewise(4)_pw_t15_CC_bin_y(2)
  state_piecewise(4)_pw_t16_CC_bin_y(1)
  state_piecewise(4)_pw_t16_CC_bin_y(2)
  state_piecewise(4)_pw_t17_CC_bin_y(1)
  state_piecewise(4)_pw_t17_CC_bin_y(2)
  state_piecewise(4)_pw_t18_CC_bin_y(1)
  state_piecewise(4)_pw_t18_CC_bin_y(2)
  state_piecewise(4)_pw_t19_CC_bin_y(1)
  state_piecewise(4)_pw_t19_CC_bin_y(2)
  state_piecewise(4)_pw_t20_CC_bin_y(1)
  state_piecewise(4)_pw_t20_CC_bin_y(2)
  state_piecewise(4)_pw_t21_CC_bin_y(1)
  state_piecewise(4)_pw_t21_CC_bin_y(2)
  state_piecewise(4)_pw_t22_CC_bin_y(1)
  state_piecewise(4)_pw_t22_CC_bin_y(2)
  state_piecewise(4)_pw_t23_CC_bin_y(1)
  state_piecewise(4)_pw_t23_CC_bin_y(2)
  state_piecewise(4)_pw_t24_CC_bin_y(1)
  state_piecewise(4)_pw_t24_CC_bin_y(2)
  state_piecewise(5)_pw_t01_CC_bin_y(1)
  state_piecewise(5)_pw_t01_CC_bin_y(2)
  state_piecewise(5)_pw_t02_CC_bin_y(1)
  state_piecewise(5)_pw_t02_CC_bin_y(2)
  state_piecewise(5)_pw_t03_CC_bin_y(1)
  state_piecewise(5)_pw_t03_CC_bin_y(2)
  state_piecewise(5)_pw_t04_CC_bin_y(1)
  state_piecewise(5)_pw_t04_CC_bin_y(2)
  state_piecewise(5)_pw_t05_CC_bin_y(1)
  state_piecewise(5)_pw_t05_CC_bin_y(2)
  state_piecewise(5)_pw_t06_CC_bin_y(1)
  state_piecewise(5)_pw_t06_CC_bin_y(2)
  state_piecewise(5)_pw_t07_CC_bin_y(1)
  state_piecewise(5)_pw_t07_CC_bin_y(2)
  state_piecewise(5)_pw_t08_CC_bin_y(1)
  state_piecewise(5)_pw_t08_CC_bin_y(2)
  state_piecewise(5)_pw_t09_CC_bin_y(1)
  state_piecewise(5)_pw_t09_CC_bin_y(2)
  state_piecewise(5)_pw_t10_CC_bin_y(1)
  state_piecewise(5)_pw_t10_CC_bin_y(2)
  state_piecewise(5)_pw_t11_CC_bin_y(1)
  state_piecewise(5)_pw_t11_CC_bin_y(2)
  state_piecewise(5)_pw_t12_CC_bin_y(1)
  state_piecewise(5)_pw_t12_CC_bin_y(2)
  state_piecewise(5)_pw_t13_CC_bin_y(1)
  state_piecewise(5)_pw_t13_CC_bin_y(2)
  state_piecewise(5)_pw_t14_CC_bin_y(1)
  state_piecewise(5)_pw_t14_CC_bin_y(2)
  state_piecewise(5)_pw_t15_CC_bin_y(1)
  state_piecewise(5)_pw_t15_CC_bin_y(2)
  state_piecewise(5)_pw_t16_CC_bin_y(1)
  state_piecewise(5)_pw_t16_CC_bin_y(2)
  state_piecewise(5)_pw_t17_CC_bin_y(1)
  state_piecewise(5)_pw_t17_CC_bin_y(2)
  state_piecewise(5)_pw_t18_CC_bin_y(1)
  state_piecewise(5)_pw_t18_CC_bin_y(2)
  state_piecewise(5)_pw_t19_CC_bin_y(1)
  state_piecewise(5)_pw_t19_CC_bin_y(2)
  state_piecewise(5)_pw_t20_CC_bin_y(1)
  state_piecewise(5)_pw_t20_CC_bin_y(2)
  state_piecewise(5)_pw_t21_CC_bin_y(1)
  state_piecewise(5)_pw_t21_CC_bin_y(2)
  state_piecewise(5)_pw_t22_CC_bin_y(1)
  state_piecewise(5)_pw_t22_CC_bin_y(2)
  state_piecewise(5)_pw_t23_CC_bin_y(1)
  state_piecewise(5)_pw_t23_CC_bin_y(2)
  state_piecewise(5)_pw_t24_CC_bin_y(1)
  state_piecewise(5)_pw_t24_CC_bin_y(2)
  state_piecewise(6)_pw_t01_CC_bin_y(1)
  state_piecewise(6)_pw_t01_CC_bin_y(2)
  state_piecewise(6)_pw_t02_CC_bin_y(1)
  state_piecewise(6)_pw_t02_CC_bin_y(2)
  state_piecewise(6)_pw_t03_CC_bin_y(1)
  state_piecewise(6)_pw_t03_CC_bin_y(2)
  state_piecewise(6)_pw_t04_CC_bin_y(1)
  state_piecewise(6)_pw_t04_CC_bin_y(2)
  state_piecewise(6)_pw_t05_CC_bin_y(1)
  state_piecewise(6)_pw_t05_CC_bin_y(2)
  state_piecewise(6)_pw_t06_CC_bin_y(1)
  state_piecewise(6)_pw_t06_CC_bin_y(2)
  state_piecewise(6)_pw_t07_CC_bin_y(1)
  state_piecewise(6)_pw_t07_CC_bin_y(2)
  state_piecewise(6)_pw_t08_CC_bin_y(1)
  state_piecewise(6)_pw_t08_CC_bin_y(2)
  state_piecewise(6)_pw_t09_CC_bin_y(1)
  state_piecewise(6)_pw_t09_CC_bin_y(2)
  state_piecewise(6)_pw_t10_CC_bin_y(1)
  state_piecewise(6)_pw_t10_CC_bin_y(2)
  state_piecewise(6)_pw_t11_CC_bin_y(1)
  state_piecewise(6)_pw_t11_CC_bin_y(2)
  state_piecewise(6)_pw_t12_CC_bin_y(1)
  state_piecewise(6)_pw_t12_CC_bin_y(2)
  state_piecewise(6)_pw_t13_CC_bin_y(1)
  state_piecewise(6)_pw_t13_CC_bin_y(2)
  state_piecewise(6)_pw_t14_CC_bin_y(1)
  state_piecewise(6)_pw_t14_CC_bin_y(2)
  state_piecewise(6)_pw_t15_CC_bin_y(1)
  state_piecewise(6)_pw_t15_CC_bin_y(2)
  state_piecewise(6)_pw_t16_CC_bin_y(1)
  state_piecewise(6)_pw_t16_CC_bin_y(2)
  state_piecewise(6)_pw_t17_CC_bin_y(1)
  state_piecewise(6)_pw_t17_CC_bin_y(2)
  state_piecewise(6)_pw_t18_CC_bin_y(1)
  state_piecewise(6)_pw_t18_CC_bin_y(2)
  state_piecewise(6)_pw_t19_CC_bin_y(1)
  state_piecewise(6)_pw_t19_CC_bin_y(2)
  state_piecewise(6)_pw_t20_CC_bin_y(1)
  state_piecewise(6)_pw_t20_CC_bin_y(2)
  state_piecewise(6)_pw_t21_CC_bin_y(1)
  state_piecewise(6)_pw_t21_CC_bin_y(2)
  state_piecewise(6)_pw_t22_CC_bin_y(1)
  state_piecewise(6)_pw_t22_CC_bin_y(2)
  state_piecewise(6)_pw_t23_CC_bin_y(1)
  state_piecewise(6)_pw_t23_CC_bin_y(2)
  state_piecewise(6)_pw_t24_CC_bin_y(1)
  state_piecewise(6)_pw_t24_CC_bin_y(2)
  gate_piecewise(gate1_t01)_CC_bin_y(1)
  gate_piecewise(gate1_t01)_CC_bin_y(2)
  gate_piecewise(gate1_t01)_CC_bin_y(3)
  gate_piecewise(gate1_t01)_CC_bin_y(4)
  gate_piecewise(gate1_t02)_CC_bin_y(1)
  gate_piecewise(gate1_t02)_CC_bin_y(2)
  gate_piecewise(gate1_t02)_CC_bin_y(3)
  gate_piecewise(gate1_t02)_CC_bin_y(4)
  gate_piecewise(gate1_t03)_CC_bin_y(1)
  gate_piecewise(gate1_t03)_CC_bin_y(2)
  gate_piecewise(gate1_t03)_CC_bin_y(3)
  gate_piecewise(gate1_t03)_CC_bin_y(4)
  gate_piecewise(gate1_t04)_CC_bin_y(1)
  gate_piecewise(gate1_t04)_CC_bin_y(2)
  gate_piecewise(gate1_t04)_CC_bin_y(3)
  gate_piecewise(gate1_t04)_CC_bin_y(4)
  gate_piecewise(gate1_t05)_CC_bin_y(1)
  gate_piecewise(gate1_t05)_CC_bin_y(2)
  gate_piecewise(gate1_t05)_CC_bin_y(3)
  gate_piecewise(gate1_t05)_CC_bin_y(4)
  gate_piecewise(gate1_t06)_CC_bin_y(1)
  gate_piecewise(gate1_t06)_CC_bin_y(2)
  gate_piecewise(gate1_t06)_CC_bin_y(3)
  gate_piecewise(gate1_t06)_CC_bin_y(4)
  gate_piecewise(gate1_t07)_CC_bin_y(1)
  gate_piecewise(gate1_t07)_CC_bin_y(2)
  gate_piecewise(gate1_t07)_CC_bin_y(3)
  gate_piecewise(gate1_t07)_CC_bin_y(4)
  gate_piecewise(gate1_t08)_CC_bin_y(1)
  gate_piecewise(gate1_t08)_CC_bin_y(2)
  gate_piecewise(gate1_t08)_CC_bin_y(3)
  gate_piecewise(gate1_t08)_CC_bin_y(4)
  gate_piecewise(gate1_t09)_CC_bin_y(1)
  gate_piecewise(gate1_t09)_CC_bin_y(2)
  gate_piecewise(gate1_t09)_CC_bin_y(3)
  gate_piecewise(gate1_t09)_CC_bin_y(4)
  gate_piecewise(gate1_t10)_CC_bin_y(1)
  gate_piecewise(gate1_t10)_CC_bin_y(2)
  gate_piecewise(gate1_t10)_CC_bin_y(3)
  gate_piecewise(gate1_t10)_CC_bin_y(4)
  gate_piecewise(gate1_t11)_CC_bin_y(1)
  gate_piecewise(gate1_t11)_CC_bin_y(2)
  gate_piecewise(gate1_t11)_CC_bin_y(3)
  gate_piecewise(gate1_t11)_CC_bin_y(4)
  gate_piecewise(gate1_t12)_CC_bin_y(1)
  gate_piecewise(gate1_t12)_CC_bin_y(2)
  gate_piecewise(gate1_t12)_CC_bin_y(3)
  gate_piecewise(gate1_t12)_CC_bin_y(4)
  gate_piecewise(gate1_t13)_CC_bin_y(1)
  gate_piecewise(gate1_t13)_CC_bin_y(2)
  gate_piecewise(gate1_t13)_CC_bin_y(3)
  gate_piecewise(gate1_t13)_CC_bin_y(4)
  gate_piecewise(gate1_t14)_CC_bin_y(1)
  gate_piecewise(gate1_t14)_CC_bin_y(2)
  gate_piecewise(gate1_t14)_CC_bin_y(3)
  gate_piecewise(gate1_t14)_CC_bin_y(4)
  gate_piecewise(gate1_t15)_CC_bin_y(1)
  gate_piecewise(gate1_t15)_CC_bin_y(2)
  gate_piecewise(gate1_t15)_CC_bin_y(3)
  gate_piecewise(gate1_t15)_CC_bin_y(4)
  gate_piecewise(gate1_t16)_CC_bin_y(1)
  gate_piecewise(gate1_t16)_CC_bin_y(2)
  gate_piecewise(gate1_t16)_CC_bin_y(3)
  gate_piecewise(gate1_t16)_CC_bin_y(4)
  gate_piecewise(gate1_t17)_CC_bin_y(1)
  gate_piecewise(gate1_t17)_CC_bin_y(2)
  gate_piecewise(gate1_t17)_CC_bin_y(3)
  gate_piecewise(gate1_t17)_CC_bin_y(4)
  gate_piecewise(gate1_t18)_CC_bin_y(1)
  gate_piecewise(gate1_t18)_CC_bin_y(2)
  gate_piecewise(gate1_t18)_CC_bin_y(3)
  gate_piecewise(gate1_t18)_CC_bin_y(4)
  gate_piecewise(gate1_t19)_CC_bin_y(1)
  gate_piecewise(gate1_t19)_CC_bin_y(2)
  gate_piecewise(gate1_t19)_CC_bin_y(3)
  gate_piecewise(gate1_t19)_CC_bin_y(4)
  gate_piecewise(gate1_t20)_CC_bin_y(1)
  gate_piecewise(gate1_t20)_CC_bin_y(2)
  gate_piecewise(gate1_t20)_CC_bin_y(3)
  gate_piecewise(gate1_t20)_CC_bin_y(4)
  gate_piecewise(gate1_t21)_CC_bin_y(1)
  gate_piecewise(gate1_t21)_CC_bin_y(2)
  gate_piecewise(gate1_t21)_CC_bin_y(3)
  gate_piecewise(gate1_t21)_CC_bin_y(4)
  gate_piecewise(gate1_t22)_CC_bin_y(1)
  gate_piecewise(gate1_t22)_CC_bin_y(2)
  gate_piecewise(gate1_t22)_CC_bin_y(3)
  gate_piecewise(gate1_t22)_CC_bin_y(4)
  gate_piecewise(gate1_t23)_CC_bin_y(1)
  gate_piecewise(gate1_t23)_CC_bin_y(2)
  gate_piecewise(gate1_t23)_CC_bin_y(3)
  gate_piecewise(gate1_t23)_CC_bin_y(4)
  gate_piecewise(gate1_t24)_CC_bin_y(1)
  gate_piecewise(gate1_t24)_CC_bin_y(2)
  gate_piecewise(gate1_t24)_CC_bin_y(3)
  gate_piecewise(gate1_t24)_CC_bin_y(4)
  gate_piecewise(gate2_t01)_CC_bin_y(1)
  gate_piecewise(gate2_t01)_CC_bin_y(2)
  gate_piecewise(gate2_t01)_CC_bin_y(3)
  gate_piecewise(gate2_t02)_CC_bin_y(1)
  gate_piecewise(gate2_t02)_CC_bin_y(2)
  gate_piecewise(gate2_t02)_CC_bin_y(3)
  gate_piecewise(gate2_t03)_CC_bin_y(1)
  gate_piecewise(gate2_t03)_CC_bin_y(2)
  gate_piecewise(gate2_t03)_CC_bin_y(3)
  gate_piecewise(gate2_t04)_CC_bin_y(1)
  gate_piecewise(gate2_t04)_CC_bin_y(2)
  gate_piecewise(gate2_t04)_CC_bin_y(3)
  gate_piecewise(gate2_t05)_CC_bin_y(1)
  gate_piecewise(gate2_t05)_CC_bin_y(2)
  gate_piecewise(gate2_t05)_CC_bin_y(3)
  gate_piecewise(gate2_t06)_CC_bin_y(1)
  gate_piecewise(gate2_t06)_CC_bin_y(2)
  gate_piecewise(gate2_t06)_CC_bin_y(3)
  gate_piecewise(gate2_t07)_CC_bin_y(1)
  gate_piecewise(gate2_t07)_CC_bin_y(2)
  gate_piecewise(gate2_t07)_CC_bin_y(3)
  gate_piecewise(gate2_t08)_CC_bin_y(1)
  gate_piecewise(gate2_t08)_CC_bin_y(2)
  gate_piecewise(gate2_t08)_CC_bin_y(3)
  gate_piecewise(gate2_t09)_CC_bin_y(1)
  gate_piecewise(gate2_t09)_CC_bin_y(2)
  gate_piecewise(gate2_t09)_CC_bin_y(3)
  gate_piecewise(gate2_t10)_CC_bin_y(1)
  gate_piecewise(gate2_t10)_CC_bin_y(2)
  gate_piecewise(gate2_t10)_CC_bin_y(3)
  gate_piecewise(gate2_t11)_CC_bin_y(1)
  gate_piecewise(gate2_t11)_CC_bin_y(2)
  gate_piecewise(gate2_t11)_CC_bin_y(3)
  gate_piecewise(gate2_t12)_CC_bin_y(1)
  gate_piecewise(gate2_t12)_CC_bin_y(2)
  gate_piecewise(gate2_t12)_CC_bin_y(3)
  gate_piecewise(gate2_t13)_CC_bin_y(1)
  gate_piecewise(gate2_t13)_CC_bin_y(2)
  gate_piecewise(gate2_t13)_CC_bin_y(3)
  gate_piecewise(gate2_t14)_CC_bin_y(1)
  gate_piecewise(gate2_t14)_CC_bin_y(2)
  gate_piecewise(gate2_t14)_CC_bin_y(3)
  gate_piecewise(gate2_t15)_CC_bin_y(1)
  gate_piecewise(gate2_t15)_CC_bin_y(2)
  gate_piecewise(gate2_t15)_CC_bin_y(3)
  gate_piecewise(gate2_t16)_CC_bin_y(1)
  gate_piecewise(gate2_t16)_CC_bin_y(2)
  gate_piecewise(gate2_t16)_CC_bin_y(3)
  gate_piecewise(gate2_t17)_CC_bin_y(1)
  gate_piecewise(gate2_t17)_CC_bin_y(2)
  gate_piecewise(gate2_t17)_CC_bin_y(3)
  gate_piecewise(gate2_t18)_CC_bin_y(1)
  gate_piecewise(gate2_t18)_CC_bin_y(2)
  gate_piecewise(gate2_t18)_CC_bin_y(3)
  gate_piecewise(gate2_t19)_CC_bin_y(1)
  gate_piecewise(gate2_t19)_CC_bin_y(2)
  gate_piecewise(gate2_t19)_CC_bin_y(3)
  gate_piecewise(gate2_t20)_CC_bin_y(1)
  gate_piecewise(gate2_t20)_CC_bin_y(2)
  gate_piecewise(gate2_t20)_CC_bin_y(3)
  gate_piecewise(gate2_t21)_CC_bin_y(1)
  gate_piecewise(gate2_t21)_CC_bin_y(2)
  gate_piecewise(gate2_t21)_CC_bin_y(3)
  gate_piecewise(gate2_t22)_CC_bin_y(1)
  gate_piecewise(gate2_t22)_CC_bin_y(2)
  gate_piecewise(gate2_t22)_CC_bin_y(3)
  gate_piecewise(gate2_t23)_CC_bin_y(1)
  gate_piecewise(gate2_t23)_CC_bin_y(2)
  gate_piecewise(gate2_t23)_CC_bin_y(3)
  gate_piecewise(gate2_t24)_CC_bin_y(1)
  gate_piecewise(gate2_t24)_CC_bin_y(2)
  gate_piecewise(gate2_t24)_CC_bin_y(3)
end
