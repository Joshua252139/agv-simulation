import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Natural Driving AGV", layout="wide")

st.title(" AGV Simulation")


# --- Sidebar (UNCHANGED) ---
st.sidebar.header("1. Vehicle Specs")
mass_empty = st.sidebar.number_input("AGV Empty Weight (kg)", value=20000)
mass_load = st.sidebar.number_input("Container Load (kg)", value=30000)
m_total = mass_empty + mass_load

A = st.sidebar.number_input("Frontal Area A (m²)", value=10.0)
Cd = st.sidebar.slider("Air Drag Coefficient (Cd)", 0.5, 1.2, 0.9)
Crr = st.sidebar.slider("Rolling Resistance Coeff (Crr)", 0.01, 0.03, 0.015)
delta = st.sidebar.slider("Rotational Mass Factor (δ)", 1.0, 1.2, 1.05)
rho = st.sidebar.number_input("Air Density ρ (kg/m³)", value=1.225)
g = 9.81
regen_eff = st.sidebar.slider("Regen. Braking Efficiency (%)", 0, 100, 50) / 100.0

st.sidebar.header("2. Route Configuration")
total_distance = st.sidebar.number_input("Total Route Distance (m)", min_value=200, max_value=10000, value=1000, step=50)
max_speed_kph = st.sidebar.slider("Max Straight Speed (km/h)", 10, 40, 25)
max_speed = max_speed_kph / 3.6
accel_rate = st.sidebar.slider("Acceleration Rate (m/s²)", 0.1, 1.0, 0.3)
decel_comfort = st.sidebar.slider("Braking Rate (m/s²)", 0.1, 1.0, 0.2, help="Lower value = Smoother, longer braking distance.")
slope_deg = st.sidebar.slider("Slope Gradient (°)", 0.0, 5.0, 0.0)

st.sidebar.subheader("3. Turn Settings")
num_turns = st.sidebar.number_input("Number of Turns", min_value=0, max_value=10, value=1)

turns_config = []
c_cornering_global = 0.05 

for i in range(num_turns):
    with st.sidebar.expander(f" Turn #{i+1} Setup", expanded=True):
        default_pos = (total_distance / (num_turns + 1)) * (i + 1)
        t_pos = st.number_input(f"Start Position #{i+1} (m)", value=int(default_pos), min_value=0, max_value=int(total_distance-50), key=f"p_{i}")
        t_rad = st.number_input(f"Radius #{i+1} (m)", value=30.0, min_value=5.0, key=f"r_{i}")
        t_angle = st.number_input(f"Turn Angle #{i+1} (deg)", value=90.0, min_value=1.0, max_value=180.0, step=45.0, key=f"deg_{i}")
        t_spd_pct = st.slider(f"Speed Limit #{i+1} (%)", 30, 90, 60, key=f"s_{i}") / 100.0
        
        target_v = max_speed * t_spd_pct
        arc_length = 2 * np.pi * t_rad * (t_angle / 360.0)
        calculated_duration = arc_length / target_v if target_v > 0 else 0
        
        turns_config.append({
            "start_dist": t_pos,
            "radius": t_rad,
            "duration": calculated_duration,
            "target_speed": target_v,
            "id": i + 1,
            "angle": t_angle
        })

turns_config.sort(key=lambda x: x["start_dist"])

# --- PHYSICS ENGINE (UNCHANGED) ---
def simulate_segment(start_v, end_v, dist_start, dist_end, dt=0.5):
    segment_data = []
    current_dist = dist_start
    total_dist = dist_end - dist_start
    if total_dist <= 0: return []
    v = start_v
    is_braking = False
    
    while current_dist < dist_end:
        d_rem = dist_end - current_dist
        if v > end_v:
            req_brake_dist = (v**2 - end_v**2) / (2 * decel_comfort)
        else:
            req_brake_dist = 0
            
        if (d_rem <= req_brake_dist) or is_braking:
            is_braking = True
            mode = "Brake"
            if d_rem > 0.1:
                a_needed = (end_v**2 - v**2) / (2 * d_rem)
            else:
                a_needed = 0; v = end_v
            a = a_needed
        else:
            if v < max_speed:
                mode = "Accel"
                a = accel_rate
                v_next_est = v + a*dt
                req_dist_next = (v_next_est**2 - end_v**2) / (2 * decel_comfort)
                if (d_rem - v*dt) < req_dist_next:
                    a = 0; mode = "Cruise"
            else:
                mode = "Cruise"; a = 0; v = max_speed
                
        v_new = v + a * dt
        if v_new < 0: v_new = 0
        if mode == "Brake" and end_v > 0 and v_new < end_v:
            v_new = end_v; a = 0
        
        d_step = v * dt + 0.5 * a * (dt**2)
        if d_step < 0: d_step = 0
        if current_dist + d_step > dist_end: d_step = dist_end - current_dist
        
        current_dist += d_step
        v = v_new
        segment_data.append({"v": v, "a": a, "dist": current_dist, "is_turn": False, "radius": 0, "phase": mode})
        if d_step < 0.001 and abs(v - end_v) < 0.1 and current_dist >= dist_end - 0.5: break
    return segment_data

def simulate_turn(start_v, duration, start_dist, radius, turn_id, dt=0.5):
    segment_data = []
    t = 0; v = start_v; current_dist = start_dist
    while t < duration:
        t += dt
        d_step = v * dt
        current_dist += d_step
        segment_data.append({"v": v, "a": 0, "dist": current_dist, "is_turn": True, "radius": radius, "phase": f"Turn #{turn_id}"})
    return segment_data, current_dist

# --- SIMULATION RUNNER (UNCHANGED) ---
full_simulation_data = []
curr_time = 0; curr_v = 0; curr_dist = 0

for turn in turns_config:
    target_start_dist = turn["start_dist"]
    target_turn_v = turn["target_speed"]
    straight_data = simulate_segment(curr_v, target_turn_v, curr_dist, target_start_dist)
    for step in straight_data:
        full_simulation_data.append({
            "Time (s)": curr_time, "Speed (m/s)": step["v"], "Accel": step["a"], 
            "Distance (m)": step["dist"], "Phase": step["phase"], "Is Turn": False, "Radius": 0
        })
        curr_time += 0.5; curr_v = step["v"]; curr_dist = step["dist"]
    turn_data, final_turn_dist = simulate_turn(curr_v, turn["duration"], curr_dist, turn["radius"], turn["id"])
    for step in turn_data:
        full_simulation_data.append({
            "Time (s)": curr_time, "Speed (m/s)": step["v"], "Accel": step["a"], 
            "Distance (m)": step["dist"], "Phase": step["phase"], "Is Turn": True, "Radius": step["radius"]
        })
        curr_time += 0.5; curr_v = step["v"]; curr_dist = step["dist"]

final_data = simulate_segment(curr_v, 0, curr_dist, total_distance)
for step in final_data:
    full_simulation_data.append({
        "Time (s)": curr_time, "Speed (m/s)": step["v"], "Accel": step["a"], 
        "Distance (m)": step["dist"], "Phase": step["phase"], "Is Turn": False, "Radius": 0
    })
    curr_time += 0.5

# --- PROCESS RESULTS ---
df = pd.DataFrame(full_simulation_data)
if df.empty: st.stop()

theta = np.radians(slope_deg)
results = []
for index, row in df.iterrows():
    v = row["Speed (m/s)"]
    a = row["Accel"]
    is_curving = row["Is Turn"]
    radius = row["Radius"]
    
    F_rf = Crr * m_total * g * np.cos(theta)
    F_air = 0.5 * rho * Cd * A * (v ** 2)
    F_grade = m_total * g * np.sin(theta)
    F_acc = m_total * a * delta
    F_curve = 0.0
    if is_curving and radius > 0: F_curve = (c_cornering_global * m_total * (v ** 2)) / radius
    F_total = F_rf + F_air + F_grade + F_acc + F_curve
    P_mech = (F_total * v) / 1000.0
    if P_mech >= 0: P_battery = P_mech; mode = "Consumption"
    else: P_battery = P_mech * regen_eff; mode = "Regeneration"
    results.append({**row, "Speed (km/h)": v * 3.6, "Net Battery Power (kW)": P_battery, "Mode": mode})

df_final = pd.DataFrame(results)

# --- VIZ ---
st.subheader(" Simulation Results")

# --- UPDATED METRICS SECTION (ADDED TIME & POWER) ---
net_energy = df_final["Net Battery Power (kW)"].sum() * 0.5 / 3600.0
total_time = df_final["Time (s)"].max()
avg_power = df_final["Net Battery Power (kW)"].mean()
peak_power = df_final["Net Battery Power (kW)"].max()

# Create 5 columns for comprehensive metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Time", f"{total_time:.1f} s")
c2.metric("Total Distance", f"{df_final['Distance (m)'].max():.1f} m")
c3.metric("Net Energy", f"{net_energy:.4f} kWh")
c4.metric("Avg Power", f"{avg_power:.2f} kW")
c5.metric("Peak Power", f"{peak_power:.1f} kW")

# Chart 1: Velocity
st.markdown("### 1. Velocity Profile")
fig_speed = px.line(df_final, x="Time (s)", y="Speed (km/h)", title="Speed Profile (Smooth Braking)")
fig_speed.update_traces(line_color="#1f77b4", line_width=3)
if num_turns > 0:
    for t_conf in turns_config:
        phase_data = df_final[df_final["Phase"] == f"Turn #{t_conf['id']}"]
        if not phase_data.empty:
            fig_speed.add_vrect(x0=phase_data["Time (s)"].min(), x1=phase_data["Time (s)"].max(), fillcolor="orange", opacity=0.2, annotation_text=f"Turn")
st.plotly_chart(fig_speed, use_container_width=True)

# Chart 2: Distance (KEPT)
st.markdown("### 2. Distance Traveled")
fig_dist = px.line(df_final, x="Time (s)", y="Distance (m)", title="Distance vs Time")
fig_dist.update_traces(line_color="#2ca02c", line_width=3)
st.plotly_chart(fig_dist, use_container_width=True)

# Chart 3: Power (KEPT)
st.markdown("### 3. Power Consumption")
fig_power = px.area(df_final, x="Time (s)", y="Net Battery Power (kW)", color="Mode", color_discrete_map={"Consumption": "#EF553B", "Regeneration": "#00CC96"})
st.plotly_chart(fig_power, use_container_width=True)

st.dataframe(df_final, use_container_width=True, height=200)
csv = df_final.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "agv_final_complete.csv", "text/csv")