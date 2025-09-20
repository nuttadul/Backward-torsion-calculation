
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ---------- Math utilities ----------
def deg2rad(d): return np.deg2rad(d)
def rad2deg(r): return np.rad2deg(r)

def rot_x(theta_deg):
    t = deg2rad(theta_deg); c, s = np.cos(t), np.sin(t)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rot_y(theta_deg):
    t = deg2rad(theta_deg); c, s = np.cos(t), np.sin(t)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rot_z(theta_deg):
    t = deg2rad(theta_deg); c, s = np.cos(t), np.sin(t)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def normalize(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v/n

def skew(v):
    x,y,z = v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])

def rot_from_a_to_b(a, b):
    a = normalize(a); b = normalize(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.isclose(c, -1.0, atol=1e-8):
        axis = normalize(np.array([1,0,0]) if abs(a[0]) < 0.9 else np.array([0,1,0]))
        v = normalize(np.cross(a, axis))
        K = skew(v)
        return -np.eye(3) + 2 * np.outer(v, v)
    if np.allclose(v, 0):
        return np.eye(3)
    K = skew(v)
    s = np.linalg.norm(v)
    return np.eye(3) + K + K@K * ((1 - c) / (s**2 + 1e-12))

def yaw_from_R(R):
    return np.arctan2(R[1,0], R[0,0])

# ---------- Model ----------
def compose_body_fixed(alpha_yz, beta_zx, gamma_yx):
    return rot_y(beta_zx) @ rot_x(alpha_yz) @ rot_z(gamma_yx)

def compute_true_torsion(alpha_yz_deg, beta_zx_deg, gamma_yx_deg):
    R = compose_body_fixed(alpha_yz_deg, beta_zx_deg, gamma_yx_deg)
    d = R @ np.array([0., 0., -1.])
    S = rot_from_a_to_b(d, np.array([0., 0., -1.]))
    R_aligned = S @ R
    torsion_rad = yaw_from_R(R_aligned)
    torsion_deg = rad2deg(torsion_rad)
    if torsion_deg <= -180: torsion_deg += 360
    if torsion_deg > 180: torsion_deg -= 360
    return torsion_deg, R, R_aligned, d

# ---------- Plotly helpers ----------
def arrow_trace(start, vec, color, name, width=8):
    s = np.array(start); v = np.array(vec); tip = s + v
    line = go.Scatter3d(x=[s[0], tip[0]], y=[s[1], tip[1]], z=[s[2], tip[2]],
                        mode='lines', line=dict(color=color, width=width), name=name, showlegend=True)
    ln = np.linalg.norm(v); size = max(0.08, 0.15*ln)
    cone = go.Cone(x=[tip[0]], y=[tip[1]], z=[tip[2]],
                   u=[v[0]], v=[v[1]], w=[v[2]], sizemode='absolute', sizeref=size,
                   showscale=False, colorscale=[[0, color],[1, color]], anchor='tip', name=name)
    return [line, cone]

def axes_traces(limit=1.2):
    x = go.Scatter3d(x=[0,limit], y=[0,0], z=[0,0], mode='lines', line=dict(color='gray', width=4), showlegend=False)
    y = go.Scatter3d(x=[0,0], y=[0,limit], z=[0,0], mode='lines', line=dict(color='gray', width=4), showlegend=False)
    z = go.Scatter3d(x=[0,0], y=[0,0], z=[0,limit], mode='lines', line=dict(color='gray', width=4), showlegend=False)
    return [x,y,z]

def plane_surfaces(limit=1.2, opacity=0.07):
    rng = np.linspace(-limit, limit, 2)
    X, Y = np.meshgrid(rng, rng)
    s_xy = go.Surface(x=X, y=Y, z=np.zeros_like(X), opacity=opacity, showscale=False, name='XY (z=0)')
    s_zx = go.Surface(x=X, y=np.zeros_like(X), z=Y, opacity=opacity, showscale=False, name='ZX (y=0)')
    s_zy = go.Surface(x=np.zeros_like(X), y=X, z=Y, opacity=opacity, showscale=False, name='ZY (x=0)')
    return [s_xy, s_zx, s_zy]

# ---------- Streamlit UI ----------
st.set_page_config(page_title="True Torsion Calculator (De-angulation)", layout="wide")
st.title("True Torsion from Mixed Deformities")

c1, c2, c3 = st.columns(3)
with c1:
    alpha = st.slider("YZ angulation α (about X) [deg]", -90.0, 90.0, 0.0, 1.0)
with c2:
    beta  = st.slider("ZX angulation β (about Y) [deg]", -90.0, 90.0, 0.0, 1.0)
with c3:
    gamma = st.number_input("Input torsion γ (about Z) [deg]", value=0.0, step=1.0, format="%.2f")

torsion_deg, R, R_aligned, d = compute_true_torsion(alpha, beta, gamma)

st.metric("Computed TRUE torsion (deg) after de-angulation", f"{torsion_deg:.2f}")

limit=1.2
origin = np.zeros(3)
Z_prox = np.array([0,0,1])
Z_dist = d
A_prox = np.array([0,1,0])
A_dist = (R @ np.array([0,1,0]))

traces = []
traces += axes_traces(limit=limit)
traces += plane_surfaces(limit=limit, opacity=0.07)
traces += arrow_trace(origin, Z_prox, 'royalblue',  'Z_prox')
traces += arrow_trace(origin, Z_dist, 'darkorange', 'Z_dist')
traces += arrow_trace(origin, A_prox, 'seagreen',   'A_prox')
traces += arrow_trace(origin, A_dist, 'crimson',    'A_dist')

fig3d = go.Figure(data=traces)
fig3d.update_layout(scene=dict(
    xaxis=dict(title='X', range=[-limit,limit]),
    yaxis=dict(title='Y', range=[-limit,limit]),
    zaxis=dict(title='Z', range=[-limit,limit]),
    aspectmode='cube'
), margin=dict(l=0,r=0,t=30,b=0), legend=dict(orientation='h', y=0.95))

st.plotly_chart(fig3d, use_container_width=True)

st.caption("Proximal shaft = +Z, distal starts = -Z. Distal rotated by R_y(β)·R_x(α)·R_z(γ).\n"
           "We align distal shaft back to -Z (de-angulate) then read residual twist about Z as true torsion.")
