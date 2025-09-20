
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ---------------------------
# Math helpers
# ---------------------------
def deg2rad(d): return np.deg2rad(d)
def rad2deg(r): return np.rad2deg(r)
def sgn(x): return 1.0 if x >= 0 else -1.0

def rot_x(a):
    t=deg2rad(a); c,s=np.cos(t),np.sin(t)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rot_y(b):
    t=deg2rad(b); c,s=np.cos(t),np.sin(t)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rot_z(g):
    t=deg2rad(g); c,s=np.cos(t),np.sin(t)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def normalize(v):
    n=np.linalg.norm(v); 
    return v if n==0 else v/n

# Apparent torsion formula (body-fixed order R = Ry(b) Rx(a) Rz(g))
# φ_app = atan2( sin g * cos b − cos g * sin a * sin b , cos g * cos a )
def apparent_from_true(a, b, g):
    ca,sa = np.cos(deg2rad(a)), np.sin(deg2rad(a))
    cb,sb = np.cos(deg2rad(b)), np.sin(deg2rad(b))
    cg,sg = np.cos(deg2rad(g)), np.sin(deg2rad(g))
    num = sg*cb - cg*sa*sb
    den = cg*ca
    return rad2deg(np.arctan2(num, den))

# Inversion for true torsion from apparent:
# tan φ = (sin g * cos b − cos g * sin a * sin b) / (cos g * cos a)
# => tan g = (tan φ * cos a + sin a * sin b) / cos b
def true_from_apparent_phi(phi_app, a, b):
    ca,sa = np.cos(deg2rad(a)), np.sin(deg2rad(a))
    cb,sb = np.cos(deg2rad(b)), np.sin(deg2rad(b))
    A = np.tan(deg2rad(phi_app))
    tg = (A*ca + sa*sb) / (cb + 1e-12)
    g = rad2deg(np.arctan(tg))
    # Normalize to (-180,180]
    if g <= -180: g += 360
    if g > 180: g -= 360
    return g

# Recover actual 3D angulations (about X and Y) from projected measurements
# AP (ZX): b = theta_ap
# Lateral (ZY): a = atan( tan(theta_lat) * cos b )
def recover_axial_rotations_from_projections(theta_lat, theta_ap):
    b = theta_ap  # coronal (about Y)
    a = rad2deg(np.arctan(np.tan(deg2rad(theta_lat)) * np.cos(deg2rad(b))))
    return a, b

# ---------------------------
# Plotly helpers
# ---------------------------
def arrow_trace(start, vec, color, name, width=8):
    s=np.array(start); v=np.array(vec); tip=s+v
    line=go.Scatter3d(x=[s[0],tip[0]], y=[s[1],tip[1]], z=[s[2],tip[2]],
                      mode='lines', line=dict(color=color, width=width), name=name, showlegend=True)
    ln=np.linalg.norm(v); size=max(0.08, 0.15*ln)
    cone=go.Cone(x=[tip[0]], y=[tip[1]], z=[tip[2]],
                 u=[v[0]], v=[v[1]], w=[v[2]], sizemode='absolute', sizeref=size,
                 showscale=False, colorscale=[[0, color],[1, color]], anchor='tip', name=name)
    return [line, cone]

def axes_traces(limit=1.2):
    x=go.Scatter3d(x=[0,limit], y=[0,0], z=[0,0], mode='lines',
                   line=dict(color='gray', width=4), showlegend=False)
    y=go.Scatter3d(x=[0,0], y=[0,limit], z=[0,0], mode='lines',
                   line=dict(color='gray', width=4), showlegend=False)
    z=go.Scatter3d(x=[0,0], y=[0,0], z=[0,limit], mode='lines',
                   line=dict(color='gray', width=4), showlegend=False)
    return [x,y,z]

def planes_through_origin(limit=1.2, opacity=0.07):
    rng=np.linspace(-limit, limit, 2)
    X,Y=np.meshgrid(rng, rng)
    s_xy=go.Surface(x=X,y=Y,z=np.zeros_like(X),opacity=opacity,showscale=False,name='XY (z=0)')
    s_zx=go.Surface(x=X,y=np.zeros_like(X),z=Y,opacity=opacity,showscale=False,name='ZX (y=0)')
    s_zy=go.Surface(x=np.zeros_like(X),y=X,z=Y,opacity=opacity,showscale=False,name='ZY (x=0)')
    return [s_xy,s_zx,s_zy]

def plane_xy_bottom(limit=1.2, opacity=0.10):
    rng=np.linspace(-limit, limit, 2)
    X,Y=np.meshgrid(rng, rng)
    return go.Surface(x=X,y=Y,z=np.full_like(X,-limit),opacity=opacity,showscale=False,name='XY bottom')

def projected_line_xy_bottom(start, vec, color='rgba(0,0,0,0.65)', width=6, limit=1.2):
    s=np.array(start); v=np.array(vec); tip=s+v
    eps=1e-3; z_plane=-limit+eps
    sP=np.array([s[0],s[1],z_plane]); tP=np.array([tip[0],tip[1],z_plane])
    return go.Scatter3d(x=[sP[0],tP[0]], y=[sP[1],tP[1]], z=[sP[2],tP[2]],
                        mode='lines+markers', line=dict(color=color, width=width),
                        marker=dict(size=3, color=color), name='proj', showlegend=False)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="True Torsion from Projected Inputs (XR + CT)", layout="wide")
st.title("True Torsion from Projected AP/Lateral Angulation and Apparent Axial Torsion")

st.markdown(
    "Enter **AP (coronal) angulation** and **lateral (sagittal) angulation** from radiographs (2D, projected), "
    "and **apparent torsion** from axial CT. The app de‑angulates and computes the **true torsion**."
)

c1,c2,c3 = st.columns(3)
with c1:
    theta_ap   = st.slider("AP (coronal) angulation – ZX projection [deg]", -90.0, 90.0, 0.0, 1.0)
with c2:
    theta_lat  = st.slider("Lateral (sagittal) angulation – ZY projection [deg]", -90.0, 90.0, 0.0, 1.0)
with c3:
    phi_app    = st.number_input("Apparent torsion on axial CT – XY projection [deg]",
                                 value=0.0, step=1.0, format="%.2f")

# Recover actual 3D rotations about X (sagittal) & Y (coronal) from projections
alpha_sag, beta_cor = recover_axial_rotations_from_projections(theta_lat, theta_ap)

# Compute true torsion from apparent torsion and recovered α, β
gamma_true = true_from_apparent_phi(phi_app, alpha_sag, beta_cor)

st.subheader("Results")
colA, colB, colC = st.columns(3)
colA.metric("Recovered sagittal angulation α (about X)", f"{alpha_sag:.2f}°")
colB.metric("Recovered coronal angulation β (about Y)", f"{beta_cor:.2f}°")
colC.metric("TRUE torsion after de‑angulation (about Z)", f"{gamma_true:.2f}°")

# Self-consistency check
phi_check = apparent_from_true(alpha_sag, beta_cor, gamma_true)
st.caption(f"Consistency check: predicted apparent torsion from (α,β,γ_true) = {phi_check:.2f}° "
           f"(should match input {phi_app:.2f}° within rounding).")

# ------------- Visualization -------------
R = rot_y(beta_cor) @ rot_x(alpha_sag) @ rot_z(gamma_true)

origin=np.zeros(3)
Z_prox=np.array([0,0,1])
Z_dist = normalize(R @ np.array([0,0,-1]))
A_prox=np.array([0,1,0])
A_dist = normalize(R @ np.array([0,1,0]))

limit=1.2
traces=[]
traces += axes_traces(limit=limit)
traces += planes_through_origin(limit=limit, opacity=0.07)
traces.append(plane_xy_bottom(limit=limit, opacity=0.10))

traces += arrow_trace(origin, Z_prox, 'royalblue',  'Z_prox')
traces += arrow_trace(origin, Z_dist, 'darkorange', 'Z_dist')
traces += arrow_trace(origin, A_prox, 'seagreen',   'A_prox')
traces += arrow_trace(origin, A_dist, 'crimson',    'A_dist')

traces += [projected_line_xy_bottom(origin,    Z_prox, color='rgba(65,105,225,0.6)', limit=limit)]
traces += [projected_line_xy_bottom(origin,    Z_dist, color='rgba(255,140,0,0.6)', limit=limit)]
traces += [projected_line_xy_bottom(origin,    A_prox, color='rgba(46,139,87,0.6)', limit=limit)]
traces += [projected_line_xy_bottom(origin,    A_dist, color='rgba(220,20,60,0.6)', limit=limit)]

fig3d=go.Figure(data=traces)
fig3d.update_layout(scene=dict(
    xaxis=dict(title='X', range=[-limit,limit]),
    yaxis=dict(title='Y', range=[-limit,limit]),
    zaxis=dict(title='Z', range=[-limit,limit]),
    aspectmode='cube'
), margin=dict(l=0,r=0,t=30,b=0))

st.plotly_chart(fig3d, use_container_width=True)

st.info("Assumes body‑fixed order R = R_y(β) R_x(α) R_z(γ). Signs of AP/lateral angles follow your XR conventions. "
        "If your measurement convention differs (e.g., varus vs valgus positive), we can add sign toggles.")

