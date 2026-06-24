"""
Well Control & Influx Analysis App
===================================
Calculates flow potential, blowout risk, kill parameters,
and influx volumes for a candidate well.

Equinor visual identity: Torch Red #EB0037 | Dark Navy #00243D |
                          Karry #FFE7D6    | Pistachio #9DBA00
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Well Control & Influx Analyser",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────────────────────
TORCH_RED   = "#EB0037"
DARK_NAVY   = "#00243D"
KARRY       = "#FFE7D6"
PISTACHIO   = "#9DBA00"
LIGHT_GREY  = "#F4F6F9"
MID_GREY    = "#B0B8C1"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {{
      font-family: 'Inter', sans-serif;
      background-color: {LIGHT_GREY};
  }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
      background-color: {DARK_NAVY} !important;
  }}
  section[data-testid="stSidebar"] * {{
      color: {KARRY} !important;
  }}
  section[data-testid="stSidebar"] .stSlider > div > div > div > div {{
      background-color: {TORCH_RED} !important;
  }}
  section[data-testid="stSidebar"] label {{
      color: {KARRY} !important;
      font-size: 0.80rem;
      font-weight: 600;
      letter-spacing: 0.04em;
      text-transform: uppercase;
  }}

  /* Header banner */
  .hero {{
      background: linear-gradient(135deg, {DARK_NAVY} 0%, #003a5c 100%);
      border-left: 6px solid {TORCH_RED};
      padding: 1.4rem 2rem;
      border-radius: 0 8px 8px 0;
      margin-bottom: 1.5rem;
  }}
  .hero h1 {{
      color: white; font-size: 1.8rem; font-weight: 700; margin: 0;
  }}
  .hero p {{
      color: {KARRY}; font-size: 0.9rem; margin: 0.3rem 0 0;
  }}

  /* KPI cards */
  .kpi-row {{ display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }}
  .kpi-card {{
      flex: 1; min-width: 150px;
      background: white;
      border-radius: 8px;
      padding: 1.1rem 1.4rem;
      border-top: 4px solid {DARK_NAVY};
      box-shadow: 0 2px 8px rgba(0,0,0,0.07);
  }}
  .kpi-card.red   {{ border-top-color: {TORCH_RED}; }}
  .kpi-card.green {{ border-top-color: {PISTACHIO}; }}
  .kpi-card.amber {{ border-top-color: #F5A623; }}
  .kpi-value {{ font-size: 1.55rem; font-weight: 700; color: {DARK_NAVY}; }}
  .kpi-label {{ font-size: 0.72rem; font-weight: 600; color: {MID_GREY};
                letter-spacing: 0.06em; text-transform: uppercase; margin-top: 2px; }}

  /* Section headers */
  .section-hdr {{
      border-bottom: 2px solid {TORCH_RED};
      padding-bottom: 0.3rem; margin-top: 1.6rem; margin-bottom: 1rem;
      color: {DARK_NAVY}; font-weight: 700; font-size: 1.05rem;
  }}

  /* Risk badge */
  .badge {{
      display: inline-block; padding: 4px 14px; border-radius: 20px;
      font-size: 0.78rem; font-weight: 700; letter-spacing: 0.07em;
      text-transform: uppercase;
  }}
  .badge-ok   {{ background: #e8f5e9; color: #2e7d32; }}
  .badge-warn {{ background: #fff3e0; color: #e65100; }}
  .badge-crit {{ background: #ffebee; color: #b71c1c; }}

  div[data-testid="stTabs"] button {{ color: {DARK_NAVY} !important; font-weight: 600; }}
  div[data-testid="stTabs"] button[aria-selected="true"] {{
      border-bottom: 3px solid {TORCH_RED} !important; color: {TORCH_RED} !important;
  }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# UNIT SYSTEM HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def unit_labels(si: bool) -> dict:
    if si:
        return dict(
            depth="m", pressure="bar", sg_label="S.G. (g/cm³)",
            perm="mD", thick="m", poros="%",
            vol="m³", rate="m³/day", grad="bar/m",
            mw_unit="S.G.", tvd_unit="m", pres_unit="bar",
        )
    else:
        return dict(
            depth="ft", pressure="psi", sg_label="S.G. (ppg equiv.)",
            perm="mD", thick="ft", poros="%",
            vol="bbl", rate="bbl/day", grad="psi/ft",
            mw_unit="ppg", tvd_unit="ft", pres_unit="psi",
        )

# Conversion helpers (SI ↔ field)
def sg_to_ppg(sg): return sg * 8.3454
def ppg_to_sg(ppg): return ppg / 8.3454
def m_to_ft(m): return m * 3.28084
def ft_to_m(ft): return ft / 3.28084
def bar_to_psi(b): return b * 14.5038
def psi_to_bar(p): return p / 14.5038
def m3_to_bbl(m3): return m3 * 6.28981
def bbl_to_m3(bbl): return bbl / 6.28981
def md_to_m2(md): return md * 9.869233e-16   # for Darcy calc (not displayed)

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def pore_pressure_bar(formation_sg: float, tvd_m: float) -> float:
    """Pore pressure in bar from S.G. and TVD (m)."""
    return formation_sg * 0.09806650 * tvd_m   # SG * g * h / 1e5  (water=1 → 0.0981 bar/m)

def hydrostatic_bar(mud_sg: float, tvd_m: float) -> float:
    return mud_sg * 0.09806650 * tvd_m

def overbalance_bar(mud_sg: float, formation_sg: float, tvd_m: float) -> float:
    return hydrostatic_bar(mud_sg, tvd_m) - pore_pressure_bar(formation_sg, tvd_m)

def kill_mud_sg(formation_sg: float, safety_margin_sg: float = 0.05) -> float:
    return formation_sg + safety_margin_sg

def darcy_flow_rate_m3_per_day(
    perm_md: float, thick_m: float,
    dp_bar: float,
    visc_cp: float, re_m: float, rw_m: float
) -> float:
    """
    Radial Darcy flow rate [m³/day]
    Q = (k·h·ΔP) / (μ · ln(re/rw)) × unit_factor
    k in mD, h in m, ΔP in bar, μ in cP
    """
    if re_m <= rw_m or visc_cp <= 0 or perm_md <= 0:
        return 0.0
    k_m2 = perm_md * 9.869233e-16
    dp_pa = dp_bar * 1e5
    ln_r  = math.log(re_m / rw_m)
    q_m3s = (k_m2 * thick_m * dp_pa) / (visc_cp * 1e-3 * ln_r)
    return q_m3s * 86400   # m³/day

def pv_influx_m3(
    poros_frac: float, thick_m: float, area_m2: float,
    influx_frac: float = 1.0
) -> float:
    """Static pore-volume influx."""
    return poros_frac * thick_m * area_m2 * influx_frac

def wellbore_volume_m3(sections: list) -> float:
    """Sum of cylindrical wellbore section volumes."""
    total = 0.0
    for (od_m, length_m) in sections:
        total += math.pi * (od_m / 2) ** 2 * length_m
    return total

def blowout_index(dp_bar: float, perm_md: float, thick_m: float) -> float:
    """Dimensionless severity index [0–1 normalised]."""
    raw = dp_bar * perm_md * thick_m
    return min(raw / (raw + 1000), 1.0)

def pit_gain_m3(influx_rate_m3day: float, reaction_time_hr: float) -> float:
    return influx_rate_m3day * reaction_time_hr / 24.0

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — INPUTS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='font-size:1.2rem;font-weight:700;color:white;margin-bottom:0.5rem;'>"
                f"⚙️ Well Parameters</div>", unsafe_allow_html=True)

    use_si = st.toggle("SI units  (off = Field units)", value=True)
    UL = unit_labels(use_si)

    st.markdown("---")

    # ── Well geometry ──
    st.markdown("**Well Geometry**")
    if use_si:
        tvd    = st.number_input("TVD to reservoir top (m)", 500.0, 8000.0, 2800.0, 50.0)
        wh_tvd = st.number_input("Wellhead / seabed depth (m)", 0.0, 3000.0, 0.0, 10.0)
        rw     = st.number_input("Wellbore radius rw (m)", 0.05, 0.50, 0.108, 0.005, format="%.3f")
        re     = st.number_input("Drainage radius re (m)", 100.0, 5000.0, 1000.0, 50.0)
    else:
        tvd_ft    = st.number_input("TVD to reservoir top (ft)", 1640.0, 26250.0, 9186.0, 164.0)
        tvd       = ft_to_m(tvd_ft)
        wh_tvd_ft = st.number_input("Wellhead / seabed depth (ft)", 0.0, 9843.0, 0.0, 33.0)
        wh_tvd    = ft_to_m(wh_tvd_ft)
        rw_in     = st.number_input("Wellbore radius rw (in)", 2.0, 20.0, 4.25, 0.25)
        rw        = rw_in * 0.0254
        re_ft     = st.number_input("Drainage radius re (ft)", 328.0, 16400.0, 3281.0, 164.0)
        re        = ft_to_m(re_ft)

    st.markdown("---")

    # ── Fluid / pressure ──
    st.markdown("**Pressure & Mud**")
    form_sg  = st.slider("Formation pressure S.G.", 0.90, 2.50, 1.50, 0.01)
    mud_sg   = st.slider("Current mud weight S.G.", 1.00, 2.50, 1.55, 0.01)
    visc_cp  = st.number_input("Fluid viscosity (cP)", 0.1, 100.0, 1.0, 0.1)
    safety_m = st.slider("Kill mud safety margin (S.G.)", 0.01, 0.20, 0.05, 0.01)

    st.markdown("---")

    # ── Reservoir properties ──
    st.markdown("**Reservoir Properties**")
    if use_si:
        thick  = st.number_input("Net pay thickness (m)", 1.0, 500.0, 25.0, 1.0)
        extent = st.number_input("Reservoir extent / length (m)", 50.0, 10000.0, 1000.0, 50.0)
    else:
        thick_ft  = st.number_input("Net pay thickness (ft)", 3.0, 1640.0, 82.0, 3.0)
        thick     = ft_to_m(thick_ft)
        extent_ft = st.number_input("Reservoir extent / length (ft)", 164.0, 32808.0, 3281.0, 164.0)
        extent    = ft_to_m(extent_ft)

    poros_pct = st.slider("Porosity (%)", 1.0, 40.0, 20.0, 0.5)
    perm_md   = st.number_input("Permeability (mD)", 0.01, 10000.0, 100.0, 1.0)
    influx_fr = st.slider("Influx fraction (% of PV)", 1.0, 100.0, 30.0, 1.0) / 100.0

    st.markdown("---")

    # ── Operational ──
    st.markdown("**Operational**")
    react_hr = st.slider("Kick detection time (hr)", 0.1, 6.0, 0.5, 0.1)

# ─────────────────────────────────────────────────────────────────────────────
# DERIVED CALCULATIONS (all in SI internally)
# ─────────────────────────────────────────────────────────────────────────────
poros_frac  = poros_pct / 100.0
area_m2     = math.pi * extent ** 2   # circular drainage area

pore_pres   = pore_pressure_bar(form_sg, tvd)
hydro_pres  = hydrostatic_bar(mud_sg, tvd)
overbal     = overbalance_bar(mud_sg, form_sg, tvd)
kill_sg     = kill_mud_sg(form_sg, safety_m)
kill_pres   = pore_pressure_bar(kill_sg, tvd)

# Flow rate (Darcy radial)
dp_bar      = max(pore_pres - hydro_pres, 0.0)   # underbalance → flow
q_m3day     = darcy_flow_rate_m3_per_day(perm_md, thick, dp_bar, visc_cp, re, rw)

# Pore-volume influx
pv_m3       = pv_influx_m3(poros_frac, thick, area_m2, influx_fr)

# Estimated pit gain
pit_m3      = pit_gain_m3(q_m3day, react_hr)

# Blowout risk index
bo_idx      = blowout_index(dp_bar, perm_md, thick)

# Wellbore volume estimate (simplified 3-section)
wbs_m3      = wellbore_volume_m3([(rw*2, tvd - wh_tvd)])

# Risk level
if overbal < 0:
    risk_level, badge_cls = "CRITICAL — Underbalanced", "badge-crit"
elif overbal < hydro_pres * 0.02:
    risk_level, badge_cls = "WARNING — Near-balance", "badge-warn"
else:
    risk_level, badge_cls = "OK — Overbalanced", "badge-ok"

# Unit display helpers
def disp_pres(bar_val):
    return f"{bar_to_psi(bar_val):.1f} psi" if not use_si else f"{bar_val:.1f} bar"

def disp_sg(sg_val):
    return f"{sg_to_ppg(sg_val):.2f} ppg" if not use_si else f"{sg_val:.3f} S.G."

def disp_vol(m3_val):
    return f"{m3_to_bbl(m3_val):.1f} bbl" if not use_si else f"{m3_val:.1f} m³"

def disp_rate(m3day):
    return f"{m3_to_bbl(m3day):.0f} bbl/day" if not use_si else f"{m3day:.1f} m³/day"

def disp_depth(m_val):
    return f"{m_to_ft(m_val):.0f} ft" if not use_si else f"{m_val:.0f} m"

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h1>🛢️ Well Control & Influx Analyser</h1>
  <p>Flow potential · Blowout risk · Kill parameters · Sensitivity analysis</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5, col6 = st.columns(6)
kpis = [
    (col1, "Pore Pressure",    disp_pres(pore_pres),  "red"),
    (col2, "Hydrostatic",      disp_pres(hydro_pres), ""),
    (col3, "Overbalance",      disp_pres(overbal),    "green" if overbal>=0 else "red"),
    (col4, "Kill Mud Weight",  disp_sg(kill_sg),      "amber"),
    (col5, "Influx Rate",      disp_rate(q_m3day),    "red" if q_m3day>0 else ""),
    (col6, "Pit Gain",         disp_vol(pit_m3),      "red" if pit_m3>0 else ""),
]
for col, label, val, cls in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-card {cls}">
          <div class="kpi-value">{val}</div>
          <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

# Risk badge
st.markdown(
    f"<br>Well status: <span class='badge {badge_cls}'>{risk_level}</span> &nbsp;"
    f"Blowout severity index: <b>{bo_idx:.3f}</b> / 1.000",
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Pressure Analysis",
    "🌊 Flow & Influx Volumes",
    "🔫 Kill Parameters",
    "🏗️ Well Sketch",
    "🔬 Sensitivity Analysis",
    "📋 Summary Table",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRESSURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("<div class='section-hdr'>Pressure Profile vs Depth</div>", unsafe_allow_html=True)

    depths_m = np.linspace(0, tvd * 1.05, 300)

    pp    = [pore_pressure_bar(form_sg, d)  for d in depths_m]
    hp    = [hydrostatic_bar(mud_sg, d)     for d in depths_m]
    kp    = [pore_pressure_bar(kill_sg, d)  for d in depths_m]

    if not use_si:
        x_pp = [bar_to_psi(v) for v in pp]
        x_hp = [bar_to_psi(v) for v in hp]
        x_kp = [bar_to_psi(v) for v in kp]
        y_d  = [m_to_ft(d) for d in depths_m]
        x_lbl = "Pressure (psi)"
        y_lbl = "Depth (ft)"
    else:
        x_pp, x_hp, x_kp = pp, hp, kp
        y_d  = list(depths_m)
        x_lbl = "Pressure (bar)"
        y_lbl = "Depth (m)"

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=x_pp, y=y_d, name="Pore Pressure",
                              line=dict(color=TORCH_RED, width=2.5, dash="solid")))
    fig1.add_trace(go.Scatter(x=x_hp, y=y_d, name="Hydrostatic (mud)",
                              line=dict(color=DARK_NAVY, width=2.5)))
    fig1.add_trace(go.Scatter(x=x_kp, y=y_d, name="Kill mud hydrostatic",
                              line=dict(color=PISTACHIO, width=2, dash="dash")))

    # Reservoir marker
    res_pp_disp = bar_to_psi(pore_pres) if not use_si else pore_pres
    res_d_disp  = m_to_ft(tvd) if not use_si else tvd
    fig1.add_hline(y=res_d_disp, line_dash="dot", line_color=MID_GREY, opacity=0.6,
                   annotation_text="Reservoir top", annotation_position="right")

    # Shaded overbalance / underbalance
    if overbal >= 0:
        fig1.add_trace(go.Scatter(
            x=x_hp + x_pp[::-1], y=y_d + y_d[::-1],
            fill='toself', fillcolor='rgba(157,186,0,0.10)',
            line=dict(width=0), name="Overbalance zone", showlegend=True
        ))
    else:
        fig1.add_trace(go.Scatter(
            x=x_pp + x_hp[::-1], y=y_d + y_d[::-1],
            fill='toself', fillcolor='rgba(235,0,55,0.10)',
            line=dict(width=0), name="Underbalance zone", showlegend=True
        ))

    fig1.update_layout(
        yaxis=dict(autorange="reversed", title=y_lbl, gridcolor="#e0e0e0"),
        xaxis=dict(title=x_lbl, gridcolor="#e0e0e0"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.05),
        height=500, margin=dict(t=30)
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Pressure gradient table
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='section-hdr'>Gradient Summary</div>", unsafe_allow_html=True)
        g_form = form_sg * 0.09806650   # bar/m
        g_mud  = mud_sg  * 0.09806650
        g_kill = kill_sg * 0.09806650
        if not use_si:
            g_form = form_sg * 0.4335
            g_mud  = mud_sg  * 0.4335
            g_kill = kill_sg * 0.4335
            g_unit = "psi/ft"
        else:
            g_unit = "bar/m"
        grad_df = pd.DataFrame({
            "Parameter": ["Formation pressure gradient", "Mud weight gradient", "Kill mud gradient"],
            f"Gradient ({g_unit})": [f"{g_form:.4f}", f"{g_mud:.4f}", f"{g_kill:.4f}"],
            "S.G.": [f"{form_sg:.3f}", f"{mud_sg:.3f}", f"{kill_sg:.3f}"],
        })
        st.dataframe(grad_df, hide_index=True, use_container_width=True)

    with col_b:
        st.markdown("<div class='section-hdr'>Pressure at Reservoir Top</div>", unsafe_allow_html=True)
        pres_df = pd.DataFrame({
            "Pressure": ["Pore", "Hydrostatic (mud)", "Overbalance", "Kill mud hydrostatic"],
            "bar" if use_si else "psi": [
                f"{pore_pres:.2f}" if use_si else f"{bar_to_psi(pore_pres):.1f}",
                f"{hydro_pres:.2f}" if use_si else f"{bar_to_psi(hydro_pres):.1f}",
                f"{overbal:.2f}" if use_si else f"{bar_to_psi(overbal):.1f}",
                f"{kill_pres:.2f}" if use_si else f"{bar_to_psi(kill_pres):.1f}",
            ]
        })
        st.dataframe(pres_df, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FLOW & INFLUX VOLUMES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("<div class='section-hdr'>Flow Potential & Influx into Wellbore</div>",
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Darcy rate vs ΔP
        dp_range = np.linspace(0, max(abs(overbal) * 3, 10), 100)
        q_range  = [darcy_flow_rate_m3_per_day(perm_md, thick, dp, visc_cp, re, rw)
                    for dp in dp_range]
        if not use_si:
            x_dp = [bar_to_psi(dp) for dp in dp_range]
            y_q  = [m3_to_bbl(q)  for q  in q_range]
            x_l, y_l = "ΔP (psi)", "Flow rate (bbl/day)"
            mark_x = bar_to_psi(abs(overbal))
            mark_y = m3_to_bbl(q_m3day)
        else:
            x_dp, y_q = list(dp_range), q_range
            x_l, y_l = "ΔP (bar)", "Flow rate (m³/day)"
            mark_x = abs(overbal)
            mark_y = q_m3day

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x_dp, y=y_q, mode="lines",
                                  line=dict(color=DARK_NAVY, width=2.5),
                                  name="Darcy flow rate"))
        fig2.add_trace(go.Scatter(x=[mark_x], y=[mark_y], mode="markers",
                                  marker=dict(color=TORCH_RED, size=12, symbol="star"),
                                  name=f"Current ΔP → {disp_rate(q_m3day)}"))
        fig2.update_layout(
            xaxis_title=x_l, yaxis_title=y_l,
            plot_bgcolor="white", paper_bgcolor="white",
            height=360, margin=dict(t=20),
            legend=dict(orientation="h", y=1.05)
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Cumulative influx over time
        time_hr   = np.linspace(0, max(react_hr * 10, 2), 200)
        cum_m3    = q_m3day / 24.0 * time_hr   # m³
        cum_disp  = [m3_to_bbl(v) if not use_si else v for v in cum_m3]
        y_vol_lbl = "Cumulative influx (bbl)" if not use_si else "Cumulative influx (m³)"

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=list(time_hr), y=cum_disp, mode="lines",
                                  line=dict(color=TORCH_RED, width=2.5),
                                  name="Cumulative influx"))
        # Detection time marker
        det_vol = m3_to_bbl(pit_m3) if not use_si else pit_m3
        fig3.add_vline(x=react_hr, line_dash="dot", line_color=PISTACHIO,
                       annotation_text=f"Detection at {react_hr:.1f} hr", annotation_position="top right")
        fig3.add_hline(y=det_vol, line_dash="dot", line_color="#F5A623",
                       annotation_text=f"Pit gain = {disp_vol(pit_m3)}")
        # Wellbore capacity line
        wb_disp = m3_to_bbl(wbs_m3) if not use_si else wbs_m3
        fig3.add_hline(y=wb_disp, line_dash="dash", line_color=DARK_NAVY, opacity=0.4,
                       annotation_text="Wellbore capacity", annotation_position="right")

        fig3.update_layout(
            xaxis_title="Time (hr)", yaxis_title=y_vol_lbl,
            plot_bgcolor="white", paper_bgcolor="white",
            height=360, margin=dict(t=20),
            legend=dict(orientation="h", y=1.05)
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Volume summary
    st.markdown("<div class='section-hdr'>Volume Summary</div>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    vol_items = [
        (col_a, "Pore Volume (total)", pv_m3 / influx_fr),
        (col_b, f"Influx volume ({influx_fr*100:.0f}% PV)", pv_m3),
        (col_c, "Wellbore volume",  wbs_m3),
    ]
    for col, label, val in vol_items:
        with col:
            st.metric(label, disp_vol(val))

    influx_ratio = pv_m3 / wbs_m3 if wbs_m3 > 0 else 0
    st.info(
        f"**Influx-to-wellbore ratio: {influx_ratio:.1f}×** — "
        f"The static pore-volume influx is {influx_ratio:.1f} times the wellbore capacity. "
        f"({'Surface equipment will be required to handle volumes.' if influx_ratio > 1 else 'Within wellbore capacity.'})"
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — KILL PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("<div class='section-hdr'>Kill Weight Mud & BHP Analysis</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Kill mud weight vs safety margin
        margins = np.linspace(0.01, 0.30, 100)
        kill_sgs = [kill_mud_sg(form_sg, m) for m in margins]

        if not use_si:
            y_kill = [sg_to_ppg(k) for k in kill_sgs]
            y_lbl2 = "Kill mud weight (ppg)"
            cur_y  = sg_to_ppg(kill_sg)
            cur_m  = safety_m
        else:
            y_kill = kill_sgs
            y_lbl2 = "Kill mud weight (S.G.)"
            cur_y  = kill_sg
            cur_m  = safety_m

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=list(margins), y=y_kill,
                                  line=dict(color=DARK_NAVY, width=2.5)))
        fig4.add_trace(go.Scatter(x=[cur_m], y=[cur_y], mode="markers",
                                  marker=dict(color=TORCH_RED, size=12, symbol="star"),
                                  name=f"Selected: {disp_sg(kill_sg)}"))
        fig4.update_layout(
            xaxis_title="Safety margin (S.G.)",
            yaxis_title=y_lbl2,
            plot_bgcolor="white", paper_bgcolor="white",
            height=350, margin=dict(t=20),
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col2:
        # Blowout index vs permeability
        k_range   = np.logspace(-1, 4, 200)
        bi_range  = [blowout_index(dp_bar, k, thick) for k in k_range]

        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=list(k_range), y=bi_range,
            line=dict(color=TORCH_RED, width=2.5), name="BO severity index"
        ))
        fig5.add_hline(y=0.7, line_dash="dot", line_color="#F5A623",
                       annotation_text="High risk threshold")
        fig5.add_vline(x=perm_md, line_dash="dash", line_color=PISTACHIO,
                       annotation_text=f"k={perm_md:.0f} mD", annotation_position="top right")
        fig5.update_layout(
            xaxis_title="Permeability (mD)", xaxis_type="log",
            yaxis_title="Blowout severity index",
            plot_bgcolor="white", paper_bgcolor="white",
            height=350, margin=dict(t=20),
        )
        st.plotly_chart(fig5, use_container_width=True)

    # Kill procedure steps
    st.markdown("<div class='section-hdr'>Kill Procedure Overview</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
**Driller's Method (two-circulation)**

1. **Shut-in** well immediately on kick detection
2. Record SIDPP = {disp_pres(pore_pres - hydrostatic_bar(mud_sg, tvd - wh_tvd))} · SICP (read at surface)
3. **First circulation** — circulate kick out with original mud weight at slow circulation rate
4. **Second circulation** — pump kill-weight mud ({disp_sg(kill_sg)}) down drill string
5. Monitor BHP = {disp_pres(kill_pres)} maintained throughout
6. Verify static kill when SIDPP → 0
""")
    with c2:
        st.markdown(f"""
**Wait & Weight (Engineer's Method)**

1. **Shut-in** well on kick detection
2. Calculate kill mud weight = {disp_sg(kill_sg)}
3. Mix kill mud to required weight
4. **Single circulation** — pump kill mud down drill string, circulate influx up annulus
5. Maintain constant BHP = {disp_pres(kill_pres)}
6. Choke management: adjust choke to maintain CITHP while pumping
7. Influx volume circulated out: ~ {disp_vol(pit_m3)}
""")

    kill_df = pd.DataFrame({
        "Parameter": [
            "Formation S.G.", "Current mud S.G.", "Pressure overbalance",
            "Kill mud S.G.", "Safety margin", "Influx rate at detection",
            "Pit gain (estimated)", "Blowout severity index"
        ],
        "Value": [
            disp_sg(form_sg), disp_sg(mud_sg), disp_pres(overbal),
            disp_sg(kill_sg), disp_sg(safety_m),
            disp_rate(q_m3day), disp_vol(pit_m3),
            f"{bo_idx:.3f}"
        ]
    })
    st.dataframe(kill_df, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — WELL SKETCH
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("<div class='section-hdr'>Well Schematic (simplified vertical section)</div>",
                unsafe_allow_html=True)

    d_surf = wh_tvd        # surface / seabed
    d_res  = tvd           # reservoir top
    d_bot  = tvd + thick   # reservoir bottom

    if not use_si:
        d_surf_d, d_res_d, d_bot_d = m_to_ft(d_surf), m_to_ft(d_res), m_to_ft(d_bot)
        depth_unit = "ft"
    else:
        d_surf_d, d_res_d, d_bot_d = d_surf, d_res, d_bot
        depth_unit = "m"

    well_x = 0
    bh_d   = d_bot_d * 1.05
    wx     = 0.08  # half-width of wellbore sketch

    fig_sk = go.Figure()

    # ── Geological layers ──
    # Sea / air
    if d_surf_d > 0:
        fig_sk.add_shape(type="rect", x0=-1, x1=1, y0=0, y1=d_surf_d,
                         fillcolor="rgba(100,180,255,0.25)", line=dict(width=0))
        fig_sk.add_annotation(x=-0.75, y=d_surf_d/2, text="Sea water",
                               showarrow=False, font=dict(color="#1565C0", size=11))

    # Overburden
    fig_sk.add_shape(type="rect", x0=-1, x1=1, y0=d_surf_d, y1=d_res_d,
                     fillcolor="rgba(210,180,140,0.35)", line=dict(width=0))
    fig_sk.add_annotation(x=-0.75, y=(d_surf_d+d_res_d)/2, text="Overburden",
                           showarrow=False, font=dict(color="#5D4037", size=11))

    # Reservoir
    fig_sk.add_shape(type="rect", x0=-1, x1=1, y0=d_res_d, y1=d_bot_d,
                     fillcolor="rgba(245,166,35,0.30)", line=dict(width=0))
    fig_sk.add_annotation(x=-0.75, y=(d_res_d+d_bot_d)/2,
                           text=f"Reservoir ({poros_pct:.0f}% φ, {perm_md:.0f} mD)",
                           showarrow=False, font=dict(color="#E65100", size=11))

    # Below reservoir
    fig_sk.add_shape(type="rect", x0=-1, x1=1, y0=d_bot_d, y1=bh_d*1.02,
                     fillcolor="rgba(150,150,150,0.20)", line=dict(width=0))

    # ── Wellbore ──
    # Casing / wellbore walls (dark navy)
    fig_sk.add_shape(type="rect", x0=-wx*1.5, x1=-wx, y0=d_surf_d, y1=d_bot_d,
                     fillcolor=DARK_NAVY, line=dict(color=DARK_NAVY))
    fig_sk.add_shape(type="rect", x0=wx, x1=wx*1.5, y0=d_surf_d, y1=d_bot_d,
                     fillcolor=DARK_NAVY, line=dict(color=DARK_NAVY))

    # Mud column
    fig_sk.add_shape(type="rect", x0=-wx, x1=wx, y0=d_surf_d, y1=d_res_d,
                     fillcolor="rgba(0,36,61,0.40)",
                     line=dict(color=DARK_NAVY, width=0.5))

    # Influx zone (red if underbalanced)
    influx_color = "rgba(235,0,55,0.50)" if overbal < 0 else "rgba(157,186,0,0.35)"
    fig_sk.add_shape(type="rect", x0=-wx, x1=wx, y0=d_res_d, y1=d_bot_d,
                     fillcolor=influx_color, line=dict(color=TORCH_RED if overbal<0 else PISTACHIO))

    # Bottom of well
    fig_sk.add_shape(type="line", x0=-wx, x1=wx, y0=d_bot_d, y1=d_bot_d,
                     line=dict(color=DARK_NAVY, width=3))

    # Annotations with arrows
    ann_common = dict(arrowhead=2, arrowcolor=TORCH_RED, arrowwidth=1.5,
                      font=dict(size=10, color=DARK_NAVY),
                      bgcolor="rgba(255,255,255,0.85)", bordercolor=MID_GREY)

    # Mud weight label
    fig_sk.add_annotation(x=wx*2, y=(d_surf_d + d_res_d)/2,
                           text=f"Mud: {disp_sg(mud_sg)}<br>{disp_pres(hydro_pres)} @ res",
                           ax=60, ay=0, **ann_common)

    # Reservoir label
    fig_sk.add_annotation(x=wx*2, y=(d_res_d + d_bot_d)/2,
                           text=f"PP: {disp_pres(pore_pres)}<br>{'← FLOW' if overbal<0 else '✓ Static'}",
                           ax=60, ay=0, **ann_common)

    # Kill mud
    fig_sk.add_annotation(x=-wx*2, y=d_res_d,
                           text=f"Kill MW: {disp_sg(kill_sg)}",
                           ax=-60, ay=0, **ann_common)

    # Flow arrows if underbalanced
    if overbal < 0:
        for dy in [0.2, 0.5, 0.8]:
            yp = d_res_d + (d_bot_d - d_res_d) * dy
            fig_sk.add_annotation(x=0, y=yp, text="↑",
                                   showarrow=False,
                                   font=dict(size=18, color=TORCH_RED))

    # Surface equipment (BOP stack)
    bop_h = d_surf_d * 0.04 if d_surf_d > 0 else d_res_d * 0.02
    bop_y = d_surf_d - bop_h if d_surf_d > 0 else 0
    fig_sk.add_shape(type="rect", x0=-wx*3, x1=wx*3, y0=bop_y - bop_h, y1=bop_y,
                     fillcolor="rgba(0,36,61,0.80)", line=dict(color=DARK_NAVY))
    fig_sk.add_annotation(x=0, y=bop_y - bop_h/2, text="BOP",
                           showarrow=False, font=dict(color="white", size=10))

    # Depth markers (right side)
    for label, depth in [("Surface/seabed", d_surf_d), ("Res. top", d_res_d),
                          ("Res. base", d_bot_d)]:
        fig_sk.add_shape(type="line", x0=0.85, x1=1.0, y0=depth, y1=depth,
                         line=dict(color=MID_GREY, dash="dot"))
        fig_sk.add_annotation(x=1.05, y=depth,
                               text=f"{label}<br>{depth:.0f} {depth_unit}",
                               showarrow=False, font=dict(size=9, color="#555"),
                               xanchor="left")

    fig_sk.update_layout(
        xaxis=dict(visible=False, range=[-1.3, 1.6]),
        yaxis=dict(autorange="reversed", title=f"Depth ({depth_unit})",
                   range=[-bh_d*0.05, bh_d*1.05]),
        plot_bgcolor="white", paper_bgcolor="white",
        height=650, margin=dict(t=20, l=20, r=130),
        showlegend=False,
    )
    st.plotly_chart(fig_sk, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
**Well dimensions**
- TVD to reservoir: **{disp_depth(tvd)}**
- Reservoir thickness: **{disp_depth(thick)}**
- Wellbore radius: **{rw:.3f} m ({rw/0.0254:.2f} in)**
- Drainage radius: **{disp_depth(re)}**
        """)
    with col2:
        st.markdown(f"""
**Wellbore volumes**
- Annular / open hole: **{disp_vol(wbs_m3)}**
- Reservoir PV: **{disp_vol(pv_m3/influx_fr)}**
- Influx at detection: **{disp_vol(pit_m3)}**
- PV / wellbore ratio: **{pv_m3/influx_fr/wbs_m3:.1f}×**
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("<div class='section-hdr'>Sensitivity Analysis</div>", unsafe_allow_html=True)

    sens_param = st.selectbox("Primary parameter to sweep:", [
        "Formation S.G. (pore pressure)",
        "Mud weight S.G.",
        "Permeability (mD)",
        "Net pay thickness",
        "Porosity (%)",
        "Reservoir extent",
        "Reaction time (hr)",
    ])

    col_s1, col_s2 = st.columns([1, 3])
    with col_s1:
        n_pts = st.slider("Number of points", 10, 100, 40)
        show_q     = st.checkbox("Flow rate", True)
        show_bo    = st.checkbox("Blowout index", True)
        show_pit   = st.checkbox("Pit gain", True)
        show_overb = st.checkbox("Overbalance", True)

    # Build sweep range
    def sweep(param_name):
        ranges = {
            "Formation S.G. (pore pressure)":   (0.90,  2.20, form_sg),
            "Mud weight S.G.":                   (1.00,  2.30, mud_sg),
            "Permeability (mD)":                 (0.1,   5000, perm_md),
            "Net pay thickness":                  (1.0,   200,  thick),
            "Porosity (%)":                       (1.0,   40,   poros_pct),
            "Reservoir extent":                   (100,   5000, extent),
            "Reaction time (hr)":                 (0.05,  8.0,  react_hr),
        }
        lo, hi, base = ranges[param_name]
        return np.linspace(lo, hi, n_pts), base

    x_arr, base_val = sweep(sens_param)

    # Compute outputs
    q_sens, bo_sens, pit_sens, ob_sens = [], [], [], []
    for xv in x_arr:
        if sens_param == "Formation S.G. (pore pressure)":
            fs, ms, kd, th, pp2, ex, rt = xv, mud_sg, perm_md, thick, poros_pct, extent, react_hr
        elif sens_param == "Mud weight S.G.":
            fs, ms, kd, th, pp2, ex, rt = form_sg, xv, perm_md, thick, poros_pct, extent, react_hr
        elif sens_param == "Permeability (mD)":
            fs, ms, kd, th, pp2, ex, rt = form_sg, mud_sg, xv, thick, poros_pct, extent, react_hr
        elif sens_param == "Net pay thickness":
            fs, ms, kd, th, pp2, ex, rt = form_sg, mud_sg, perm_md, xv, poros_pct, extent, react_hr
        elif sens_param == "Porosity (%)":
            fs, ms, kd, th, pp2, ex, rt = form_sg, mud_sg, perm_md, thick, xv, extent, react_hr
        elif sens_param == "Reservoir extent":
            fs, ms, kd, th, pp2, ex, rt = form_sg, mud_sg, perm_md, thick, poros_pct, xv, react_hr
        else:  # Reaction time
            fs, ms, kd, th, pp2, ex, rt = form_sg, mud_sg, perm_md, thick, poros_pct, extent, xv

        dp2   = max(pore_pressure_bar(fs, tvd) - hydrostatic_bar(ms, tvd), 0.0)
        q2    = darcy_flow_rate_m3_per_day(kd, th, dp2, visc_cp, re, rw)
        bo2   = blowout_index(dp2, kd, th)
        area2 = math.pi * ex ** 2
        pv2   = pv_influx_m3(pp2/100.0, th, area2, influx_fr)
        pit2  = pit_gain_m3(q2, rt)
        ob2   = overbalance_bar(ms, fs, tvd)

        q_sens.append(m3_to_bbl(q2) if not use_si else q2)
        bo_sens.append(bo2)
        pit_sens.append(m3_to_bbl(pit2) if not use_si else pit2)
        ob_sens.append(bar_to_psi(ob2) if not use_si else ob2)

    # X-axis label
    x_lbl_map = {
        "Formation S.G. (pore pressure)":  "Formation S.G.",
        "Mud weight S.G.":                 "Mud weight S.G.",
        "Permeability (mD)":               "Permeability (mD)",
        "Net pay thickness":               f"Thickness ({UL['thick']})",
        "Porosity (%)":                    "Porosity (%)",
        "Reservoir extent":                f"Extent ({UL['depth']})",
        "Reaction time (hr)":              "Reaction time (hr)",
    }
    x_lbl_s = x_lbl_map[sens_param]

    with col_s2:
        fig_sens = make_subplots(rows=2, cols=2, shared_xaxes=False,
                                 subplot_titles=[
                                     f"Flow rate ({UL['rate']})",
                                     "Blowout severity index",
                                     f"Pit gain ({UL['vol']})",
                                     f"Overbalance ({UL['pres_unit']})"
                                 ])

        colors  = [TORCH_RED, DARK_NAVY, PISTACHIO, "#F5A623"]
        flags   = [show_q, show_bo, show_pit, show_overb]
        y_lists = [q_sens, bo_sens, pit_sens, ob_sens]
        pos     = [(1,1),(1,2),(2,1),(2,2)]

        for (row, col_p), y_data, color, flag in zip(pos, y_lists, colors, flags):
            if flag:
                fig_sens.add_trace(go.Scatter(
                    x=list(x_arr), y=y_data, mode="lines",
                    line=dict(color=color, width=2.5)
                ), row=row, col=col_p)
                # Baseline marker
                fig_sens.add_vline(x=base_val, line_dash="dot",
                                   line_color=MID_GREY, opacity=0.7,
                                   row=row, col=col_p)

        fig_sens.update_xaxes(title_text=x_lbl_s)
        fig_sens.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=600, margin=dict(t=40),
            showlegend=False,
        )
        st.plotly_chart(fig_sens, use_container_width=True)

    # Tornado chart (±20% sensitivity)
    st.markdown("<div class='section-hdr'>Tornado: Influx Rate Sensitivity (±20% on each parameter)</div>",
                unsafe_allow_html=True)
    tornado_params = {
        "Formation S.G.":   ("form_sg",   form_sg),
        "Mud weight S.G.":  ("mud_sg",    mud_sg),
        "Permeability":     ("perm_md",   perm_md),
        "Thickness":        ("thick",     thick),
        "Porosity":         ("poros_frac",poros_frac),
        "Extent":           ("extent",    extent),
    }
    tornado_rows = []
    base_q = darcy_flow_rate_m3_per_day(perm_md, thick, dp_bar, visc_cp, re, rw)

    for label, (pname, base) in tornado_params.items():
        lo_v, hi_v = base * 0.80, base * 1.20
        results = []
        for variant in (lo_v, hi_v):
            kwargs = dict(perm_md=perm_md, thick=thick, form_sg=form_sg, mud_sg=mud_sg)
            if pname == "form_sg":
                dp2 = max(pore_pressure_bar(variant, tvd) - hydro_pres, 0)
                q2 = darcy_flow_rate_m3_per_day(perm_md, thick, dp2, visc_cp, re, rw)
            elif pname == "mud_sg":
                dp2 = max(pore_pres - hydrostatic_bar(variant, tvd), 0)
                q2 = darcy_flow_rate_m3_per_day(perm_md, thick, dp2, visc_cp, re, rw)
            elif pname == "perm_md":
                q2 = darcy_flow_rate_m3_per_day(variant, thick, dp_bar, visc_cp, re, rw)
            elif pname == "thick":
                q2 = darcy_flow_rate_m3_per_day(perm_md, variant, dp_bar, visc_cp, re, rw)
            elif pname == "poros_frac":
                q2 = base_q   # porosity doesn't affect Darcy rate
            elif pname == "extent":
                q2 = darcy_flow_rate_m3_per_day(perm_md, thick, dp_bar, visc_cp, variant, rw)
            results.append(q2)
        lo_q, hi_q = results
        if not use_si:
            lo_q, hi_q, base_q_d = m3_to_bbl(lo_q), m3_to_bbl(hi_q), m3_to_bbl(base_q)
        else:
            base_q_d = base_q
        tornado_rows.append((label, lo_q - base_q_d, hi_q - base_q_d))

    tornado_rows.sort(key=lambda r: abs(r[2] - r[1]))

    fig_torn = go.Figure()
    for label, delta_lo, delta_hi in tornado_rows:
        fig_torn.add_trace(go.Bar(
            y=[label], x=[min(delta_lo, delta_hi)],
            orientation="h", marker_color=DARK_NAVY, showlegend=False,
            base=max(delta_lo, delta_hi),
        ))
        fig_torn.add_trace(go.Bar(
            y=[label], x=[max(delta_hi - delta_lo, delta_lo - delta_hi)],
            base=min(delta_lo, delta_hi),
            orientation="h", marker_color=TORCH_RED, showlegend=False,
        ))

    fig_torn.add_vline(x=0, line_color=DARK_NAVY, line_width=2)
    fig_torn.update_layout(
        barmode="overlay",
        xaxis_title=f"Δ Influx rate ({UL['rate']}) vs base",
        plot_bgcolor="white", paper_bgcolor="white",
        height=350, margin=dict(t=10)
    )
    st.plotly_chart(fig_torn, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("<div class='section-hdr'>Full Parameter & Results Summary</div>",
                unsafe_allow_html=True)

    summary = pd.DataFrame({
        "Category": [
            "Well Geometry", "Well Geometry", "Well Geometry", "Well Geometry",
            "Pressure", "Pressure", "Pressure", "Pressure",
            "Reservoir", "Reservoir", "Reservoir", "Reservoir", "Reservoir",
            "Flow", "Flow", "Flow",
            "Kill", "Kill", "Kill",
            "Volumes", "Volumes", "Volumes",
            "Risk",
        ],
        "Parameter": [
            "TVD to reservoir", "Wellbore radius", "Drainage radius", "Wellbore volume",
            "Pore pressure", "Hydrostatic pressure", "Overbalance", "Kill mud pressure",
            "Net pay thickness", "Porosity", "Permeability", "Reservoir extent", "Pore volume",
            "Influx rate (Darcy)", "Pit gain at detection", "Influx fraction",
            "Kill mud S.G.", "Safety margin", "Kill method",
            "PV influx volume", "Wellbore capacity", "PV / wellbore ratio",
            "Risk level",
        ],
        "Value": [
            disp_depth(tvd), f"{rw:.3f} m ({rw/0.0254:.2f} in)", disp_depth(re), disp_vol(wbs_m3),
            disp_pres(pore_pres), disp_pres(hydro_pres), disp_pres(overbal), disp_pres(kill_pres),
            disp_depth(thick), f"{poros_pct:.1f}%", f"{perm_md:.1f} mD",
            disp_depth(extent), disp_vol(pv_m3/influx_fr),
            disp_rate(q_m3day), disp_vol(pit_m3), f"{influx_fr*100:.0f}%",
            disp_sg(kill_sg), disp_sg(safety_m), "Driller's / Wait & Weight",
            disp_vol(pv_m3), disp_vol(wbs_m3), f"{pv_m3/influx_fr/wbs_m3:.1f}×",
            risk_level,
        ],
    })

    st.dataframe(summary, use_container_width=True, hide_index=True,
                 column_config={"Category": st.column_config.TextColumn(width="medium")})

    st.download_button(
        label="⬇️ Download summary CSV",
        data=summary.to_csv(index=False).encode(),
        file_name="well_control_summary.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='color:{MID_GREY};font-size:0.75rem;text-align:center;'>"
    f"Well Control & Influx Analyser · MIT licence · Engineering use only — not a substitute for "
    f"certified well control procedures (IWCF/IADC). All calculations based on Darcy radial flow, "
    f"hydrostatic pressure balance, and standard well control principles."
    f"</div>",
    unsafe_allow_html=True
)
