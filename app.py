# app.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st

# ------------------ NumPy integration compatibility ------------------
if hasattr(np, "trapezoid"):
    TRAPZ = np.trapezoid
elif hasattr(np, "trapz"):
    TRAPZ = np.trapz
else:
    raise RuntimeError("NumPy has neither trapezoid nor trapz. Please update NumPy.")


# ===================== HELPERS (UNITS + DESIGN) =====================
def compute_phiVc(unit_system: str, phi: float, fc: float, bw: float, d: float) -> float:
    """
    φVc = 2 φ √fc' bw d / 1000
    Returns:
      - Imperial: kip
      - SI: kN
    NOTE: This is your chosen simplified expression (keep consistent with your PDF work).
    """
    return 2.0 * phi * (fc ** 0.5) * bw * d / 1000.0


def d_to_beam_length(unit_system: str, d: float) -> float:
    # d in inches -> ft ; d in mm -> m
    return d / 12.0 if "Imperial" in unit_system else d / 1000.0


def stirrup_spacing_length(unit_system: str, s: float) -> float:
    # s in inches -> ft ; s in mm -> m
    return s / 12.0 if "Imperial" in unit_system else s / 1000.0


def compute_spacing(Av: float, fy: float, d_eff: float, Vs_req: float, smax: float, smin: float) -> float:
    """
    Simple spacing: Vs = Av*fy*d/s  (units consistent with your input convention)
    cap = Av*fy*d/1000 => (kip or kN) when Av in in² or mm², fy in psi or MPa, d in in or mm
    """
    if Vs_req <= 1e-9:
        return smax
    cap = (Av * fy * d_eff) / 1000.0
    s = cap / Vs_req
    return max(smin, min(smax, s))


def first_x_where_absV_crossing(x_vals, V_vals, target, x_start=0.0):
    """Return the x-location where |V(x)| first crosses DOWN through 'target'
    (from > target to <= target) starting from x_start, using linear interpolation
    between bracketing samples for better accuracy.
    """
    x_vals = np.asarray(x_vals, dtype=float)
    V_vals = np.asarray(V_vals, dtype=float)

    mask = x_vals >= float(x_start)
    xs = x_vals[mask]
    vs = np.abs(V_vals[mask])

    if xs.size < 2:
        return None

    diff = vs - float(target)

    # first downward crossing
    for i in range(1, len(xs)):
        if diff[i] <= 0.0 and diff[i - 1] > 0.0:
            x0, x1 = xs[i - 1], xs[i]
            d0, d1 = diff[i - 1], diff[i]
            if abs(d1 - d0) < 1e-12:
                return float(x1)
            t = d0 / (d0 - d1)  # in (0,1]
            return float(x0 + t * (x1 - x0))

    if diff[0] <= 0.0:
        return float(xs[0])

    return None


# ===================== SOLVER (FEM + SFD polyline w/ jumps) =====================
def solve_beam(L, E, I, supports, loads):
    # ---- nodes ----
    xs = [0.0, L]
    for s in supports:
        xs.append(float(s["x"]))
    for ld in loads:
        if ld["type"] == "udl":
            xs += [float(ld["a"]), float(ld["b"])]
        elif ld["type"] == "point":
            xs.append(float(ld["x"]))

    nodes = sorted(set(round(x, 10) for x in xs))
    n = len(nodes)
    ndof = 2 * n

    def dof(i):
        return [2 * i, 2 * i + 1]

    # ---- element stiffness ----
    def k_beam(EI, Le):
        return EI / Le**3 * np.array(
            [
                [12, 6 * Le, -12, 6 * Le],
                [6 * Le, 4 * Le**2, -6 * Le, 2 * Le**2],
                [-12, -6 * Le, 12, -6 * Le],
                [6 * Le, 2 * Le**2, -6 * Le, 4 * Le**2],
            ],
            dtype=float,
        )

    K = np.zeros((ndof, ndof), dtype=float)
    F = np.zeros(ndof, dtype=float)

    for e in range(n - 1):
        Le = nodes[e + 1] - nodes[e]
        if Le <= 0:
            continue
        ke = k_beam(E * I, Le)
        idx = dof(e) + dof(e + 1)
        for i in range(4):
            for j in range(4):
                K[idx[i], idx[j]] += ke[i, j]

    # ---- loads to global ----
    def add_point(P, x):
        i = nodes.index(round(x, 10))
        F[2 * i] += P

    def add_udl_on_element(e, w, Le):
        fe = np.array([w * Le / 2, w * Le**2 / 12, w * Le / 2, -w * Le**2 / 12], dtype=float)
        idx = dof(e) + dof(e + 1)
        F[idx] += fe

    for ld in loads:
        if ld["type"] == "point":
            add_point(float(ld["P"]), float(ld["x"]))

    for ld in loads:
        if ld["type"] == "udl":
            w = float(ld["w"])
            a = float(ld["a"])
            b = float(ld["b"])
            for e in range(n - 1):
                x1, x2 = nodes[e], nodes[e + 1]
                if x1 >= a and x2 <= b:
                    add_udl_on_element(e, w, x2 - x1)

    # ---- boundary conditions ----
    fixed = []
    for s in supports:
        xi = round(float(s["x"]), 10)
        i = nodes.index(xi)
        if s["type"] in ("pin", "roller"):
            fixed.append(2 * i)          # vertical disp = 0
        elif s["type"] == "fixed":
            fixed += [2 * i, 2 * i + 1]  # disp and rotation = 0

    fixed = sorted(set(fixed))
    free = [i for i in range(ndof) if i not in fixed]

    d = np.zeros(ndof, dtype=float)
    if free:
        d[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])

    R = K @ d - F

    reactions = []
    for s in supports:
        xi = round(float(s["x"]), 10)
        i = nodes.index(xi)
        reactions.append((float(s["x"]), float(R[2 * i])))

    # ---- SFD polyline (events cause vertical jumps) ----
    events = {}
    for rx, rr in reactions:
        x0 = round(rx, 10)
        events[x0] = events.get(x0, 0.0) + rr
    for ld in loads:
        if ld["type"] == "point":
            x0 = round(float(ld["x"]), 10)
            events[x0] = events.get(x0, 0.0) + float(ld["P"])

    jump_xs = sorted(events.keys())

    def w_at(x):
        wt = 0.0
        for ld in loads:
            if ld["type"] == "udl":
                a = float(ld["a"])
                b = float(ld["b"])
                if a <= x <= b:
                    wt += float(ld["w"])
        return wt

    def add_udl_contrib(V_now, x1, x2):
        if x2 <= x1:
            return V_now
        xs_int = np.linspace(x1, x2, 120)
        ws_int = np.array([w_at(xi) for xi in xs_int], dtype=float)
        return V_now + TRAPZ(ws_int, xs_int)

    eps = max(1e-6, L * 1e-7)

    x_vals, V_vals = [0.0], [0.0]
    V = 0.0
    x_prev = 0.0

    if 0.0 in jump_xs:
        V += events[0.0]
        x_vals.append(0.0)
        V_vals.append(V)

    stations = sorted([x for x in jump_xs if 0.0 < x < L] + [L])

    for x_next in stations:
        if x_next > x_prev:
            sample = np.linspace(x_prev + eps, x_next, 80)
            for xv in sample:
                x_vals.append(float(xv))
                V_vals.append(float(add_udl_contrib(V, x_prev, xv)))

        V = add_udl_contrib(V, x_prev, x_next)
        x_prev = x_next

        xk = round(x_next, 10)
        if xk in events:
            V += events[xk]
            x_vals.append(float(x_next))
            V_vals.append(float(V))

    return np.array(x_vals), np.array(V_vals), reactions


# ===================== PLOTS =====================
def fig_beam_diagram(L, supports, loads, unit_system):
    fig, ax = plt.subplots(figsize=(10, 2.9))
    ax.set_title("Beam Diagram")

    # beam at y=0
    ax.plot([0, L], [0, 0], linewidth=10)

    pad = 0.05 * L
    ax.set_xlim(-pad, L + pad)

    support_color = "#0B3D91"  # deep blue
    fixed_color =  "#FF5100"  # deep blue
    udl_color = "#2ca02c"      # green
    pt_color = "#d62728"       # red

    sym = max(0.18, 0.02 * L)

    # PIN touches beam: apex at y=0
    def draw_pin(x):
        apex_y = -0.07
        base_y = apex_y - 0.22
        tri = patches.Polygon(
            [[x, apex_y], [x - sym, base_y], [x + sym, base_y]],
            closed=True, 
            facecolor=support_color, 
            edgecolor=support_color,
            linewidth=2.8, 
        )
        tri.set_clip_on(False)
        ax.add_patch(tri)

    # ROLLER touches beam: top tangent at y=0
    def draw_roller(x):
        r_y = 0.11
        y_shift = -0.07
        ax.plot(
            [x], 
            [-r_y+y_shift],
            marker="o", markersize=20,
            markerfacecolor=support_color,
            markeredgewidth=3.2,
            markeredgecolor=support_color,
        )

    def draw_fixed(x):
        ax.plot([x, x], [0.3, -0.3], linewidth=10, color=fixed_color)

    for s in supports:
        x = float(s["x"])
        t = s["type"]
        if t == "pin":
            draw_pin(x)
        elif t == "roller":
            draw_roller(x)
        elif t == "fixed":
            draw_fixed(x)

    # point loads (RED)
    point_xs = []
    for ld in loads:
        if ld["type"] == "point":
            x = float(ld["x"])
            Pmag = abs(float(ld["P"]))
            point_xs.append(x)

            ax.annotate(
                "",
                xy=(x, 0.01),
                xytext=(x, 1.05),
                arrowprops=dict(arrowstyle="->", lw=3, color=pt_color),
            )
            ax.text(
                x, 1.05, f"{Pmag:g}",
                ha="center", va="bottom",
                fontsize=14, fontweight="bold",
                color=pt_color
            )

    # UDL (GREEN)
    for ld in loads:
        if ld["type"] == "udl":
            a, b = float(ld["a"]), float(ld["b"])
            wmag = abs(float(ld["w"]))

            ax.plot([a, b], [0.75, 0.75], linewidth=3, color=udl_color)

            for x in np.linspace(a, b, 14):
                ax.annotate(
                    "",
                    xy=(x, 0.05),
                    xytext=(x, 0.75),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color=udl_color),
                )

            x_mid = (a + b) / 2
            y_text = 0.8

            # push UDL label away from point load if it overlaps
            thresh = max(0.35, 0.03 * L)
            if any(abs(px - x_mid) < thresh for px in point_xs):
                x_mid = min(b - 0.05 * (b - a), x_mid + 0.10 * (b - a))

            w_unit = "kip/ft" if "Imperial" in unit_system else "kN/m"
            ax.text(
                x_mid, y_text, f"{wmag:g} {w_unit}",
                ha="center", va="bottom",
                fontsize=13, fontweight="bold",
                color=udl_color
            )

    ax.set_ylim(-0.60, 1.30)
    ax.set_yticks([])
    ax.set_xlabel("x")
    ax.grid(True, axis="x", alpha=0.35)
    return fig


def fig_sfd(L, x_vals, V_vals, unit_system):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_title("Shear Force Diagram (SFD)")

    ax.plot(x_vals, V_vals, linewidth=2.5)
    ax.axhline(0, linewidth=1)

    ax.fill_between(x_vals, V_vals, 0, where=(V_vals >= 0), alpha=0.15, interpolate=True)
    ax.fill_between(x_vals, V_vals, 0, where=(V_vals <= 0), alpha=0.15, interpolate=True)

    ax.set_xlim(0, L)
    ax.set_xlabel("x")
    ax.set_ylabel("V(x)")
    ax.grid(True, alpha=0.35)

    force_unit = "k" if "Imperial" in unit_system else "kN"

    # label at ends and at jumps
    jump_indices = []
    for i in range(1, len(x_vals)):
        if abs(x_vals[i] - x_vals[i - 1]) < 1e-10 and abs(V_vals[i] - V_vals[i - 1]) > 1e-6:
            jump_indices.append(i)

    label_points = {0, len(x_vals) - 1}
    for i in jump_indices:
        label_points.add(i - 1)
        label_points.add(i)

    for i in sorted(label_points):
        x = x_vals[i]
        V = V_vals[i]
        dy = 4 if V >= 0 else -6
        ax.plot([x], [V], marker="o", markersize=4)
        ax.text(x, V + dy, f"{V:.1f} {force_unit}", fontsize=10)

    return fig


def fig_sfd_design_marks(L, x_vals, V_vals, d_len, phi, phiVc, unit_system):
    """
    Diagram 3 (updated to match your PDF-style sketch):
      - Shear marks Vu at x=d, x2, x3
      - ALSO shows bottom dimension lines:
            d (0 -> x1), x2 (0 -> x2), x3 (0 -> x3)
      - Vu labels use mathtext subscripts: V_{u,1}, V_{u,2}, V_{u,3}
    """
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.set_title("SFD + Shear Design Marks")

    ax.plot(x_vals, V_vals, linewidth=3.5, color="black")
    ax.axhline(0, linewidth=1.2, color="black")
    ax.set_xlim(0, L)
    ax.set_xlabel("x")
    ax.set_ylabel("V(x)")
    ax.grid(True, alpha=0.25)

    mark_blue = "#0000FF"
    circle_red = "#FF0000"
    guide_grey = "#777777"
    dim_color = "black"

    def V_at(x):
        return float(np.interp(x, x_vals, V_vals))

    # x1 is the critical section at distance d from the left support (x=0)
    x1 = max(0.0, min(L, float(d_len)))
    x2 = first_x_where_absV_crossing(x_vals, V_vals, phiVc, x_start=x1)
    x3 = first_x_where_absV_crossing(x_vals, V_vals, phiVc / 2.0, x_start=x1)

    Vu1 = abs(V_at(x1))
    Vu2 = phiVc
    Vu3 = phiVc / 2.0

    LABEL_FS = 12
    CIRCLE_SIZE = 8
    CIRCLE_EDGE = 2.0

    placed_bboxes = []

    def place_label_with_arrow(x, y, text, dx, dy):
        candidates = [(dx, dy), (dx, dy * 1.2), (dx * 1.2, dy), (dx * 1.4, dy * 1.2)]
        for ddx, ddy in candidates:
            ann = ax.annotate(
                text,
                xy=(x, y),
                xytext=(x + ddx, y + ddy),
                fontsize=LABEL_FS,
                color=mark_blue,
                arrowprops=dict(arrowstyle="->", lw=2.0, color=mark_blue),
            )
            fig.canvas.draw()
            bb = ann.get_window_extent()
            if all(not bb.overlaps(prev) for prev in placed_bboxes):
                placed_bboxes.append(bb)
                return
            ann.remove()

        ann = ax.annotate(
            text,
            xy=(x, y),
            xytext=(x + dx, y + dy),
            fontsize=LABEL_FS,
            color=mark_blue,
            arrowprops=dict(arrowstyle="->", lw=1.5, color=mark_blue),
        )
        fig.canvas.draw()
        placed_bboxes.append(ann.get_window_extent())

    def draw_mark(x, label_text):
        if x is None:
            return
        y = V_at(x)

        # blue vertical line from axis to point
        ax.vlines(x, 0, y, linewidth=2.5, color=mark_blue)

        # red circle marker
        ax.plot(
            [x], [y],
            marker="o",
            markersize=CIRCLE_SIZE,
            markerfacecolor="white",
            markeredgecolor=circle_red,
            markeredgewidth=CIRCLE_EDGE,
            zorder=5,
        )

        dx = 0.07 * L
        dy = 0.16 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        if x > 0.65 * L:
            dx = -0.22 * L

        place_label_with_arrow(x, y, label_text, dx, dy)

    # ---- marks (x1 solid guide; x2/x3 dashed guides like your sketch) ----
    draw_mark(x1, rf"$V_{{u,1}} = {Vu1:.2f}$")
    if x2 is not None:
        draw_mark(x2, rf"$V_{{u,2}} = {Vu2:.2f}$")
    if x3 is not None:
        draw_mark(x3, rf"$V_{{u,3}} = {Vu3:.2f}$")

    # ---- bottom dimensions: d, x2, x3 (stacked) ----
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin if (ymax - ymin) > 1e-9 else 1.0

    # extend lower ylim to make room for dimensions
    need_min = ymin - 0.45 * yr
    need_max = ymax + 0.20 * yr
    ax.set_ylim(need_min, need_max)

    # recompute after set_ylim
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin

    y_dim1 = ymin + 0.45 * yr  # closest to x-axis (higher)
    y_dim2 = ymin + 0.3 * yr
    y_dim3 = ymin + 0.15 * yr  # lowest

    if "Imperial" in unit_system:
        unit_suffix = "′"  # feet prime
    else:
        unit_suffix = " m"

    def draw_dim(xa, xb, y, text):
        ax.annotate(
            "",
            xy=(xa, y),
            xytext=(xb, y),
            arrowprops=dict(arrowstyle="<->", lw=2.4, color=dim_color, mutation_scale=18, shrinkA=0, shrinkB=0),
        )
        ax.text((xa + xb) / 2.0, y - 0.03 * yr, text, ha="center", va="top", fontsize=14, color=dim_color)

    # d = x1 (0 -> x1)
    draw_dim(0.0, x1, y_dim1, f"d = {x1:.2f}{unit_suffix}")

    # x2 and x3 are absolute distances from the left support (0 -> x2/x3)
    if x2 is not None:
        draw_dim(0.0, x2, y_dim2, f"x\u2082 = {x2:.2f}{unit_suffix}")
    if x3 is not None:
        draw_dim(0.0, x3, y_dim3, f"x\u2083 = {x3:.2f}{unit_suffix}")

    
    # ---- extend dashed guide lines down to touch the dimension arrows ----
    ymin_g, ymax_g = ax.get_ylim()
    if x1 is not None:
        ax.vlines(
            x1, y_dim1, ymax_g,
            linestyles=(0, (4, 4)),
            linewidth=1.6,
            color=guide_grey,
            alpha=0.8,
            zorder=0,
        )
    if x2 is not None:
        ax.vlines(
            x2, y_dim2, ymax_g,
            linestyles=(0, (4, 4)),
            linewidth=1.6,
            color=guide_grey,
            alpha=0.8,
            zorder=0,
        )
    if x3 is not None:
        ax.vlines(
            x3, y_dim3, ymax_g,
            linestyles=(0, (4, 4)),
            linewidth=1.6,
            color=guide_grey,
            alpha=0.8,
            zorder=0,
        )

    return fig, x1, x2, x3, Vu1

def fig_shear_reinf_design(L, unit_system, x1, x2, x3, s_req, s_max):
    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.set_title("Shear Reinforcement Design Diagram")

    pad = 0.05 * L
    ax.set_xlim(-pad, L + pad)
    ax.set_ylim(-0.25, 1.15)
    ax.set_yticks([])
    ax.set_xlabel("x")
    ax.grid(True, axis="x", alpha=0.25)

    beam_y0 = 0.15
    beam_h = 0.55
    rect = patches.Rectangle((0, beam_y0), L, beam_h, fill=False, linewidth=2)
    ax.add_patch(rect)

    def dashed(x):
        if x is None:
            return
        ax.vlines(x, 0, 1.05, linestyles="dashed", linewidth=1)

    dashed(x1)
    dashed(x2)
    dashed(x3)

    # (Diagram 4 uses d/x2/x3 labels; Diagram 3 has NO d text)
    if x1 is not None:
        ax.text(x1, 1.02, "d", ha="center", va="bottom")
    if x2 is not None:
        ax.text(x2, 1.02, "x₂", ha="center", va="bottom")
    if x3 is not None:
        ax.text(x3, 1.02, "x₃", ha="center", va="bottom")

    x2_eff = x2 if x2 is not None else L
    x3_eff = x3 if x3 is not None else L

    s1_len = stirrup_spacing_length(unit_system, s_req)
    smax_len = stirrup_spacing_length(unit_system, s_max)

    def draw_stirrups(x_start, x_end, spacing_len):
        if spacing_len <= 1e-9:
            return
        x = x_start + 0.15 * spacing_len
        while x < x_end - 1e-9:
            ax.vlines(x, beam_y0, beam_y0 + beam_h, linewidth=3)
            x += spacing_len

    draw_stirrups(0.0, x2_eff, s1_len)
    if x3_eff > x2_eff + 1e-9:
        draw_stirrups(x2_eff, x3_eff, smax_len)

    y_dim = 0.03
    ax.annotate("", xy=(0, y_dim), xytext=(x2_eff, y_dim), arrowprops=dict(arrowstyle="<->", lw=1.5))
    ax.text(x2_eff / 2, y_dim - 0.06, f"Required shear (s={s_req:.0f})", ha="center", va="top")

    if x3_eff > x2_eff + 1e-9:
        ax.annotate("", xy=(x2_eff, y_dim), xytext=(x3_eff, y_dim), arrowprops=dict(arrowstyle="<->", lw=1.5))
        ax.text((x2_eff + x3_eff) / 2, y_dim - 0.06, f"Min shear (s={s_max:.0f})", ha="center", va="top")

    if x3_eff < L - 1e-9:
        ax.annotate("", xy=(x3_eff, y_dim), xytext=(L, y_dim), arrowprops=dict(arrowstyle="<->", lw=1.5))
        ax.text((x3_eff + L) / 2, y_dim - 0.06, "No shear reinforcement", ha="center", va="top")

    return fig


# ===================== STREAMLIT UI =====================
st.set_page_config(page_title="Beam Solver", layout="wide")
st.title("RC Beam Analysis Input Form")

st.header("0) Units")
unit_system = st.selectbox("Select Unit System", ["Imperial (kip/ft, ft, psi, in)", "SI (kN/m, m, MPa, mm)"])

if "Imperial" in unit_system:
    L_label = "Beam Length L (ft)"
    w_label = "UDL w (kip/ft) downward"
    P_label = "Point Load P (kip) downward"
    fc_label = "fc' (psi)"
    bw_label = "bw (in)"
    d_label = "d (in)"
    fy_label = "fy (psi)"
    Av_label = "Av per stirrup (in²) (total legs)"
    smax_label = "Smax (in)"
    smin_label = "Smin (in)"
    force_unit = "kip"
else:
    L_label = "Beam Length L (m)"
    w_label = "UDL w (kN/m) downward"
    P_label = "Point Load P (kN) downward"
    fc_label = "fc' (MPa)"
    bw_label = "bw (mm)"
    d_label = "d (mm)"
    fy_label = "fy (MPa)"
    Av_label = "Av per stirrup (mm²) (total legs)"
    smax_label = "Smax (mm)"
    smin_label = "Smin (mm)"
    force_unit = "kN"

st.divider()

st.header("1) Beam Specifications")
c1, c2, c3 = st.columns(3)
with c1:
    L = st.number_input(L_label, min_value=0.1, value=18.0, step=0.5)
with c2:
    E = st.number_input("E (relative ok)", min_value=0.0, value=1.0, step=1.0)
with c3:
    I = st.number_input("I (relative ok)", min_value=0.0, value=1.0, step=1.0)

st.divider()

st.header("2) Supports")
n_supports = st.number_input("Number of supports", min_value=1, max_value=10, value=2, step=1)

supports = []
default_positions = [0.0, float(L)]
for i in range(int(n_supports)):
    st.subheader(f"Support {i + 1}")
    sc1, sc2 = st.columns(2)
    default_x = default_positions[i] if i < len(default_positions) else float(L) * (i / max(1, (n_supports - 1)))
    with sc1:
        sx = st.number_input(f"Support location x (0 to {L})", min_value=0.0, max_value=float(L),
                             value=float(default_x), key=f"sx{i}")
    with sc2:
        stype = st.selectbox("Support type", ["roller", "pin", "fixed"], key=f"st{i}")
    supports.append({"x": float(sx), "type": stype})

if len({s["x"] for s in supports}) != len(supports):
    st.error("Two supports cannot be at the same x-location.")

st.divider()

st.header("3) Loads")

st.subheader("Point Loads")
nP = st.number_input("Number of point loads", min_value=0, max_value=20, value=1, step=1)

point_loads = []
for i in range(int(nP)):
    pc1, pc2 = st.columns(2)
    with pc1:
        P = st.number_input(P_label, value=12.0, step=1.0, key=f"P{i}")
    with pc2:
        xp = st.number_input(f"x location (0 to {L})", min_value=0.0, max_value=float(L),
                             value=float(L / 2), step=0.5, key=f"xp{i}")
    point_loads.append({"type": "point", "P": -abs(float(P)), "x": float(xp)})

st.subheader("UDL Loads")
nU = st.number_input("Number of UDLs", min_value=0, max_value=20, value=1, step=1)

udls = []
for i in range(int(nU)):
    uc1, uc2, uc3 = st.columns(3)
    with uc1:
        w = st.number_input(w_label, value=5.3 if "Imperial" in unit_system else 5.0, step=0.5, key=f"w{i}")
    with uc2:
        a = st.number_input("start a", min_value=0.0, max_value=float(L), value=0.0, step=0.5, key=f"a{i}")
    with uc3:
        b = st.number_input("end b", min_value=0.0, max_value=float(L), value=float(L), step=0.5, key=f"b{i}")

    udls.append({"type": "udl", "w": -abs(float(w)), "a": float(min(a, b)), "b": float(max(a, b))})

loads = point_loads + udls

st.divider()

st.header("4) Shear Design Inputs")
dcol1, dcol2, dcol3, dcol4 = st.columns(4)
with dcol1:
    bw = st.number_input(bw_label, value=16.0, step=1.0)
with dcol2:
    d_eff = st.number_input(d_label, value=22.0, step=1.0)
with dcol3:
    fc = st.number_input(fc_label, value=4000.0 if "Imperial" in unit_system else 28.0, step=100.0)
with dcol4:
    phi = st.number_input("phi", value=0.75, step=0.05)

phiVc = compute_phiVc(unit_system, phi, fc, bw, d_eff)
d_len = d_to_beam_length(unit_system, d_eff)

st.divider()

st.header("5) Shear Reinforcement Inputs (Stirrups)")
r1, r2, r3, r4 = st.columns(4)
with r1:
    Av = st.number_input(Av_label, value=0.40 if "Imperial" in unit_system else 200.0, step=0.05)
with r2:
    fy = st.number_input(fy_label, value=60000.0 if "Imperial" in unit_system else 420.0, step=10.0)
with r3:
    smax = st.number_input(smax_label, value=12.0 if "Imperial" in unit_system else 300.0, step=1.0)
with r4:
    smin = st.number_input(smin_label, value=4.0 if "Imperial" in unit_system else 100.0, step=1.0)

st.divider()

if st.button("Run Analysis"):
    x_vals, V_vals, reactions = solve_beam(L, E, I, supports, loads)

    st.header("Result")
    st.pyplot(fig_beam_diagram(L, supports, loads, unit_system), use_container_width=True)
    st.pyplot(fig_sfd(L, x_vals, V_vals, unit_system), use_container_width=True)

    fig3, x1, x2, x3, Vu1 = fig_sfd_design_marks(L, x_vals, V_vals, d_len, phi, phiVc, unit_system)
    st.pyplot(fig3, use_container_width=True)

    # simple shear reinforcement sizing
    Vc = phiVc / max(phi, 1e-9)
    Vs_req = max(0.0, Vu1 / max(phi, 1e-9) - Vc)
    s_req = compute_spacing(Av, fy, d_eff, Vs_req, smax, smin)

    st.pyplot(fig_shear_reinf_design(L, unit_system, x1, x2, x3, s_req, smax), use_container_width=True)

    st.subheader("Support Reactions")
    st.table([{"x": round(x, 4), f"Reaction ({force_unit})": round(R, 4)} for x, R in reactions])
