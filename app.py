# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# --------------------------
# --- AQUÍ se copió tu código exactamente ---
# (he respetado nombres y fórmulas; eliminé sólo matplotlib.widgets
#  porque Streamlit provee los sliders/botones)
# --------------------------

def q_triangular(x_vals, tri_start, tri_end, w1, w2):
    q = np.zeros_like(x_vals)
    L_tri = tri_end - tri_start
    if L_tri <= 0:
        return q
    mask = (x_vals >= tri_start) & (x_vals <= tri_end)
    t = (x_vals[mask] - tri_start) / L_tri
    q[mask] = w1 + (w2 - w1) * t
    return q

def calcular_reacciones(L, b=None, w=0, P=0, a=0,
                        tri1_start=0, tri1_end=0, w1_1=0, w1_2=0,
                        tri2_start=0, tri2_end=0, w2_1=0, w2_2=0,
                        Ma=0, aM=0):
    if b is None or b <= 0:
        b = L

    # carga uniforme
    W_u = w * L
    x_u = L / 2

    # triangular 1
    L_tri1 = max(0, tri1_end - tri1_start)
    W_tri1 = 0
    x_c1 = 0
    if L_tri1 > 0 and (abs(w1_1) > 1e-12 or abs(w1_2) > 1e-12):
        W_tri1 = 0.5 * (w1_1 + w1_2) * L_tri1
        if abs(w1_1 + w1_2) < 1e-12:
            x_c1 = tri1_start + L_tri1 / 2
        else:
            x_c1 = tri1_start + L_tri1 * (w1_1 + 2 * w1_2) / (3 * (w1_1 + w1_2))

    # triangular 2
    L_tri2 = max(0, tri2_end - tri2_start)
    W_tri2 = 0
    x_c2 = 0
    if L_tri2 > 0 and (abs(w2_1) > 1e-12 or abs(w2_2) > 1e-12):
        W_tri2 = 0.5 * (w2_1 + w2_2) * L_tri2
        if abs(w2_1 + w2_2) < 1e-12:
            x_c2 = tri2_start + L_tri2 / 2
        else:
            x_c2 = tri2_start + L_tri2 * (w2_1 + 2 * w2_2) / (3 * (w2_1 + w2_2))

    # puntual
    W_p = P if abs(P) > 1e-12 else 0
    x_p = a

    # total
    W_total = W_u + W_tri1 + W_tri2 + W_p

    # momento alrededor de A (ahora INCLUYE Ma)
    M_about_A = W_u * x_u + W_tri1 * x_c1 + W_tri2 * x_c2 + W_p * x_p + Ma

    RB = M_about_A / b if abs(b) > 1e-12 else 0
    RA = W_total - RB

    return RA, RB


def cortar_y_momentear(L, b=None, w=0, P=0, a=0,
                       tri1_start=0, tri1_end=0, w1_1=0, w1_2=0,
                       tri2_start=0, tri2_end=0, w2_1=0, w2_2=0,
                       Ma=0, aM=0, n=2000):
    if b is None or b <= 0:
        b = L
    x = np.linspace(0, L, n)
    dx = x[1]-x[0]
    q_total = w*np.ones_like(x) + \
              q_triangular(x,tri1_start,tri1_end,w1_1,w1_2) + \
              q_triangular(x,tri2_start,tri2_end,w2_1,w2_2)

    RA, RB = calcular_reacciones(L,b,w,P,a,tri1_start,tri1_end,w1_1,w1_2,tri2_start,tri2_end,w2_1,w2_2,Ma,aM)

    V = np.zeros_like(x)
    V[0] = RA
    for i in range(1,n):
        V[i] = V[i-1] - q_total[i-1]*dx
        # carga puntual
        if abs(P) > 1e-12 and x[i-1] < a <= x[i]:
            V[i:] -= P
        # reacción RB
        if abs(RB) > 1e-12 and x[i-1] < b <= x[i]:
            V[i:] += RB   # ojo: sumar, no restar

    # momento
    M = np.zeros_like(x)
    for i in range(1,n):
        M[i] = M[i-1] + 0.5*(V[i]+V[i-1])*dx
        if abs(Ma) > 1e-12 and x[i-1] < aM <= x[i]:
            M[i:] += Ma

    return x, V, M

def curva_elastica(L, b=None, EI=1e7, w=0, P=0, a=0,
                   tri1_start=0, tri1_end=0, w1_1=0, w1_2=0,
                   tri2_start=0, tri2_end=0, w2_1=0, w2_2=0,
                   Ma=0, aM=0, n=2000):
    x, V, M = cortar_y_momentear(L,b,w,P,a,tri1_start,tri1_end,w1_1,w1_2,
                                 tri2_start,tri2_end,w2_1,w2_2,Ma,aM,n)
    dx = x[1]-x[0]
    theta = np.zeros_like(x)
    v = np.zeros_like(x)
    for i in range(1,n):
        theta[i] = theta[i-1] + 0.5*(M[i]+M[i-1])*dx/EI
        v[i] = v[i-1] + 0.5*(theta[i]+theta[i-1])*dx
    # corrección lineal para compatibilidad extremos
    alpha = (v[-1]-v[0])/L
    v -= (alpha*x + v[0])
    return x, v, theta

# --------------------------
# Datos de las vigas (20 vigas) - ahora con campo "b" opcional (por defecto L)
# --------------------------
vigas = {
    "Viga 1":  {"L":5.5, "b":5.5, "w":2100, "P":12000, "a":4.0,
                "tri1_start":0, "tri1_end":0, "w11":0, "w12":0,
                "tri2_start":0, "tri2_end":0, "w21":0, "w22":0,
                "M":0, "aM":0},
    "Viga 2":  {"L":5.5, "b":5.5, "w":2200, "P":0, "a":0,
                "tri1_start":4.0,"tri1_end":5.5,"w11":1200,"w12":0,
                "tri2_start":0, "tri2_end":0, "w21":0, "w22":0,
                "M":0, "aM":0},
    "Viga 3":  {"L":3.5, "b":3.5, "w":2200, "P":0, "a":0,
                "tri1_start":3.0,"tri1_end":3.5,"w11":0,"w12":1560,
                "tri2_start":0, "tri2_end":0, "w21":0, "w22":0,
                "M":0, "aM":0},
    "Viga 4":  {"L":4.45,"b":4.45, "w":1560, "P":0, "a":0,
                "tri1_start":0.0,"tri1_end":3.45,"w11":3560,"w12":0,
                "tri2_start":0, "tri2_end":0, "w21":0, "w22":0,
                "M":0, "aM":0},
    # ... (el resto igual, agregando "b":L por defecto)
    "Viga 5":  {"L":5.5, "b":4.0, "w":2100, "P":0, "a":0,
                "tri1_start":0, "tri1_end":0, "w11":0, "w12":0,
                "tri2_start":0, "tri2_end":0, "w21":0, "w22":0,
                "M":-12000, "aM":4.0},
    "Viga 6":  {"L":5.5, "b":4.0, "w":2200, "P":0, "a":0,
                "tri1_start":4.0, "tri1_end":5.5, "w11":1200, "w12":0,
                "tri2_start":0, "tri2_end":0, "w21":0, "w22":0,
                "M":0, "aM":0},
    "Viga 7":  {"L":3.5, "b":3.0, "w":2200, "P":0, "a":0,
                "tri1_start":3.0, "tri1_end":3.5, "w11":0, "w12":1560,
                "tri2_start":0, "tri2_end":0, "w21":0, "w22":0,
                "M":0, "aM":0},
    "Viga 8":  {"L":4.45, "b":3.45, "w":1560, "P":0, "a":0,
                "tri1_start":0.0, "tri1_end":3.45, "w11":3560, "w12":0,
                "tri2_start":0, "tri2_end":0, "w21":0, "w22":0,
                "M":0, "aM":0},
    "Viga 9":  {"L":5.5, "b":4.0, "w":2100, "P":0, "a":0,
                "tri1_start":4.0, "tri1_end":5.5, "w11":3100, "w12":3100,
                "tri2_start":0, "tri2_end":0, "w21":0, "w22":0,
                "M":0, "aM":0},
    "Viga 10": {"L":5.5, "b":4.0, "w":2200, "P":12000, "a":5.5,
                "tri1_start":0, "tri1_end":0, "w11":0, "w12":0,
                "tri2_start":0, "tri2_end":0, "w21":0, "w22":0,
                "M":0, "aM":0},
    "Viga 11": {"L":3.5, "b":3.0, "w":2200, "P":0, "a":0,
                "tri1_start":0, "tri1_end":0, "w11":0, "w12":0,
                "tri2_start":0, "tri2_end":0, "w21":0, "w22":0,
                "M":-8000, "aM":3.5},
    "Viga 12": {"L":4.45, "b":3.45, "w":1560, "P":0, "a":0,
                "tri1_start":0.0, "tri1_end":4.45, "w11":1560, "w12":0,
                "tri2_start":0, "tri2_end":0, "w21":0, "w22":0,
                "M":0, "aM":0},
    "Viga 13": {"L":5.5, "b":5.5, "w":2100, "P":15000, "a":4.0,
                "tri1_start":0.0, "tri1_end":4.0, "w11":0, "w12":4200,
                "tri2_start":4.0, "tri2_end":5.5, "w21":1200, "w22":0,
                "M":0, "aM":0},
    "Viga 14": {"L":5.5, "b":5.5, "w":2200, "P":0, "a":0,
                "tri1_start":0.0, "tri1_end":4.0, "w11":3360, "w12":0,
                "tri2_start":4.0, "tri2_end":5.5, "w21":1200, "w22":0,
                "M":0, "aM":0},
    "Viga 15": {"L":3.5, "b":3.5, "w":2200, "P":0, "a":0,
                "tri1_start":0.0, "tri1_end":3.0, "w11":3360, "w12":0,
                "tri2_start":3.0, "tri2_end":3.5, "w21":0, "w22":1560,
                "M":0, "aM":0},
    "Viga 16": {"L":4.45, "b":4.45, "w":0, "P":0, "a":0,
                "tri1_start":0.0, "tri1_end":4.45, "w11":3360, "w12":0,
                "tri2_start":3.45, "tri2_end":4.45, "w21":2560, "w22":0,
                "M":0, "aM":0},
    "Viga 17": {"L":5.5, "b":4.0, "w":2100, "P":15000, "a":4.0,
                "tri1_start":0.0, "tri1_end":4.0, "w11":0, "w12":4200,
                "tri2_start":4.0, "tri2_end":5.5, "w21":1200, "w22":0,
                "M":0, "aM":0},
    "Viga 18": {"L":5.5, "b":4.0, "w":2200, "P":0, "a":0,
                "tri1_start":0.0, "tri1_end":4.0, "w11":3360, "w12":0,
                "tri2_start":4.0, "tri2_end":5.5, "w21":1200, "w22":0,
                "M":0, "aM":0},
    "Viga 19": {"L":3.5, "b":3.0, "w":2200, "P":0, "a":0,
                "tri1_start":0.0, "tri1_end":3.0, "w11":3360, "w12":0,
                "tri2_start":3.0, "tri2_end":3.5, "w21":0, "w22":1560,
                "M":0, "aM":0},
    "Viga 20": {"L":4.45, "b":3.45, "w":0, "P":0, "a":0,
                "tri1_start":0.0, "tri1_end":4.45, "w11":3360, "w12":0,
                "tri2_start":3.45, "tri2_end":4.45, "w21":2560, "w22":0,
                "M":0, "aM":0}
}

def dibujar_momento_en_carga(ax, aM, Ma, L, altura_base):
    if Ma == 0:
        return
    x0 = aM
    y0 = altura_base
    if Ma < 0:
        conn = "arc3,rad=0.6"
    else:
        conn = "arc3,rad=-0.6"
    arr = FancyArrowPatch((x0 - L*0.08, y0), (x0 + L*0.08, y0),
                         connectionstyle=conn, arrowstyle='-|>', mutation_scale=12, linewidth=1.5)
    ax.add_patch(arr)
    ax.text(x0, y0 + 0.02*L, f"M={Ma:.0f}", ha='center', va='bottom', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))

def dibujar_reacciones(ax, RA, RB, b, L, altura_base):
    # dibuja RA en x=0 y RB en x=b (solo letras)
    ax.annotate("RA", xy=(0, altura_base), xytext=(0, altura_base + 0.05*L),
                ha='center', arrowprops=dict(arrowstyle='->'))
    ax.annotate("RB", xy=(b, altura_base), xytext=(b, altura_base + 0.05*L),
                ha='center', arrowprops=dict(arrowstyle='->'))

# --------------------------
# --- FIN de tu código original ---
# --------------------------

# --------------------------
# Now: Streamlit UI that uses the same variable names and functions above.
# --------------------------

st.set_page_config(layout="wide", page_title="Vigas - Interactivo (fiel al original)")

st.sidebar.title("Controles (idénticos)")
sel = st.sidebar.radio("Selecciona una viga", list(vigas.keys()))
params = vigas[sel]

# sliders / controles: mismos nombres / rangos parecidos a tu original
L = params["L"]
b = st.sidebar.slider("Pos. apoyo B (b)", 0.0, float(L), float(params.get("b", L)))
w = st.sidebar.slider("Carga uniforme", 0.0, 5000.0, float(params["w"]))
P = st.sidebar.slider("Carga puntual", 0.0, 20000.0, float(params["P"]))
a = st.sidebar.slider("Pos. puntual (a)", 0.0, float(L), float(params["a"]))
EI = st.sidebar.slider("EI", 1.0, 1e8, 1e7)

st.sidebar.markdown("### Triangular 1")
tri1_start = st.sidebar.slider("Tri1 x start", 0.0, float(L), float(params["tri1_start"]))
tri1_end   = st.sidebar.slider("Tri1 x end",   0.0, float(L), float(params["tri1_end"]))
w11 = st.sidebar.slider("Tri1 w1", 0.0, 5000.0, float(params.get("w11",0)))
w12 = st.sidebar.slider("Tri1 w2", 0.0, 5000.0, float(params.get("w12",0)))

st.sidebar.markdown("### Triangular 2")
tri2_start = st.sidebar.slider("Tri2 x start", 0.0, float(L), float(params["tri2_start"]))
tri2_end   = st.sidebar.slider("Tri2 x end",   0.0, float(L), float(params["tri2_end"]))
w21 = st.sidebar.slider("Tri2 w1", 0.0, 5000.0, float(params.get("w21",0)))
w22 = st.sidebar.slider("Tri2 w2", 0.0, 5000.0, float(params.get("w22",0)))

# Ma/aM se mantienen desde params (tal cual en tu código original)
Ma = params.get("M", 0)
aM = params.get("aM", 0)

# asegurar orden correcto (igual que en tu función actualizar)
if tri1_end < tri1_start:
    tri1_start, tri1_end = tri1_end, tri1_start
if tri2_end < tri2_start:
    tri2_start, tri2_end = tri2_end, tri2_start

# calcular
n = 2000
x, V, M = cortar_y_momentear(L, b, w, P, a,
                             tri1_start,tri1_end,w11,w12,
                             tri2_start,tri2_end,w21,w22,
                             Ma,aM,n)
xv, v, dv = curva_elastica(L, b, EI, w, P, a,
                           tri1_start,tri1_end,w11,w12,
                           tri2_start,tri2_end,w21,w22,
                           Ma,aM,n)
RA, RB = calcular_reacciones(L, b, w, P, a,
                             tri1_start,tri1_end,w11,w12,
                             tri2_start,tri2_end,w21,w22,Ma,aM)

# Resultados (misma info que el txt en tu figura)
st.subheader(f"Viga: {sel}")
st.markdown(f"- RA = {RA:.2f} @ 0")
st.markdown(f"- RB = {RB:.2f} @ {b:.2f}")
st.markdown(f"- |V|_max = {np.max(np.abs(V)):.2f}")
st.markdown(f"- |M|_max = {np.max(np.abs(M)):.2f}")
st.markdown(f"- v_max = {np.max(np.abs(v)):.6f}")

# Dibujar las 4 gráficas (idénticas en contenido)
fig, axes = plt.subplots(2,2, figsize=(12,8))
plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.07, hspace=0.3, wspace=0.25)
ax_carga, ax_V, ax_M, ax_v = axes[0,0], axes[0,1], axes[1,0], axes[1,1]

# diagrama de cargas (incluye triangulares y puntual)
ax_carga.plot(x, w * np.ones_like(x), label='Uniforme')
if tri1_end > tri1_start and (abs(w11) > 1e-12 or abs(w12) > 1e-12):
    tri_x = np.linspace(tri1_start, tri1_end, 200)
    tri_y = w11 + (w12 - w11) * (tri_x - tri1_start)/(tri1_end - tri1_start)
    ax_carga.fill_between(tri_x, 0, tri_y, color='orange', alpha=0.5, label='Triangular 1')
if tri2_end > tri2_start and (abs(w21) > 1e-12 or abs(w22) > 1e-12):
    tri_x2 = np.linspace(tri2_start, tri2_end, 200)
    tri_y2 = w21 + (w22 - w21) * (tri_x2 - tri2_start)/(tri2_end - tri2_start)
    ax_carga.fill_between(tri_x2, 0, tri_y2, color='peru', alpha=0.5, label='Triangular 2')
if P != 0 and 0 <= a <= L:
    ax_carga.plot([a,a],[0,max(P, w)],'r',label='Puntual')

altura_base = max(0.05 * L, 0.1 * max(np.max([w, w11, w12, w21, w22, P]), 1))
if Ma != 0:
    dibujar_momento_en_carga(ax_carga, aM, Ma, L, altura_base)
dibujar_reacciones(ax_carga, RA, RB, b, L, altura_base)

ax_carga.set_title("Diagrama de cargas")
ax_carga.grid(True)
ax_carga.legend(fontsize=8)

# cortante, momento, curva elástica
ax_V.plot(x, V, 'r'); ax_V.set_title("Diagrama de cortante V(x)"); ax_V.grid(True)
ax_M.plot(x, M, 'g'); ax_M.set_title("Diagrama de momento M(x)"); ax_M.grid(True)
ax_v.plot(xv, v, 'm'); ax_v.set_title("Curva elástica v(x)"); ax_v.grid(True)

st.pyplot(fig)

st.markdown("---")
st.caption("La implementación mantiene las funciones y cálculos del script original; los controles son equivalentes y actualizan las gráficas en tiempo real (interfaz web).")
