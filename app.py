from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DATA = Path(__file__).parent / "data"

# Los Spaces de Gradio gratuitos corren en ZeroGPU, que exige declarar al menos una función con
# @spaces.GPU. Esta app NO usa GPU (solo lee predicciones ya calculadas): la función de abajo
# nunca se llama, así que no consume cuota de GPU. Solo sirve para pasar la comprobación de arranque.
try:
    import spaces

    @spaces.GPU(duration=1)
    def _zerogpu_noop():
        return None
except ImportError:  # en local o en hardware sin ZeroGPU
    pass

# ---------------------------------------------------------------- datos
d = pd.read_parquet(DATA / "valenbisi_demo_predicciones.parquet")
d["estacion"] = d["estacion"].astype(str)
d = d.sort_values("ts_origen", kind="stable").reset_index(drop=True)
TS = d["ts_origen"].to_numpy()

info = pd.read_parquet(DATA / "valenbisi_estaciones.parquet")
info.index = info.index.astype(str)

FECHAS = sorted(pd.Series(pd.unique(d["ts_origen"])).dt.strftime("%Y-%m-%d").unique())
ESTACIONES = sorted(d["estacion"].unique(), key=lambda s: (len(s), s))
CHOICES_EST = [(f"Estación {s} (capacidad {int(info.capacidad.get(s, 0))})", s) for s in ESTACIONES]
HORAS = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]  # los datos de la demo están cada 30 min
FECHA_DEF = "2025-03-17" if "2025-03-17" in FECHAS else FECHAS[len(FECHAS) // 2]


def dia(fecha):
    """Filas de un día (d está ordenado por tiempo)."""
    ini = np.datetime64(pd.Timestamp(fecha), "ns")
    fin = ini + np.timedelta64(1, "D")
    lo, hi = np.searchsorted(TS, [ini, fin])
    return d.iloc[lo:hi]


def figura_vacia(texto):
    fig = go.Figure()
    fig.add_annotation(text=texto, showarrow=False, font=dict(size=16))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


# ---------------------------------------------------------- pestaña 1
def estacion_dia(fecha, est):
    dd = dia(fecha)
    dd = dd[dd["estacion"] == est]
    if dd.empty:
        vacio = figura_vacia("Sin datos para esta estación y día")
        return vacio, vacio, "No hay datos ese día (faltan algunos días en el histórico). Elige otro."
    x = dd["ts_origen"] + pd.Timedelta(hours=1)

    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=x, y=dd["real"], name="Real", line=dict(color="#222", width=3)))
    f1.add_trace(go.Scatter(x=x, y=dd["pred_gru"], name="Predicción GRU", line=dict(color="#e4572e", width=2)))
    f1.add_trace(go.Scatter(x=x, y=dd["pred_base4"], name="Baseline (persistencia + cambio típico)",
                            line=dict(color="#2e86ab", width=1.5, dash="dot")))
    f1.add_trace(go.Scatter(x=x, y=dd["ahora"], name="Persistencia (lo de ahora)",
                            line=dict(color="#999", width=1.5, dash="dash")))
    f1.update_layout(title=f"Bicis disponibles: real frente a lo previsto 1 hora antes (estación {est})",
                     xaxis_title="Hora", yaxis_title="Bicis", height=420,
                     legend=dict(orientation="h", y=-0.25), margin=dict(t=50))

    f2 = go.Figure(go.Scatter(x=x, y=dd["prob_vacia"], fill="tozeroy", line=dict(color="#c0392b"),
                              name="Riesgo"))
    f2.update_layout(title="Riesgo estimado de quedarse sin bicis dentro de 1 hora",
                     xaxis_title="Hora", yaxis_title="Riesgo (0 a 1)", yaxis_range=[0, 1],
                     height=300, margin=dict(t=50))

    mae = lambda c: float(np.abs(dd[c] - dd["real"]).mean())
    vacias = int((dd["real"] == 0).sum())
    texto = (f"**Error medio ese día** · GRU: **{mae('pred_gru'):.2f}** bicis · "
             f"Baseline: {mae('pred_base4'):.2f} · Persistencia: {mae('ahora'):.2f}\n\n"
             f"La estación estuvo realmente vacía en {vacias} de {len(dd)} consultas (una cada 30 min).\n\n"
             "_En un solo día y una estación el resultado es ruidoso: el modelo no gana siempre. "
             "Las cifras globales están en la pestaña 'Sobre el proyecto'._")
    return f1, f2, texto


# ---------------------------------------------------------- pestaña 2
def mapa_riesgo(fecha, hora):
    ts = pd.Timestamp(f"{fecha} {hora}")
    dd = dia(fecha)
    r = dd[dd["ts_origen"] == ts].copy()
    if r.empty:
        return figura_vacia("Sin datos en ese instante"), "No hay datos en ese instante. Prueba otra hora u otro día.", pd.DataFrame()
    r = r.join(info[["latitude", "longitude"]], on="estacion").dropna(subset=["latitude", "longitude"])
    r["Riesgo %"] = (100 * r["prob_vacia"]).round(0)
    r["Estación"] = r["estacion"]

    fig = px.scatter(r, x="longitude", y="latitude", color="Riesgo %", range_color=(0, 50),
                     color_continuous_scale="RdYlGn_r", hover_name="Estación",
                     hover_data={"latitude": False, "longitude": False, "ahora": True, "Riesgo %": True},
                     height=560)
    fig.update_traces(marker=dict(size=13, line=dict(width=0.5, color="#333")))
    fig.update_xaxes(title="Longitud", showgrid=True)
    fig.update_yaxes(title="Latitud", showgrid=True, scaleanchor="x",
                     scaleratio=1 / np.cos(np.radians(39.47)))  # proporciones reales de la ciudad
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))

    top = r.sort_values("prob_vacia", ascending=False).head(10)
    aciertos = int((top["real"] == 0).sum())
    hora_obj = (ts + pd.Timedelta(hours=1)).strftime("%H:%M")
    texto = (f"### Riesgo de quedarse sin bicis a las {hora_obj} (consulta a las {hora})\n"
             f"**De las 10 estaciones con más riesgo, {aciertos} estaban realmente vacías una hora después.**\n\n"
             f"Tasa base: en ese momento, {int((r['real'] == 0).sum())} de {len(r)} estaciones estaban vacías.")
    tabla = pd.DataFrame({
        "Estación": top["estacion"].values,
        "Bicis ahora": top["ahora"].round(0).astype(int).values,
        "Previsión GRU (1 h)": top["pred_gru"].astype("float64").round(1).values,
        "Riesgo": (100 * top["prob_vacia"]).round(0).astype(int).astype(str).values + " %",
        "Real (1 h después)": top["real"].round(0).astype(int).values,
    })
    return fig, texto, tabla


# ---------------------------------------------------------- pestaña 3
SOBRE = """
### Qué hace
Una red neuronal recurrente (GRU) lee las últimas 4 horas de cada estación de Valenbisi y estima
cuántas bicis habrá dentro de **1 hora**, y si la estación se quedará **sin bicis**.

### Resultados en test (marzo-septiembre de 2025, 2,6 millones de casos)
| Método | Error medio (bicis) |
|---|---|
| Persistencia ("lo mismo que ahora") | 1,548 |
| Baseline (persistencia + cambio típico) | 1,499 |
| LightGBM | 1,430 |
| GRU (media de 5 entrenamientos) | 1,416 |
| Conjunto de 5 GRU (el de esta demo) | 1,414 |

Detección de estaciones que hoy tienen bicis y en 1 h se quedan vacías (4,8 % de los casos):
precisión media de 31 % para GRU y LightGBM, frente a 25 % del baseline.

### Limitaciones
- **El objetivo inicial (reducir un 15 % el error del mejor baseline) no se alcanzó**: la mejora es de ~5,5 %.
- La GRU y LightGBM empatan en la práctica: cada uno gana en una métrica, por menos de un 1 %.
- La detección de vaciados es útil pero limitada: con un 50 % de recall, solo ~3 de cada 10 avisos aciertan.
- Los saltos bruscos (reposiciones de bicis, averías) no son predecibles con estos datos.
- Faltan unos 78 días del histórico (por ejemplo, las Fallas de 2024).
- No usa meteorología ni eventos.
- Los datos terminan en septiembre de 2025; esta demo **no es en directo**.
- El objetivo de alerta temprana se definió tras la exploración de datos, no al inicio.

### Fuentes
Datos: Ajuntament de València (portal de datos abiertos, CC-BY 4.0), recopilados por el autor del dataset
`femartip/valencia-urban-mobility` en Hugging Face.
"""

# ---------------------------------------------------------- interfaz
with gr.Blocks(title="Valenbisi: alerta de estaciones vacías") as demo:
    gr.Markdown("# 🚲 Valenbisi: ¿qué estaciones se quedarán sin bicis en la próxima hora?\n"
                "Demo con datos históricos de test (mar-sep 2025). Es una repetición de datos pasados, "
                "**no una predicción en directo**.")
    fecha = gr.Dropdown(FECHAS, value=FECHA_DEF, label="Día",
                        info="Prueba el 17 de marzo de 2025 (Fallas) o un día laborable cualquiera")

    with gr.Tabs():
        with gr.Tab("📈 Estación y día"):
            est = gr.Dropdown(CHOICES_EST, value=ESTACIONES[0], label="Estación")
            p1 = gr.Plot()
            t1 = gr.Markdown()
            p2 = gr.Plot()
            for c in (fecha, est):
                c.change(estacion_dia, [fecha, est], [p1, p2, t1])

        with gr.Tab("🗺️ Mapa de riesgo") as tab_mapa:
            hora = gr.Dropdown(HORAS, value="08:00", label="Hora de la consulta")
            t2 = gr.Markdown()
            mapa = gr.Plot()
            tabla = gr.Dataframe(label="Las 10 estaciones con más riesgo", interactive=False)
            for c in (fecha, hora):
                c.change(mapa_riesgo, [fecha, hora], [mapa, t2, tabla])
            tab_mapa.select(mapa_riesgo, [fecha, hora], [mapa, t2, tabla])

        with gr.Tab("ℹ️ Sobre el proyecto"):
            gr.Markdown(SOBRE)

    demo.load(estacion_dia, [fecha, est], [p1, p2, t1])
    demo.load(mapa_riesgo, [fecha, hora], [mapa, t2, tabla])

if __name__ == "__main__":
    demo.launch()