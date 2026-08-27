"""Dashboard interactivo: consulta -> conexiones priorizadas -> evidencia ->
oportunidades. Cumple la secuencia mínima de demostración de la Guía Oficial
§5: consultar/procesar -> descubrir relación -> valorar pertinencia ->
consultar evidencia -> generar oportunidad.

El motor (grafo + embeddings + facetas) se construye UNA vez al arrancar el
proceso (module-level), no por request: instanciarlo de nuevo en cada
callback recargaría ~3200 vectores innecesariamente.
"""
from __future__ import annotations

import os

import dash
import dash_bootstrap_components as dbc
import networkx as nx
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

from src import config
from src.embeddings import obtener_o_construir_indice
from src.explicacion import enriquecer_con_explicacion
from src.facetas import cargar_vocabulario
from src.grafo import GrafoConocimiento
from src.ingesta import RepositorioInstitucional
from src.motor import TIPOS_DOCUMENTALES, MotorConexiones
from src.oportunidades import GeneradorOportunidades

# --------------------------------------------------------------------- #
# Estado del proceso: construido una sola vez al arrancar
# --------------------------------------------------------------------- #
print("Cargando Data V1.0 y el índice de conocimiento...")
_repo = RepositorioInstitucional().cargar()
_grafo = GrafoConocimiento.cargar()
_vocabulario = cargar_vocabulario()
_textos = {eid: e.texto_indexable() for eid, e in _repo.entidades.items()}
_indice = obtener_o_construir_indice(_textos)
_motor = MotorConexiones(_repo, _grafo, _indice, _vocabulario)
print(
    f"Listo: {len(_repo.entidades)} entidades, motor semántico "
    f"{'disponible' if _indice.motor_semantico_disponible else 'NO disponible (fallback TF-IDF)'}."
)

_NECESIDADES = sorted(_repo.por_tipo("NEED"), key=lambda e: e.id)
_OPCIONES_NECESIDAD = [{"label": f"{n.id} — {n.valor('title')}", "value": n.id} for n in _NECESIDADES]

_COLOR_TIPO = {
    "PROJECT": "#2E86AB", "THESIS": "#A23B72", "PUBLICATION": "#F18F01", "LINE": "#6A994E",
    "RESEARCHER": "#C73E1D", "GROUP": "#8338EC", "SUBJECT": "#3A86FF", "CAPABILITY": "#5E548E",
}

_ETIQUETA_TIPO = {
    "PROJECT": "Proyecto", "THESIS": "Tesis", "PUBLICATION": "Publicación", "LINE": "Línea",
    "RESEARCHER": "Investigador", "GROUP": "Grupo", "SUBJECT": "Asignatura", "CAPABILITY": "Capacidad",
}

_COLOR_BANDA = {"Alta": "success", "Media": "warning", "Baja": "secondary"}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "Knowledge Nexus LATAM"
server = app.server


# --------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------- #
def _slider_peso(id_, etiqueta, valor):
    return html.Div(
        [
            html.Small(etiqueta, className="text-muted"),
            dcc.Slider(id=id_, min=0, max=1, step=0.05, value=valor, marks=None,
                       tooltip={"placement": "right", "always_visible": False}),
        ],
        className="mb-2",
    )


sidebar = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Consulta", className="card-title"),
            dcc.Dropdown(
                id="dd-necesidad",
                options=_OPCIONES_NECESIDAD,
                placeholder="Cargar una necesidad institucional (opcional)...",
                className="mb-2",
            ),
            dcc.Textarea(
                id="txt-consulta",
                placeholder="Escribe cualquier problema institucional nuevo, en tus propias palabras...",
                style={"width": "100%", "height": "110px"},
                className="mb-3",
            ),
            dbc.Button("Buscar", id="btn-buscar", color="primary", className="w-100 mb-3"),
            html.Hr(),
            html.H6("Tipos de antecedente"),
            dcc.Checklist(
                id="chk-tipos",
                options=[{"label": f" {_ETIQUETA_TIPO[t]}", "value": t} for t in TIPOS_DOCUMENTALES],
                value=list(TIPOS_DOCUMENTALES),
                className="mb-3",
            ),
            html.H6("Pesos del score (re-rankean en vivo)"),
            _slider_peso("sl-w_sem", "Semántico (embeddings)", config.PESOS_SCORE["w_sem"]),
            _slider_peso("sl-w_lex", "Léxico (TF-IDF)", config.PESOS_SCORE["w_lex"]),
            _slider_peso("sl-w_tema", "Tema", config.PESOS_SCORE["w_tema"]),
            _slider_peso("sl-w_dominio", "Dominio", config.PESOS_SCORE["w_dominio"]),
            _slider_peso("sl-w_metodo", "Método", config.PESOS_SCORE["w_metodo"]),
            _slider_peso("sl-w_evidencia", "Fuerza de evidencia", config.PESOS_SCORE["w_evidencia"]),
            html.Hr(),
            html.Div(
                [html.Small("Resultados por categoría: "), dcc.Input(id="in-topk", type="number", value=8, min=3, max=20, style={"width": "60px"})],
            ),
        ]
    ),
    className="mb-3",
)

badge_motor = dbc.Alert(
    [
        html.B("Motor: "),
        "embeddings locales en español (BETO, ONNX, offline)" if _indice.motor_semantico_disponible
        else "TF-IDF (fallback — motor semántico no disponible en esta máquina)",
        html.Span(f"  ·  {len(_repo.entidades)} entidades  ·  {_grafo.g.number_of_edges()} aristas explícitas", className="text-muted"),
    ],
    color="info" if _indice.motor_semantico_disponible else "warning",
    className="py-2 mb-3",
)

app.layout = dbc.Container(
    fluid=True,
    children=[
        dcc.Store(id="store-resultados"),
        html.H3("Knowledge Nexus LATAM", className="mt-3"),
        html.P("Conectar el conocimiento: de una necesidad institucional a oportunidades priorizadas y trazables.", className="text-muted"),
        badge_motor,
        dbc.Row(
            [
                dbc.Col(sidebar, md=3),
                dbc.Col(
                    [
                        dbc.Tabs(
                            id="tabs",
                            active_tab="tab-conexiones",
                            children=[
                                dbc.Tab(label="Conexiones", tab_id="tab-conexiones"),
                                dbc.Tab(label="Grafo", tab_id="tab-grafo"),
                                dbc.Tab(label="Oportunidades", tab_id="tab-oportunidades"),
                                dbc.Tab(label="¿Por qué A antes que B?", tab_id="tab-comparador"),
                            ],
                        ),
                        html.Div(id="contenido-tab", className="mt-3"),
                    ],
                    md=9,
                ),
            ]
        ),
    ],
)


# --------------------------------------------------------------------- #
# Autocompletar consulta al elegir una necesidad del catálogo
# --------------------------------------------------------------------- #
@app.callback(Output("txt-consulta", "value"), Input("dd-necesidad", "value"))
def _autocompletar_consulta(necesidad_id):
    if not necesidad_id:
        return dash.no_update
    necesidad = _repo.entidades[necesidad_id]
    return f"{necesidad.valor('title')}. {necesidad.valor('description')}"


# --------------------------------------------------------------------- #
# Búsqueda principal: consulta -> conexiones -> oportunidades
# --------------------------------------------------------------------- #
@app.callback(
    Output("store-resultados", "data"),
    Input("btn-buscar", "n_clicks"),
    State("txt-consulta", "value"),
    State("dd-necesidad", "value"),
    State("chk-tipos", "value"),
    State("in-topk", "value"),
    State("sl-w_sem", "value"), State("sl-w_lex", "value"), State("sl-w_tema", "value"),
    State("sl-w_dominio", "value"), State("sl-w_metodo", "value"), State("sl-w_evidencia", "value"),
    prevent_initial_call=True,
)
def _buscar(n_clicks, texto, necesidad_id, tipos, top_k, w_sem, w_lex, w_tema, w_dominio, w_metodo, w_evidencia):
    if not texto or not texto.strip():
        return dash.no_update
    top_k = int(top_k or config.TOP_K_DEFECTO)
    tipos = tuple(tipos) if tipos else TIPOS_DOCUMENTALES
    pesos = {
        "w_sem": w_sem, "w_lex": w_lex, "w_tema": w_tema,
        "w_dominio": w_dominio, "w_metodo": w_metodo, "w_evidencia": w_evidencia,
    }
    prioridad = _repo.entidades[necesidad_id].valor("priority") if necesidad_id else "MEDIUM"

    resultados_a = _motor.buscar(texto, tipos_destino=tipos, top_k=top_k, pesos=pesos)
    investigadores = _motor.propagar_a_investigadores(resultados_a, texto, top_k=top_k)
    grupos = _motor.propagar_a_grupos(resultados_a, texto, top_k=top_k)
    asignaturas = _motor.articular_curriculo(resultados_a, texto, top_k=top_k)
    capacidades = _motor.emparejar_capacidades(texto, top_k=top_k)

    todas = resultados_a + investigadores + grupos + asignaturas + capacidades
    enriquecer_con_explicacion(todas, _repo)

    oportunidades = GeneradorOportunidades(_repo, _motor).generar(
        prioridad, resultados_a, investigadores, capacidades, asignaturas
    )

    return {
        "consulta": texto,
        "resultados_a": [c.to_dict() for c in resultados_a],
        "investigadores": [c.to_dict() for c in investigadores],
        "grupos": [c.to_dict() for c in grupos],
        "asignaturas": [c.to_dict() for c in asignaturas],
        "capacidades": [c.to_dict() for c in capacidades],
        "oportunidades": [o.to_dict() for o in oportunidades],
    }


# --------------------------------------------------------------------- #
# Render por pestaña
# --------------------------------------------------------------------- #
def _grafico_desglose(desglose: dict) -> go.Figure:
    componentes = desglose["componentes"]
    etiquetas = list(componentes.keys())
    valores = list(componentes.values())
    colores = ["#2E86AB" if v >= 0 else "#C73E1D" for v in valores]
    fig = go.Figure(go.Bar(x=valores, y=etiquetas, orientation="h", marker_color=colores))
    fig.update_layout(
        height=40 * max(3, len(etiquetas)) + 40, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="contribución al score", template="plotly_white",
    )
    return fig


def _tarjeta_conexion(c: dict, idx_key: str):
    destino = c["destino"]
    titulo = f"{destino['nombre']}  ·  {_ETIQUETA_TIPO.get(destino['tipo'], destino['tipo'])}  ·  {destino['id']}"
    return dbc.AccordionItem(
        [
            html.P(c["explicacion"]),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=_grafico_desglose(c["desglose"]), config={"displayModeBar": False}), md=6),
                    dbc.Col(
                        [
                            html.H6("Evidencia (trazabilidad)"),
                            html.Ul([html.Li([html.Code(ev["procedencia"]["cita"]), html.Br(), html.Small(ev["texto"][:220])]) for ev in c["evidencia"]]),
                        ],
                        md=6,
                    ),
                ]
            ),
        ],
        title=[
            dbc.Badge(c["banda"], color=_COLOR_BANDA.get(c["banda"], "secondary"), className="me-2"),
            dbc.Badge(f"{c['relevancia']:.3f}", color="dark", className="me-2"),
            dbc.Badge(c["naturaleza"], color="light", text_color="dark", className="me-2 border"),
            titulo,
        ],
        item_id=idx_key,
    )


def _render_conexiones(data: dict):
    if not data:
        return dbc.Alert("Escribe una consulta o elige una necesidad del catálogo y presiona Buscar.", color="light")
    secciones = [
        ("Antecedentes (proyectos / tesis / publicaciones / líneas)", data["resultados_a"]),
        ("Investigadores (propagado por autoría real)", data["investigadores"]),
        ("Grupos de investigación", data["grupos"]),
        ("Currículo (asignaturas)", data["asignaturas"]),
        ("Capacidades institucionales (inferido)", data["capacidades"]),
    ]
    bloques = []
    for titulo_sec, items in secciones:
        if not items:
            continue
        bloques.append(html.H6(titulo_sec, className="mt-3"))
        bloques.append(
            dbc.Accordion(
                [_tarjeta_conexion(c, f"{titulo_sec}-{i}") for i, c in enumerate(items)],
                start_collapsed=True,
                flush=True,
            )
        )
    return html.Div(bloques) if bloques else dbc.Alert("Sin resultados para esta consulta.", color="light")


def _render_oportunidades(data: dict):
    if not data or not data["oportunidades"]:
        return dbc.Alert("Sin oportunidades generadas para esta consulta todavía.", color="light")
    tarjetas = []
    for o in data["oportunidades"]:
        color = {"Alta": "success", "Media": "warning", "Baja": "secondary"}.get(o["priority"], "secondary")
        tarjetas.append(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div([dbc.Badge(o["type"], color="dark", className="me-2"), dbc.Badge(o["priority"], color=color)]),
                        html.H6(o["opportunity"], className="mt-2"),
                        html.P(o["reason"]),
                        html.Small("Entidades: " + ", ".join(o["related_entities"]), className="text-muted d-block"),
                        html.Ul([html.Li(html.Code(ev["procedencia"]["cita"])) for ev in o["evidence"]], className="mt-2"),
                    ]
                ),
                className="mb-2",
            )
        )
    return html.Div(tarjetas)


def _figura_grafo(data: dict) -> go.Figure:
    g = nx.Graph()
    g.add_node("CONSULTA", tipo="CONSULTA", nombre="Tu consulta")
    todas = data["resultados_a"] + data["investigadores"] + data["grupos"] + data["asignaturas"] + data["capacidades"]
    for c in todas:
        destino = c["destino"]
        if destino["id"] not in g:
            g.add_node(destino["id"], tipo=destino["tipo"], nombre=destino["nombre"])
        if not g.has_edge("CONSULTA", destino["id"]):
            g.add_edge("CONSULTA", destino["id"], naturaleza=c["naturaleza"])

    # aristas explícitas reales del grafo institucional entre nodos ya presentes
    incluidos = set(g.nodes())
    for u, v, datos in _grafo.g.edges(data=True):
        if u in incluidos and v in incluidos and u != v and not g.has_edge(u, v):
            g.add_edge(u, v, naturaleza="explicita")

    pos = nx.spring_layout(g, seed=42, k=0.7)

    fig = go.Figure()
    estilo_linea = {"explicita": ("solid", "#2E86AB"), "mixta": ("dash", "#8338EC"), "inferida": ("dot", "#bbb")}
    for naturaleza, (dash_estilo, color) in estilo_linea.items():
        xs, ys = [], []
        for u, v, datos in g.edges(data=True):
            if datos["naturaleza"] != naturaleza:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            xs += [x0, x1, None]
            ys += [y0, y1, None]
        if xs:
            fig.add_trace(
                go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=1.5, dash=dash_estilo, color=color),
                           hoverinfo="skip", showlegend=False)
            )

    paleta = dict(_COLOR_TIPO, CONSULTA="#111111")
    for tipo, color in paleta.items():
        xs, ys, hover, textos = [], [], [], []
        for n, d in g.nodes(data=True):
            if d["tipo"] != tipo:
                continue
            x, y = pos[n]
            xs.append(x)
            ys.append(y)
            hover.append(f"{d['nombre']} ({n})")
            textos.append("" if tipo != "CONSULTA" else "CONSULTA")
        if xs:
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="markers+text",
                    marker=dict(size=26 if tipo == "CONSULTA" else 15, color=color,
                                symbol="diamond" if tipo == "CONSULTA" else "circle",
                                line=dict(width=1, color="white")),
                    text=textos, textposition="top center", hovertext=hover, hoverinfo="text",
                    name=_ETIQUETA_TIPO.get(tipo, tipo),
                )
            )

    fig.update_layout(
        showlegend=True, height=620, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor="white",
    )
    return fig


def _render_grafo(data: dict):
    if not data:
        return dbc.Alert("Ejecuta una búsqueda primero.", color="light")
    return html.Div(
        [
            html.P(
                "Rombo negro = tu consulta. Línea sólida = relación explícita del dataset. "
                "Línea punteada = inferida. Pasa el mouse sobre un nodo para ver su nombre.",
                className="text-muted",
            ),
            dcc.Graph(figure=_figura_grafo(data), config={"displayModeBar": False}),
        ]
    )


def _render_comparador(data: dict):
    if not data:
        return dbc.Alert("Ejecuta una búsqueda primero.", color="light")
    todas = data["resultados_a"] + data["investigadores"] + data["grupos"] + data["asignaturas"] + data["capacidades"]
    if len(todas) < 2:
        return dbc.Alert("Se necesitan al menos 2 resultados para comparar.", color="light")
    opciones = [{"label": f"{c['destino']['nombre']} ({c['destino']['id']}) — {c['relevancia']:.3f}", "value": i} for i, c in enumerate(todas)]
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(dcc.Dropdown(id="dd-comp-a", options=opciones, value=0), md=6),
                    dbc.Col(dcc.Dropdown(id="dd-comp-b", options=opciones, value=1 if len(opciones) > 1 else 0), md=6),
                ]
            ),
            html.Div(id="contenido-comparador", className="mt-3"),
        ]
    )


@app.callback(Output("contenido-tab", "children"), Input("tabs", "active_tab"), Input("store-resultados", "data"))
def _renderizar_tab(tab, data):
    if tab == "tab-conexiones":
        return _render_conexiones(data)
    if tab == "tab-grafo":
        return _render_grafo(data)
    if tab == "tab-oportunidades":
        return _render_oportunidades(data)
    if tab == "tab-comparador":
        return _render_comparador(data)
    return html.Div()


@app.callback(
    Output("contenido-comparador", "children"),
    Input("dd-comp-a", "value"), Input("dd-comp-b", "value"), State("store-resultados", "data"),
)
def _comparar(idx_a, idx_b, data):
    if idx_a is None or idx_b is None or not data:
        return dash.no_update
    todas = data["resultados_a"] + data["investigadores"] + data["grupos"] + data["asignaturas"] + data["capacidades"]
    a, b = todas[idx_a], todas[idx_b]
    claves = sorted(set(a["desglose"]["componentes"]) | set(b["desglose"]["componentes"]))
    filas = []
    for k in claves:
        va = a["desglose"]["componentes"].get(k, 0.0)
        vb = b["desglose"]["componentes"].get(k, 0.0)
        filas.append(html.Tr([html.Td(k), html.Td(f"{va:.3f}"), html.Td(f"{vb:.3f}"), html.Td(f"{va - vb:+.3f}")]))
    tabla = dbc.Table(
        [html.Thead(html.Tr([html.Th("Componente"), html.Th("A"), html.Th("B"), html.Th("Diferencia")]))]
        + [html.Tbody(filas)],
        bordered=True, size="sm",
    )
    ganador = a["destino"]["nombre"] if a["relevancia"] >= b["relevancia"] else b["destino"]["nombre"]
    return html.Div(
        [
            dbc.Alert(
                f"{ganador} queda primero (A={a['relevancia']:.3f} vs B={b['relevancia']:.3f}). "
                "La tabla muestra qué componente explica la diferencia.",
                color="info",
            ),
            tabla,
        ]
    )


if __name__ == "__main__":
    # Hugging Face Spaces (Docker) espera 0.0.0.0:7860 por convención;
    # PORT y DASH_DEBUG son configurables para otros hosts (Render, Railway, local).
    _puerto = int(os.environ.get("PORT", 7860))
    _debug = os.environ.get("DASH_DEBUG", "false").strip().lower() == "true"
    app.run(host="0.0.0.0", port=_puerto, debug=_debug)
