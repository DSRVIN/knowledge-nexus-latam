"""Asistente conversacional para la pestaña "Asistente" del dashboard.

Diseño: el LLM (vía OpenRouter) NUNCA decide qué es relevante ni inventa
conexiones — solo narra, en tono cálido y sin jerga técnica, resultados que
ya calculó el motor real (src/motor.py, src/oportunidades.py). El contexto
que se le pasa al modelo es siempre evidencia real del dataset; se le
instruye explícitamente a no salirse de ahí.

Robustez para la demo en vivo: se intenta una lista de modelos gratuitos de
OpenRouter en cascada (el roster de modelos :free rota sin aviso). Si todos
fallan o no hay API key configurada, se cae a una narración determinista de
respaldo sobre los MISMOS resultados reales — la pestaña nunca deja de
funcionar por depender de un tercero.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import requests

from src import config

_SYSTEM_PROMPT = """Sos el asistente conversacional de Knowledge Nexus LATAM, una herramienta que ayuda a estudiantes y personal administrativo de una universidad a descubrir conexiones útiles entre necesidades institucionales, proyectos, tesis, investigadores, grupos y oportunidades académicas.

Tu tono: cálido, cercano, en español, tuteando de forma natural, sin tecnicismos innecesarios. Muchas personas que te escriben nunca usaron una herramienta así y pueden sentirse inseguras — tu trabajo es darles confianza, no abrumarlas. Evitá palabras como "score", "embedding", "vector" o "ranking" salvo que te pregunten explícitamente cómo funciona el sistema.

Reglas que NO podés romper bajo ninguna circunstancia:
1. Solo podés mencionar personas, proyectos, tesis, grupos u oportunidades que aparezcan literalmente en el CONTEXTO que te paso. Nunca inventes un nombre, un ID o un dato que no esté ahí — es la regla más importante de todas.
2. Si el contexto no tiene resultados relevantes para lo que te preguntan, decilo con honestidad ("no encontré antecedentes claros para esto") y sugerí reformular la consulta con otras palabras. Nunca inventes una respuesta solo para parecer útil.
3. Sé breve: 3 a 5 oraciones, con los nombres más relevantes destacados. No escribas un ensayo.
4. Cerrá siempre recordando, en una frase corta, que el detalle completo y la evidencia exacta de cada dato están disponibles en las otras pestañas del panel (Conexiones, Grafo, Oportunidades)."""


@dataclass
class RespuestaAsistente:
    texto: str
    generado_por_ia: bool
    citas: list[str] = field(default_factory=list)


def _llamar_openrouter(mensajes: list[dict]) -> str | None:
    if not config.ASISTENTE_HABILITADO:
        return None
    for modelo in config.OPENROUTER_MODELOS:
        try:
            respuesta = requests.post(
                config.OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"model": modelo, "messages": mensajes, "temperature": 0.4, "max_tokens": 500},
                timeout=config.OPENROUTER_TIMEOUT_SEG,
            )
            if respuesta.status_code != 200:
                continue  # modelo saturado/no disponible: probar el siguiente de la lista
            texto = respuesta.json()["choices"][0]["message"]["content"]
            if texto and texto.strip():
                return texto.strip()
        except (requests.RequestException, KeyError, IndexError, ValueError):
            continue
    return None


def _resumen_items(items: list[dict], max_items: int = 4) -> str:
    lineas = []
    for item in items[:max_items]:
        destino = item["destino"]
        lineas.append(
            f"- {destino['nombre']} ({destino['tipo']}, id {destino['id']}), "
            f"confianza {item['banda']} (score {item['relevancia']:.2f}). {item['explicacion']}"
        )
    return "\n".join(lineas) if lineas else "(sin resultados en esta categoría)"


def construir_contexto(resultados_a: list[dict], investigadores: list[dict], oportunidades: list[dict]) -> str:
    partes = [
        "ANTECEDENTES (proyectos/tesis/publicaciones/líneas de investigación):",
        _resumen_items(resultados_a),
        "",
        "INVESTIGADORES relacionados (por autoría o dirección real, no solo similitud):",
        _resumen_items(investigadores),
        "",
        "OPORTUNIDADES institucionales generadas a partir de lo anterior:",
    ]
    if oportunidades:
        for o in oportunidades[:4]:
            partes.append(f"- [{o['type']}] {o['opportunity']}: {o['reason']}")
    else:
        partes.append("(sin oportunidades generadas para esta consulta)")
    return "\n".join(partes)


def _citas_principales(resultados_a: list[dict], investigadores: list[dict], max_citas: int = 5) -> list[str]:
    citas: list[str] = []
    for item in resultados_a + investigadores:
        for cita in item.get("sources", []):
            if cita not in citas:
                citas.append(cita)
        if len(citas) >= max_citas:
            break
    return citas[:max_citas]


def _respuesta_respaldo(resultados_a: list[dict], investigadores: list[dict], oportunidades: list[dict]) -> str:
    """Narración cálida sin IA, sobre los mismos resultados reales. Se usa si
    OpenRouter no está configurado o todos los modelos de la lista fallan."""
    if not resultados_a and not investigadores:
        return (
            "No encontré antecedentes claros para lo que me contaste. ¿Podrías darme un poco más "
            "de detalle o probar con otras palabras? A veces ayuda mencionar el área (por ejemplo "
            "salud, educación, tecnología) o el tipo de problema que querés resolver."
        )
    partes = ["¡Encontré algunas conexiones que pueden servirte!"]
    if resultados_a:
        top = resultados_a[0]
        partes.append(
            f"El antecedente más relevante que encontré es \"{top['destino']['nombre']}\" "
            f"(confianza {top['banda'].lower()})."
        )
    if investigadores:
        top_inv = investigadores[0]
        partes.append(f"La persona más conectada con este tema parece ser {top_inv['destino']['nombre']}.")
    if oportunidades:
        partes.append(
            f"También identifiqué {len(oportunidades)} oportunidad(es) institucional(es), "
            f"como \"{oportunidades[0]['opportunity']}\"."
        )
    partes.append(
        "Podés ver el detalle completo y la evidencia exacta de cada dato en las pestañas de "
        "Conexiones y Oportunidades."
    )
    return " ".join(partes)


def generar_respuesta_asistente(
    mensaje_usuario: str,
    resultados_a: list[dict],
    investigadores: list[dict],
    oportunidades: list[dict],
    historial: list[dict] | None = None,
) -> RespuestaAsistente:
    contexto = construir_contexto(resultados_a, investigadores, oportunidades)
    citas = _citas_principales(resultados_a, investigadores)

    mensajes = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for turno in (historial or [])[-4:]:
        mensajes.append(turno)
    mensajes.append(
        {
            "role": "user",
            "content": f"Consulta de la persona: {mensaje_usuario}\n\nCONTEXTO (resultados reales ya encontrados por el sistema):\n{contexto}",
        }
    )

    texto = _llamar_openrouter(mensajes)
    if texto:
        return RespuestaAsistente(texto=texto, generado_por_ia=True, citas=citas)
    return RespuestaAsistente(
        texto=_respuesta_respaldo(resultados_a, investigadores, oportunidades),
        generado_por_ia=False,
        citas=citas,
    )
