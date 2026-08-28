import pytest

from src import asistente, config

RESULTADO_A = {
    "destino": {"id": "PRJ-001", "tipo": "PROJECT", "nombre": "Análisis de permanencia estudiantil"},
    "tipo_relacion": "antecedente_relevante",
    "relevancia": 0.48,
    "banda": "Media",
    "naturaleza": "inferida",
    "explicacion": "Antecedente relevante por similitud semántica.",
    "desglose": {"componentes": {}, "penalizaciones": {}, "total": 0.48},
    "evidencia": [],
    "sources": ["Data V1.0 / projects.csv / PRJ-001 / methodology"],
}
INVESTIGADOR = {
    "destino": {"id": "INV-001", "tipo": "RESEARCHER", "nombre": "Daniel Moreno Valencia"},
    "tipo_relacion": "propagada_por_autoria",
    "relevancia": 0.55,
    "banda": "Alta",
    "naturaleza": "explicita",
    "explicacion": "Autor principal del proyecto relacionado.",
    "desglose": {"componentes": {}, "penalizaciones": {}, "total": 0.55},
    "evidencia": [],
    "sources": ["Data V1.0 / researcher_project.csv / INV-001|PRJ-001 / role"],
}
OPORTUNIDAD = {
    "opportunity": "Continuidad investigativa a partir de PRJ-001",
    "type": "RESEARCH_CONTINUITY",
    "related_entities": ["PRJ-001", "INV-001"],
    "reason": "El proyecto tiene un investigador activo con grupo vigente.",
    "priority": "Media",
    "evidence": [],
}


def test_asistente_deshabilitado_sin_api_key(monkeypatch):
    monkeypatch.setattr(config, "ASISTENTE_HABILITADO", False)
    respuesta = asistente.generar_respuesta_asistente("necesito ayuda con deserción estudiantil", [], [], [])
    assert respuesta.generado_por_ia is False
    assert "no encontré" in respuesta.texto.lower()


def test_respaldo_determinista_con_resultados_reales(monkeypatch):
    monkeypatch.setattr(config, "ASISTENTE_HABILITADO", False)
    respuesta = asistente.generar_respuesta_asistente(
        "deserción estudiantil", [RESULTADO_A], [INVESTIGADOR], [OPORTUNIDAD]
    )
    assert respuesta.generado_por_ia is False
    assert "Análisis de permanencia estudiantil" in respuesta.texto
    assert "Daniel Moreno Valencia" in respuesta.texto
    assert respuesta.citas == ["Data V1.0 / projects.csv / PRJ-001 / methodology", "Data V1.0 / researcher_project.csv / INV-001|PRJ-001 / role"]


def test_construir_contexto_incluye_las_tres_categorias():
    contexto = asistente.construir_contexto([RESULTADO_A], [INVESTIGADOR], [OPORTUNIDAD])
    assert "Análisis de permanencia estudiantil" in contexto
    assert "Daniel Moreno Valencia" in contexto
    assert "RESEARCH_CONTINUITY" in contexto


def test_usa_ia_cuando_openrouter_responde(monkeypatch):
    monkeypatch.setattr(config, "ASISTENTE_HABILITADO", True)
    monkeypatch.setattr(asistente, "_llamar_openrouter", lambda mensajes: "¡Hola! Encontré justo lo que buscabas.")
    respuesta = asistente.generar_respuesta_asistente("deserción estudiantil", [RESULTADO_A], [], [])
    assert respuesta.generado_por_ia is True
    assert respuesta.texto == "¡Hola! Encontré justo lo que buscabas."


def test_cae_a_respaldo_si_openrouter_falla(monkeypatch):
    monkeypatch.setattr(config, "ASISTENTE_HABILITADO", True)
    monkeypatch.setattr(asistente, "_llamar_openrouter", lambda mensajes: None)
    respuesta = asistente.generar_respuesta_asistente("deserción estudiantil", [RESULTADO_A], [], [])
    assert respuesta.generado_por_ia is False
    assert "Análisis de permanencia estudiantil" in respuesta.texto


def test_llamar_openrouter_prueba_siguiente_modelo_si_uno_falla(monkeypatch):
    monkeypatch.setattr(config, "ASISTENTE_HABILITADO", True)
    monkeypatch.setattr(config, "OPENROUTER_MODELOS", ["modelo-a:free", "modelo-b:free"])
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "fake-key")

    llamadas = []

    class RespuestaFalsa:
        def __init__(self, status_code, payload=None):
            self.status_code = status_code
            self._payload = payload or {}

        def json(self):
            return self._payload

    def post_falso(url, headers, json, timeout):
        llamadas.append(json["model"])
        if json["model"] == "modelo-a:free":
            return RespuestaFalsa(429)  # simula rate limit
        return RespuestaFalsa(200, {"choices": [{"message": {"content": "respuesta del segundo modelo"}}]})

    monkeypatch.setattr(asistente.requests, "post", post_falso)
    resultado = asistente._llamar_openrouter([{"role": "user", "content": "hola"}])
    assert resultado == "respuesta del segundo modelo"
    assert llamadas == ["modelo-a:free", "modelo-b:free"]
