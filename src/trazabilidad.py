"""Núcleo de trazabilidad: toda afirmación del sistema debe poder citar de
dónde viene. Procedencia y Evidencia son los únicos vehículos permitidos
para eso; ningún otro módulo debe inventar su propio formato de cita.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Procedencia:
    """Apunta a un campo concreto de un registro concreto en Data V1.0."""

    capa: str          # "03_knowledge_needs"
    archivo: str       # "projects.csv"
    registro_id: str   # "PRJ-081"
    campo: str         # "methodology"

    def cita(self) -> str:
        return f"Data V1.0 / {self.archivo} / {self.registro_id} / {self.campo}"

    def to_dict(self) -> dict:
        return {
            "capa": self.capa,
            "archivo": self.archivo,
            "registro_id": self.registro_id,
            "campo": self.campo,
            "cita": self.cita(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Procedencia":
        return cls(capa=d["capa"], archivo=d["archivo"], registro_id=d["registro_id"], campo=d["campo"])


@dataclass
class Evidencia:
    """Un fragmento verificable que sustenta una conexión u oportunidad.

    tipo="recuperada": el texto es literal del dataset (el caso normal).
    tipo="inferida": la relación se dedujo (p. ej. semántica/grafo), pero el
        texto citado sigue siendo literal del dataset, no generado.
    tipo="generada": contenido producido por el narrador LLM opcional; nunca
        se cuenta como evidencia institucional y la UI lo distingue siempre.
    """

    procedencia: Procedencia
    texto: str
    tipo: str = "recuperada"

    def __post_init__(self) -> None:
        if self.tipo not in {"recuperada", "inferida", "generada"}:
            raise ValueError(f"tipo de evidencia inválido: {self.tipo!r}")
        if self.tipo == "generada" and self.procedencia.archivo != "narrador_llm":
            raise ValueError(
                "una Evidencia tipo 'generada' no puede citar procedencia institucional"
            )

    def to_dict(self) -> dict:
        return {
            "procedencia": self.procedencia.to_dict(),
            "texto": self.texto,
            "tipo": self.tipo,
        }


@dataclass
class DesgloseScore:
    """Cada sumando de S(Q,E), guardado para que la UI y el comparador
    'por qué A antes que B' puedan mostrarlo componente a componente.
    """

    componentes: dict = field(default_factory=dict)
    penalizaciones: dict = field(default_factory=dict)
    total: float = 0.0

    def to_dict(self) -> dict:
        return {
            "componentes": dict(self.componentes),
            "penalizaciones": dict(self.penalizaciones),
            "total": round(self.total, 4),
        }
