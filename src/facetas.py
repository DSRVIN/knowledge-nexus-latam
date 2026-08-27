"""Vocabulario controlado de facetas TEMA / MÉTODO / DOMINIO, construido
desde el propio dataset (nunca hardcodeado) y aplicado de forma idéntica
a las entidades y a la consulta del evaluador.

Un mismo término puede aparecer en más de una faceta (p. ej. "series
temporales" es a la vez keyword de proyecto, application_domain de
investigador y valor de methodological_expertise). No se deduplica entre
facetas: se etiqueta según el campo de origen, que es la lectura honesta
del dato (Documento Técnico §3).

Hallazgo relevante para el diseño: 22 de las 42 necesidades institucionales
(NEED-021..042) usan una plantilla de descripción que NO enumera términos
explícitos ("La institución requiere consolidar información distribuida
para {título}..."), a diferencia de NEED-001..020. Verificado: sus títulos
casi no comparten vocabulario léxico con keywords de proyectos/tesis. Por
eso extraer_facetas() debe aceptar con naturalidad un resultado disperso o
vacío en esos casos — es una señal real, no un bug a forzar. El score
semántico (embeddings) es quien carga el peso ahí.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field

from src import config
from src.ingesta import RepositorioInstitucional


def normalizar(texto: str) -> str:
    texto = (texto or "").lower().strip()
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c))


def _compilar(terminos: set[str]) -> list[tuple[str, re.Pattern]]:
    return [(t, re.compile(r"\b" + re.escape(t) + r"\b")) for t in terminos if t]


@dataclass
class VocabularioFacetas:
    tema: set[str] = field(default_factory=set)
    metodo: set[str] = field(default_factory=set)
    dominio: set[str] = field(default_factory=set)
    _patrones: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._patrones = {
            "tema": _compilar(self.tema),
            "metodo": _compilar(self.metodo),
            "dominio": _compilar(self.dominio),
        }

    def extraer(self, texto: str) -> dict[str, set[str]]:
        t = normalizar(texto)
        return {
            faceta: {term for term, patron in patrones if patron.search(t)}
            for faceta, patrones in self._patrones.items()
        }

    def to_dict(self) -> dict:
        return {
            "tema": sorted(self.tema),
            "metodo": sorted(self.metodo),
            "dominio": sorted(self.dominio),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VocabularioFacetas":
        return cls(tema=set(d["tema"]), metodo=set(d["metodo"]), dominio=set(d["dominio"]))


def construir_vocabulario(repo: RepositorioInstitucional) -> VocabularioFacetas:
    tema: set[str] = set()
    for tipo, campo in [
        ("PROJECT", "keywords"),
        ("THESIS", "keywords"),
        ("PUBLICATION", "keywords"),
        ("SUBJECT", "main_topics"),
    ]:
        for entidad in repo.por_tipo(tipo):
            for valor in entidad.valores_multivalor(campo):
                if valor:
                    tema.add(normalizar(valor))
    # research_lines.keywords no es multivalor: ver config.CAMPOS_FRAGMENTADOS_POR_PALABRA.
    # Cada línea aporta una única frase-tema ya reconstruida por ingesta.py.
    for entidad in repo.por_tipo("LINE"):
        valor = entidad.valor("keywords")
        if valor:
            tema.add(normalizar(valor))

    metodo: set[str] = set()
    for entidad in repo.por_tipo("RESEARCHER"):
        for valor in entidad.valores_multivalor("methodological_expertise"):
            if valor:
                metodo.add(normalizar(valor))

    dominio: set[str] = set()
    for entidad in repo.por_tipo("RESEARCHER"):
        for valor in entidad.valores_multivalor("application_domains"):
            if valor:
                dominio.add(normalizar(valor))
    for entidad in repo.por_tipo("CAPABILITY"):
        for valor in entidad.valores_multivalor("application_domains"):
            if valor:
                dominio.add(normalizar(valor))
    for tipo in ("PROJECT", "THESIS"):
        for entidad in repo.por_tipo(tipo):
            valor = entidad.valor("application_context")
            if valor:
                dominio.add(normalizar(valor))

    return VocabularioFacetas(tema=tema, metodo=metodo, dominio=dominio)


def guardar_vocabulario(vocabulario: VocabularioFacetas, ruta=None) -> None:
    ruta = ruta or config.RUTA_VOCABULARIO_FACETAS
    ruta.write_text(
        json.dumps(vocabulario.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def cargar_vocabulario(ruta=None) -> VocabularioFacetas:
    ruta = ruta or config.RUTA_VOCABULARIO_FACETAS
    return VocabularioFacetas.from_dict(json.loads(ruta.read_text(encoding="utf-8")))


def facetas_por_entidad(
    repo: RepositorioInstitucional, vocabulario: VocabularioFacetas
) -> dict[str, dict[str, set[str]]]:
    return {eid: vocabulario.extraer(e.texto_indexable()) for eid, e in repo.entidades.items()}
