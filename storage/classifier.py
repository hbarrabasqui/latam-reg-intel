"""
Clasificador de dominio para documentos regulatorios.

Antes de indexar un documento en el RAG, este módulo decide a qué dominio
pertenece (telecom, electrico, etc.) o si debe descartarse.

Funciona por puntaje de palabras clave: si el texto contiene suficientes
términos del dominio, se clasifica ahí. Es simple, rápido, sin costo de API,
y fácil de auditar (podés ver exactamente por qué un documento entró o no).

Dominios disponibles:
  - "telecom"   → homologación, certificación, equipos, espectro, redes
  - "electrico" → seguridad eléctrica, baja tensión, IRAM, IEC (a futuro)
  - None        → no pertenece a ningún dominio conocido, se descarta
"""
from __future__ import annotations

import logging
import re

from storage.models import RegulatoryDocument

logger = logging.getLogger(__name__)

# ─── KEYWORDS POR DOMINIO ─────────────────────────────────────────────────────
# Cada entrada es (keyword, peso). Mayor peso = más relevante para el dominio.
# Los keywords se buscan sin distinguir mayúsculas ni acentos.

_TELECOM_KEYWORDS: list[tuple[str, int]] = [
    # Procesos de certificación/homologación
    ("homologaci", 3),       # cubre homologación/homologacao
    ("certificaci", 3),      # cubre certificación/certificacao
    ("habilitaci", 2),

    # Organismos y marcos legales típicos de telecom
    ("RAMATEL", 4),
    ("enacom-q", 4),          # normas técnicas ENACOM-Q2-XX
    ("norma técnica", 3),
    ("norma tecnica", 3),

    # Equipos y dispositivos
    ("equipo terminal", 3),
    ("dispositivo", 2),
    ("aparato", 1),
    ("interfaz", 2),

    # Tecnologías
    ("telecomunicaci", 3),   # cubre telecomunicaciones/telecomunicação
    ("radiocomunicaci", 3),
    ("radioelectrico", 2),
    ("radioeléctrico", 2),
    ("espectro", 2),
    ("frecuencia", 2),
    ("banda", 1),
    ("wifi", 3),
    ("bluetooth", 3),
    ("lte", 2),
    ("5g", 2),
    ("4g", 1),
    ("gsm", 2),
    ("umts", 2),
    ("wlan", 3),
    ("wan", 1),
    ("router", 3),
    ("modem", 2),
    ("módem", 2),
    ("antena", 2),
    ("transmisor", 2),
    ("receptor", 1),

    # Redes
    ("red de telecomunicaci", 3),
    ("internet", 1),
    ("banda ancha", 2),
    ("fibra óptica", 2),
    ("fibra optica", 2),

    # Interferencias y compatibilidad electromagnética
    ("interferencia", 2),
    ("compatibilidad electromagnética", 3),
    ("compatibilidad electromagnetica", 3),
    ("cem", 2),
    ("sar", 2),              # exposición a campos electromagnéticos

    # Marco legal telecom Argentina
    ("ley 27.078", 3),
    ("ley 27078", 3),
    ("decreto 267", 2),

    # Términos en portugués (Brasil/Anatel)
    ("homologaç", 3),
    ("certificaç", 3),
    ("anatel", 3),
    ("produto de telecomunicaç", 4),
    ("resolução", 1),

    # Marco legal telecom Chile (SUBTEL)
    ("subtel", 4),
    ("alcance reducido", 3),
    ("p.i.r.e", 2),
    ("pire", 1),
    ("ley 18.168", 3),
    ("ley 18168", 3),
]

_TELECOM_STRONG_SIGNALS: list[str] = [
    # Términos lo bastante específicos de telecom como para no aparecer por
    # casualidad en normas de otros rubros (gas, tránsito, penal, laboral,
    # etc.). Se exige al menos uno de estos ADEMÁS del puntaje por keywords
    # genéricas — esto es lo que filtra falsos positivos como "certificación"
    # o "habilitación" o "aparato", que aparecen en cualquier norma administrativa.
    "telecomunicaci", "radiocomunicaci", "radioelectric", "radioeléctric",
    "espectro radioel", "homologaci", "enacom-q", "ramatel",
    "wifi", "bluetooth", "wlan", "lte", "umts", "gsm",
    "antena", "transmisor", "transceptor",
    "banda ancha", "fibra óptic", "fibra optic",
    "equipo terminal", "compatibilidad electromagnét", "compatibilidad electromagnet",
    "ondas métricas", "ondas metricas", "ondas decimétricas", "ondas decimetricas",
    "red de telecomunicaci",
    "homologaç", "certificaç", "anatel", "produto de telecomunicaç",
    "subtel", "alcance reducido",
]

_ELECTRICO_KEYWORDS: list[tuple[str, int]] = [
    ("seguridad eléctrica", 4),
    ("seguridad electrica", 4),
    ("baja tensión", 3),
    ("baja tension", 3),
    ("iram", 3),
    ("iec 60", 3),
    ("norma iram", 3),
    ("instalación eléctrica", 3),
    ("instalacion electrica", 3),
    ("disyuntor", 2),
    ("protección eléctrica", 2),
    ("proteccion electrica", 2),
]

_DOMAIN_KEYWORDS: dict[str, list[tuple[str, int]]] = {
    "telecom": _TELECOM_KEYWORDS,
    "electrico": _ELECTRICO_KEYWORDS,
}

# Umbral mínimo de puntaje para clasificar en un dominio
_MIN_SCORE: dict[str, int] = {
    "telecom": 5,
    "electrico": 6,
}

# ─── PALABRAS QUE EXCLUYEN el dominio telecom ─────────────────────────────────
# Si el documento menciona SOLO estos temas sin keywords de telecom, se descarta.
_TELECOM_EXCLUSION_SIGNALS: list[str] = [
    "servicio postal",
    "correo postal",
    "encomienda",
    "franqueo",
    "radiodifusión sonora",
    "radiodifusion sonora",
    "servicio de radiodifusión",
    "contenidos audiovisuales",
    "señal de televisión",
    "senal de television",
    # Documentos administrativos del Estado (GDE, modernización)
    "gestion documental",
    "expediente electronico",
    "modernizacion administrativa",
]


# ─── CLASIFICADOR ─────────────────────────────────────────────────────────────

class DomainClassifier:
    """
    Clasifica un documento regulatorio en un dominio temático.

    Ejemplo de uso:
        clf = DomainClassifier()
        domain = clf.classify(doc)
        if domain == "telecom":
            # indexar en el RAG de telecom
        elif domain is None:
            # descartar
    """

    def classify(self, doc: RegulatoryDocument) -> str | None:
        """
        Devuelve el nombre del dominio ("telecom", "electrico", etc.)
        o None si el documento no pertenece a ningún dominio conocido.
        """
        text = _normalize(doc.raw_text + " " + doc.title)

        scores: dict[str, int] = {}
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            scores[domain] = _score_text(text, keywords)

        # Log para auditoría (muy útil para debugging y para el TP)
        logger.debug(
            "Clasificación '%s %s': %s",
            doc.organism,
            doc.number,
            scores,
        )

        # Encontrar el dominio con mayor puntaje que supere el umbral
        best_domain = max(scores, key=lambda d: scores[d])
        best_score = scores[best_domain]

        if best_score < _MIN_SCORE.get(best_domain, 5):
            logger.info(
                "DESCARTADO '%s %s' — puntaje insuficiente: %s",
                doc.organism,
                doc.number,
                scores,
            )
            return None

        # Verificar señales de exclusión para telecom
        if best_domain == "telecom" and _has_exclusion_signals(text, scores["telecom"]):
            logger.info(
                "DESCARTADO '%s %s' — señales de exclusión (postal/audiovisual)",
                doc.organism,
                doc.number,
            )
            return None

        # Exigir al menos un término fuerte y específico de telecom. Sin esto,
        # normas de otros rubros (gas, tránsito, penal, laboral) que solo usan
        # palabras administrativas genéricas ("certificación", "habilitación",
        # "aparato") pasaban el umbral por puntaje acumulado sin tener nada que
        # ver con telecomunicaciones.
        if best_domain == "telecom" and not any(sig in text for sig in _TELECOM_STRONG_SIGNALS):
            logger.info(
                "DESCARTADO '%s %s' — puntaje suficiente pero sin término fuerte de telecom: %s",
                doc.organism,
                doc.number,
                scores,
            )
            return None

        logger.info(
            "CLASIFICADO '%s %s' → dominio='%s' (puntaje=%d)",
            doc.organism,
            doc.number,
            best_domain,
            best_score,
        )
        return best_domain

    def explain(self, doc: RegulatoryDocument) -> dict:
        """
        Devuelve el detalle del proceso de clasificación.
        Útil para debugging y para el TP (mostrar cómo funciona).
        """
        text = _normalize(doc.raw_text + " " + doc.title)
        result = {}
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            matched = [kw for kw, _ in keywords if re.search(r"\b" + re.escape(kw.lower()), text)]
            score = _score_text(text, keywords)
            result[domain] = {
                "score": score,
                "threshold": _MIN_SCORE.get(domain, 5),
                "matched_keywords": matched,
                "classifies": score >= _MIN_SCORE.get(domain, 5),
            }
        return result


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase y eliminación de acentos para búsqueda robusta."""
    text = text.lower()
    replacements = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "ã": "a", "ê": "e", "ô": "o", "ç": "c", "ñ": "n",
        "ü": "u",
    }
    for accented, plain in replacements.items():
        text = text.replace(accented, plain)
    return text


def _score_text(text: str, keywords: list[tuple[str, int]]) -> int:
    """Suma los pesos de los keywords que aparecen en el texto.
    Usa word boundary al inicio para evitar matches dentro de otras palabras."""
    score = 0
    for keyword, weight in keywords:
        if re.search(r"\b" + re.escape(keyword.lower()), text):
            score += weight
    return score


def _has_exclusion_signals(text: str, telecom_score: int) -> bool:
    """
    Devuelve True si el documento tiene señales de exclusión Y el puntaje
    de telecom no es lo suficientemente alto como para compensarlas.
    """
    exclusion_count = sum(1 for signal in _TELECOM_EXCLUSION_SIGNALS if signal in text)
    # Si hay señales de exclusión fuertes y el puntaje telecom es bajo, descartar
    return exclusion_count >= 1 and telecom_score < 15
