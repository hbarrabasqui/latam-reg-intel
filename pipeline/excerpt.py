"""Extracción de un fragmento "saliente" de un texto largo: el preámbulo
(fecha/organismo, útil de contexto aunque sea boilerplate) más una ventana
alrededor de la primera mención de un término fuerte de dominio, sin importar
qué tan profundo esté en el documento.

Por qué hace falta esto y no alcanza con un límite fijo de caracteres: las
resoluciones de ENACOM suelen tener un preámbulo largo ("VISTO... los
Decretos... CONSIDERANDO...") antes de llegar al contenido sustantivo — se
encontró un caso real (Resolución 25/2026, acreditación de un laboratorio)
donde la mención relevante ("Registro de Laboratorios Acreditados",
"homologación") aparecía recién en el carácter 1724, muy después de
cualquier límite fijo razonable. Buscar el término en todo el documento (no
solo al principio) es la única forma de garantizar que se vea el contenido
real cuando existe, sin mandar el documento entero.

Extraído de `scripts/semantic_cleanup.py` (donde se usa para armar el
excerpt que ve el juez Haiku) y generalizado el 5/8/2026 para reusarlo
también en `pipeline/embedder.py` — el mismo problema de fondo (preámbulo
largo antes del contenido sustantivo) afecta al modelo de embeddings, que
trunca internamente a ~128 tokens (~600-700 caracteres): un chunk de varios
miles de caracteres solo "veía" el preámbulo administrativo, nunca el
contenido real, y por eso no había señal semántica para preguntas puntuales
como "¿de qué trata la Resolución 25/26?"."""
from __future__ import annotations

_ACCENT_MAP = str.maketrans("áéíóúãêôçñü", "aeiouaeocnu")

PREAMBLE_CHARS = 300
MATCH_WINDOW_BEFORE = 120
MATCH_WINDOW_AFTER = 350


def normalize_for_search(text: str) -> str:
    return text.lower().translate(_ACCENT_MAP)


def build_salient_excerpt(
    raw_text: str,
    signal_terms: list[str],
    preamble_chars: int = PREAMBLE_CHARS,
    window_before: int = MATCH_WINDOW_BEFORE,
    window_after: int = MATCH_WINDOW_AFTER,
) -> str:
    """Arma un excerpt: el inicio del texto (preámbulo) MÁS una ventana
    alrededor de la primera mención de algún término de `signal_terms` en
    TODO el texto. Si no hay match, o el match cae dentro del preámbulo,
    devuelve solo el preámbulo."""
    preamble = raw_text[:preamble_chars].replace("\n", " ")
    normalized = normalize_for_search(raw_text)

    match_pos = None
    for term in signal_terms:
        idx = normalized.find(normalize_for_search(term))
        if idx != -1 and (match_pos is None or idx < match_pos):
            match_pos = idx

    if match_pos is None or match_pos < preamble_chars:
        return preamble

    start = max(0, match_pos - window_before)
    end = min(len(raw_text), match_pos + window_after)
    snippet = raw_text[start:end].replace("\n", " ")
    return f"{preamble} [...] {snippet}"
