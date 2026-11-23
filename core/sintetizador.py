"""Módulo de síntesis ejecutiva e interpretación de gráficos con Google Gemini."""

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv(override=True)
os.environ.pop("GOOGLE_API_KEY", None)
logger = logging.getLogger(__name__)

MODELO_GEMINI = "gemini-2.5-flash"
TIMEOUT_SEGUNDOS = 10.0
TIMEOUT_MILISEGUNDOS = 10000

MENSAJE_IA_NO_DISPONIBLE = (
    "Consulta a la IA no disponible. Verifica tu conexión a internet o la clave GEMINI_API_KEY."
)


class _RespuestaContextoSchema(BaseModel):
    """Estructura JSON esperada desde Gemini para el encabezado del análisis."""

    titulo_reporte: str
    resumen_ejecutivo: str


def _normalizar_texto_plano(texto: str, max_lineas: int = 3) -> str:
    """Limpia marcas Markdown y une el texto en un máximo de líneas."""
    sin_md = texto.replace("**", "").replace("__", "").replace("`", "")
    lineas = [
        linea.lstrip("#-*• ").strip()
        for linea in sin_md.strip().splitlines()
        if linea.strip()
    ]
    if not lineas:
        return ""
    return " ".join(lineas[:max_lineas])


def _respuesta_ia_no_disponible() -> dict[str, Any]:
    """Retorna el estado de aviso cuando la consulta a Gemini no está disponible."""
    return {
        "titulo_reporte": "Diagnóstico Estructural del Dataset",
        "resumen_ejecutivo": MENSAJE_IA_NO_DISPONIBLE,
        "ia_disponible": False,
    }


def _invocar_gemini(
    api_key: str,
    esquema: list[dict[str, Any]],
    muestra_filas: list[dict[str, Any]],
) -> dict[str, Any]:
    """Envía el esquema y 3 filas de muestra a Gemini con límite de 10 segundos."""
    payload = {
        "esquema": esquema,
        "muestra_filas": muestra_filas[:3],
    }
    prompt = (
        "Actúa como Analista de Datos Senior. Analiza el esquema y las 3 filas "
        "de muestra proporcionadas y responde en JSON con dos campos: "
        "'titulo_reporte' (título técnico claro en español latinoamericano) y "
        "'resumen_ejecutivo' (máximo 3 líneas describiendo qué contiene el dataset "
        "y qué aspectos de calidad como valores nulos o duplicados se observan).\n\n"
        f"Datos:\n{json.dumps(payload, ensure_ascii=False, default=str)}"
    )

    cliente = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=TIMEOUT_MILISEGUNDOS,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    respuesta = cliente.models.generate_content(
        model=MODELO_GEMINI,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=_RespuestaContextoSchema,
        ),
    )

    texto_respuesta = getattr(respuesta, "text", None) or ""
    if not texto_respuesta.strip():
        raise ValueError("La respuesta de Google GenAI llegó vacía.")

    datos = json.loads(texto_respuesta)
    titulo = str(datos.get("titulo_reporte", "")).strip()
    resumen = _normalizar_texto_plano(
        str(datos.get("resumen_ejecutivo", "")), max_lineas=3
    )

    if not titulo or not resumen:
        raise ValueError(
            "El JSON devuelto por Google GenAI no contiene campos válidos."
        )

    return {
        "titulo_reporte": titulo,
        "resumen_ejecutivo": resumen,
        "ia_disponible": True,
    }


def generar_contexto_ejecutivo(
    auditoria: dict[str, Any], muestra_filas: list[dict[str, Any]]
) -> dict[str, Any]:
    """Genera título y resumen con Gemini (timeout 10s) o aviso de no disponible."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "tu_api_key_aqui":
        return _respuesta_ia_no_disponible()

    esquema = auditoria.get("esquema", [])
    muestra = muestra_filas[:3] if isinstance(muestra_filas, list) else []

    try:
        return _invocar_gemini(api_key, esquema, muestra)
    except Exception as exc:
        logger.warning("Consulta a Gemini no disponible: %s", exc)
        return _respuesta_ia_no_disponible()


def _invocar_gemini_grafico(
    api_key: str,
    titulo_reporte: str,
    tipo_grafico: str,
    datos_grafico: dict[str, Any],
) -> str:
    """Consulta Gemini para interpretar una gráfica estadística en máximo 3 líneas."""
    payload = {
        "contexto_dataset": titulo_reporte,
        "tipo_grafico": tipo_grafico,
        "metricas_grafica": datos_grafico,
    }
    prompt = (
        "Actúa como Analista de Datos Senior. Redacta una interpretación clara, "
        "precisa y directa en español latinoamericano basada en las métricas "
        "exactas de esta gráfica. Menciona las cifras principales y qué significan "
        "en la práctica. Responde en texto plano sin Markdown, sin listas "
        "y en máximo 3 líneas.\n\n"
        f"Datos:\n{json.dumps(payload, ensure_ascii=False, default=str)}"
    )

    cliente = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=TIMEOUT_MILISEGUNDOS,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    respuesta = cliente.models.generate_content(
        model=MODELO_GEMINI,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    texto = _normalizar_texto_plano(
        getattr(respuesta, "text", None) or "", max_lineas=3
    )
    if not texto:
        raise ValueError("La respuesta de interpretación llegó vacía.")
    return texto


def explicar_grafico_ia(
    contexto_dataset: dict[str, Any],
    tipo_grafico: str,
    datos_grafico: dict[str, Any],
) -> str:
    """Interpreta una gráfica con Gemini (timeout 10s) o retorna aviso de error."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "tu_api_key_aqui":
        return MENSAJE_IA_NO_DISPONIBLE

    titulo_reporte = str(
        contexto_dataset.get("titulo_reporte", "Análisis de Dataset")
    )
    try:
        return _invocar_gemini_grafico(
            api_key,
            titulo_reporte,
            tipo_grafico,
            datos_grafico,
        )
    except Exception as exc:
        logger.warning(
            "Consulta a Gemini no disponible al explicar gráfico (%s): %s",
            tipo_grafico,
            exc,
        )
        return MENSAJE_IA_NO_DISPONIBLE
