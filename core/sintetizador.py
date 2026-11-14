"""Módulo de síntesis ejecutiva contextual con Google GenAI y respaldo local."""

from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv(override=False)

logger = logging.getLogger(__name__)

MODELO_GEMINI = "gemini-2.5-flash"
TIMEOUT_SEGUNDOS = 4.0
TIMEOUT_MILISEGUNDOS = 4000


class _RespuestaContextoSchema(BaseModel):
    """Esquema estructurado esperado desde la API de Google GenAI."""

    titulo_reporte: str
    resumen_ejecutivo: str


def _limitar_lineas(texto: str, max_lineas: int = 3) -> str:
    """Normaliza saltos de línea y recorta el resumen a un máximo de 3 líneas."""
    lineas = [linea.strip() for linea in texto.strip().splitlines() if linea.strip()]
    if not lineas:
        return ""
    return " ".join(lineas[:max_lineas])


def _generar_respaldo_determinista(
    auditoria: dict[str, Any],
) -> dict[str, str]:
    """Construye un título y resumen técnico local sin conexión externa."""
    filas = int(auditoria.get("filas", 0))
    columnas = int(auditoria.get("columnas", 0))
    duplicados = int(auditoria.get("duplicados", 0))
    esquema = auditoria.get("esquema", [])

    total_nulos = sum(int(col.get("nulos", 0)) for col in esquema)
    cols_numericas = [
        str(col.get("columna", ""))
        for col in esquema
        if col.get("tipo_logico") == "numerica"
    ]
    cols_categoricas = [
        str(col.get("columna", ""))
        for col in esquema
        if col.get("tipo_logico") in {"categorica", "fecha"}
    ]

    outliers_iqr = auditoria.get("outliers_iqr", {})
    total_outliers = sum(
        int(info.get("cantidad", 0))
        for info in outliers_iqr.values()
        if isinstance(info, dict)
    )

    muestra_num = ", ".join(cols_numericas[:3]) if cols_numericas else "sin métricas"
    muestra_cat = (
        ", ".join(cols_categoricas[:3]) if cols_categoricas else "sin dimensiones"
    )

    titulo_reporte = (
        f"Reporte Técnico de Auditoría y Análisis ({filas} registros, "
        f"{columnas} variables)"
    )
    resumen_ejecutivo = (
        f"El conjunto de datos analizado comprende {filas} filas y {columnas} "
        f"columnas (métricas: {muestra_num}; dimensiones: {muestra_cat}).\n"
        f"La inspección estructural identificó {total_nulos} valor(es) nulo(s), "
        f"{duplicados} registro(s) duplicado(s) y {total_outliers} valor(es) "
        f"atípico(s) bajo el criterio IQR.\n"
        "Tras aplicar el flujo funcional de depuración, los indicadores descriptivos "
        "y visualizaciones quedan habilitados para la toma de decisiones."
    )

    return {
        "titulo_reporte": titulo_reporte,
        "resumen_ejecutivo": resumen_ejecutivo,
    }


def _invocar_gemini(
    api_key: str, esquema: list[dict[str, Any]], muestra_filas: list[dict[str, Any]]
) -> dict[str, str]:
    """Ejecuta la consulta a Gemini enviando únicamente esquema y 3 filas."""
    payload = {
        "esquema": esquema,
        "muestra_filas": muestra_filas[:3],
    }
    prompt = (
        "Actúa como Arquitecto y Analista de Datos Senior. Analiza únicamente "
        "el esquema y las 3 filas de muestra proporcionadas y responde en JSON "
        "con exactamente dos campos: 'titulo_reporte' (título técnico y formal "
        "en español latinoamericano) y 'resumen_ejecutivo' (máximo 3 líneas, "
        "tono técnico y formal describiendo el dominio y estructura del dataset).\n\n"
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
    resumen = _limitar_lineas(str(datos.get("resumen_ejecutivo", "")), max_lineas=3)

    if not titulo or not resumen:
        raise ValueError(
            "El JSON devuelto por Google GenAI no contiene campos válidos."
        )

    return {
        "titulo_reporte": titulo,
        "resumen_ejecutivo": resumen,
    }


def generar_contexto_ejecutivo(
    auditoria: dict[str, Any], muestra_filas: list[dict[str, Any]]
) -> dict[str, str]:
    """Genera título y resumen ejecutivo vía Gemini con respaldo determinista.

    Aplica degradación elegante con un timeout estricto de 4 segundos cuando
    existe ``GEMINI_API_KEY`` válida, enviando únicamente el esquema y hasta
    3 filas de muestra. Ante ausencia de clave, errores de red, timeout o
    respuestas inválidas, retorna un resumen determinista local.

    Args:
        auditoria: Diccionario retornado por ``DatasetAuditor.auditar_estructura()``.
        muestra_filas: Lista de registros de muestra del dataset.

    Returns:
        Diccionario con las claves ``titulo_reporte`` y ``resumen_ejecutivo``.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "tu_api_key_aqui":
        return _generar_respaldo_determinista(auditoria)

    esquema = auditoria.get("esquema", [])
    muestra = muestra_filas[:3] if isinstance(muestra_filas, list) else []

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        futuro = executor.submit(_invocar_gemini, api_key, esquema, muestra)
        return futuro.result(timeout=TIMEOUT_SEGUNDOS)
    except (FuturesTimeoutError, TimeoutError) as exc:
        logger.warning(
            "Timeout (> %.1fs) al consultar Google GenAI; usando respaldo local: %s",
            TIMEOUT_SEGUNDOS,
            exc,
        )
        return _generar_respaldo_determinista(auditoria)
    except Exception as exc:
        logger.warning(
            "Fallo en síntesis con Google GenAI (%s); usando respaldo local.",
            exc,
        )
        return _generar_respaldo_determinista(auditoria)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
