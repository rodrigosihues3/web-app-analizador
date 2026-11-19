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

load_dotenv(override=True)
os.environ.pop("GOOGLE_API_KEY", None)
logger = logging.getLogger(__name__)

MODELO_GEMINI = "gemini-2.5-flash"
TIMEOUT_SEGUNDOS = 10.0
# La API de Google GenAI exige un deadline de servidor >= 10s (10000 ms);
# el corte estricto de 4.0s se mantiene en Python via ThreadPoolExecutor.
TIMEOUT_MILISEGUNDOS = 10000


class _RespuestaContextoSchema(BaseModel):
    """Esquema Pydantic que obliga a Gemini a responder un JSON con 2 campos."""

    titulo_reporte: str
    resumen_ejecutivo: str


def _normalizar_texto_plano(texto: str, max_lineas: int = 3) -> str:
    """Elimina marcas Markdown básicas y limita el texto a un máximo de líneas."""
    sin_md = texto.replace("**", "").replace("__", "").replace("`", "")
    lineas = [
        linea.lstrip("#-*• ").strip()
        for linea in sin_md.strip().splitlines()
        if linea.strip()
    ]
    if not lineas:
        return ""
    return " ".join(lineas[:max_lineas])


def _generar_respaldo_determinista(
    auditoria: dict[str, Any],
) -> dict[str, str]:
    """Construye un título y resumen ejecutivo sobrio de 2 líneas sin red."""
    esquema = auditoria.get("esquema", [])
    duplicados = int(auditoria.get("duplicados", 0))
    total_nulos = sum(int(col.get("nulos", 0)) for col in esquema)

    cols_numericas = [
        str(col.get("columna", ""))
        for col in esquema
        if col.get("tipo_logico") == "numerica"
    ]
    cols_dimension = [
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

    muestra_num = (
        ", ".join(cols_numericas[:3]) if cols_numericas else "sin métricas numéricas"
    )
    muestra_dim = (
        ", ".join(cols_dimension[:3])
        if cols_dimension
        else "sin dimensiones categóricas"
    )

    titulo_reporte = "Diagnóstico Estructural y Variables del Conjunto de Datos"

    linea_1 = (
        f"El archivo integra dimensiones descriptivas ({muestra_dim}) junto con "
        f"variables cuantitativas ({muestra_num}) orientadas al análisis comparativo."
    )

    hallazgos: list[str] = []
    if total_nulos > 0:
        hallazgos.append("valores faltantes")
    if duplicados > 0:
        hallazgos.append("registros repetidos")
    if total_outliers > 0:
        hallazgos.append("valores atípicos marcados")

    if hallazgos:
        detalle_calidad = ", ".join(hallazgos[:-1]) + (
            f" y {hallazgos[-1]}" if len(hallazgos) > 1 else hallazgos[0]
        )
        linea_2 = (
            f"Se observan {detalle_calidad} que conviene depurar antes de "
            "consolidar estadísticas y visualizaciones."
        )
    else:
        linea_2 = (
            "La estructura presenta consistencia completa para generar "
            "estadísticas descriptivas y visualizaciones comparativas."
        )

    return {
        "titulo_reporte": titulo_reporte,
        "resumen_ejecutivo": f"{linea_1}\n{linea_2}",
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

    # ThreadPoolExecutor ejecuta la llamada en segundo plano para cortar a los 4.0s exactos
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


def _generar_explicacion_grafico_local(
    tipo_grafico: str, datos_grafico: dict[str, Any]
) -> str:
    """Genera una interpretación estadística determinista cuando no hay API externa."""
    tipo = tipo_grafico.strip().lower()

    if tipo == "histograma":
        col = str(datos_grafico.get("columna", "la métrica"))
        media = float(datos_grafico.get("media", 0.0))
        mediana = float(datos_grafico.get("mediana", 0.0))
        moda = datos_grafico.get("moda")
        q25 = float(datos_grafico.get("q25", 0.0))
        q75 = float(datos_grafico.get("q75", 0.0))
        outliers = int(datos_grafico.get("outliers", 0))

        diferencia_rel = abs(media - mediana) / max(abs(mediana), 1e-6)
        if diferencia_rel < 0.03:
            sesgo_txt = (
                f"La distribución de {col} es prácticamente simétrica "
                f"(media {media:.2f} cercana a la mediana {mediana:.2f})."
            )
        elif media > mediana:
            sesgo_txt = (
                f"La distribución de {col} presenta sesgo positivo: la media "
                f"({media:.2f}) supera a la mediana ({mediana:.2f}) por efecto de valores altos."
            )
        else:
            sesgo_txt = (
                f"La distribución de {col} muestra sesgo negativo: la media "
                f"({media:.2f}) queda por debajo de la mediana ({mediana:.2f})."
            )

        moda_txt = (
            f" El valor más frecuente se ubica en {float(moda):.2f} y el"
            if moda is not None
            else " El"
        )
        rango_txt = (
            f"{moda_txt} 50% central de los registros se concentra entre "
            f"{q25:.2f} y {q75:.2f}"
        )
        extra_outliers = (
            f", con {outliers} valor(es) fuera del rango típico."
            if outliers > 0
            else "."
        )
        return f"{sesgo_txt}{rango_txt}{extra_outliers}"

    if tipo in {"barras", "barras_categoria", "boxplot"}:
        col_met = str(datos_grafico.get("col_metrica", "la métrica"))
        col_cat = str(datos_grafico.get("col_categoria", "la categoría"))
        ranking = datos_grafico.get("ranking_categorias", [])
        promedio_global = datos_grafico.get("promedio_general")

        if not ranking:
            return (
                f"No hay suficientes registros válidos para comparar {col_met} "
                f"entre las categorías de {col_cat}."
            )

        lider = ranking[0]
        menor = ranking[-1]
        nombre_lider = str(lider.get("categoria", ""))
        val_lider = float(lider.get("promedio", 0.0))
        nombre_menor = str(menor.get("categoria", ""))
        val_menor = float(menor.get("promedio", 0.0))

        brecha = val_lider - val_menor
        linea_1 = (
            f"En {col_met} por {col_cat}, '{nombre_lider}' lidera con un "
            f"promedio de {val_lider:.2f}, superando en {brecha:.2f} unidades a "
            f"'{nombre_menor}' ({val_menor:.2f})."
        )
        if promedio_global is not None:
            linea_2 = (
                f" La referencia general del conjunto se sitúa en "
                f"{float(promedio_global):.2f}, permitiendo distinguir grupos "
                f"por encima y por debajo de la media."
            )
        else:
            linea_2 = ""
        return f"{linea_1}{linea_2}"

    correlaciones = datos_grafico.get("correlaciones", [])
    if not correlaciones:
        return (
            "El conjunto actual requiere al menos dos variables numéricas con "
            "variación para identificar asociaciones lineales relevantes."
        )

    mayor = correlaciones[0]
    var_a = str(mayor.get("var_a", ""))
    var_b = str(mayor.get("var_b", ""))
    coef = float(mayor.get("coeficiente", 0.0))
    intensidad = (
        "fuerte" if abs(coef) >= 0.7 else ("moderada" if abs(coef) >= 0.4 else "débil")
    )
    sentido = "positiva" if coef >= 0 else "negativa"
    return (
        f"La relación lineal más destacada ocurre entre {var_a} y {var_b}, "
        f"con una correlación {intensidad} {sentido} (r = {coef:.2f}). "
        "Los valores cercanos a cero indican variables que evolucionan de forma independiente."
    )


def _invocar_gemini_grafico(
    api_key: str,
    titulo_reporte: str,
    tipo_grafico: str,
    datos_grafico: dict[str, Any],
) -> str:
    """Consulta Gemini para redactar una interpretación breve de una gráfica."""
    payload = {
        "contexto_dataset": titulo_reporte,
        "tipo_grafico": tipo_grafico,
        "metricas_grafica": datos_grafico,
    }
    prompt = (
        "Actúa como Analista de Datos Senior. Redacta una interpretación clara "
        "y elocuente en español latinoamericano basada únicamente en las métricas "
        "exactas de esta gráfica. Responde en texto plano sin Markdown, sin listas "
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
    """Genera una interpretación breve en español sobre una gráfica estadística.

    Usa ``gemini-2.5-flash`` con timeout estricto de 4 segundos cuando existe
    ``GEMINI_API_KEY`` válida; ante cualquier fallo o modo offline retorna una
    explicación determinista basada en las métricas recibidas.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "tu_api_key_aqui":
        return _generar_explicacion_grafico_local(tipo_grafico, datos_grafico)

    titulo_reporte = str(contexto_dataset.get("titulo_reporte", "Análisis de Dataset"))
    # ThreadPoolExecutor ejecuta la llamada en segundo plano para cortar a los 4.0s exactos
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        futuro = executor.submit(
            _invocar_gemini_grafico,
            api_key,
            titulo_reporte,
            tipo_grafico,
            datos_grafico,
        )
        return futuro.result(timeout=TIMEOUT_SEGUNDOS)
    except Exception as exc:
        logger.warning("Respaldo local al explicar gráfico (%s): %s", tipo_grafico, exc)
        return _generar_explicacion_grafico_local(tipo_grafico, datos_grafico)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
