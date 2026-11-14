"""Servidor MCP para exponer el núcleo analítico como herramientas."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:
    from mcp.server.mcpserver import MCPServer as FastMCP

from core.auditor import DatasetAuditor
from core.estadistica import analizar_valores_clave, resumen_descriptivo
from core.exportador import generar_reporte_pdf
from core.sintetizador import generar_contexto_ejecutivo
from core.transformador import limpiar_dataset
from core.visualizador import generar_graficos_estaticos_pdf

mcp = FastMCP("AnalizadorMultiparadigma")


def _detectar_columnas_clave(auditoria: dict[str, Any]) -> tuple[str | None, str | None]:
    """Selecciona automáticamente la primera columna categórica y numérica."""
    esquema = auditoria.get("esquema", [])
    col_categoria = next(
        (
            str(col["columna"])
            for col in esquema
            if col.get("tipo_logico") == "categorica"
        ),
        None,
    )
    col_metrica = next(
        (
            str(col["columna"])
            for col in esquema
            if col.get("tipo_logico") == "numerica"
        ),
        None,
    )
    return col_categoria, col_metrica


@mcp.tool()
def auditar_dataset_mcp(ruta_archivo: str) -> str:
    """Audita un archivo CSV o Excel y devuelve el diagnóstico en JSON."""
    try:
        auditor = DatasetAuditor(ruta_archivo)
        diagnostico = auditor.auditar_estructura()
        return json.dumps(diagnostico, ensure_ascii=False, indent=2)
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError, OSError) as exc:
        return json.dumps(
            {"error": f"No fue posible auditar el archivo: {exc}"},
            ensure_ascii=False,
            indent=2,
        )


@mcp.tool()
def metricas_clave_mcp(
    ruta_archivo: str, col_categoria: str, col_metrica: str
) -> str:
    """Limpia nulos por mediana y retorna estadísticas descriptivas y valores clave."""
    try:
        auditor = DatasetAuditor(ruta_archivo)
        df_limpio, logs = limpiar_dataset(
            auditor.df_raw,
            eliminar_duplicados=True,
            estrategia_nulos="mediana",
            recortar_outliers=False,
        )
        estadisticas = resumen_descriptivo(df_limpio)
        valores_clave = analizar_valores_clave(
            df_limpio, col_categoria=col_categoria, col_metrica=col_metrica
        )
        resultado = {
            "logs_limpieza": logs,
            "resumen_descriptivo": estadisticas,
            "valores_clave": valores_clave,
        }
        return json.dumps(resultado, ensure_ascii=False, indent=2)
    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        pd.errors.EmptyDataError,
        OSError,
    ) as exc:
        return json.dumps(
            {"error": f"No fue posible calcular las métricas clave: {exc}"},
            ensure_ascii=False,
            indent=2,
        )


@mcp.tool()
def exportar_reporte_pdf_mcp(ruta_archivo: str, ruta_salida_pdf: str) -> str:
    """Ejecuta el pipeline analítico completo y guarda el reporte PDF en disco."""
    try:
        if not ruta_salida_pdf or not isinstance(ruta_salida_pdf, str):
            raise ValueError("La ruta de salida del PDF debe ser no vacía.")

        auditor = DatasetAuditor(ruta_archivo)
        auditoria = auditor.auditar_estructura()
        df_limpio, logs = limpiar_dataset(
            auditor.df_raw,
            eliminar_duplicados=True,
            estrategia_nulos="mediana",
            recortar_outliers=False,
        )
        estadisticas = resumen_descriptivo(df_limpio)

        col_cat, col_met = _detectar_columnas_clave(auditoria)
        valores_clave: dict[str, Any] | None = None
        if col_cat and col_met:
            try:
                valores_clave = analizar_valores_clave(
                    df_limpio, col_categoria=col_cat, col_metrica=col_met
                )
            except (KeyError, ValueError):
                valores_clave = None

        muestra_filas = df_limpio.head(3).to_dict(orient="records")
        contexto = generar_contexto_ejecutivo(auditoria, muestra_filas)
        imagenes_graficos = generar_graficos_estaticos_pdf(df_limpio)

        pdf_buffer = generar_reporte_pdf(
            contexto=contexto,
            auditoria=auditoria,
            logs_limpieza=logs,
            estadisticas=estadisticas,
            valores_clave=valores_clave,
            imagenes_graficos=imagenes_graficos,
        )

        destino = Path(ruta_salida_pdf)
        destino.parent.mkdir(parents=True, exist_ok=True)
        datos_pdf = pdf_buffer.getvalue()
        destino.write_bytes(datos_pdf)

        return (
            f"Reporte PDF generado correctamente en: {destino.resolve()} "
            f"({len(datos_pdf)} bytes)."
        )
    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        pd.errors.EmptyDataError,
        OSError,
    ) as exc:
        return f"Error al exportar el reporte PDF: {exc}"


if __name__ == "__main__":
    mcp.run()
