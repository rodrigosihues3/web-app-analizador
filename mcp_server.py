"""Servidor MCP interactivo (2 pasos) para exponer el núcleo analítico."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Literal

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:
    from mcp.server.mcpserver import MCPServer as FastMCP

from core.auditor import DatasetAuditor
from core.estadistica import analizar_valores_clave, resumen_descriptivo
from core.exportador import exportar_dataframe, generar_reporte_pdf
from core.sintetizador import generar_contexto_ejecutivo
from core.transformador import limpiar_dataset
from core.visualizador import generar_graficos_estaticos_pdf

mcp = FastMCP("AnalizadorMultiparadigma")

EstrategiaNulos = Literal["eliminar", "media", "mediana", "moda", "ninguna"]


def _detectar_columnas_clave(
    auditoria: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Sugiere automáticamente una columna categórica y una numérica útil."""
    esquema = auditoria.get("esquema", [])
    constantes = set(auditoria.get("columnas_varianza_cero", []))

    candidatas_cat = [
        str(col["columna"])
        for col in esquema
        if col.get("tipo_logico") == "categorica"
        and 2 <= int(col.get("valores_unicos", 0)) <= 12
    ]
    todas_cat = [
        str(col["columna"])
        for col in esquema
        if col.get("tipo_logico") in {"categorica", "fecha"}
    ]
    col_categoria = (
        candidatas_cat[0]
        if candidatas_cat
        else (todas_cat[0] if todas_cat else None)
    )

    todas_num = [
        str(col["columna"])
        for col in esquema
        if col.get("tipo_logico") == "numerica"
    ]
    num_utiles = [c for c in todas_num if c not in constantes] or todas_num
    col_metrica = num_utiles[0] if num_utiles else None

    return col_categoria, col_metrica


@mcp.tool()
def auditar_dataset_mcp(ruta_archivo: str) -> str:
    """Paso 1 obligatorio: Audita la estructura, calidad, valores nulos, filas duplicadas y valores atípicos de un archivo CSV o XLSX. Tras recibir este diagnóstico, DEBES presentar un resumen claro de los problemas encontrados al usuario y preguntarle explícitamente: (1) qué estrategia prefiere para valores nulos ('mediana', 'media', 'moda', 'eliminar' o 'ninguna'), (2) si desea eliminar duplicados (True/False), (3) si desea acotar valores extremos IQR (True/False), y opcionalmente (4) qué columna categórica y numérica del esquema desea comparar antes de ejecutar 'procesar_y_exportar_mcp'."""
    try:
        auditor = DatasetAuditor(ruta_archivo)
        diagnostico = auditor.auditar_estructura()
        return json.dumps(diagnostico, ensure_ascii=False, indent=2)
    except (FileNotFoundError, ValueError, OSError) as exc:
        return json.dumps(
            {"error": f"No fue posible auditar el archivo: {exc}"},
            ensure_ascii=False,
            indent=2,
        )


@mcp.tool()
def procesar_y_exportar_mcp(
    ruta_archivo: str,
    carpeta_salida: str,
    eliminar_duplicados: bool = True,
    estrategia_nulos: EstrategiaNulos = "mediana",
    recortar_outliers: bool = False,
    col_categoria: str = "",
    col_metrica: str = "",
) -> str:
    """Paso 2: Aplica la limpieza configurada por el usuario sobre el dataset, calcula estadísticas descriptivas y genera tanto el archivo CSV limpio como el reporte técnico en PDF dentro de 'carpeta_salida'. Retorna las rutas de los archivos generados y las métricas clave para redactar el resumen final."""
    try:
        if not carpeta_salida or not isinstance(carpeta_salida, str):
            raise ValueError("La carpeta de salida debe ser una ruta no vacía.")

        auditor = DatasetAuditor(ruta_archivo)
        auditoria = auditor.auditar_estructura()

        df_limpio, logs = limpiar_dataset(
            auditor.df_raw,
            eliminar_duplicados=eliminar_duplicados,
            estrategia_nulos=estrategia_nulos,
            recortar_outliers=recortar_outliers,
        )
        estadisticas = resumen_descriptivo(df_limpio)

        cat_default, met_default = _detectar_columnas_clave(auditoria)
        cat_final = col_categoria.strip() or cat_default
        met_final = col_metrica.strip() or met_default

        valores_clave: dict[str, Any] | None = None
        if cat_final and met_final:
            try:
                valores_clave = analizar_valores_clave(
                    df_limpio, col_categoria=cat_final, col_metrica=met_final
                )
            except (KeyError, ValueError, ZeroDivisionError):
                valores_clave = None

        contexto = generar_contexto_ejecutivo(
            auditoria, auditor.obtener_muestra(3)
        )
        imagenes_graficos = generar_graficos_estaticos_pdf(df_limpio)

        directorio = Path(carpeta_salida)
        directorio.mkdir(parents=True, exist_ok=True)
        nombre_base = Path(ruta_archivo).stem

        ruta_csv = directorio / f"{nombre_base}_limpio.csv"
        csv_buffer = exportar_dataframe(df_limpio, formato="csv")
        ruta_csv.write_bytes(csv_buffer.getvalue())

        ruta_pdf = directorio / f"{nombre_base}_reporte.pdf"
        pdf_buffer = generar_reporte_pdf(
            contexto=contexto,
            auditoria=auditoria,
            logs_limpieza=logs,
            estadisticas=estadisticas,
            valores_clave=valores_clave,
            imagenes_graficos=imagenes_graficos,
        )
        ruta_pdf.write_bytes(pdf_buffer.getvalue())

        resultado = {
            "ruta_csv_limpio": str(ruta_csv.resolve()),
            "ruta_reporte_pdf": str(ruta_pdf.resolve()),
            "logs_limpieza": logs,
            "estadisticas": estadisticas,
            "valores_clave": valores_clave,
        }
        return json.dumps(resultado, ensure_ascii=False, indent=2)
    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        ZeroDivisionError,
        TimeoutError,
        OSError,
    ) as exc:
        return json.dumps(
            {"error": f"No fue posible procesar y exportar el dataset: {exc}"},
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    mcp.run()
