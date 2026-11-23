"""Servidor MCP interactivo (2 pasos) para auditar, limpiar y exportar datasets."""

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

EstrategiaNulos = Literal[
    "eliminar", "media", "mediana", "moda", "personalizado", "ninguna"
]


def _resolver_ruta_archivo(ruta_o_nombre: str) -> Path:
    """Localiza un archivo CSV o XLSX por ruta completa o por nombre en carpetas del usuario."""
    texto = ruta_o_nombre.strip().strip('"').strip("'")
    if not texto:
        raise ValueError("Debes indicar el nombre o la ruta del archivo.")

    candidato_directo = Path(texto).expanduser()
    try:
        if candidato_directo.exists() and candidato_directo.is_file():
            return candidato_directo.resolve()
    except OSError:
        pass

    nombre = candidato_directo.name
    posibles_nombres = [nombre]
    if Path(nombre).suffix.lower() not in (".csv", ".xlsx"):
        posibles_nombres.append(f"{nombre}.csv")
        posibles_nombres.append(f"{nombre}.xlsx")

    home = Path.home()
    directorios_busqueda = [
        BASE_DIR / "datasets",
        BASE_DIR,
        home / "Downloads",
        home / "Descargas",
        home / "Desktop",
        home / "Escritorio",
        home / "Documents",
        home / "Documentos",
    ]

    for carpeta in directorios_busqueda:
        try:
            if not carpeta.exists():
                continue
            for nom in posibles_nombres:
                ruta_posible = carpeta / nom
                if ruta_posible.exists() and ruta_posible.is_file():
                    return ruta_posible.resolve()
        except OSError:
            continue

    for carpeta_profunda in (home / "Downloads", home / "Documents"):
        try:
            if not carpeta_profunda.exists():
                continue
            for nom in posibles_nombres:
                for encontrado in carpeta_profunda.rglob(nom):
                    if encontrado.is_file():
                        return encontrado.resolve()
        except OSError:
            continue

    raise FileNotFoundError(
        f"No se encontró '{ruta_o_nombre}' en la ruta indicada ni en Descargas, "
        "Documentos, Escritorio o datasets/."
    )


def _detectar_columnas_clave(
    auditoria: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Sugiere automáticamente una columna categórica y una numérica del esquema."""
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
    """Paso 1 obligatorio: Audita la estructura, columnas, valores nulos y filas duplicadas de un archivo CSV o XLSX. Puedes pasar la ruta completa O simplemente el nombre del archivo (ej. 'Lista_Alumnos_Notas.csv' o 'Lista_Alumnos_Notas') y esta herramienta lo buscará automáticamente en Descargas, Documentos, Escritorio y datasets/. Tras recibir el resultado, DEBES mostrar al usuario el resumen de nulos y duplicados y preguntarle antes de llamar a 'procesar_y_exportar_mcp': (1) cómo desea tratar los valores nulos ('mediana', 'media', 'moda', 'eliminar', 'ninguna' o 'personalizado' indicando un valor manual como 0), (2) si desea eliminar duplicados (True/False), y (3) qué columna categórica y qué columna numérica desea analizar."""
    try:
        ruta_resuelta = _resolver_ruta_archivo(ruta_archivo)
        auditor = DatasetAuditor(str(ruta_resuelta))
        diagnostico = auditor.auditar_estructura()
        diagnostico["ruta_resuelta"] = str(ruta_resuelta)
        diagnostico["opciones_nulos_disponibles"] = [
            "mediana",
            "media",
            "moda",
            "personalizado (permite rellenar con 0 u otro valor en 'valor_nulos')",
            "eliminar",
            "ninguna",
        ]
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
    carpeta_salida: str = "",
    eliminar_duplicados: bool = True,
    estrategia_nulos: str = "mediana",
    valor_nulos: str = "0",
    col_categoria: str = "",
    col_metrica: str = "",
) -> str:
    """Paso 2: Limpia el dataset según las opciones elegidas por el usuario, calcula estadísticas descriptivas (incluyendo Media, Mediana y Moda) y exporta el CSV limpio y el reporte PDF en 'carpeta_salida' (si se deja vacía, los guarda en la misma carpeta del archivo original). Para reemplazar nulos con un número fijo como 0, usa estrategia_nulos='personalizado' y valor_nulos='0'."""
    try:
        ruta_resuelta = _resolver_ruta_archivo(ruta_archivo)
        auditor = DatasetAuditor(str(ruta_resuelta))
        auditoria = auditor.auditar_estructura()

        estrategia_limpia = estrategia_nulos.strip().lower()
        valor_manual = valor_nulos
        if estrategia_limpia not in (
            "eliminar",
            "media",
            "mediana",
            "moda",
            "personalizado",
            "ninguna",
        ):
            try:
                float(estrategia_limpia)
                valor_manual = estrategia_limpia
                estrategia_limpia = "personalizado"
            except ValueError:
                pass

        df_limpio, logs = limpiar_dataset(
            auditor.df_raw,
            eliminar_duplicados=eliminar_duplicados,
            estrategia_nulos=estrategia_limpia,
            valor_nulos=valor_manual,
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

        directorio = (
            Path(carpeta_salida).expanduser()
            if carpeta_salida and carpeta_salida.strip()
            else ruta_resuelta.parent
        )
        directorio.mkdir(parents=True, exist_ok=True)
        nombre_base = ruta_resuelta.stem

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
