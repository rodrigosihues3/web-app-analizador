"""Aplicación monolítica web Offline-First (FastAPI + Jinja2 + HTMX)."""

from __future__ import annotations

import logging
from pathlib import Path
import shutil
from typing import Any
import uuid

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core.auditor import DatasetAuditor
from core.estadistica import analizar_valores_clave, resumen_descriptivo
from core.exportador import exportar_dataframe, generar_reporte_pdf
from core.sintetizador import generar_contexto_ejecutivo
from core.transformador import limpiar_dataset
from core.visualizador import (
    generar_boxplot_plotly,
    generar_graficos_estaticos_pdf,
    generar_heatmap_plotly,
    generar_histograma_plotly,
)

logger = logging.getLogger(__name__)

BASE_DIR: Path = Path(__file__).resolve().parent.parent
WEB_DIR: Path = Path(__file__).resolve().parent
STATIC_DIR: Path = WEB_DIR / "static"
TEMPLATES_DIR: Path = WEB_DIR / "templates"
DATASETS_DIR: Path = BASE_DIR / "datasets"
TEMP_UPLOADS_DIR: Path = BASE_DIR / "temp_uploads"

TEMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS_DEMO: dict[str, dict[str, str]] = {
    "notas_estudiantes.csv": {
        "archivo": "notas_estudiantes.csv",
        "etiqueta": "notas_estudiantes.csv — Registro Académico (25 filas, nulos, duplicados y outlier IQR)",
        "dominio": "Académico / Educación Superior",
    },
    "retail_sales.csv": {
        "archivo": "retail_sales.csv",
        "etiqueta": "retail_sales.csv — Ventas Corporativas (30 filas, series de fecha, nulos y outlier extremo)",
        "dominio": "Comercial / Retail B2B",
    },
}

# Almacén en memoria del estado por sesión activa
SESIONES: dict[str, dict[str, Any]] = {}

app = FastAPI(
    title="Analizador Multiparadigma de Datos",
    description=(
        "Plataforma analítica B2B Offline-First con arquitectura híbrida "
        "POO + Programación Funcional."
    ),
    version="2.0.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _interpretar_booleano(valor: str | bool | None) -> bool:
    """Convierte valores de checkboxes HTML o formularios a booleano nativo."""
    if valor is None:
        return False
    if isinstance(valor, bool):
        return valor
    return str(valor).strip().lower() in {"true", "1", "on", "si", "sí", "yes"}


def _seleccionar_columnas_por_tipo(
    esquema: list[dict[str, Any]],
) -> tuple[list[str], list[str], str, str]:
    """Clasifica columnas categóricas y numéricas sugiriendo valores por defecto.

    Args:
        esquema: Lista de diccionarios generada por ``auditar_estructura()``.

    Returns:
        Tupla ``(cols_categoricas, cols_numericas, cat_default, num_default)``.
    """
    cols_numericas = [
        str(col["columna"])
        for col in esquema
        if col.get("tipo_logico") == "numerica"
    ]
    cols_categoricas = [
        str(col["columna"])
        for col in esquema
        if col.get("tipo_logico") in {"categorica", "fecha"}
    ]

    if not cols_categoricas:
        cols_categoricas = [str(col["columna"]) for col in esquema]

    # Priorizar como categoría sugerida aquella con cardinalidad moderada (2 a 12)
    candidatas_agrupacion = [
        str(col["columna"])
        for col in esquema
        if col.get("tipo_logico") == "categorica"
        and 2 <= int(col.get("valores_unicos", 0)) <= 12
    ]
    cat_default = (
        candidatas_agrupacion[0]
        if candidatas_agrupacion
        else (cols_categoricas[0] if cols_categoricas else "")
    )

    # Priorizar métricas principales conocidas o la primera numérica disponible
    metricas_preferidas = {"Nota", "Total_Venta"}
    num_preferidas = [c for c in cols_numericas if c in metricas_preferidas]
    num_default = (
        num_preferidas[0]
        if num_preferidas
        else (cols_numericas[0] if cols_numericas else "")
    )

    return cols_categoricas, cols_numericas, cat_default, num_default


def _renderizar_alerta_error(
    request: Request,
    titulo: str,
    detalle: str,
    codigo_estado: int = 200,
) -> HTMLResponse:
    """Renderiza el fragmento HTML controlado de alerta técnica sin trazas crudas."""
    return templates.TemplateResponse(
        request=request,
        name="partials/alerta_error.html",
        context={
            "titulo_error": titulo,
            "mensaje_error": detalle,
        },
        status_code=codigo_estado,
    )


async def _materializar_archivo_sesion(
    session_id: str,
    archivo: UploadFile | None,
    dataset_demo: str | None,
) -> tuple[Path, str]:
    """Almacena el archivo subido o copia el dataset demo hacia ``temp_uploads/``.

    Args:
        session_id: Identificador UUID4 asignado a la sesión.
        archivo: Archivo recibido por ``multipart/form-data``.
        dataset_demo: Nombre del dataset de demostración seleccionado.

    Returns:
        Tupla ``(ruta_guardada, nombre_original)``.

    Raises:
        ValueError: Si no se suministró fuente válida o el archivo está vacío.
        FileNotFoundError: Si el dataset de demostración no existe en disco.
    """
    TEMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    if archivo is not None and archivo.filename and archivo.filename.strip():
        nombre_original = Path(archivo.filename.strip()).name
        contenido = await archivo.read()
        if not contenido:
            raise ValueError(
                f"El archivo cargado '{nombre_original}' está completamente vacío."
            )
        ruta_destino = TEMP_UPLOADS_DIR / f"{session_id}_{nombre_original}"
        ruta_destino.write_bytes(contenido)
        return ruta_destino, nombre_original

    if dataset_demo is not None and dataset_demo.strip():
        nombre_demo = Path(dataset_demo.strip()).name
        if nombre_demo not in DATASETS_DEMO:
            raise ValueError(
                f"El dataset de prueba '{nombre_demo}' no está registrado."
            )
        ruta_origen = DATASETS_DIR / nombre_demo
        if not ruta_origen.exists():
            raise FileNotFoundError(
                f"No se encontró el dataset de demostración '{nombre_demo}'."
            )
        ruta_destino = TEMP_UPLOADS_DIR / f"{session_id}_{nombre_demo}"
        shutil.copyfile(ruta_origen, ruta_destino)
        return ruta_destino, nombre_demo

    raise ValueError(
        "Debe adjuntar un archivo tabular (.csv o .xlsx) o seleccionar un "
        "dataset de demostración."
    )


@app.get("/", response_class=HTMLResponse)
async def pagina_inicio(request: Request) -> HTMLResponse:
    """Renderiza la interfaz principal de ingesta y consola analítica."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "datasets_demo": list(DATASETS_DEMO.values()),
        },
    )


@app.post("/analizar/cargar", response_class=HTMLResponse)
async def cargar_y_auditar_dataset(
    request: Request,
    archivo: UploadFile | None = File(default=None),
    dataset_demo: str | None = Form(default=None),
) -> HTMLResponse:
    """Ingesta un archivo o dataset demo, audita su estructura y genera síntesis IA."""
    session_id = str(uuid.uuid4())
    ruta_guardada: Path | None = None

    try:
        ruta_guardada, nombre_original = await _materializar_archivo_sesion(
            session_id=session_id,
            archivo=archivo,
            dataset_demo=dataset_demo,
        )

        auditor = DatasetAuditor(str(ruta_guardada))
        auditoria = auditor.auditar_estructura()
        muestra_filas = auditor.df_raw.head(3).to_dict(orient="records")
        contexto_ia = generar_contexto_ejecutivo(auditoria, muestra_filas)

        (
            cols_categoricas,
            cols_numericas,
            cat_default,
            num_default,
        ) = _seleccionar_columnas_por_tipo(auditoria["esquema"])

        total_nulos = sum(
            int(item.get("nulos", 0)) for item in auditoria["esquema"]
        )
        total_outliers = sum(
            int(info.get("cantidad", 0))
            for info in auditoria.get("outliers_iqr", {}).values()
        )

        SESIONES[session_id] = {
            "session_id": session_id,
            "nombre_archivo": nombre_original,
            "ruta_archivo": str(ruta_guardada),
            "df_raw": auditor.df_raw,
            "auditoria": auditoria,
            "contexto": contexto_ia,
            "columnas_categoricas": cols_categoricas,
            "columnas_numericas": cols_numericas,
            "col_categoria_default": cat_default,
            "col_numerica_default": num_default,
            "config_limpieza": {
                "eliminar_duplicados": True,
                "estrategia_nulos": "mediana",
                "recortar_outliers": True,
                "col_categoria": cat_default,
                "col_numerica": num_default,
            },
            "df_limpio": None,
            "logs_limpieza": [],
            "estadisticas": [],
            "valores_clave": None,
        }

        return templates.TemplateResponse(
            request=request,
            name="partials/panel_auditoria.html",
            context={
                "session_id": session_id,
                "nombre_archivo": nombre_original,
                "contexto": contexto_ia,
                "auditoria": auditoria,
                "total_nulos": total_nulos,
                "total_outliers": total_outliers,
                "columnas_categoricas": cols_categoricas,
                "columnas_numericas": cols_numericas,
                "col_categoria_default": cat_default,
                "col_numerica_default": num_default,
            },
        )
    except (
        UnicodeDecodeError,
        FileNotFoundError,
        KeyError,
        ZeroDivisionError,
        ValueError,
        TimeoutError,
        OSError,
    ) as exc:
        logger.warning("Error controlado en /analizar/cargar: %s", exc)
        if ruta_guardada is not None and ruta_guardada.exists():
            try:
                ruta_guardada.unlink()
            except OSError:
                pass
        return _renderizar_alerta_error(
            request=request,
            titulo="ERROR DE INGESTA Y AUDITORÍA ESTRUCTURAL",
            detalle=str(exc),
            codigo_estado=200,
        )


@app.post("/analizar/procesar", response_class=HTMLResponse)
async def procesar_pipeline_analitico(
    request: Request,
    session_id: str = Form(default=""),
    eliminar_duplicados: str | None = Form(default=None),
    estrategia_nulos: str = Form(default="mediana"),
    recortar_outliers: str | None = Form(default=None),
    col_categoria: str | None = Form(default=None),
    col_numerica: str | None = Form(default=None),
    col_metrica: str | None = Form(default=None),
) -> HTMLResponse:
    """Ejecuta el pipeline funcional de limpieza, estadística y visualización Plotly."""
    try:
        id_limpio = session_id.strip()
        if not id_limpio or id_limpio not in SESIONES:
            raise KeyError(
                "La sesión de análisis no existe o ha expirado. "
                "Cargue nuevamente el archivo para continuar."
            )

        sesion = SESIONES[id_limpio]
        df_raw = sesion["df_raw"]

        bool_duplicados = _interpretar_booleano(eliminar_duplicados)
        bool_outliers = _interpretar_booleano(recortar_outliers)
        estrategia_norm = estrategia_nulos.strip().lower() or "mediana"

        columna_cat = (
            col_categoria
            or sesion.get("col_categoria_default")
            or ""
        ).strip()
        columna_num = (
            col_numerica
            or col_metrica
            or sesion.get("col_numerica_default")
            or ""
        ).strip()

        if not columna_num:
            raise ValueError(
                "El dataset no dispone de columnas numéricas válidas para "
                "ejecutar el análisis estadístico."
            )
        if not columna_cat:
            raise ValueError(
                "Debe seleccionar una columna categórica válida para la "
                "agrupación analítica."
            )

        df_limpio, logs_limpieza = limpiar_dataset(
            df_raw,
            eliminar_duplicados=bool_duplicados,
            estrategia_nulos=estrategia_norm,
            recortar_outliers=bool_outliers,
        )

        estadisticas = resumen_descriptivo(df_limpio)
        valores_clave = analizar_valores_clave(
            df_limpio, columna_cat, columna_num
        )

        grafico_histograma = generar_histograma_plotly(df_limpio, columna_num)
        grafico_boxplot = generar_boxplot_plotly(
            df_limpio, columna_num, col_grupo=columna_cat
        )
        grafico_heatmap = generar_heatmap_plotly(df_limpio)

        config_actual = {
            "eliminar_duplicados": bool_duplicados,
            "estrategia_nulos": estrategia_norm,
            "recortar_outliers": bool_outliers,
            "col_categoria": columna_cat,
            "col_numerica": columna_num,
        }
        sesion["config_limpieza"] = config_actual
        sesion["df_limpio"] = df_limpio
        sesion["logs_limpieza"] = logs_limpieza
        sesion["estadisticas"] = estadisticas
        sesion["valores_clave"] = valores_clave

        return templates.TemplateResponse(
            request=request,
            name="partials/panel_resultados.html",
            context={
                "session_id": id_limpio,
                "nombre_archivo": sesion["nombre_archivo"],
                "config_limpieza": config_actual,
                "filas_originales": sesion["auditoria"]["filas"],
                "filas_limpias": len(df_limpio),
                "logs_limpieza": logs_limpieza,
                "estadisticas": estadisticas,
                "valores_clave": valores_clave,
                "grafico_histograma": grafico_histograma,
                "grafico_boxplot": grafico_boxplot,
                "grafico_heatmap": grafico_heatmap,
            },
        )
    except (
        KeyError,
        ValueError,
        ZeroDivisionError,
        TimeoutError,
        FileNotFoundError,
        OSError,
    ) as exc:
        mensaje = exc.args[0] if exc.args else str(exc)
        logger.warning("Error controlado en /analizar/procesar: %s", mensaje)
        return _renderizar_alerta_error(
            request=request,
            titulo="ERROR EN PIPELINE DE LIMPIEZA Y ANÁLISIS",
            detalle=str(mensaje),
            codigo_estado=200,
        )


def _asegurar_estado_procesado(sesion: dict[str, Any]) -> None:
    """Garantiza que la sesión cuente con DataFrame limpio y métricas antes de exportar."""
    if sesion.get("df_limpio") is not None:
        return

    config = sesion.get("config_limpieza", {})
    df_limpio, logs = limpiar_dataset(
        sesion["df_raw"],
        eliminar_duplicados=bool(config.get("eliminar_duplicados", True)),
        estrategia_nulos=str(config.get("estrategia_nulos", "mediana")),
        recortar_outliers=bool(config.get("recortar_outliers", True)),
    )
    estadisticas = resumen_descriptivo(df_limpio)

    col_cat = str(config.get("col_categoria") or sesion.get("col_categoria_default") or "")
    col_num = str(config.get("col_numerica") or sesion.get("col_numerica_default") or "")
    valores_clave: dict[str, Any] | None = None
    if col_cat and col_num:
        try:
            valores_clave = analizar_valores_clave(df_limpio, col_cat, col_num)
        except (KeyError, ValueError, ZeroDivisionError):
            valores_clave = None

    sesion["df_limpio"] = df_limpio
    sesion["logs_limpieza"] = logs
    sesion["estadisticas"] = estadisticas
    sesion["valores_clave"] = valores_clave


@app.get("/export/dataset/{session_id}")
async def descargar_dataset_limpio(
    request: Request,
    session_id: str,
    formato: str = Query(default="csv"),
) -> Response:
    """Exporta el DataFrame procesado de la sesión en formato CSV o XLSX."""
    try:
        id_limpio = session_id.strip()
        if not id_limpio or id_limpio not in SESIONES:
            raise KeyError(
                f"La sesión '{id_limpio}' no existe o ha expirado."
            )

        sesion = SESIONES[id_limpio]
        _asegurar_estado_procesado(sesion)

        formato_norm = formato.strip().lower()
        buffer = exportar_dataframe(sesion["df_limpio"], formato=formato_norm)

        base_nombre = Path(sesion["nombre_archivo"]).stem
        if formato_norm in {"xlsx", "excel"}:
            ext = "xlsx"
            media_type = (
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            )
        else:
            ext = "csv"
            media_type = "text/csv; charset=utf-8"

        nombre_descarga = f"{base_nombre}_limpio.{ext}"
        headers = {
            "Content-Disposition": f'attachment; filename="{nombre_descarga}"'
        }
        return StreamingResponse(
            buffer, media_type=media_type, headers=headers
        )
    except (
        KeyError,
        ValueError,
        FileNotFoundError,
        OSError,
    ) as exc:
        mensaje = exc.args[0] if exc.args else str(exc)
        logger.warning("Error controlado en /export/dataset: %s", mensaje)
        return _renderizar_alerta_error(
            request=request,
            titulo="ERROR DE EXPORTACIÓN DE DATASET",
            detalle=str(mensaje),
            codigo_estado=400,
        )


@app.get("/export/pdf/{session_id}")
async def descargar_reporte_pdf(
    request: Request,
    session_id: str,
) -> Response:
    """Genera y transmite el informe técnico ejecutivo en formato PDF (A4)."""
    try:
        id_limpio = session_id.strip()
        if not id_limpio or id_limpio not in SESIONES:
            raise KeyError(
                f"La sesión '{id_limpio}' no existe o ha expirado."
            )

        sesion = SESIONES[id_limpio]
        _asegurar_estado_procesado(sesion)

        df_limpio = sesion["df_limpio"]
        imagenes_graficos = generar_graficos_estaticos_pdf(df_limpio)
        pdf_buffer = generar_reporte_pdf(
            contexto=sesion["contexto"],
            auditoria=sesion["auditoria"],
            logs_limpieza=sesion["logs_limpieza"],
            estadisticas=sesion["estadisticas"],
            valores_clave=sesion["valores_clave"],
            imagenes_graficos=imagenes_graficos,
        )

        base_nombre = Path(sesion["nombre_archivo"]).stem
        nombre_pdf = f"reporte_tecnico_{base_nombre}.pdf"
        headers = {
            "Content-Disposition": f'attachment; filename="{nombre_pdf}"'
        }
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers=headers,
        )
    except (
        KeyError,
        ValueError,
        ZeroDivisionError,
        FileNotFoundError,
        OSError,
    ) as exc:
        mensaje = exc.args[0] if exc.args else str(exc)
        logger.warning("Error controlado en /export/pdf: %s", mensaje)
        return _renderizar_alerta_error(
            request=request,
            titulo="ERROR AL GENERAR REPORTE PDF",
            detalle=str(mensaje),
            codigo_estado=400,
        )
