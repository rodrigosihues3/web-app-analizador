"""Aplicación web (FastAPI + Jinja2 + HTMX)."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any
import uuid

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core.auditor import DatasetAuditor
from core.estadistica import (
    analizar_valores_clave,
    extraer_metricas_grafico,
    resumen_descriptivo,
)
from core.exportador import exportar_dataframe, generar_reporte_pdf
from core.sintetizador import (
    MENSAJE_IA_NO_DISPONIBLE,
    explicar_grafico_ia,
    generar_contexto_ejecutivo,
)
from core.transformador import limpiar_dataset
from core.visualizador import (
    generar_barras_categoria_plotly,
    generar_graficos_estaticos_pdf,
    generar_heatmap_plotly,
    generar_histograma_plotly,
)

logger = logging.getLogger(__name__)

BASE_DIR: Path = Path(__file__).resolve().parent.parent
WEB_DIR: Path = Path(__file__).resolve().parent
STATIC_DIR: Path = WEB_DIR / "static"
TEMPLATES_DIR: Path = WEB_DIR / "templates"
TEMP_UPLOADS_DIR: Path = BASE_DIR / "temp_uploads"

TEMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Almacén en memoria del estado por sesión activa
SESIONES: dict[str, dict[str, Any]] = {}

app = FastAPI(
    title="Analizador de Datos",
    description=(
        "Plataforma analítica Offline-First con arquitectura híbrida "
        "POO + Programación Funcional."
    ),
    version="2.1.0",
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
    columnas_varianza_cero: list[str] | None = None,
) -> tuple[list[str], list[str], str, str]:
    """Clasifica columnas categóricas y numéricas sugiriendo valores iniciales."""
    constantes = set(columnas_varianza_cero or [])
    cols_numericas = [
        str(col["columna"]) for col in esquema if col.get("tipo_logico") == "numerica"
    ]
    cols_numericas_utiles = [
        c for c in cols_numericas if c not in constantes
    ] or cols_numericas

    cols_categoricas = [
        str(col["columna"])
        for col in esquema
        if col.get("tipo_logico") in {"categorica", "fecha"}
    ]

    if not cols_categoricas:
        cols_categoricas = [str(col["columna"]) for col in esquema]

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

    metricas_preferidas = {"Nota", "Total_Venta"}
    num_preferidas = [c for c in cols_numericas_utiles if c in metricas_preferidas]
    num_default = (
        num_preferidas[0]
        if num_preferidas
        else (cols_numericas_utiles[0] if cols_numericas_utiles else "")
    )

    return cols_categoricas, cols_numericas, cat_default, num_default


def _renderizar_alerta_error(
    request: Request,
    titulo: str,
    detalle: str,
    codigo_estado: int = 200,
) -> HTMLResponse:
    """Renderiza el fragmento HTML controlado de alerta sin trazas crudas."""
    return templates.TemplateResponse(
        request=request,
        name="partials/alerta_error.html",
        context={
            "titulo_error": titulo,
            "mensaje_error": detalle,
        },
        status_code=codigo_estado,
    )


def _eliminar_sesion(session_id: str | None) -> None:
    """Elimina una sesión de memoria y borra su archivo temporal en disco."""
    if not session_id:
        return
    id_limpio = session_id.strip()
    sesion = SESIONES.pop(id_limpio, None)
    if sesion and sesion.get("ruta_archivo"):
        ruta_temp = Path(str(sesion["ruta_archivo"]))
        if ruta_temp.exists():
            try:
                ruta_temp.unlink()
            except OSError as exc:
                logger.warning(
                    "No se pudo eliminar archivo temporal '%s': %s",
                    ruta_temp,
                    exc,
                )


async def _materializar_archivo_sesion(
    session_id: str,
    archivo: UploadFile | None,
) -> tuple[Path, str]:
    """Almacena el archivo subido hacia ``temp_uploads/`` validando contenido."""
    TEMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    if archivo is not None and archivo.filename and archivo.filename.strip():
        nombre_original = Path(archivo.filename.strip()).name
        contenido = await archivo.read()
        if not contenido:
            raise ValueError(
                f"El archivo '{nombre_original}' está vacío. Selecciona un archivo con registros."
            )
        ruta_destino = TEMP_UPLOADS_DIR / f"{session_id}_{nombre_original}"
        ruta_destino.write_bytes(contenido)
        return ruta_destino, nombre_original

    raise ValueError(
        "Adjunta un archivo tabular en formato .csv o .xlsx para iniciar el análisis."
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Retorna un icono SVG minimalista local para evitar 404 en navegadores."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
        '<circle cx="8" cy="8" r="6" fill="#0f766e"/></svg>'
    )
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/", response_class=HTMLResponse)
async def pagina_inicio(request: Request) -> HTMLResponse:
    """Renderiza la vista principal limpia lista para carga manual."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.post("/analizar/cargar", response_class=HTMLResponse)
async def cargar_y_auditar_dataset(
    request: Request,
    archivo: UploadFile | None = File(default=None),
    session_anterior: str | None = Form(default=None, alias="session_id"),
) -> HTMLResponse:
    """Ingesta un archivo tabular, audita su estructura y genera síntesis inicial."""
    inicio_analisis = time.perf_counter()
    session_id = str(uuid.uuid4())
    ruta_guardada: Path | None = None

    try:
        ruta_guardada, nombre_original = await _materializar_archivo_sesion(
            session_id=session_id,
            archivo=archivo,
        )

        auditor = DatasetAuditor(str(ruta_guardada))
        auditoria = auditor.auditar_estructura()
        muestra_filas = auditor.obtener_muestra(3)
        contexto_ia = generar_contexto_ejecutivo(auditoria, muestra_filas)
        tiempo_analisis_s = round(time.perf_counter() - inicio_analisis, 2)

        # Limpia la sesión anterior solo cuando el nuevo archivo se validó bien
        _eliminar_sesion(session_anterior)

        (
            cols_categoricas,
            cols_numericas,
            cat_default,
            num_default,
        ) = _seleccionar_columnas_por_tipo(
            auditoria["esquema"], auditoria.get("columnas_varianza_cero", [])
        )

        total_nulos = sum(int(item.get("nulos", 0)) for item in auditoria["esquema"])

        SESIONES[session_id] = {
            "session_id": session_id,
            "nombre_archivo": nombre_original,
            "ruta_archivo": str(ruta_guardada),
            "df_raw": auditor.df_raw,
            "muestra_filas": muestra_filas,
            "auditoria": auditoria,
            "contexto": contexto_ia,
            "columnas_categoricas": cols_categoricas,
            "columnas_numericas": cols_numericas,
            "col_categoria_default": cat_default,
            "col_numerica_default": num_default,
            "config_limpieza": {
                "eliminar_duplicados": True,
                "estrategia_nulos": "mediana",
                "valor_nulos": "0",
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
                "tiempo_analisis_s": tiempo_analisis_s,
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
            titulo="No se pudo leer el archivo",
            detalle=str(exc),
            codigo_estado=200,
        )


@app.post("/analizar/restaurar", response_class=HTMLResponse)
async def restaurar_estado_inicial(
    request: Request,
    session_id: str | None = Form(default=None),
) -> HTMLResponse:
    """Limpia la sesión activa y su archivo temporal, restaurando el inicio."""
    _eliminar_sesion(session_id)

    return templates.TemplateResponse(
        request=request,
        name="partials/estado_inicial_oob.html",
        context={},
    )


@app.post("/analizar/regenerar-resumen", response_class=HTMLResponse)
async def regenerar_resumen_ejecutivo(
    request: Request,
    session_id: str = Form(default=""),
) -> HTMLResponse:
    """Regenera el título y resumen ejecutivo para la sesión activa."""
    try:
        id_limpio = session_id.strip()
        if not id_limpio or id_limpio not in SESIONES:
            raise KeyError(
                "La sesión expiró o no es válida. Sube nuevamente el archivo."
            )

        sesion = SESIONES[id_limpio]
        contexto_nuevo = generar_contexto_ejecutivo(
            sesion["auditoria"], sesion["muestra_filas"]
        )
        sesion["contexto"] = contexto_nuevo

        return templates.TemplateResponse(
            request=request,
            name="partials/resumen_ejecutivo.html",
            context={
                "contexto": contexto_nuevo,
                "animar_escritura": True,
            },
        )
    except (KeyError, ValueError, TimeoutError, OSError) as exc:
        mensaje = exc.args[0] if exc.args else str(exc)
        logger.warning("Error controlado en /analizar/regenerar-resumen: %s", mensaje)
        return _renderizar_alerta_error(
            request=request,
            titulo="No se pudo regenerar el resumen",
            detalle=str(mensaje),
            codigo_estado=200,
        )


@app.post("/analizar/procesar", response_class=HTMLResponse)
async def procesar_pipeline_analitico(
    request: Request,
    session_id: str = Form(default=""),
    eliminar_duplicados: str | None = Form(default=None),
    estrategia_nulos: str = Form(default="mediana"),
    valor_nulos: str = Form(default="0"),
    col_categoria: str | None = Form(default=None),
    col_numerica: str | None = Form(default=None),
) -> HTMLResponse:
    """Ejecuta limpieza funcional, estadística descriptiva y gráficos Plotly."""
    try:
        id_limpio = session_id.strip()
        if not id_limpio or id_limpio not in SESIONES:
            raise KeyError(
                "La sesión de análisis expiró. Sube nuevamente el archivo para continuar."
            )

        sesion = SESIONES[id_limpio]
        df_raw = sesion["df_raw"]

        bool_duplicados = _interpretar_booleano(eliminar_duplicados)
        estrategia_norm = estrategia_nulos.strip().lower() or "mediana"
        valor_manual = valor_nulos.strip() if valor_nulos is not None else "0"

        columna_cat = (
            col_categoria or sesion.get("col_categoria_default") or ""
        ).strip()
        columna_num = (col_numerica or sesion.get("col_numerica_default") or "").strip()

        if not columna_num:
            raise ValueError(
                "El archivo no contiene columnas numéricas disponibles para analizar."
            )
        if not columna_cat:
            raise ValueError(
                "Selecciona una columna categórica para comparar los datos."
            )

        df_limpio, logs_limpieza = limpiar_dataset(
            df_raw,
            eliminar_duplicados=bool_duplicados,
            estrategia_nulos=estrategia_norm,
            valor_nulos=valor_manual,
        )

        estadisticas = resumen_descriptivo(df_limpio)
        valores_clave = analizar_valores_clave(df_limpio, columna_cat, columna_num)

        grafico_histograma = generar_histograma_plotly(df_limpio, columna_num)
        grafico_barras = generar_barras_categoria_plotly(
            df_limpio, columna_num, columna_cat
        )
        grafico_heatmap = generar_heatmap_plotly(df_limpio)

        config_actual = {
            "eliminar_duplicados": bool_duplicados,
            "estrategia_nulos": estrategia_norm,
            "valor_nulos": valor_manual,
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
                "grafico_barras": grafico_barras,
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
            titulo="No se pudo completar el análisis",
            detalle=str(mensaje),
            codigo_estado=200,
        )


def _asegurar_estado_procesado(sesion: dict[str, Any]) -> None:
    """Garantiza que la sesión cuente con DataFrame limpio antes de exportar o explicar."""
    if sesion.get("df_limpio") is not None:
        return

    config = sesion.get("config_limpieza", {})
    df_limpio, logs = limpiar_dataset(
        sesion["df_raw"],
        eliminar_duplicados=bool(config.get("eliminar_duplicados", True)),
        estrategia_nulos=str(config.get("estrategia_nulos", "mediana")),
        valor_nulos=config.get("valor_nulos", "0"),
    )
    estadisticas = resumen_descriptivo(df_limpio)

    col_cat = str(
        config.get("col_categoria") or sesion.get("col_categoria_default") or ""
    )
    col_num = str(
        config.get("col_numerica") or sesion.get("col_numerica_default") or ""
    )
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


@app.post("/analizar/explicar-grafico", response_class=HTMLResponse)
async def explicar_grafico_endpoint(
    request: Request,
    session_id: str = Form(default=""),
    tipo_grafico: str = Form(default="histograma"),
    col_numerica: str | None = Form(default=None),
    col_categoria: str | None = Form(default=None),
) -> HTMLResponse:
    """Genera una interpretación analítica breve para una visualización específica."""
    try:
        id_limpio = session_id.strip()
        if not id_limpio or id_limpio not in SESIONES:
            raise KeyError("La sesión de análisis expiró. Sube nuevamente el archivo.")

        sesion = SESIONES[id_limpio]
        _asegurar_estado_procesado(sesion)

        config = sesion.get("config_limpieza", {})
        metrica = (
            col_numerica
            or config.get("col_numerica")
            or sesion.get("col_numerica_default")
            or ""
        ).strip()
        categoria = (
            col_categoria
            or config.get("col_categoria")
            or sesion.get("col_categoria_default")
            or ""
        ).strip()

        datos_grafico = extraer_metricas_grafico(
            df=sesion["df_limpio"],
            tipo_grafico=tipo_grafico,
            col_metrica=metrica or None,
            col_categoria=categoria or None,
        )
        interpretacion = explicar_grafico_ia(
            contexto_dataset=sesion["contexto"],
            tipo_grafico=tipo_grafico,
            datos_grafico=datos_grafico,
        )

        return templates.TemplateResponse(
            request=request,
            name="partials/explicacion_grafico.html",
            context={
                "interpretacion": interpretacion,
                "ia_disponible": interpretacion != MENSAJE_IA_NO_DISPONIBLE,
            },
        )
    except (
        KeyError,
        ValueError,
        ZeroDivisionError,
        TimeoutError,
        OSError,
    ) as exc:
        mensaje = exc.args[0] if exc.args else str(exc)
        logger.warning("Error controlado en /analizar/explicar-grafico: %s", mensaje)
        return _renderizar_alerta_error(
            request=request,
            titulo="No se pudo interpretar la gráfica",
            detalle=str(mensaje),
            codigo_estado=200,
        )


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
            raise KeyError("La sesión solicitada no existe o ha expirado.")

        sesion = SESIONES[id_limpio]
        _asegurar_estado_procesado(sesion)

        formato_norm = formato.strip().lower()
        buffer = exportar_dataframe(sesion["df_limpio"], formato=formato_norm)

        base_nombre = Path(sesion["nombre_archivo"]).stem
        if formato_norm in {"xlsx", "excel"}:
            ext = "xlsx"
            media_type = (
                "application/vnd.openxmlformats-officedocument." "spreadsheetml.sheet"
            )
        else:
            ext = "csv"
            media_type = "text/csv; charset=utf-8"

        nombre_descarga = f"{base_nombre}_limpio.{ext}"
        headers = {"Content-Disposition": f'attachment; filename="{nombre_descarga}"'}
        return StreamingResponse(buffer, media_type=media_type, headers=headers)
    except (KeyError, ValueError, FileNotFoundError, OSError) as exc:
        mensaje = exc.args[0] if exc.args else str(exc)
        logger.warning("Error controlado en /export/dataset: %s", mensaje)
        return _renderizar_alerta_error(
            request=request,
            titulo="No se pudo exportar el archivo",
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
            raise KeyError("La sesión solicitada no existe o ha expirado.")

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
        nombre_pdf = f"reporte_{base_nombre}.pdf"
        headers = {"Content-Disposition": f'attachment; filename="{nombre_pdf}"'}
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
            titulo="No se pudo generar el reporte PDF",
            detalle=str(mensaje),
            codigo_estado=400,
        )
