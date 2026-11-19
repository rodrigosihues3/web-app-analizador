"""Módulo de exportación tabular (CSV / Excel) y generación de reportes PDF."""

from __future__ import annotations

from datetime import datetime
import io
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

COLOR_PRIMARIO_HEX = "#0f172a"
COLOR_ACENTO_HEX = "#2563eb"
COLOR_SECUNDARIO_HEX = "#475569"
COLOR_FONDO_SUAVE_HEX = "#f8fafc"
COLOR_BORDE_HEX = "#cbd5e1"


def exportar_dataframe(df: pd.DataFrame, formato: str = "csv") -> io.BytesIO:
    """Exporta un DataFrame hacia un buffer en memoria en CSV o Excel.

    Args:
        df: DataFrame procesado a exportar.
        formato: Formato destino en ``{"csv", "xlsx"}``.

    Returns:
        Buffer ``io.BytesIO`` posicionado al inicio.

    Raises:
        ValueError: Si ``df`` es inválido o el formato no está soportado.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("El parámetro 'df' debe ser un pandas.DataFrame.")

    formato_norm = formato.strip().lower()
    buffer = io.BytesIO()

    try:
        if formato_norm == "csv":
            contenido_csv = df.to_csv(index=False)
            # utf-8-sig añade BOM para que Excel en Windows muestre tildes y ñ sin errores
            buffer.write(contenido_csv.encode("utf-8-sig"))
        elif formato_norm in {"xlsx", "excel"}:
            # ExcelWriter escribe el archivo .xlsx directamente en el buffer de memoria
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(
                    writer, index=False, sheet_name="Datos_Procesados"
                )
        else:
            raise ValueError(
                f"Formato de exportación no soportado: '{formato}'. "
                "Use 'csv' o 'xlsx'."
            )
    except (OSError, ValueError, TypeError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(
            f"No fue posible exportar el dataset en formato '{formato}': {exc}"
        ) from exc

    buffer.seek(0)
    return buffer


def _estilo_tabla_corporativa() -> TableStyle:
    """Retorna el estilo visual estándar para las tablas del informe PDF."""
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(COLOR_PRIMARIO_HEX)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("ALIGN", (0, 0), (-1, 0), "LEFT"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("TOPPADDING", (0, 0), (-1, 0), 6),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 7.5),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor(COLOR_PRIMARIO_HEX)),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [colors.white, colors.HexColor(COLOR_FONDO_SUAVE_HEX)],
            ),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(COLOR_BORDE_HEX)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 1), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        ]
    )


def _formatear_numero(valor: Any, decimales: int = 2) -> str:
    """Formatea valores numéricos de manera uniforme para celdas PDF."""
    if valor is None:
        return "N/A"
    if isinstance(valor, (int, float)):
        return f"{valor:,.{decimales}f}"
    return str(valor)


def _escapar_texto(valor: Any) -> str:
    """Escapa caracteres XML especiales para uso seguro en Paragraph."""
    return escape(str(valor))


def generar_reporte_pdf(
    contexto: dict[str, Any],
    auditoria: dict[str, Any],
    logs_limpieza: list[str],
    estadisticas: list[dict[str, Any]],
    valores_clave: dict[str, Any] | None,
    imagenes_graficos: list[tuple[str, io.BytesIO]],
) -> io.BytesIO:
    """Construye un documento PDF A4 formal usando ReportLab Platypus.

    Args:
        contexto: Diccionario con ``titulo_reporte`` y ``resumen_ejecutivo``.
        auditoria: Resultado de ``DatasetAuditor.auditar_estructura()``.
        logs_limpieza: Registro textual de transformaciones aplicadas.
        estadisticas: Salida de ``resumen_descriptivo()``.
        valores_clave: Salida opcional de ``analizar_valores_clave()``.
        imagenes_graficos: Lista de tuplas ``(titulo, buffer_png)``.

    Returns:
        Buffer ``io.BytesIO`` con el PDF generado y cursor al inicio.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title=str(
            contexto.get(
                "titulo_reporte", "Reporte Analizador Multiparadigma"
            )
        ),
    )

    estilos = getSampleStyleSheet()
    estilo_institucional = ParagraphStyle(
        "Institucional",
        parent=estilos["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        textColor=colors.HexColor(COLOR_ACENTO_HEX),
        spaceAfter=4,
    )
    estilo_titulo = ParagraphStyle(
        "TituloPrincipal",
        parent=estilos["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor(COLOR_PRIMARIO_HEX),
        spaceAfter=6,
    )
    estilo_subtitulo = ParagraphStyle(
        "Seccion",
        parent=estilos["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=14,
        textColor=colors.HexColor(COLOR_PRIMARIO_HEX),
        spaceBefore=12,
        spaceAfter=6,
    )
    estilo_cuerpo = ParagraphStyle(
        "Cuerpo",
        parent=estilos["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor(COLOR_SECUNDARIO_HEX),
        spaceAfter=6,
    )
    estilo_celda = ParagraphStyle(
        "Celda",
        parent=estilos["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor(COLOR_PRIMARIO_HEX),
    )

    elementos: list[Any] = []
    fecha_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 1. Encabezado institucional y título
    elementos.append(
        Paragraph(
            f"ANALIZADOR MULTIPARADIGMA DE DATOS | INFORME TÉCNICO ({fecha_str})",
            estilo_institucional,
        )
    )
    titulo_reporte = str(
        contexto.get("titulo_reporte", "Informe de Auditoría y Análisis de Datos")
    )
    resumen_ejecutivo = str(
        contexto.get(
            "resumen_ejecutivo",
            "Diagnóstico técnico y estadístico generado automáticamente.",
        )
    )
    elementos.append(Paragraph(_escapar_texto(titulo_reporte), estilo_titulo))
    elementos.append(
        Paragraph(_escapar_texto(resumen_ejecutivo), estilo_cuerpo)
    )
    elementos.append(Spacer(1, 0.25 * cm))

    # 2. Auditoría de calidad y esquema
    elementos.append(
        Paragraph("1. Auditoría de Calidad y Estructura Original", estilo_subtitulo)
    )
    var_cero = auditoria.get("columnas_varianza_cero", [])
    var_cero_txt = ", ".join(str(c) for c in var_cero) if var_cero else "Ninguna"
    resumen_audit_data = [
        ["Filas", "Columnas", "Memoria (KB)", "Duplicados", "Varianza Cero"],
        [
            str(auditoria.get("filas", 0)),
            str(auditoria.get("columnas", 0)),
            _formatear_numero(auditoria.get("memoria_kb", 0.0), 2),
            str(auditoria.get("duplicados", 0)),
            Paragraph(_escapar_texto(var_cero_txt), estilo_celda),
        ],
    ]
    tabla_resumen = Table(
        resumen_audit_data,
        colWidths=[2.5 * cm, 2.5 * cm, 3.0 * cm, 2.5 * cm, 6.9 * cm],
    )
    tabla_resumen.setStyle(_estilo_tabla_corporativa())
    elementos.append(tabla_resumen)
    elementos.append(Spacer(1, 0.3 * cm))

    esquema = auditoria.get("esquema", [])
    if esquema:
        filas_esquema = [
            ["Columna", "Tipo Lógico", "Dtype", "Nulos", "% Nulos", "Únicos"]
        ]
        for item in esquema:
            filas_esquema.append(
                [
                    Paragraph(_escapar_texto(item.get("columna", "")), estilo_celda),
                    str(item.get("tipo_logico", "")),
                    str(item.get("dtype", "")),
                    str(item.get("nulos", 0)),
                    f"{item.get('porcentaje_nulos', 0.0):.2f}%",
                    str(item.get("valores_unicos", 0)),
                ]
            )
        tabla_esquema = Table(
            filas_esquema,
            colWidths=[4.8 * cm, 2.6 * cm, 2.5 * cm, 2.0 * cm, 2.5 * cm, 3.0 * cm],
            repeatRows=1,
        )
        tabla_esquema.setStyle(_estilo_tabla_corporativa())
        elementos.append(tabla_esquema)
        elementos.append(Spacer(1, 0.25 * cm))

    outliers_iqr = auditoria.get("outliers_iqr", {})
    if outliers_iqr:
        filas_outliers = [
            ["Columna Numérica", "Outliers IQR", "Límite Inferior", "Límite Superior"]
        ]
        for col_name, info in outliers_iqr.items():
            filas_outliers.append(
                [
                    Paragraph(_escapar_texto(col_name), estilo_celda),
                    str(info.get("cantidad", 0)),
                    _formatear_numero(info.get("limite_inferior"), 2),
                    _formatear_numero(info.get("limite_superior"), 2),
                ]
            )
        tabla_outliers = Table(
            filas_outliers,
            colWidths=[5.8 * cm, 3.2 * cm, 4.2 * cm, 4.2 * cm],
            repeatRows=1,
        )
        tabla_outliers.setStyle(_estilo_tabla_corporativa())
        elementos.append(tabla_outliers)

    # 3. Bitácora de limpieza funcional
    elementos.append(
        Paragraph("2. Bitácora de Limpieza Funcional", estilo_subtitulo)
    )
    if logs_limpieza:
        for paso in logs_limpieza:
            elementos.append(
                Paragraph(f"&bull; {_escapar_texto(paso)}", estilo_cuerpo)
            )
    else:
        elementos.append(
            Paragraph("No se registraron operaciones de limpieza.", estilo_cuerpo)
        )

    # 4. Métricas estadísticas descriptivas
    elementos.append(
        Paragraph("3. Estadísticas Descriptivas (Dataset Procesado)", estilo_subtitulo)
    )
    if estadisticas:
        filas_stats = [
            [
                "Columna",
                "N",
                "Media",
                "Mediana",
                "Desv.Est",
                "Mín",
                "Máx",
                "Q25",
                "Q75",
                "Asim.",
            ]
        ]
        for st in estadisticas:
            filas_stats.append(
                [
                    Paragraph(_escapar_texto(st.get("columna", "")), estilo_celda),
                    str(st.get("conteo", 0)),
                    _formatear_numero(st.get("media"), 2),
                    _formatear_numero(st.get("mediana"), 2),
                    _formatear_numero(st.get("desviacion_estandar"), 2),
                    _formatear_numero(st.get("minimo"), 2),
                    _formatear_numero(st.get("maximo"), 2),
                    _formatear_numero(st.get("q25"), 2),
                    _formatear_numero(st.get("q75"), 2),
                    _formatear_numero(st.get("asimetria"), 2),
                ]
            )
        tabla_stats = Table(
            filas_stats,
            colWidths=[
                3.2 * cm,
                1.1 * cm,
                1.7 * cm,
                1.7 * cm,
                1.7 * cm,
                1.6 * cm,
                1.6 * cm,
                1.6 * cm,
                1.6 * cm,
                1.6 * cm,
            ],
            repeatRows=1,
        )
        tabla_stats.setStyle(_estilo_tabla_corporativa())
        elementos.append(tabla_stats)
    else:
        elementos.append(
            Paragraph(
                "No se encontraron columnas numéricas para resumir.", estilo_cuerpo
            )
        )

    # 5. Hallazgos clave
    if valores_clave:
        elementos.append(
            Paragraph("4. Hallazgos Clave y Liderazgo por Categoría", estilo_subtitulo)
        )
        col_cat = _escapar_texto(valores_clave.get("col_categoria", ""))
        col_met = _escapar_texto(valores_clave.get("col_metrica", ""))
        lider_prom = valores_clave.get("categoria_lider_promedio", {})
        lider_acum = valores_clave.get("categoria_lider_acumulado", {})

        elementos.append(
            Paragraph(
                f"Métrica evaluada: <b>{col_met}</b> agrupada por <b>{col_cat}</b>. "
                f"Valor máximo individual: <b>{_formatear_numero(valores_clave.get('valor_maximo'), 2)}</b> | "
                f"Valor mínimo individual: <b>{_formatear_numero(valores_clave.get('valor_minimo'), 2)}</b>.",
                estilo_cuerpo,
            )
        )
        elementos.append(
            Paragraph(
                f"Categoría líder en promedio: <b>{_escapar_texto(lider_prom.get('categoria', 'N/A'))}</b> "
                f"({_formatear_numero(lider_prom.get('promedio'), 2)}). "
                f"Categoría líder en acumulado: <b>{_escapar_texto(lider_acum.get('categoria', 'N/A'))}</b> "
                f"({_formatear_numero(lider_acum.get('acumulado'), 2)}).",
                estilo_cuerpo,
            )
        )

        tabla_agrupada = valores_clave.get("tabla_agrupada", [])
        if tabla_agrupada:
            filas_grp = [
                ["Categoría", "Promedio", "Acumulado", "Registros", "Máximo"]
            ]
            for row in tabla_agrupada:
                filas_grp.append(
                    [
                        Paragraph(_escapar_texto(row.get("categoria", "")), estilo_celda),
                        _formatear_numero(row.get("mean"), 2),
                        _formatear_numero(row.get("sum"), 2),
                        str(row.get("count", 0)),
                        _formatear_numero(row.get("max"), 2),
                    ]
                )
            tabla_clave = Table(
                filas_grp,
                colWidths=[5.4 * cm, 3.0 * cm, 3.2 * cm, 2.8 * cm, 3.0 * cm],
                repeatRows=1,
            )
            tabla_clave.setStyle(_estilo_tabla_corporativa())
            elementos.append(tabla_clave)

    # 6. Gráficos estáticos incrustados
    if imagenes_graficos:
        elementos.append(
            Paragraph("5. Visualizaciones Estadísticas", estilo_subtitulo)
        )
        ancho_max = 16.5 * cm
        for titulo_img, img_buffer in imagenes_graficos:
            try:
                img_buffer.seek(0)
                datos_img = img_buffer.read()
                img_buffer.seek(0)
                copia_stream = io.BytesIO(datos_img)
                imagen_rl = Image(copia_stream)
                if imagen_rl.drawWidth > ancho_max and imagen_rl.drawWidth > 0:
                    escala = ancho_max / float(imagen_rl.drawWidth)
                    imagen_rl.drawWidth = ancho_max
                    imagen_rl.drawHeight = imagen_rl.drawHeight * escala
                imagen_rl.hAlign = "CENTER"
                elementos.append(
                    KeepTogether(
                        [
                            Paragraph(
                                f"<b>{_escapar_texto(titulo_img)}</b>",
                                estilo_cuerpo,
                            ),
                            imagen_rl,
                            Spacer(1, 0.35 * cm),
                        ]
                    )
                )
            except (OSError, ValueError, TypeError):
                continue

    doc.build(elementos)
    buffer.seek(0)
    return buffer
