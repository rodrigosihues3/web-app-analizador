"""Módulo de visualización dual: gráficos web Plotly y PNG estáticos para PDF."""

from __future__ import annotations

import io
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

COLOR_PRIMARIO = "#0f172a"
COLOR_ACENTO = "#2563eb"
COLOR_SECUNDARIO = "#475569"
COLOR_GRILLA = "#e2e8f0"
COLOR_FONDO = "#ffffff"
PALETA_CORPORATIVA: tuple[str, ...] = (
    "#2563eb",
    "#0f172a",
    "#475569",
    "#3b82f6",
    "#1e293b",
    "#64748b",
)


def _validar_columna_numerica(df: pd.DataFrame, columna: str) -> pd.Series:
    """Valida existencia, tipo numérico y presencia de datos no nulos."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("El parámetro 'df' debe ser un pandas.DataFrame.")
    if columna not in df.columns:
        raise KeyError(f"La columna '{columna}' no existe en el dataset.")
    if (
        not pd.api.types.is_numeric_dtype(df[columna])
        or pd.api.types.is_bool_dtype(df[columna])
    ):
        raise ValueError(f"La columna '{columna}' debe ser numérica.")

    serie = df[columna].dropna().astype(float)
    if serie.empty:
        raise ValueError(
            f"La columna '{columna}' no contiene valores válidos para graficar."
        )
    return serie


def _aplicar_layout_corporativo(fig: go.Figure, titulo: str) -> go.Figure:
    """Aplica estilo sobrio corporativo con fondo blanco a figuras Plotly."""
    fig.update_layout(
        title={
            "text": titulo,
            "font": {"size": 16, "color": COLOR_PRIMARIO, "family": "Arial"},
            "x": 0.02,
        },
        paper_bgcolor=COLOR_FONDO,
        plot_bgcolor=COLOR_FONDO,
        font={"color": COLOR_PRIMARIO, "family": "Arial", "size": 12},
        margin={"l": 50, "r": 25, "t": 55, "b": 45},
        xaxis={
            "gridcolor": COLOR_GRILLA,
            "linecolor": COLOR_SECUNDARIO,
            "zerolinecolor": COLOR_GRILLA,
        },
        yaxis={
            "gridcolor": COLOR_GRILLA,
            "linecolor": COLOR_SECUNDARIO,
            "zerolinecolor": COLOR_GRILLA,
        },
    )
    return fig


def generar_histograma_plotly(df: pd.DataFrame, columna: str) -> str:
    """Genera un fragmento HTML de histograma interactivo Plotly.

    Args:
        df: DataFrame fuente.
        columna: Columna numérica a graficar.

    Returns:
        Fragmento HTML generado mediante ``to_html(include_plotlyjs=False,
        full_html=False)``.
    """
    serie = _validar_columna_numerica(df, columna)

    fig = go.Figure(
        data=[
            go.Histogram(
                x=serie.tolist(),
                marker={
                    "color": COLOR_ACENTO,
                    "line": {"color": COLOR_PRIMARIO, "width": 1},
                },
                opacity=0.88,
                name=columna,
            )
        ]
    )
    _aplicar_layout_corporativo(fig, f"Distribución de {columna}")
    fig.update_xaxes(title_text=columna)
    fig.update_yaxes(title_text="Frecuencia")
    return fig.to_html(include_plotlyjs=False, full_html=False)


def generar_boxplot_plotly(
    df: pd.DataFrame, columna: str, col_grupo: str | None = None
) -> str:
    """Genera un fragmento HTML de diagrama de caja interactivo Plotly.

    Args:
        df: DataFrame fuente.
        columna: Columna numérica a analizar.
        col_grupo: Columna categórica opcional para comparar distribuciones.

    Returns:
        Fragmento HTML generado mediante ``to_html(include_plotlyjs=False,
        full_html=False)``.
    """
    _validar_columna_numerica(df, columna)

    fig = go.Figure()
    if col_grupo is not None:
        if col_grupo not in df.columns:
            raise KeyError(
                f"La columna de agrupación '{col_grupo}' no existe en el dataset."
            )

        df_val = df[[columna, col_grupo]].dropna()
        if df_val.empty:
            raise ValueError(
                f"No hay datos válidos para '{columna}' agrupados por '{col_grupo}'."
            )

        grupos = sorted(df_val[col_grupo].astype(str).unique().tolist())
        for idx, grupo in enumerate(grupos):
            valores_grupo = (
                df_val[df_val[col_grupo].astype(str) == grupo][columna]
                .astype(float)
                .tolist()
            )
            color = PALETA_CORPORATIVA[idx % len(PALETA_CORPORATIVA)]
            fig.add_trace(
                go.Box(
                    y=valores_grupo,
                    name=grupo,
                    marker_color=color,
                    line_color=COLOR_PRIMARIO,
                    boxmean=True,
                )
            )
        titulo = f"Distribución de {columna} por {col_grupo}"
        fig.update_xaxes(title_text=col_grupo)
    else:
        serie = df[columna].dropna().astype(float)
        fig.add_trace(
            go.Box(
                y=serie.tolist(),
                name=columna,
                marker_color=COLOR_ACENTO,
                line_color=COLOR_PRIMARIO,
                boxmean=True,
            )
        )
        titulo = f"Diagrama de Caja — {columna}"

    _aplicar_layout_corporativo(fig, titulo)
    fig.update_yaxes(title_text=columna)
    fig.update_layout(showlegend=bool(col_grupo))
    return fig.to_html(include_plotlyjs=False, full_html=False)


def generar_heatmap_plotly(df: pd.DataFrame) -> str:
    """Genera un mapa de calor de correlación de Pearson en HTML Plotly.

    Args:
        df: DataFrame fuente.

    Returns:
        Fragmento HTML generado mediante ``to_html(include_plotlyjs=False,
        full_html=False)``.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("El parámetro 'df' debe ser un pandas.DataFrame.")

    cols_num = [
        str(col)
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
        and not pd.api.types.is_bool_dtype(df[col])
    ]

    if len(cols_num) < 2:
        fig = go.Figure()
        _aplicar_layout_corporativo(fig, "Matriz de Correlación (Pearson)")
        fig.add_annotation(
            text="Se requieren al menos 2 columnas numéricas para el heatmap.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 13, "color": COLOR_SECUNDARIO},
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig.to_html(include_plotlyjs=False, full_html=False)

    corr = df[cols_num].corr(method="pearson").round(3).fillna(0.0)
    z_vals = corr.values.tolist()
    etiquetas_texto = [[f"{val:.2f}" for val in fila] for fila in z_vals]

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z_vals,
                x=cols_num,
                y=cols_num,
                text=etiquetas_texto,
                texttemplate="%{text}",
                textfont={"color": COLOR_PRIMARIO, "size": 11},
                colorscale=[
                    [0.0, "#475569"],
                    [0.5, "#f8fafc"],
                    [1.0, "#2563eb"],
                ],
                zmin=-1.0,
                zmax=1.0,
                colorbar={"title": "r"},
            )
        ]
    )
    _aplicar_layout_corporativo(fig, "Matriz de Correlación (Pearson)")
    fig.update_yaxes(autorange="reversed")
    return fig.to_html(include_plotlyjs=False, full_html=False)


def _guardar_figura_bytes(fig: plt.Figure) -> io.BytesIO:
    """Serializa una figura Matplotlib a PNG en memoria a 150 DPI."""
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=150,
        bbox_inches="tight",
        facecolor=COLOR_FONDO,
    )
    buffer.seek(0)
    return buffer


def generar_graficos_estaticos_pdf(
    df: pd.DataFrame,
) -> list[tuple[str, io.BytesIO]]:
    """Genera hasta 3 gráficos PNG en memoria para incrustar en el reporte PDF.

    Incluye histograma de la primera columna numérica, diagrama de caja
    comparativo y matriz de correlación cuando existen al menos 2 métricas.

    Args:
        df: DataFrame depurado o de entrada.

    Returns:
        Lista de tuplas ``(titulo_grafico, buffer_png)`` listas para ReportLab.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []

    cols_num = [
        str(col)
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
        and not pd.api.types.is_bool_dtype(df[col])
        and not df[col].dropna().empty
    ]
    if not cols_num:
        return []

    cols_cat = [
        str(col)
        for col in df.columns
        if col not in cols_num
        and 2 <= int(df[col].nunique(dropna=True)) <= 12
    ]

    graficos: list[tuple[str, io.BytesIO]] = []
    col_principal = cols_num[0]

    # 1. Histograma de la primera variable numérica
    fig_hist, ax_hist = plt.subplots(figsize=(6.8, 3.6), facecolor=COLOR_FONDO)
    try:
        valores = df[col_principal].dropna().to_numpy(dtype=float)
        bins = min(15, max(5, int(np.sqrt(valores.size))))
        ax_hist.hist(
            valores,
            bins=bins,
            color=COLOR_ACENTO,
            edgecolor=COLOR_PRIMARIO,
            alpha=0.88,
        )
        ax_hist.set_title(
            f"Distribución de {col_principal}",
            fontsize=11,
            fontweight="bold",
            color=COLOR_PRIMARIO,
            pad=10,
        )
        ax_hist.set_xlabel(col_principal, fontsize=9, color=COLOR_SECUNDARIO)
        ax_hist.set_ylabel("Frecuencia", fontsize=9, color=COLOR_SECUNDARIO)
        ax_hist.grid(axis="y", color=COLOR_GRILLA, linestyle="--", linewidth=0.7)
        ax_hist.spines["top"].set_visible(False)
        ax_hist.spines["right"].set_visible(False)
        graficos.append(
            (f"Histograma de {col_principal}", _guardar_figura_bytes(fig_hist))
        )
    except (ValueError, TypeError) as exc:
        logger.warning("No se pudo generar histograma estático: %s", exc)
    finally:
        plt.close(fig_hist)

    # 2. Boxplot comparativo (por categoría si existe, o entre columnas numéricas)
    fig_box, ax_box = plt.subplots(figsize=(6.8, 3.8), facecolor=COLOR_FONDO)
    try:
        if cols_cat:
            col_grupo = cols_cat[0]
            df_val = df[[col_principal, col_grupo]].dropna()
            etiquetas = sorted(df_val[col_grupo].astype(str).unique().tolist())
            series_grupos = [
                df_val[df_val[col_grupo].astype(str) == g][col_principal]
                .astype(float)
                .to_numpy()
                for g in etiquetas
            ]
            titulo_box = f"Comparativa de {col_principal} por {col_grupo}"
            etiqueta_x = col_grupo
        else:
            cols_box = cols_num[:4]
            etiquetas = cols_box
            series_grupos = [
                df[c].dropna().astype(float).to_numpy() for c in cols_box
            ]
            titulo_box = "Distribución Comparativa (Boxplot)"
            etiqueta_x = "Variables Numéricas"

        if series_grupos:
            bp = ax_box.boxplot(
                series_grupos,
                tick_labels=etiquetas,
                patch_artist=True,
                medianprops={"color": COLOR_PRIMARIO, "linewidth": 1.6},
                whiskerprops={"color": COLOR_SECUNDARIO},
                capprops={"color": COLOR_SECUNDARIO},
            )
            for idx, parche in enumerate(bp["boxes"]):
                parche.set_facecolor(
                    PALETA_CORPORATIVA[idx % len(PALETA_CORPORATIVA)]
                )
                parche.set_alpha(0.78)
                parche.set_edgecolor(COLOR_PRIMARIO)

            ax_box.set_title(
                titulo_box,
                fontsize=11,
                fontweight="bold",
                color=COLOR_PRIMARIO,
                pad=10,
            )
            ax_box.set_xlabel(etiqueta_x, fontsize=9, color=COLOR_SECUNDARIO)
            ax_box.set_ylabel(col_principal, fontsize=9, color=COLOR_SECUNDARIO)
            ax_box.tick_params(axis="x", labelsize=8, rotation=18)
            ax_box.grid(
                axis="y", color=COLOR_GRILLA, linestyle="--", linewidth=0.7
            )
            ax_box.spines["top"].set_visible(False)
            ax_box.spines["right"].set_visible(False)
            graficos.append((titulo_box, _guardar_figura_bytes(fig_box)))
    except (ValueError, TypeError) as exc:
        logger.warning("No se pudo generar boxplot estático: %s", exc)
    finally:
        plt.close(fig_box)

    # 3. Heatmap de correlación (si hay al menos 2 variables numéricas)
    if len(cols_num) >= 2:
        fig_hm, ax_hm = plt.subplots(figsize=(6.4, 4.4), facecolor=COLOR_FONDO)
        try:
            corr = df[cols_num].corr(method="pearson").round(2).fillna(0.0)
            im = ax_hm.imshow(
                corr.values, cmap="Blues", vmin=-1.0, vmax=1.0, aspect="auto"
            )
            fig_hm.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

            ax_hm.set_xticks(range(len(cols_num)))
            ax_hm.set_yticks(range(len(cols_num)))
            ax_hm.set_xticklabels(
                cols_num, fontsize=8, rotation=25, ha="right", color=COLOR_PRIMARIO
            )
            ax_hm.set_yticklabels(cols_num, fontsize=8, color=COLOR_PRIMARIO)

            for i in range(len(cols_num)):
                for j in range(len(cols_num)):
                    val = float(corr.iloc[i, j])
                    color_texto = "#ffffff" if val >= 0.55 else COLOR_PRIMARIO
                    ax_hm.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color=color_texto,
                        fontsize=8,
                        fontweight="bold",
                    )

            ax_hm.set_title(
                "Matriz de Correlación de Pearson",
                fontsize=11,
                fontweight="bold",
                color=COLOR_PRIMARIO,
                pad=10,
            )
            graficos.append(
                ("Matriz de Correlación (Pearson)", _guardar_figura_bytes(fig_hm))
            )
        except (ValueError, TypeError) as exc:
            logger.warning("No se pudo generar heatmap estático: %s", exc)
        finally:
            plt.close(fig_hm)

    return graficos
