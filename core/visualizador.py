"""Módulo de visualización dual: gráficos web Plotly y PNG estáticos para PDF."""

import io
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.utils import es_columna_numerica, obtener_columnas_numericas

logger = logging.getLogger(__name__)

COLOR_PRIMARIO = "#18181b"
COLOR_ACENTO = "#0f766e"
COLOR_SECUNDARIO = "#52525b"
COLOR_TERCIARIO = "#71717a"
COLOR_GRILLA = "#e7e5e4"
COLOR_FONDO = "#ffffff"
PALETA_CORPORATIVA: tuple[str, ...] = (
    "#0f766e",
    "#18181b",
    "#52525b",
    "#0d9488",
    "#27272a",
    "#71717a",
)
PLOTLY_CONFIG: dict[str, bool] = {
    "displayModeBar": False,
    "responsive": True,
}


def _validar_columna_numerica(df: pd.DataFrame, columna: str) -> pd.Series:
    """Valida existencia, tipo numérico y presencia de datos no nulos."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("El parámetro 'df' debe ser un pandas.DataFrame.")
    if columna not in df.columns:
        raise KeyError(f"La columna '{columna}' no existe en el dataset.")
    if not es_columna_numerica(df[columna]):
        raise ValueError(f"La columna '{columna}' debe ser numérica.")

    serie = df[columna].dropna().astype(float)
    if serie.empty:
        raise ValueError(
            f"La columna '{columna}' no contiene valores válidos para graficar."
        )
    return serie


def _aplicar_layout_corporativo(fig: go.Figure, titulo: str) -> go.Figure:
    """Aplica estilo sobrio minimalista con fondo transparente a figuras Plotly."""
    fuente = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"
    fig.update_layout(
        title={
            "text": titulo,
            "font": {"size": 14, "color": COLOR_PRIMARIO, "family": fuente},
            "x": 0.01,
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLOR_PRIMARIO, "family": fuente, "size": 12},
        hovermode="closest",
        hoverlabel={
            "bgcolor": "#18181b",
            "bordercolor": "#27272a",
            "font": {"color": "#fafafa", "family": fuente, "size": 12},
        },
        margin={"l": 48, "r": 20, "t": 48, "b": 44},
        xaxis={
            "gridcolor": COLOR_GRILLA,
            "linecolor": COLOR_GRILLA,
            "zerolinecolor": COLOR_GRILLA,
            "tickfont": {"color": COLOR_SECUNDARIO, "size": 11},
        },
        yaxis={
            "gridcolor": COLOR_GRILLA,
            "linecolor": COLOR_GRILLA,
            "zerolinecolor": COLOR_GRILLA,
            "tickfont": {"color": COLOR_SECUNDARIO, "size": 11},
        },
    )
    return fig


def generar_histograma_plotly(df: pd.DataFrame, columna: str) -> str:
    """Genera un fragmento HTML de histograma interactivo con Media, Mediana y Moda.

    Args:
        df: DataFrame fuente.
        columna: Columna numérica a graficar.

    Returns:
        Fragmento HTML generado mediante ``to_html(include_plotlyjs=False,
        full_html=False, config=PLOTLY_CONFIG)``.
    """
    serie = _validar_columna_numerica(df, columna)
    media = float(serie.mean())
    mediana = float(serie.median())
    modas = serie.mode(dropna=True)
    moda = float(modas.iloc[0]) if not modas.empty else None

    fig = go.Figure(
        data=[
            go.Histogram(
                x=serie.tolist(),
                marker={
                    "color": "rgba(15, 118, 110, 0.78)",
                    "line": {"color": "#115e59", "width": 1},
                },
                name=columna,
                hovertemplate=f"{columna}: %{{x}}<br>Frecuencia: %{{y}}<extra></extra>",
            )
        ]
    )

    marcadores: list[tuple[str, float, str, str, float]] = [
        ("Media", media, COLOR_ACENTO, "dash", 0.99),
        ("Mediana", mediana, COLOR_PRIMARIO, "solid", 0.88),
    ]
    if moda is not None:
        marcadores.append(("Moda", moda, COLOR_TERCIARIO, "dot", 0.77))

    for nombre, valor, color, trazo, pos_y in marcadores:
        fig.add_vline(
            x=valor,
            line_width=2,
            line_dash=trazo,
            line_color=color,
        )
        fig.add_annotation(
            x=valor,
            y=pos_y,
            yref="paper",
            text=f"{nombre}: {valor:.2f}",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            xshift=5,
            font={"size": 11, "color": color},
            bgcolor="rgba(255,255,255,0.88)",
            borderpad=2,
        )

    _aplicar_layout_corporativo(fig, f"Distribución de {columna}")
    fig.update_xaxes(title_text=columna)
    fig.update_yaxes(title_text="Frecuencia")
    return fig.to_html(
        include_plotlyjs=False,
        full_html=False,
        config=PLOTLY_CONFIG,
    )


def generar_barras_categoria_plotly(
    df: pd.DataFrame, col_metrica: str, col_grupo: str
) -> str:
    """Genera un gráfico de barras ordenado por promedio de categoría (top 12).

    Incluye etiquetas directas sobre cada barra y una línea punteada de
    referencia para el promedio general.
    """
    _validar_columna_numerica(df, col_metrica)
    if col_grupo not in df.columns:
        raise KeyError(
            f"La columna de agrupación '{col_grupo}' no existe en el dataset."
        )

    df_val = df[[col_metrica, col_grupo]].dropna()
    if df_val.empty:
        raise ValueError(
            f"No hay datos válidos para '{col_metrica}' agrupados por '{col_grupo}'."
        )

    promedio_general = float(df_val[col_metrica].astype(float).mean())
    agrupado = (
        df_val.groupby(df_val[col_grupo].astype(str), observed=True)[col_metrica]
        .mean()
        .sort_values(ascending=False)
        .head(12)
    )

    categorias = agrupado.index.tolist()
    promedios = [round(float(v), 2) for v in agrupado.values]

    fig = go.Figure(
        data=[
            go.Bar(
                x=categorias,
                y=promedios,
                text=[f"{v:.2f}" for v in promedios],
                textposition="outside",
                textfont={"size": 11, "color": COLOR_SECUNDARIO},
                marker={"color": COLOR_ACENTO, "opacity": 0.9},
                hovertemplate="%{x}<br>Promedio: %{y:.2f}<extra></extra>",
                name=col_metrica,
            )
        ]
    )

    fig.add_hline(
        y=promedio_general,
        line_width=1.5,
        line_dash="dot",
        line_color=COLOR_PRIMARIO,
    )
    fig.add_annotation(
        x=0.99,
        xref="paper",
        y=promedio_general,
        text=f"Promedio general: {promedio_general:.2f}",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        yshift=4,
        font={"size": 11, "color": COLOR_PRIMARIO},
        bgcolor="rgba(255,255,255,0.88)",
        borderpad=2,
    )

    _aplicar_layout_corporativo(
        fig, f"Promedio de {col_metrica} por {col_grupo}"
    )
    max_y = max(promedios + [promedio_general])
    min_y = min(promedios + [0.0])
    techo_y = max_y * 1.18 if max_y > 0 else 1.0
    piso_y = min_y * 1.12 if min_y < 0 else 0.0

    fig.update_xaxes(title_text=col_grupo)
    fig.update_yaxes(title_text=f"Promedio ({col_metrica})", range=[piso_y, techo_y])
    return fig.to_html(
        include_plotlyjs=False,
        full_html=False,
        config=PLOTLY_CONFIG,
    )


def generar_boxplot_plotly(
    df: pd.DataFrame, columna: str, col_grupo: str | None = None
) -> str:
    """Genera un fragmento HTML de diagrama de caja interactivo Plotly.

    Conservada para compatibilidad con la suite ``test_core.py`` y uso directo.
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
    return fig.to_html(
        include_plotlyjs=False,
        full_html=False,
        config=PLOTLY_CONFIG,
    )


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

    cols_num = obtener_columnas_numericas(df)

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
        return fig.to_html(
            include_plotlyjs=False,
            full_html=False,
            config=PLOTLY_CONFIG,
        )

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
                    [0.0, "#71717a"],
                    [0.5, "#f5f5f4"],
                    [1.0, "#0f766e"],
                ],
                zmin=-1.0,
                zmax=1.0,
                colorbar={"title": "r"},
            )
        ]
    )
    _aplicar_layout_corporativo(fig, "Matriz de Correlación (Pearson)")
    fig.update_yaxes(autorange="reversed")
    return fig.to_html(
        include_plotlyjs=False,
        full_html=False,
        config=PLOTLY_CONFIG,
    )


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
    """Genera hasta 3 gráficos PNG en memoria para incrustar en el reporte PDF."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []

    cols_num = [
        col
        for col in obtener_columnas_numericas(df)
        if not df[col].dropna().empty
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
