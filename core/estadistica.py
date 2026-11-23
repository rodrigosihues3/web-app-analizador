"""Módulo funcional de análisis estadístico descriptivo y agrupaciones."""

import logging
from typing import Any

import numpy as np
import pandas as pd

from core.utils import es_columna_numerica, obtener_columnas_numericas

logger = logging.getLogger(__name__)


def _a_valor_nativo(valor: Any) -> Any:
    """Convierte valores de NumPy o Pandas a tipos básicos de Python para JSON."""
    if pd.isna(valor):
        return None
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, np.floating):
        return round(float(valor), 4)
    if isinstance(valor, np.bool_):
        return bool(valor)
    if isinstance(valor, pd.Timestamp):
        return valor.isoformat()
    return valor


def resumen_descriptivo(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Calcula estadísticas descriptivas para cada columna numérica del DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("El parámetro 'df' debe ser un pandas.DataFrame.")

    resultados: list[dict[str, Any]] = []
    for col in obtener_columnas_numericas(df):
        try:
            serie = df[col].dropna().astype(float)
            if serie.empty:
                raise ValueError(
                    f"La columna '{col}' no posee valores numéricos no nulos."
                )

            valores = serie.to_numpy(dtype=float)
            conteo = int(valores.size)
            ddof = 1 if conteo > 1 else 0

            modas = serie.mode(dropna=True)
            moda = round(float(modas.iloc[0]), 4) if not modas.empty else 0.0

            asimetria_raw = serie.skew()
            asimetria = 0.0 if pd.isna(asimetria_raw) else float(asimetria_raw)

            resultados.append(
                {
                    "columna": col,
                    "conteo": conteo,
                    "media": round(float(np.mean(valores)), 4),
                    "mediana": round(float(np.median(valores)), 4),
                    "moda": moda,
                    "desviacion_estandar": round(
                        float(np.std(valores, ddof=ddof)), 4
                    ),
                    "varianza": round(float(np.var(valores, ddof=ddof)), 4),
                    "minimo": round(float(np.min(valores)), 4),
                    "maximo": round(float(np.max(valores)), 4),
                    "q25": round(float(np.percentile(valores, 25)), 4),
                    "q75": round(float(np.percentile(valores, 75)), 4),
                    "asimetria": round(asimetria, 4),
                }
            )
        except (ValueError, ZeroDivisionError, TypeError) as exc:
            logger.warning(
                "Columna numérica omitida en resumen descriptivo (%s): %s",
                col,
                exc,
            )
            continue

    return resultados


def analizar_valores_clave(
    df: pd.DataFrame, col_categoria: str, col_metrica: str
) -> dict[str, Any]:
    """Obtiene valores extremos, medidas centrales y agrupación por categoría."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("El parámetro 'df' debe ser un pandas.DataFrame.")
    if col_categoria not in df.columns:
        raise KeyError(
            f"La columna de categoría '{col_categoria}' no existe en el dataset."
        )
    if col_metrica not in df.columns:
        raise KeyError(
            f"La columna métrica '{col_metrica}' no existe en el dataset."
        )
    if not es_columna_numerica(df[col_metrica]):
        raise ValueError(
            f"La columna métrica '{col_metrica}' debe ser de tipo numérico."
        )

    df_valido = df.dropna(subset=[col_categoria, col_metrica])
    if df_valido.empty:
        raise ValueError(
            "No existen filas válidas (no nulas) para las columnas "
            f"'{col_categoria}' y '{col_metrica}'."
        )

    serie_metrica = df_valido[col_metrica].astype(float)
    valores_num = serie_metrica.to_numpy(dtype=float)
    media_metrica = round(float(np.mean(valores_num)), 4)
    mediana_metrica = round(float(np.median(valores_num)), 4)
    modas_metrica = serie_metrica.mode(dropna=True)
    moda_metrica = (
        round(float(modas_metrica.iloc[0]), 4)
        if not modas_metrica.empty
        else 0.0
    )

    idx_max = df_valido[col_metrica].idxmax()
    idx_min = df_valido[col_metrica].idxmin()

    fila_max = df_valido.loc[idx_max]
    if isinstance(fila_max, pd.DataFrame):
        fila_max = fila_max.iloc[0]

    fila_min = df_valido.loc[idx_min]
    if isinstance(fila_min, pd.DataFrame):
        fila_min = fila_min.iloc[0]

    registro_maximo = {
        str(k): _a_valor_nativo(v) for k, v in fila_max.to_dict().items()
    }
    registro_minimo = {
        str(k): _a_valor_nativo(v) for k, v in fila_min.to_dict().items()
    }

    agrupado = (
        df_valido.groupby(col_categoria, observed=True)[col_metrica]
        .agg(["mean", "sum", "count", "max"])
        .sort_values(by="sum", ascending=False)
    )
    if agrupado.empty:
        raise ValueError(f"No fue posible agrupar por '{col_categoria}'.")

    cat_lider_promedio = str(agrupado["mean"].idxmax())
    val_lider_promedio = round(float(agrupado["mean"].max()), 4)
    cat_lider_acumulado = str(agrupado["sum"].idxmax())
    val_lider_acumulado = round(float(agrupado["sum"].max()), 4)

    tabla_agrupada = [
        {
            "categoria": str(categoria),
            "mean": round(float(fila["mean"]), 4),
            "sum": round(float(fila["sum"]), 4),
            "count": int(fila["count"]),
            "max": round(float(fila["max"]), 4),
        }
        for categoria, fila in agrupado.iterrows()
    ]

    return {
        "col_categoria": col_categoria,
        "col_metrica": col_metrica,
        "media_metrica": media_metrica,
        "mediana_metrica": mediana_metrica,
        "moda_metrica": moda_metrica,
        "registro_maximo": registro_maximo,
        "valor_maximo": round(float(fila_max[col_metrica]), 4),
        "registro_minimo": registro_minimo,
        "valor_minimo": round(float(fila_min[col_metrica]), 4),
        "categoria_lider_promedio": {
            "categoria": cat_lider_promedio,
            "promedio": val_lider_promedio,
        },
        "categoria_lider_acumulado": {
            "categoria": cat_lider_acumulado,
            "acumulado": val_lider_acumulado,
        },
        "tabla_agrupada": tabla_agrupada,
    }


def matriz_correlacion(df: pd.DataFrame) -> dict[str, Any]:
    """Calcula la matriz de correlación de Pearson entre columnas numéricas."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("El parámetro 'df' debe ser un pandas.DataFrame.")

    cols_num = obtener_columnas_numericas(df)
    if len(cols_num) < 2:
        return {"columnas": cols_num, "matriz": {}}

    try:
        corr_df = df[cols_num].corr(method="pearson").round(3).fillna(0.0)
        matriz = {
            str(col): {
                str(idx): round(float(corr_df.loc[idx, col]), 3)
                for idx in corr_df.index
            }
            for col in corr_df.columns
        }
        return {"columnas": cols_num, "matriz": matriz}
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        logger.warning("Error al calcular matriz de correlación: %s", exc)
        return {"columnas": cols_num, "matriz": {}}


def extraer_metricas_grafico(
    df: pd.DataFrame,
    tipo_grafico: str,
    col_metrica: str | None = None,
    col_categoria: str | None = None,
) -> dict[str, Any]:
    """Extrae las cifras resumidas de una gráfica para enviarlas a la IA."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("El parámetro 'df' debe ser un pandas.DataFrame.")

    tipo = tipo_grafico.strip().lower()
    if tipo == "histograma":
        if not col_metrica or col_metrica not in df.columns:
            raise KeyError(
                f"La columna numérica '{col_metrica}' no existe en el dataset."
            )
        serie = df[col_metrica].dropna().astype(float)
        if serie.empty:
            raise ValueError(
                f"La columna '{col_metrica}' no tiene valores válidos."
            )
        valores = serie.to_numpy(dtype=float)
        modas = serie.mode(dropna=True)
        moda = round(float(modas.iloc[0]), 4) if not modas.empty else None
        return {
            "columna": col_metrica,
            "conteo": int(valores.size),
            "media": round(float(np.mean(valores)), 4),
            "mediana": round(float(np.median(valores)), 4),
            "moda": moda,
            "minimo": round(float(np.min(valores)), 4),
            "maximo": round(float(np.max(valores)), 4),
            "q25": round(float(np.percentile(valores, 25)), 4),
            "q75": round(float(np.percentile(valores, 75)), 4),
        }

    if tipo in {"barras", "barras_categoria", "boxplot"}:
        if not col_metrica or col_metrica not in df.columns:
            raise KeyError(
                f"La columna numérica '{col_metrica}' no existe en el dataset."
            )
        if not col_categoria or col_categoria not in df.columns:
            raise KeyError(
                f"La columna categórica '{col_categoria}' no existe en el dataset."
            )
        df_val = df[[col_metrica, col_categoria]].dropna()
        if df_val.empty:
            raise ValueError("No hay registros válidos para agrupar.")
        promedio_general = round(
            float(df_val[col_metrica].astype(float).mean()), 4
        )
        agrupado = (
            df_val.groupby(col_categoria, observed=True)[col_metrica]
            .agg(["mean", "count"])
            .sort_values(by="mean", ascending=False)
            .head(12)
        )
        ranking = [
            {
                "categoria": str(cat),
                "promedio": round(float(fila["mean"]), 4),
                "registros": int(fila["count"]),
            }
            for cat, fila in agrupado.iterrows()
        ]
        return {
            "col_metrica": col_metrica,
            "col_categoria": col_categoria,
            "promedio_general": promedio_general,
            "ranking_categorias": ranking,
        }

    if tipo in {"heatmap", "correlacion"}:
        corr = matriz_correlacion(df)
        cols = corr["columnas"]
        matriz = corr["matriz"]
        pares: list[dict[str, Any]] = []
        for idx_a, var_a in enumerate(cols):
            for var_b in cols[idx_a + 1 :]:
                coef = float(matriz.get(var_a, {}).get(var_b, 0.0))
                pares.append(
                    {
                        "var_a": var_a,
                        "var_b": var_b,
                        "coeficiente": round(coef, 3),
                    }
                )
        pares.sort(key=lambda item: abs(item["coeficiente"]), reverse=True)
        return {
            "columnas": cols,
            "correlaciones": pares[:6],
        }

    raise ValueError(
        f"Tipo de gráfico no soportado para explicación: '{tipo_grafico}'."
    )
