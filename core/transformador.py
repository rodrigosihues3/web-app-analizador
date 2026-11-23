"""Módulo funcional de limpieza y transformación inmutable de DataFrames."""

from typing import Any

import pandas as pd

from core.utils import es_columna_numerica

ESTRATEGIAS_NULOS_VALIDAS = (
    "eliminar",
    "media",
    "mediana",
    "moda",
    "personalizado",
    "ninguna",
)


def _remover_duplicados(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Retorna un nuevo DataFrame sin filas duplicadas y su mensaje de registro."""
    cantidad_duplicados = int(df.duplicated().sum())
    if cantidad_duplicados == 0:
        return df.copy(deep=True), "Sin registros duplicados."

    df_sin_duplicados = df.drop_duplicates().reset_index(drop=True)
    return (
        df_sin_duplicados,
        f"Duplicados eliminados: {cantidad_duplicados} fila(s).",
    )


def _calcular_valor_imputacion_numerica(
    serie: pd.Series, estrategia: str, valor_nulos: Any = 0
) -> float:
    """Calcula el valor numérico con el que se rellenarán los datos faltantes."""
    if estrategia == "personalizado":
        try:
            return float(valor_nulos)
        except (ValueError, TypeError):
            raise ValueError(
                f"El valor personalizado '{valor_nulos}' no es un número válido."
            )

    serie_valida = serie.dropna().astype(float)
    if serie_valida.empty:
        return 0.0

    if estrategia == "media":
        return float(serie_valida.mean())
    if estrategia == "mediana":
        return float(serie_valida.median())
    if estrategia == "moda":
        modas = serie_valida.mode(dropna=True)
        return float(modas.iloc[0]) if not modas.empty else 0.0

    return 0.0


def _calcular_valor_imputacion_categorica(
    serie: pd.Series, estrategia: str, valor_nulos: Any = None
) -> Any:
    """Obtiene el texto de reemplazo para columnas categóricas con nulos."""
    if estrategia == "personalizado" and valor_nulos is not None:
        texto = str(valor_nulos).strip()
        if texto:
            try:
                float(texto)
            except ValueError:
                return texto

    serie_valida = serie.dropna()
    if serie_valida.empty:
        return "Desconocido"

    modas = serie_valida.mode()
    if not modas.empty:
        return modas.iloc[0]

    return "Desconocido"


def _tratar_nulos(
    df: pd.DataFrame, estrategia: str, valor_nulos: Any = 0
) -> tuple[pd.DataFrame, list[str]]:
    """Aplica la estrategia de valores faltantes sobre una copia del DataFrame."""
    df_nuevo = df.copy(deep=True)
    logs: list[str] = []

    if estrategia == "ninguna":
        logs.append("Valores faltantes conservados sin cambios.")
        return df_nuevo, logs

    if estrategia == "eliminar":
        filas_con_nulos = int(df_nuevo.isna().any(axis=1).sum())
        if filas_con_nulos == 0:
            logs.append("Sin filas con valores faltantes.")
            return df_nuevo, logs

        df_filtrado = df_nuevo.dropna().reset_index(drop=True)
        logs.append(
            f"Eliminadas {filas_con_nulos} fila(s) con datos faltantes."
        )
        return df_filtrado, logs

    columnas_con_nulos = [
        col for col in df_nuevo.columns if int(df_nuevo[col].isna().sum()) > 0
    ]
    if not columnas_con_nulos:
        logs.append("Sin valores faltantes por completar.")
        return df_nuevo, logs

    for col in columnas_con_nulos:
        serie = df_nuevo[col]
        n_nulos = int(serie.isna().sum())

        if es_columna_numerica(serie):
            valor = _calcular_valor_imputacion_numerica(
                serie, estrategia, valor_nulos
            )
            if pd.api.types.is_integer_dtype(serie) and not float(valor).is_integer():
                df_nuevo[col] = serie.astype("float64").fillna(valor)
            else:
                relleno = (
                    int(valor)
                    if pd.api.types.is_integer_dtype(serie)
                    else valor
                )
                df_nuevo[col] = serie.fillna(relleno)

            etiqueta = (
                f"valor manual ({round(valor, 4)})"
                if estrategia == "personalizado"
                else f"{estrategia} ({round(valor, 4)})"
            )
            logs.append(
                f"'{col}': {n_nulos} dato(s) faltante(s) completado(s) con {etiqueta}."
            )
        elif pd.api.types.is_datetime64_any_dtype(serie):
            modas_fecha = serie.dropna().mode()
            if not modas_fecha.empty:
                fecha_imp = modas_fecha.iloc[0]
                df_nuevo[col] = serie.fillna(fecha_imp)
                logs.append(
                    f"'{col}': {n_nulos} fecha(s) faltante(s) completada(s) con {fecha_imp}."
                )
            else:
                logs.append(f"'{col}': sin fechas válidas para completar.")
        else:
            valor_cat = _calcular_valor_imputacion_categorica(
                serie, estrategia, valor_nulos
            )
            df_nuevo[col] = serie.fillna(valor_cat)
            logs.append(
                f"'{col}': {n_nulos} dato(s) faltante(s) completado(s) con '{valor_cat}'."
            )

    return df_nuevo, logs


def limpiar_dataset(
    df: pd.DataFrame,
    eliminar_duplicados: bool = True,
    estrategia_nulos: str = "mediana",
    valor_nulos: Any = 0,
) -> tuple[pd.DataFrame, list[str]]:
    """Limpia duplicados y valores nulos sin modificar el DataFrame original."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("El parámetro 'df' debe ser un pandas.DataFrame.")

    estrategia = estrategia_nulos.strip().lower()
    if estrategia not in ESTRATEGIAS_NULOS_VALIDAS:
        raise ValueError(
            f"Estrategia de nulos inválida: '{estrategia_nulos}'. "
            f"Opciones válidas: {list(ESTRATEGIAS_NULOS_VALIDAS)}."
        )

    df_actual = df.copy(deep=True)
    logs: list[str] = []

    if eliminar_duplicados:
        df_actual, log_dup = _remover_duplicados(df_actual)
        logs.append(log_dup)
    else:
        logs.append("Duplicados conservados.")

    df_actual, logs_nulos = _tratar_nulos(df_actual, estrategia, valor_nulos)
    logs.extend(logs_nulos)

    return df_actual, logs
