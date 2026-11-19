"""Módulo de auditoría estructural de datasets (Enfoque POO)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import zipfile

import pandas as pd

from core.utils import calcular_limites_iqr, obtener_columnas_numericas


class DatasetAuditor:
    """Audita la estructura, calidad y anomalías de un archivo tabular.

    Mantiene una copia íntegra e inmutable del dataset original en
    ``self.df_raw`` para preservar la trazabilidad de la auditoría.
    """

    EXTENSIONES_SOPORTADAS: frozenset[str] = frozenset({".csv", ".xlsx"})

    def __init__(self, ruta_archivo: str) -> None:
        """Carga el archivo tabular desde disco y almacena ``self.df_raw``.

        Soporta archivos ``.csv`` (con respaldo automático ``utf-8`` ->
        ``latin-1`` y detección de separador ``;``) y ``.xlsx``.
        """
        if not ruta_archivo or not isinstance(ruta_archivo, str):
            raise ValueError("La ruta del archivo debe ser una cadena no vacía.")

        ruta = Path(ruta_archivo)
        if not ruta.exists() or not ruta.is_file():
            raise FileNotFoundError(
                f"No se encontró el archivo especificado: '{ruta_archivo}'."
            )

        extension = ruta.suffix.lower()
        if extension not in self.EXTENSIONES_SOPORTADAS:
            raise ValueError(
                f"Extensión '{extension}' no soportada. "
                "Formatos permitidos: .csv, .xlsx."
            )

        self.ruta_archivo: str = str(ruta)
        df_cargado = self._cargar_archivo(ruta, extension)

        if df_cargado.empty and len(df_cargado.columns) == 0:
            raise pd.errors.EmptyDataError(
                f"El archivo '{ruta.name}' no contiene columnas ni registros."
            )

        # copy(deep=True) asegura que self.df_raw nunca se modifique por referencia
        self.df_raw: pd.DataFrame = df_cargado.copy(deep=True)

    @staticmethod
    def _leer_csv_con_delimitador(ruta: Path, encoding: str) -> pd.DataFrame:
        """Lee un CSV detectando automáticamente separadores ';' o tabulación."""
        df = pd.read_csv(ruta, encoding=encoding)
        # Si quedó 1 sola columna con ';' o tabulación, relee detectando el separador
        if len(df.columns) == 1:
            encabezado = str(df.columns[0])
            if ";" in encabezado or "\t" in encabezado:
                return pd.read_csv(
                    ruta, encoding=encoding, sep=None, engine="python"
                )
        return df

    @staticmethod
    def _cargar_archivo(ruta: Path, extension: str) -> pd.DataFrame:
        """Lee un archivo CSV o Excel aplicando manejo estricto de errores."""
        if extension == ".csv":
            try:
                return DatasetAuditor._leer_csv_con_delimitador(ruta, "utf-8")
            except UnicodeDecodeError:
                try:
                    return DatasetAuditor._leer_csv_con_delimitador(
                        ruta, "latin-1"
                    )
                except (
                    UnicodeDecodeError,
                    pd.errors.EmptyDataError,
                    pd.errors.ParserError,
                    OSError,
                ) as exc:
                    raise ValueError(
                        f"No fue posible decodificar el CSV '{ruta.name}' "
                        f"con utf-8 ni latin-1: {exc}"
                    ) from exc
            except pd.errors.EmptyDataError as exc:
                raise pd.errors.EmptyDataError(
                    f"El archivo CSV '{ruta.name}' está vacío."
                ) from exc
            except (pd.errors.ParserError, OSError, ValueError) as exc:
                raise ValueError(
                    f"Error al analizar el archivo CSV '{ruta.name}': {exc}"
                ) from exc

        try:
            return pd.read_excel(ruta, engine="openpyxl")
        except pd.errors.EmptyDataError as exc:
            raise pd.errors.EmptyDataError(
                f"El archivo Excel '{ruta.name}' está vacío."
            ) from exc
        except (ValueError, OSError, zipfile.BadZipFile) as exc:
            raise ValueError(
                f"Error al leer el archivo Excel '{ruta.name}': {exc}"
            ) from exc

    @staticmethod
    def _inferir_tipo_logico(serie: pd.Series) -> str:
        """Determina si una columna es ``numerica``, ``fecha`` o ``categorica``."""
        if pd.api.types.is_datetime64_any_dtype(serie):
            return "fecha"
        if pd.api.types.is_bool_dtype(serie):
            return "categorica"
        if pd.api.types.is_numeric_dtype(serie):
            return "numerica"

        serie_valida = serie.dropna()
        if serie_valida.empty:
            return "categorica"

        # Verifica si al menos el 80% de la muestra de texto tiene formato de fecha
        if pd.api.types.is_object_dtype(serie) or pd.api.types.is_string_dtype(serie):
            muestra = serie_valida.astype(str).str.strip().head(50)
            tiene_separador_fecha = muestra.str.contains(
                r"[-/:]", regex=True
            ).mean()
            if tiene_separador_fecha >= 0.8:
                try:
                    # format="mixed" y errors="coerce" convierten fechas válidas y dejan NaT en errores
                    parseadas = pd.to_datetime(
                        muestra, format="mixed", errors="coerce"
                    )
                    if float(parseadas.notna().mean()) >= 0.8:
                        return "fecha"
                except (ValueError, TypeError):
                    return "categorica"

        return "categorica"

    def _calcular_outliers_iqr(self) -> dict[str, dict[str, Any]]:
        """Calcula valores atípicos por columna numérica usando la regla IQR."""
        outliers: dict[str, dict[str, Any]] = {}

        for col in obtener_columnas_numericas(self.df_raw):
            try:
                q1, q3, iqr, lim_inf, lim_sup = calcular_limites_iqr(
                    self.df_raw[col]
                )
                valores = self.df_raw[col].dropna().astype(float)
                # Filtro booleano de Pandas para contar registros fuera de límites
                cantidad = int(((valores < lim_inf) | (valores > lim_sup)).sum())

                outliers[col] = {
                    "cantidad": cantidad,
                    "q1": round(q1, 4),
                    "q3": round(q3, 4),
                    "iqr": round(iqr, 4),
                    "limite_inferior": round(lim_inf, 4),
                    "limite_superior": round(lim_sup, 4),
                }
            except (ValueError, TypeError, IndexError):
                outliers[col] = {
                    "cantidad": 0,
                    "q1": None,
                    "q3": None,
                    "iqr": None,
                    "limite_inferior": None,
                    "limite_superior": None,
                }

        return outliers

    def obtener_muestra(self, n: int = 3) -> list[dict[str, Any]]:
        """Retorna las primeras ``n`` filas como lista de diccionarios."""
        return self.df_raw.head(n).to_dict(orient="records")

    def auditar_estructura(self) -> dict[str, Any]:
        """Ejecuta una auditoría integral de estructura, nulos y atípicos."""
        filas = int(self.df_raw.shape[0])
        columnas = int(self.df_raw.shape[1])
        # memory_usage(deep=True) mide el peso real en bytes incluyendo cadenas de texto
        memoria_kb = round(
            float(self.df_raw.memory_usage(deep=True).sum()) / 1024.0, 2
        )

        esquema: list[dict[str, Any]] = []
        for col in self.df_raw.columns:
            serie = self.df_raw[col]
            nulos = int(serie.isna().sum())
            porcentaje_nulos = (
                round((nulos / filas) * 100.0, 2) if filas > 0 else 0.0
            )

            esquema.append(
                {
                    "columna": str(col),
                    "dtype": str(serie.dtype),
                    "tipo_logico": self._inferir_tipo_logico(serie),
                    "nulos": nulos,
                    "porcentaje_nulos": porcentaje_nulos,
                    # nunique(dropna=True) cuenta valores distintos ignorando NaN
                    "valores_unicos": int(serie.nunique(dropna=True)),
                }
            )

        # duplicated() marca con True las filas repetidas exactas
        duplicados = int(self.df_raw.duplicated().sum())
        columnas_varianza_cero = [
            str(col)
            for col in self.df_raw.columns
            if int(self.df_raw[col].nunique(dropna=True)) <= 1
        ]
        outliers_iqr = self._calcular_outliers_iqr()

        return {
            "filas": filas,
            "columnas": columnas,
            "memoria_kb": memoria_kb,
            "esquema": esquema,
            "duplicados": duplicados,
            "columnas_varianza_cero": columnas_varianza_cero,
            "outliers_iqr": outliers_iqr,
        }
