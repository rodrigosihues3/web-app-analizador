"""Suite de pruebas integrales para el núcleo analítico y el servidor MCP."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.auditor import DatasetAuditor
from core.estadistica import (
    analizar_valores_clave,
    extraer_metricas_grafico,
    matriz_correlacion,
    resumen_descriptivo,
)
from core.exportador import exportar_dataframe, generar_reporte_pdf
from core.sintetizador import explicar_grafico_ia, generar_contexto_ejecutivo
from core.transformador import limpiar_dataset
from core.visualizador import (
    generar_barras_categoria_plotly,
    generar_boxplot_plotly,
    generar_graficos_estaticos_pdf,
    generar_heatmap_plotly,
    generar_histograma_plotly,
)
import mcp_server

BASE_DIR = Path(__file__).resolve().parent
RUTA_NOTAS = BASE_DIR / "datasets" / "notas_estudiantes.csv"
RUTA_RETAIL = BASE_DIR / "datasets" / "retail_sales.csv"


class TestAnalizadorMultiparadigma(unittest.TestCase):
    """Valida auditoría POO, limpieza funcional, estadística, gráficos, PDF y MCP."""

    def test_auditoria_notas_estudiantes(self) -> None:
        """Verifica detección de nulos, duplicados, outlier IQR y esquema."""
        auditor = DatasetAuditor(str(RUTA_NOTAS))
        auditoria = auditor.auditar_estructura()

        self.assertEqual(auditoria["filas"], 25)
        self.assertEqual(auditoria["columnas"], 5)
        self.assertEqual(auditoria["duplicados"], 1)

        esquema_por_col = {
            item["columna"]: item for item in auditoria["esquema"]
        }
        self.assertEqual(esquema_por_col["Nota"]["nulos"], 2)
        self.assertEqual(esquema_por_col["Nota"]["tipo_logico"], "numerica")
        self.assertEqual(
            esquema_por_col["Asignatura"]["tipo_logico"], "categorica"
        )
        self.assertEqual(auditoria["outliers_iqr"]["Nota"]["cantidad"], 1)

    def test_auditoria_retail_sales(self) -> None:
        """Verifica detección de nulos en Cantidad, outlier en Total_Venta y fechas."""
        auditor = DatasetAuditor(str(RUTA_RETAIL))
        auditoria = auditor.auditar_estructura()

        self.assertEqual(auditoria["filas"], 30)
        self.assertEqual(auditoria["columnas"], 8)
        self.assertEqual(auditoria["duplicados"], 0)

        esquema_por_col = {
            item["columna"]: item for item in auditoria["esquema"]
        }
        self.assertEqual(esquema_por_col["Cantidad"]["nulos"], 2)
        self.assertEqual(esquema_por_col["Fecha"]["tipo_logico"], "fecha")
        self.assertGreaterEqual(
            auditoria["outliers_iqr"]["Total_Venta"]["cantidad"], 1
        )

    def test_auditoria_xlsx_latin1_y_errores(self) -> None:
        """Valida soporte Excel, fallback latin-1, varianza cero y errores."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # CSV en latin-1 con columna de varianza cero
            ruta_latin1 = tmp_path / "latin1.csv"
            ruta_latin1.write_bytes(
                "Nombre,Ciudad,Constante\nJosé,Bogotá,Fijo\nInés,Cali,Fijo\n".encode(
                    "latin-1"
                )
            )
            auditor_l1 = DatasetAuditor(str(ruta_latin1))
            aud_l1 = auditor_l1.auditar_estructura()
            self.assertEqual(aud_l1["filas"], 2)
            self.assertIn("Constante", aud_l1["columnas_varianza_cero"])

            # Excel .xlsx
            ruta_xlsx = tmp_path / "prueba.xlsx"
            auditor_l1.df_raw.to_excel(ruta_xlsx, index=False)
            auditor_xl = DatasetAuditor(str(ruta_xlsx))
            self.assertEqual(auditor_xl.auditar_estructura()["filas"], 2)

            # CSV delimitado por punto y coma (;)
            ruta_pc = tmp_path / "punto_coma.csv"
            ruta_pc.write_text(
                "Codigo;Nombre;Nota1;Nota2\nA1;Ana;15;18\nA2;Luis;14;16\n",
                encoding="utf-8",
            )
            auditor_pc = DatasetAuditor(str(ruta_pc))
            aud_pc = auditor_pc.auditar_estructura()
            self.assertEqual(aud_pc["columnas"], 4)
            self.assertEqual(aud_pc["filas"], 2)

            # CSV vacío
            ruta_vacia = tmp_path / "vacio.csv"
            ruta_vacia.write_text("", encoding="utf-8")
            with self.assertRaises(pd.errors.EmptyDataError):
                DatasetAuditor(str(ruta_vacia))

            # Extensión no soportada y archivo inexistente
            ruta_txt = tmp_path / "invalido.txt"
            ruta_txt.write_text("a,b\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                DatasetAuditor(str(ruta_txt))
            with self.assertRaises(FileNotFoundError):
                DatasetAuditor(str(tmp_path / "no_existe.csv"))

    def test_limpieza_funcional_sin_mutacion(self) -> None:
        """Verifica pureza funcional, estrategias de nulos y recorte IQR."""
        auditor = DatasetAuditor(str(RUTA_NOTAS))
        copia_antes = auditor.df_raw.copy(deep=True)

        df_limpio, logs = limpiar_dataset(
            auditor.df_raw,
            eliminar_duplicados=True,
            estrategia_nulos="mediana",
            recortar_outliers=True,
        )

        # df_raw permanece intacto
        pd.testing.assert_frame_equal(auditor.df_raw, copia_antes)
        self.assertEqual(len(df_limpio), 24)
        self.assertEqual(int(df_limpio.isna().sum().sum()), 0)
        self.assertLess(float(df_limpio["Nota"].max()), 99.0)
        self.assertTrue(any("Duplicados eliminados" in msg for msg in logs))

        # Otras estrategias válidas y categóricas con nulos
        df_cat = pd.DataFrame(
            {"Grupo": ["A", "A", None, "B"], "Valor": [10.0, None, 30.0, 40.0]}
        )
        df_cat_orig = df_cat.copy(deep=True)

        df_moda, _ = limpiar_dataset(
            df_cat, eliminar_duplicados=False, estrategia_nulos="moda"
        )
        self.assertEqual(df_moda.loc[2, "Grupo"], "A")

        df_elim, _ = limpiar_dataset(
            df_cat, eliminar_duplicados=False, estrategia_nulos="eliminar"
        )
        self.assertEqual(len(df_elim), 2)

        df_todo_nulo = pd.DataFrame({"Grupo": [None, None], "Valor": [1.0, 2.0]})
        df_desc, _ = limpiar_dataset(
            df_todo_nulo, eliminar_duplicados=False, estrategia_nulos="media"
        )
        self.assertTrue((df_desc["Grupo"] == "Desconocido").all())
        pd.testing.assert_frame_equal(df_cat, df_cat_orig)

        with self.assertRaises(ValueError):
            limpiar_dataset(df_cat, estrategia_nulos="desconocida")

    def test_estadistica_y_valores_clave(self) -> None:
        """Verifica resumen descriptivo, valores clave y matriz de correlación."""
        auditor = DatasetAuditor(str(RUTA_RETAIL))
        df_limpio, _ = limpiar_dataset(
            auditor.df_raw,
            eliminar_duplicados=True,
            estrategia_nulos="mediana",
            recortar_outliers=False,
        )

        stats = resumen_descriptivo(df_limpio)
        cols_stats = {s["columna"]: s for s in stats}
        self.assertIn("Total_Venta", cols_stats)
        self.assertIn("asimetria", cols_stats["Total_Venta"])

        clave = analizar_valores_clave(df_limpio, "Region", "Total_Venta")
        self.assertEqual(clave["valor_maximo"], 85000.0)
        self.assertEqual(
            clave["categoria_lider_acumulado"]["categoria"], "Norte"
        )
        self.assertGreater(len(clave["tabla_agrupada"]), 0)

        corr = matriz_correlacion(df_limpio)
        self.assertIn("Total_Venta", corr["columnas"])
        self.assertEqual(
            corr["matriz"]["Total_Venta"]["Total_Venta"], 1.0
        )

        # Manejo de columna numérica completamente vacía
        df_con_vacia = pd.DataFrame(
            {"Vac": pd.Series([np.nan, np.nan], dtype=float), "Val": [1.0, 3.0]}
        )
        stats_vacia = resumen_descriptivo(df_con_vacia)
        self.assertEqual([s["columna"] for s in stats_vacia], ["Val"])

        with self.assertRaises(KeyError):
            analizar_valores_clave(df_limpio, "Inexistente", "Total_Venta")
        with self.assertRaises(ValueError):
            analizar_valores_clave(df_limpio, "Region", "Producto")

    def test_visualizaciones_plotly_y_pdf(self) -> None:
        """Verifica generación de HTML Plotly (sin CDN) y PNG Matplotlib."""
        auditor = DatasetAuditor(str(RUTA_NOTAS))
        df_limpio, _ = limpiar_dataset(
            auditor.df_raw,
            eliminar_duplicados=True,
            estrategia_nulos="mediana",
            recortar_outliers=True,
        )

        html_hist = generar_histograma_plotly(df_limpio, "Nota")
        html_barras = generar_barras_categoria_plotly(
            df_limpio, "Nota", "Asignatura"
        )
        html_box = generar_boxplot_plotly(
            df_limpio, "Nota", col_grupo="Asignatura"
        )
        html_hm = generar_heatmap_plotly(df_limpio)

        for html_frag in (html_hist, html_barras, html_box, html_hm):
            self.assertIn("plotly", html_frag.lower())
            self.assertIn('"displaymodebar": false', html_frag.lower())
            self.assertNotIn("<html", html_frag.lower())
            self.assertNotIn("https://", html_frag.lower())

        self.assertIn("Media:", html_hist)
        self.assertIn("Mediana:", html_hist)
        self.assertIn("Promedio general:", html_barras)

        graficos_pdf = generar_graficos_estaticos_pdf(df_limpio)
        self.assertEqual(len(graficos_pdf), 3)
        for titulo, buf in graficos_pdf:
            self.assertTrue(len(titulo) > 0)
            cabecera = buf.getvalue()[:8]
            self.assertTrue(cabecera.startswith(b"\x89PNG"))

    def test_sintetizador_offline_y_degradacion_elegante(self) -> None:
        """Valida respaldo local ante ausencia de clave y ante fallo/timeout."""
        auditor = DatasetAuditor(str(RUTA_NOTAS))
        auditoria = auditor.auditar_estructura()
        muestra = auditor.obtener_muestra(3)
        self.assertEqual(len(muestra), 3)

        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            ctx_sin_key = generar_contexto_ejecutivo(auditoria, muestra)
            self.assertTrue(ctx_sin_key["titulo_reporte"])
            self.assertTrue(ctx_sin_key["resumen_ejecutivo"])

            df_limpio, _ = limpiar_dataset(auditor.df_raw)
            metricas_hist = extraer_metricas_grafico(
                df_limpio, "histograma", col_metrica="Nota"
            )
            metricas_barras = extraer_metricas_grafico(
                df_limpio,
                "barras",
                col_metrica="Nota",
                col_categoria="Asignatura",
            )
            metricas_corr = extraer_metricas_grafico(df_limpio, "correlacion")

            for tipo_g, met_g in (
                ("histograma", metricas_hist),
                ("barras", metricas_barras),
                ("correlacion", metricas_corr),
            ):
                texto_g = explicar_grafico_ia(ctx_sin_key, tipo_g, met_g)
                self.assertTrue(len(texto_g) > 20)
                self.assertNotIn("**", texto_g)

        with patch.dict(
            os.environ, {"GEMINI_API_KEY": "clave_simulada"}, clear=False
        ):
            with patch(
                "core.sintetizador._invocar_gemini",
                side_effect=TimeoutError("Timeout simulado > 4s"),
            ):
                ctx_fallback = generar_contexto_ejecutivo(auditoria, muestra)
                self.assertTrue(ctx_fallback["titulo_reporte"])
                self.assertTrue(ctx_fallback["resumen_ejecutivo"])

            with patch(
                "core.sintetizador._invocar_gemini_grafico",
                side_effect=TimeoutError("Timeout simulado > 4s"),
            ):
                exp_fallback = explicar_grafico_ia(
                    ctx_fallback, "histograma", metricas_hist
                )
                self.assertTrue(len(exp_fallback) > 20)

    def test_flujo_completo_exportacion_y_pdf_en_disco(self) -> None:
        """Ejecuta flujo integral en ambos datasets, valida test_output.pdf y lo elimina."""
        ruta_pdf_prueba = BASE_DIR / "test_output.pdf"

        for ruta_csv, col_cat, col_met in (
            (RUTA_NOTAS, "Asignatura", "Nota"),
            (RUTA_RETAIL, "Region", "Total_Venta"),
        ):
            auditor = DatasetAuditor(str(ruta_csv))
            auditoria = auditor.auditar_estructura()
            df_limpio, logs = limpiar_dataset(
                auditor.df_raw,
                eliminar_duplicados=True,
                estrategia_nulos="mediana",
                recortar_outliers=True,
            )

            csv_buf = exportar_dataframe(df_limpio, formato="csv")
            xlsx_buf = exportar_dataframe(df_limpio, formato="xlsx")
            self.assertGreater(len(csv_buf.getvalue()), 100)
            self.assertTrue(xlsx_buf.getvalue().startswith(b"PK"))

            stats = resumen_descriptivo(df_limpio)
            valores_clave = analizar_valores_clave(df_limpio, col_cat, col_met)

            with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
                contexto = generar_contexto_ejecutivo(
                    auditoria, df_limpio.head(3).to_dict(orient="records")
                )

            imagenes = generar_graficos_estaticos_pdf(df_limpio)
            pdf_buf = generar_reporte_pdf(
                contexto=contexto,
                auditoria=auditoria,
                logs_limpieza=logs,
                estadisticas=stats,
                valores_clave=valores_clave,
                imagenes_graficos=imagenes,
            )

            try:
                ruta_pdf_prueba.write_bytes(pdf_buf.getvalue())
                self.assertTrue(ruta_pdf_prueba.exists())
                tamano = ruta_pdf_prueba.stat().st_size
                self.assertGreater(tamano, 10240)
                self.assertTrue(
                    ruta_pdf_prueba.read_bytes().startswith(b"%PDF")
                )
            finally:
                if ruta_pdf_prueba.exists():
                    ruta_pdf_prueba.unlink()

    def test_herramientas_mcp(self) -> None:
        """Verifica las 2 herramientas MCP interactivas y exportación CSV/PDF."""
        import asyncio

        herramientas = asyncio.run(mcp_server.mcp.list_tools())
        nombres_tools = {t.name for t in herramientas}
        self.assertEqual(
            nombres_tools,
            {
                "auditar_dataset_mcp",
                "procesar_y_exportar_mcp",
            },
        )

        aud_json = json.loads(mcp_server.auditar_dataset_mcp(str(RUTA_NOTAS)))
        self.assertEqual(aud_json["filas"], 25)
        self.assertEqual(aud_json["duplicados"], 1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
                resp_str = mcp_server.procesar_y_exportar_mcp(
                    ruta_archivo=str(RUTA_RETAIL),
                    carpeta_salida=tmp_dir,
                    eliminar_duplicados=True,
                    estrategia_nulos="mediana",
                    recortar_outliers=True,
                    col_categoria="Categoria",
                    col_metrica="Total_Venta",
                )

            resultado = json.loads(resp_str)
            self.assertNotIn("error", resultado)
            self.assertIn("ruta_csv_limpio", resultado)
            self.assertIn("ruta_reporte_pdf", resultado)
            self.assertIn("logs_limpieza", resultado)
            self.assertIn("estadisticas", resultado)
            self.assertIn("valores_clave", resultado)

            ruta_csv = Path(resultado["ruta_csv_limpio"])
            ruta_pdf = Path(resultado["ruta_reporte_pdf"])
            self.assertTrue(ruta_csv.exists())
            self.assertTrue(ruta_pdf.exists())
            self.assertGreater(ruta_pdf.stat().st_size, 10240)
            self.assertTrue(ruta_pdf.read_bytes().startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
