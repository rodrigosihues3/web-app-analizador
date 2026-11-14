"""Suite de pruebas automatizadas para el monolito web Offline-First."""

from __future__ import annotations

import os
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from web.app import SESIONES, TEMP_UPLOADS_DIR, TEMPLATES_DIR, STATIC_DIR, app

BASE_DIR: Path = Path(__file__).resolve().parent
RUTA_NOTAS: Path = BASE_DIR / "datasets" / "notas_estudiantes.csv"
RUTA_RETAIL: Path = BASE_DIR / "datasets" / "retail_sales.csv"


def _limpiar_temp_uploads() -> None:
    """Elimina archivos temporales de sesión preservando únicamente ``.gitkeep``."""
    SESIONES.clear()
    if not TEMP_UPLOADS_DIR.exists():
        return
    for archivo in TEMP_UPLOADS_DIR.iterdir():
        if archivo.is_file() and archivo.name != ".gitkeep":
            try:
                archivo.unlink()
            except OSError:
                pass


class TestMonolitoWebOfflineFirst(unittest.TestCase):
    """Valida endpoints HTTP, fragmentos HTMX, activos locales y exportaciones."""

    @classmethod
    def setUpClass(cls) -> None:
        """Inicializa el cliente de pruebas de FastAPI en modo offline."""
        _limpiar_temp_uploads()
        cls.client = TestClient(app)

    def tearDown(self) -> None:
        """Limpia archivos generados en ``temp_uploads/`` tras cada prueba."""
        _limpiar_temp_uploads()

    def test_01_activos_estaticos_y_plantillas_sin_cdn(self) -> None:
        """Verifica que los 3 activos locales pesen > 10 KB y cero CDN en HTML."""
        activos_esperados = (
            STATIC_DIR / "css" / "tailwind.min.css",
            STATIC_DIR / "js" / "htmx.min.js",
            STATIC_DIR / "js" / "plotly.min.js",
        )
        for ruta_activo in activos_esperados:
            self.assertTrue(
                ruta_activo.exists(),
                f"No existe el activo estático requerido: {ruta_activo}",
            )
            self.assertGreater(
                ruta_activo.stat().st_size,
                10240,
                f"El activo {ruta_activo.name} pesa menos de 10 KB.",
            )

        archivos_html = list(TEMPLATES_DIR.rglob("*.html"))
        self.assertGreaterEqual(len(archivos_html), 5)
        for ruta_html in archivos_html:
            contenido = ruta_html.read_text(encoding="utf-8").lower()
            self.assertNotIn(
                "http://",
                contenido,
                f"Enlace externo http:// detectado en {ruta_html.name}",
            )
            self.assertNotIn(
                "https://",
                contenido,
                f"Enlace externo https:// detectado en {ruta_html.name}",
            )

    def test_02_get_index_referencias_locales_y_cero_urls_externas(self) -> None:
        """Valida que GET / retorne HTTP 200 con activos /static/ y sin CDN."""
        respuesta = self.client.get("/")
        self.assertEqual(respuesta.status_code, 200)

        html = respuesta.text
        self.assertIn("/static/css/tailwind.min.css", html)
        self.assertIn("/static/js/htmx.min.js", html)
        self.assertIn("/static/js/plotly.min.js", html)
        self.assertIn("notas_estudiantes.csv", html)
        self.assertIn("retail_sales.csv", html)
        self.assertNotIn("http://", html.lower())
        self.assertNotIn("https://", html.lower())

    def test_03_flujo_completo_cargar_procesar_y_exportar(self) -> None:
        """Valida carga, auditoría, procesamiento Plotly y exportaciones CSV/XLSX/PDF."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            with RUTA_NOTAS.open("rb") as archivo_csv:
                resp_carga = self.client.post(
                    "/analizar/cargar",
                    files={
                        "archivo": (
                            "notas_estudiantes.csv",
                            archivo_csv,
                            "text/csv",
                        )
                    },
                )

        self.assertEqual(resp_carga.status_code, 200)
        html_auditoria = resp_carga.text
        self.assertIn("NULOS DETECTADOS", html_auditoria)
        self.assertIn("OUTLIER IQR", html_auditoria)
        self.assertIn("25", html_auditoria)
        self.assertNotIn("http://", html_auditoria.lower())
        self.assertNotIn("https://", html_auditoria.lower())

        coincidencia = re.search(
            r'name="session_id"\s+value="([0-9a-fA-F-]+)"', html_auditoria
        )
        self.assertIsNotNone(
            coincidencia, "No se encontró el session_id en panel_auditoria.html"
        )
        session_id = coincidencia.group(1) if coincidencia else ""
        self.assertIn(session_id, SESIONES)

        # Ejecutar POST /analizar/procesar
        resp_proc = self.client.post(
            "/analizar/procesar",
            data={
                "session_id": session_id,
                "eliminar_duplicados": "true",
                "estrategia_nulos": "mediana",
                "recortar_outliers": "true",
                "col_categoria": "Asignatura",
                "col_numerica": "Nota",
            },
        )
        self.assertEqual(resp_proc.status_code, 200)
        html_resultados = resp_proc.text

        self.assertIn("contenedor-plotly-histograma", html_resultados)
        self.assertIn("contenedor-plotly-boxplot", html_resultados)
        self.assertIn("contenedor-plotly-heatmap", html_resultados)
        self.assertIn("plotly-graph-div", html_resultados)
        self.assertIn("Registro Máximo Global", html_resultados)
        self.assertIn("Categoría Líder en Promedio", html_resultados)
        self.assertIn("Duplicados eliminados", html_resultados)
        self.assertNotIn("http://", html_resultados.lower())
        self.assertNotIn("https://", html_resultados.lower())

        # Exportar Dataset Limpio (.CSV)
        resp_csv = self.client.get(
            f"/export/dataset/{session_id}?formato=csv"
        )
        self.assertEqual(resp_csv.status_code, 200)
        self.assertIn(
            "attachment",
            resp_csv.headers.get("content-disposition", "").lower(),
        )
        contenido_csv = resp_csv.content.decode("utf-8-sig")
        self.assertIn("Estudiante,Asignatura,Ciclo,Nota,Asistencia", contenido_csv)
        self.assertGreater(len(resp_csv.content), 200)

        # Exportar Dataset Limpio (.XLSX)
        resp_xlsx = self.client.get(
            f"/export/dataset/{session_id}?formato=xlsx"
        )
        self.assertEqual(resp_xlsx.status_code, 200)
        self.assertIn(
            "attachment",
            resp_xlsx.headers.get("content-disposition", "").lower(),
        )
        self.assertTrue(resp_xlsx.content.startswith(b"PK"))
        self.assertGreater(len(resp_xlsx.content), 1000)

        # Exportar Reporte Técnico (.PDF)
        resp_pdf = self.client.get(f"/export/pdf/{session_id}")
        self.assertEqual(resp_pdf.status_code, 200)
        self.assertEqual(
            resp_pdf.headers.get("content-type"), "application/pdf"
        )
        self.assertTrue(resp_pdf.content.startswith(b"%PDF"))
        self.assertGreater(len(resp_pdf.content), 10240)

    def test_04_selector_demo_y_manejo_controlado_de_errores(self) -> None:
        """Verifica carga mediante dataset_demo y alertas HTML controladas."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            resp_demo = self.client.post(
                "/analizar/cargar",
                data={"dataset_demo": "retail_sales.csv"},
            )
        self.assertEqual(resp_demo.status_code, 200)
        self.assertIn("Total_Venta", resp_demo.text)

        # Archivo vacío
        resp_vacio = self.client.post(
            "/analizar/cargar",
            files={"archivo": ("vacio.csv", b"", "text/csv")},
        )
        self.assertEqual(resp_vacio.status_code, 200)
        self.assertIn("ALERTA CONTROLADA", resp_vacio.text)

        # Sesión inexistente
        resp_sesion_invalida = self.client.post(
            "/analizar/procesar",
            data={"session_id": "sesion-inexistente"},
        )
        self.assertEqual(resp_sesion_invalida.status_code, 200)
        self.assertIn("ALERTA CONTROLADA", resp_sesion_invalida.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
