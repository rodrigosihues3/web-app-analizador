"""Suite de pruebas automatizadas para el monolito web Offline-First."""

from __future__ import annotations

import os
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from web.app import SESIONES, STATIC_DIR, TEMPLATES_DIR, TEMP_UPLOADS_DIR, app

BASE_DIR: Path = Path(__file__).resolve().parent
RUTA_NOTAS: Path = BASE_DIR / "datasets" / "notas_estudiantes.csv"


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
        """Inicializa el cliente de pruebas de FastAPI en modo offline estricto."""
        _limpiar_temp_uploads()
        cls._env_patch = patch.dict(
            os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=False
        )
        cls._env_patch.start()
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        """Cierra el cliente de pruebas y restaura las variables de entorno."""
        cls.client.close()
        cls._env_patch.stop()

    def tearDown(self) -> None:
        """Limpia archivos generados en ``temp_uploads/`` tras cada prueba."""
        _limpiar_temp_uploads()

    def test_01_activos_estaticos_y_plantillas_sin_cdn(self) -> None:
        """Verifica activos locales y ausencia total de enlaces CDN en HTML."""
        activos_esperados = (
            STATIC_DIR / "css" / "tailwind.min.css",
            STATIC_DIR / "css" / "app.css",
            STATIC_DIR / "js" / "htmx.min.js",
            STATIC_DIR / "js" / "plotly.min.js",
        )
        for ruta_activo in activos_esperados:
            self.assertTrue(
                ruta_activo.exists(),
                f"No existe el activo estático requerido: {ruta_activo}",
            )

        for ruta_min in (
            STATIC_DIR / "css" / "tailwind.min.css",
            STATIC_DIR / "js" / "htmx.min.js",
            STATIC_DIR / "js" / "plotly.min.js",
        ):
            self.assertGreater(ruta_min.stat().st_size, 10240)

        archivos_html = list(TEMPLATES_DIR.rglob("*.html"))
        self.assertGreaterEqual(len(archivos_html), 8)
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

    def test_02_get_index_referencias_locales_y_vista_manual_limpia(self) -> None:
        """Valida GET / con dropzone directo, navbar oculto y cero URLs externas."""
        respuesta = self.client.get("/")
        self.assertEqual(respuesta.status_code, 200)

        html = respuesta.text
        self.assertIn("/static/css/tailwind.min.css", html)
        self.assertIn("/static/css/app.css", html)
        self.assertIn("/static/js/htmx.min.js", html)
        self.assertIn("/static/js/plotly.min.js", html)
        self.assertIn("Analizador de Datos", html)
        self.assertIn("input-archivo-inicio", html)
        self.assertIn("Analizar este archivo", html)
        self.assertIn('id="btn-marca-inicio"', html)
        self.assertIn('id="btn-abrir-modal-carga"', html)
        self.assertIn("modal-carga", html)
        self.assertNotIn("notas_estudiantes.csv", html)
        self.assertNotIn("http://", html.lower())
        self.assertNotIn("https://", html.lower())

    def test_03_flujo_completo_cargar_regenerar_procesar_explicar_exportar_restaurar(self) -> None:
        """Valida ingesta, OOB navbar, regeneración, gráficos, explicabilidad, exportación y restaurar."""
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
        self.assertIn("nulos", html_auditoria.lower())
        self.assertIn("Calidad de los datos", html_auditoria)
        self.assertIn("Configuración del análisis", html_auditoria)
        self.assertIn('id="btn-abrir-modal-carga"', html_auditoria)
        self.assertIn("Analizar otro archivo", html_auditoria)
        self.assertIn("Regenerar resumen con IA", html_auditoria)
        self.assertIn("Procesar datos", html_auditoria)

        coincidencia = re.search(
            r'name="session_id"\s+value="([0-9a-fA-F-]+)"', html_auditoria
        )
        self.assertIsNotNone(
            coincidencia, "No se encontró el session_id en panel_auditoria.html"
        )
        session_id = coincidencia.group(1) if coincidencia else ""
        self.assertIn(session_id, SESIONES)

        # El botón OOB #btn-abrir-modal-carga no debe estar oculto al cargar datos
        btn_modal_match = re.search(
            r'<button[^>]*id="btn-abrir-modal-carga"[^>]*>', html_auditoria
        )
        self.assertIsNotNone(btn_modal_match)
        self.assertNotIn("hidden", btn_modal_match.group(0))

        # El UUID de sesión no debe exponerse visiblemente fuera de atributos internos
        texto_visible = re.sub(r"<[^>]+>", " ", html_auditoria)
        self.assertNotIn(session_id, texto_visible)

        # Regenerar resumen ejecutivo (en modo offline muestra aviso rojo de IA no disponible)
        resp_regen = self.client.post(
            "/analizar/regenerar-resumen",
            data={"session_id": session_id},
        )
        self.assertEqual(resp_regen.status_code, 200)
        self.assertIn("alert-ia-error", resp_regen.text)

        # Ejecutar POST /analizar/procesar con valor personalizado 0
        resp_proc = self.client.post(
            "/analizar/procesar",
            data={
                "session_id": session_id,
                "eliminar_duplicados": "true",
                "estrategia_nulos": "personalizado",
                "valor_nulos": "0",
                "col_categoria": "Asignatura",
                "col_numerica": "Nota",
            },
        )
        self.assertEqual(resp_proc.status_code, 200)
        html_resultados = resp_proc.text

        self.assertIn("contenedor-plotly-histograma", html_resultados)
        self.assertIn("contenedor-plotly-barras", html_resultados)
        self.assertIn("contenedor-plotly-heatmap", html_resultados)
        self.assertIn("plotly-graph-div", html_resultados)
        self.assertIn('"displaymodebar": false', html_resultados.lower())
        self.assertIn("Media (Nota)", html_resultados)
        self.assertIn("Mediana (Nota)", html_resultados)
        self.assertIn("Moda (Nota)", html_resultados)
        self.assertIn("Registro máximo global", html_resultados)
        self.assertIn("Categoría líder en promedio", html_resultados)
        self.assertIn("Duplicados eliminados", html_resultados)
        self.assertIn("Analizar con IA", html_resultados)

        # Explicabilidad IA (en modo offline muestra aviso rojo de IA no disponible)
        for tipo in ("histograma", "barras", "correlacion"):
            resp_exp = self.client.post(
                "/analizar/explicar-grafico",
                data={
                    "session_id": session_id,
                    "tipo_grafico": tipo,
                    "col_numerica": "Nota",
                    "col_categoria": "Asignatura",
                },
            )
            self.assertEqual(resp_exp.status_code, 200)
            self.assertIn("alert-ia-error", resp_exp.text)

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

        # Exportar Dataset Limpio (.XLSX)
        resp_xlsx = self.client.get(
            f"/export/dataset/{session_id}?formato=xlsx"
        )
        self.assertEqual(resp_xlsx.status_code, 200)
        self.assertTrue(resp_xlsx.content.startswith(b"PK"))

        # Exportar Reporte Técnico (.PDF)
        resp_pdf = self.client.get(f"/export/pdf/{session_id}")
        self.assertEqual(resp_pdf.status_code, 200)
        self.assertEqual(
            resp_pdf.headers.get("content-type"), "application/pdf"
        )
        self.assertTrue(resp_pdf.content.startswith(b"%PDF"))
        self.assertGreater(len(resp_pdf.content), 10240)

        # Restaurar sesión y volver al inicio limpio
        ruta_archivo_temp = Path(SESIONES[session_id]["ruta_archivo"])
        self.assertTrue(ruta_archivo_temp.exists())

        resp_restaurar = self.client.post(
            "/analizar/restaurar",
            data={"session_id": session_id},
        )
        self.assertEqual(resp_restaurar.status_code, 200)
        self.assertNotIn(session_id, SESIONES)
        self.assertFalse(ruta_archivo_temp.exists())
        self.assertIn("input-archivo-inicio", resp_restaurar.text)

        btn_modal_oculto = re.search(
            r'<button[^>]*id="btn-abrir-modal-carga"[^>]*>', resp_restaurar.text
        )
        self.assertIsNotNone(btn_modal_oculto)
        self.assertIn("hidden", btn_modal_oculto.group(0))

    def test_04_manejo_controlado_de_errores(self) -> None:
        """Verifica alertas HTML controladas ante archivo vacío y sesión inválida."""
        resp_vacio = self.client.post(
            "/analizar/cargar",
            files={"archivo": ("vacio.csv", b"", "text/csv")},
        )
        self.assertEqual(resp_vacio.status_code, 200)
        self.assertIn("No se pudo leer el archivo", resp_vacio.text)

        resp_sesion_invalida = self.client.post(
            "/analizar/procesar",
            data={"session_id": "sesion-inexistente"},
        )
        self.assertEqual(resp_sesion_invalida.status_code, 200)
        self.assertIn("No se pudo completar el análisis", resp_sesion_invalida.text)

        resp_regen_invalida = self.client.post(
            "/analizar/regenerar-resumen",
            data={"session_id": "sesion-inexistente"},
        )
        self.assertEqual(resp_regen_invalida.status_code, 200)
        self.assertIn("No se pudo regenerar el resumen", resp_regen_invalida.text)

        resp_exp_invalida = self.client.post(
            "/analizar/explicar-grafico",
            data={"session_id": "sesion-inexistente", "tipo_grafico": "histograma"},
        )
        self.assertEqual(resp_exp_invalida.status_code, 200)
        self.assertIn("No se pudo interpretar la gráfica", resp_exp_invalida.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
