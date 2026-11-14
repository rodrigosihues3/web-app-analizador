# AGENTS.md — Reglas de Arquitectura y Convenciones del Repositorio

Este documento define las directrices obligatorias para agentes de IA y desarrolladores que trabajen en el proyecto **Analizador Multiparadigma de Datos**.

## 1. Separación Estricta de Capas (`core/` vs `web/`)
- **`core/`**: Capa de dominio analítico puro. **Prohibido** importar frameworks o dependencias HTTP/web (`fastapi`, `starlette`, `uvicorn`, `jinja2`, `python-multipart`) dentro de `core/`. Todos los módulos de `core/` deben funcionar de forma independiente en CLI, pruebas unitarias y servidor MCP (`mcp_server.py`).
- **`web/`** *(Directiva 2)*: Capa de presentación y transporte HTTP (FastAPI + Jinja2). Solo debe orquestar peticiones/respuestas delegando toda la lógica analítica, estadística, gráfica y de exportación a `core/`.

## 2. Prohibición Absoluta de Enlaces CDN Externos
- Toda plantilla HTML bajo `web/templates/` debe operar en modo **100% Offline-First**.
- Queda **terminantemente prohibido** referenciar recursos externos mediante etiquetas `<script src="https://...">`, `<link href="https://...">` o fuentes remotas.
- Todos los activos estáticos (librerías JS como Plotly, hojas de estilo CSS, íconos y fuentes) deben almacenarse y servirse localmente desde `web/static/`.

## 3. Paradigma Híbrido Explícito
- **Programación Orientada a Objetos (POO)**: Reservada para la gestión de estado e inspección estructural de entrada (`DatasetAuditor` en `core/auditor.py`), manteniendo intacto `self.df_raw`.
- **Programación Funcional**: Reservada para transformación y análisis (`core/transformador.py`, `core/estadistica.py`). Toda función debe ser pura: recibe un `pd.DataFrame`, nunca lo muta en sitio y retorna nuevas estructuras o tuplas `(nuevo_df, logs)` usando operaciones vectorizadas de NumPy/Pandas y comprensiones.

## 4. Convenciones de Código y Calidad
- **Estilo**: Cumplir estrictamente con **PEP 8** (nombres claros en `snake_case` para funciones/variables y `PascalCase` para clases).
- **Anotaciones de Tipo (*Type Hints*)**: Obligatorias en todas las firmas de funciones, métodos y atributos públicos.
- **Documentación**: Incluir *docstrings* concisos en clases y funciones públicas; redactar comentarios y mensajes de usuario en Español Latinoamericano.
- **Manejo Explícito de Excepciones**: Usar bloques `try / except` específicos (`UnicodeDecodeError`, `pd.errors.EmptyDataError`, `KeyError`, `ZeroDivisionError`, `ValueError`, `TimeoutError`, `FileNotFoundError`). Prohibido usar `except:` desnudo o silenciar errores sin registro o manejo controlado.

## 5. Estrategia Offline-First y Degradación Elegante
- Ningún módulo de `core/` puede depender de conexión a internet para completar su flujo.
- La integración con Google GenAI (`core/sintetizador.py`) debe aplicar un **timeout estricto de 4 segundos** y contar siempre con un respaldo determinista local ante ausencia de `GEMINI_API_KEY`, errores de red o respuestas inválidas.
