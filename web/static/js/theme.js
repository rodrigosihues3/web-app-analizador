/* Tema claro/oscuro y adaptación de los gráficos Plotly al tema activo.
   Funciona 100% local: no depende de recursos externos. */
(function () {
    "use strict";

    var CLAVE_TEMA = "tema-analizador";
    var raiz = document.documentElement;

    /* ---------- Utilidades de color ---------- */
    function variable(nombre) {
        return getComputedStyle(raiz).getPropertyValue(nombre).trim();
    }

    function hexARgb(hex) {
        var h = hex.replace("#", "");
        if (h.length === 3) {
            h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
        }
        return [
            parseInt(h.substr(0, 2), 16),
            parseInt(h.substr(2, 2), 16),
            parseInt(h.substr(4, 2), 16)
        ];
    }

    function rgba(hex, alfa) {
        var c = hexARgb(hex);
        return "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + alfa + ")";
    }

    function mezclar(a, b, t) {
        return [0, 1, 2].map(function (i) {
            return Math.round(a[i] + (b[i] - a[i]) * t);
        });
    }

    function luminancia(rgb) {
        var v = rgb.map(function (x) {
            var s = x / 255;
            return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
        });
        return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
    }

    function contraste(l1, l2) {
        var a = Math.max(l1, l2);
        var b = Math.min(l1, l2);
        return (a + 0.05) / (b + 0.05);
    }

    /* ---------- Tema de la página ---------- */
    function temaActual() {
        return raiz.getAttribute("data-theme") === "dark" ? "dark" : "light";
    }

    function actualizarBoton() {
        var boton = document.getElementById("boton-tema");
        if (!boton) {
            return;
        }
        var oscuro = temaActual() === "dark";
        var texto = oscuro ? "Cambiar a tema claro" : "Cambiar a tema oscuro";
        boton.setAttribute("aria-label", texto);
        boton.setAttribute("title", texto);
    }

    function guardarTema(tema) {
        try {
            window.localStorage.setItem(CLAVE_TEMA, tema);
        } catch (err) {
            /* Almacenamiento no disponible: la preferencia solo dura la sesión. */
        }
    }

    function alternarTema() {
        var nuevo = temaActual() === "dark" ? "light" : "dark";
        raiz.setAttribute("data-theme", nuevo);
        guardarTema(nuevo);
        actualizarBoton();
        aplicarTemaGraficos(document);
    }

    /* ---------- Gráficos Plotly ---------- */
    function paletaTema() {
        var superficie = variable("--surface");
        var serie = [1, 2, 3, 4, 5, 6].map(function (n) {
            return variable("--chart-" + n);
        });
        return {
            texto: variable("--text"),
            apagado: variable("--muted"),
            borde: variable("--border"),
            superficie: superficie,
            acento: variable("--accent-text"),
            serie: serie,
            neg: variable("--chart-neg"),
            pos: variable("--chart-pos"),
            fuente: getComputedStyle(document.body).fontFamily
        };
    }

    function colorEscala(valor, p) {
        /* Escala divergente: gris azulado -> superficie -> acento (r en [-1, 1]). */
        var t = Math.max(0, Math.min(1, (valor + 1) / 2));
        var neg = hexARgb(p.neg);
        var med = hexARgb(p.superficie);
        var pos = hexARgb(p.pos);
        return t < 0.5 ? mezclar(neg, med, t * 2) : mezclar(med, pos, (t - 0.5) * 2);
    }

    function colorTextoSobre(rgb, p) {
        var lum = luminancia(rgb);
        var claro = [255, 255, 255];
        var oscuro = hexARgb(p.texto);
        var cClaro = contraste(lum, luminancia(claro));
        var cOscuro = contraste(lum, luminancia(oscuro));
        return cClaro >= cOscuro ? "#ffffff" : p.texto;
    }

    function tiposDeTrazas(div) {
        return (div.data || []).map(function (traza) {
            return traza.type;
        });
    }

    function construirLayout(div, p, tipos) {
        var hayHeatmap = tipos.indexOf("heatmap") !== -1;
        var cambios = {
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            "font.color": p.texto,
            "font.family": p.fuente,
            "font.size": 12,
            "title.text": "",
            "margin.t": 16,
            "margin.l": 56,
            "margin.r": 16,
            "margin.b": 48,
            "hoverlabel.bgcolor": p.superficie,
            "hoverlabel.bordercolor": p.borde,
            "hoverlabel.font.color": p.texto,
            "hoverlabel.font.family": p.fuente,
            "legend.bgcolor": "rgba(0,0,0,0)",
            "legend.font.color": p.texto,
            "modebar.bgcolor": "rgba(0,0,0,0)",
            "modebar.color": p.apagado,
            "modebar.activecolor": p.acento
        };
        ["xaxis", "yaxis"].forEach(function (eje) {
            cambios[eje + ".gridcolor"] = p.borde;
            cambios[eje + ".linecolor"] = p.borde;
            cambios[eje + ".zerolinecolor"] = p.borde;
            cambios[eje + ".tickfont.color"] = p.apagado;
            cambios[eje + ".title.font.color"] = p.apagado;
            if (hayHeatmap) {
                cambios[eje + ".showgrid"] = false;
            }
        });

        /* Anotaciones: valores de celda del heatmap y mensajes informativos. */
        var anotaciones = ((div.layout && div.layout.annotations) || [])
            .filter(function (a) {
                return a.name !== "celda-correlacion";
            })
            .map(function (a) {
                var copia = Object.assign({}, a);
                copia.font = Object.assign({}, a.font || {}, { color: p.apagado, family: p.fuente });
                return copia;
            });

        if (hayHeatmap) {
            var idx = tipos.indexOf("heatmap");
            var traza = div.data[idx];
            var z = traza.z || [];
            (traza.y || []).forEach(function (etiquetaY, i) {
                (traza.x || []).forEach(function (etiquetaX, j) {
                    var valor = Number(z[i] && z[i][j]);
                    if (isNaN(valor)) {
                        return;
                    }
                    anotaciones.push({
                        name: "celda-correlacion",
                        xref: "x",
                        yref: "y",
                        x: etiquetaX,
                        y: etiquetaY,
                        text: valor.toFixed(2),
                        showarrow: false,
                        font: {
                            color: colorTextoSobre(colorEscala(valor, p), p),
                            family: p.fuente,
                            size: 11
                        }
                    });
                });
            });
        }
        cambios.annotations = anotaciones;
        return cambios;
    }

    function construirTrazas(p, tipos) {
        /* Devuelve una lista de {indice, cambios}: un restyle por traza. */
        var grupos = [];
        var contadorCajas = 0;
        tipos.forEach(function (tipo, i) {
            if (tipo === "histogram") {
                grupos.push({
                    indice: i,
                    cambios: {
                        "marker.color": p.serie[0],
                        "marker.line.color": p.superficie,
                        "marker.line.width": 1,
                        opacity: 1
                    }
                });
            } else if (tipo === "box") {
                var color = p.serie[contadorCajas % p.serie.length];
                contadorCajas += 1;
                grupos.push({
                    indice: i,
                    cambios: {
                        "marker.color": color,
                        "line.color": color,
                        fillcolor: rgba(color, 0.22)
                    }
                });
            } else if (tipo === "heatmap") {
                grupos.push({
                    indice: i,
                    cambios: {
                        colorscale: [[0, p.neg], [0.5, p.superficie], [1, p.pos]],
                        zmin: -1,
                        zmax: 1,
                        texttemplate: "",
                        "colorbar.tickfont.color": p.apagado,
                        "colorbar.title.font.color": p.apagado,
                        "colorbar.outlinewidth": 0,
                        "colorbar.thickness": 12
                    }
                });
            }
        });
        return grupos;
    }

    function restyleTraza(div, cambios, indice) {
        /* Cada valor se envuelve en un arreglo: Plotly lo asigna a la traza indicada. */
        var envuelto = {};
        Object.keys(cambios).forEach(function (clave) {
            envuelto[clave] = [cambios[clave]];
        });
        return window.Plotly.restyle(div, envuelto, [indice]);
    }

    function adaptarGrafico(div, intento) {
        if (!window.Plotly || !div) {
            return;
        }
        var listo = div._fullLayout && div.data && div.querySelector(".main-svg");
        if (!listo) {
            if (intento < 25) {
                window.setTimeout(function () {
                    adaptarGrafico(div, intento + 1);
                }, 60);
            }
            return;
        }
        try {
            var p = paletaTema();
            var tipos = tiposDeTrazas(div);
            construirTrazas(p, tipos).forEach(function (grupo) {
                restyleTraza(div, grupo.cambios, grupo.indice);
            });
            window.Plotly.relayout(div, construirLayout(div, p, tipos));
            div.setAttribute("data-tema-aplicado", temaActual());
        } catch (err) {
            /* Si el ajuste de color falla, el gráfico conserva su estilo original. */
        }
    }

    function aplicarTemaGraficos(contexto) {
        if (!window.Plotly || !contexto || !contexto.querySelectorAll) {
            return;
        }
        contexto.querySelectorAll(".plotly-graph-div").forEach(function (div) {
            adaptarGrafico(div, 0);
        });
    }

    /* ---------- Inicialización ---------- */
    window.AnalizadorTema = {
        alternar: alternarTema,
        actual: temaActual,
        aplicarGraficos: aplicarTemaGraficos
    };

    function iniciar() {
        var boton = document.getElementById("boton-tema");
        if (boton) {
            boton.addEventListener("click", alternarTema);
        }
        actualizarBoton();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", iniciar);
    } else {
        iniciar();
    }
})();
