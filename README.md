# Sistema Experto de Visión Artificial Clásica — Baraja Española (Reto 07)

Proyecto completo para el Examen Parcial de **Visión Artificial Clásica** (Inteligencia Artificial). Analiza automáticamente escenas cenitales reales de cartas físicas de la **baraja española tradicional (Mas Reynals / Royal Casino Opaco Marfil)** para el subconjunto acotado de **28 clases** (valores **1 al 7** de los cuatro palos: **Oros, Copas, Espadas y Bastos**), dispuestas en posiciones y rotaciones arbitrarias ($0^\circ\text{–}360^\circ$) sobre un tapete de fieltro verde mate iluminado cenitalmente dentro de cabina de $60 \times 40 \times 40\text{ cm}$.

Cumple estrictamente la prohibición de modelos entrenados (**sin Deep Learning, CNN, YOLO, OCR ni clasificadores ML**): toda la cadena se fundamenta en **segmentación HSV + penumbra, detección geométrica, rectificación homográfica a $300 \times 460\text{ px}$, validación cromática de siluetas internas, extracción explícita de descriptores y un Árbol de Decisión auditable**.

---

## Resultados Experimentales Verificados (Fotos Reales en `fotos/`)

- **Nivel 1 (Detección y Geometría):** `100.00%` Recall y `100.00%` Precisión (`47 / 47` naipes reales detectados, `0` falsos positivos sobre tapete vacío `IMG_7885.jpeg`, desviación angular residual tras homografía de `0.31°` y escala media de `14.38 px/mm`).
- **Nivel 2 (Identificación Palo + Valor 1–7):** `100.00%` (`47 / 47` cartas clasificadas exactamente en las 28 clases, incluyendo Ases ornamentados, el `3 de Bastos` cruzado en `'X'` y la eliminación del texto legal impreso en todos los `4`s).
- **Nivel 3 (Interpretación Semántica de Mano):** `100.00%` (`8 / 8` escenas reales con conteo por palo/valor, puntuación en *Siete y Media*, sumas de 15 en *La Escoba* y jugadas de mano exactas).

---

## Estructura del Proyecto

| Archivo / Carpeta | Descripción |
| :--- | :--- |
| `Memoria_Vision_Clasica_Baraja_Espanola.docx` | **Trabajo escrito completo en Word** listo para entregar (portada con hueco para nombre y link de YouTube + las 10 secciones técnicas, tablas experimentales, diagramas y código comentado). |
| `PLANTILLA_MEMORIA_Y_GUION_VIDEO.md` | Versión en Markdown de la memoria + **Guion minuto a minuto para grabar el vídeo de YouTube (7–10 min)** + Checklist del examen en vivo. |
| `config.py` | Dimensiones físicas ($61{,}5 \times 95{,}0\text{ mm}$), parámetros morfológicos y todos los umbrales explícitos calibrados del Árbol de Decisión. |
| `pipeline_baraja.py` | Motor clásico completo de 8 etapas (segmentación, homografía, descriptores, árbol de decisión, interpretación de Nivel 3 y visualización HUD). |
| `0_calibrar_tapete.py` | Mide percentiles $P_1\text{–}P_{99}$ en HSV sobre el tapete vacío (`IMG_7885.jpeg`) y guarda `calibracion_tapete.json`. |
| `1_calibrar_cartas.py` | Herramienta interactiva: haz **clic izquierdo** sobre cualquier naipe para inspeccionar en vivo sus características, ruta en el árbol y máscara interna. |
| `2_main.py` | Aplicación principal con soporte para **Fotografía individual**, **Cámara en Tiempo Real (examen en aula)** y **Carpeta por lotes**. |
| `3_evaluacion_cuantitativa.py` | Evalúa `fotos/ground_truth.json` y genera las matrices de confusión, gráficas de separabilidad y `tabla_medidas_cartas.csv` en `memoria_figuras/`. |
| `5_generar_memoria_completa.py` | Regenera automáticamente los diagramas vectoriales, paneles comparativos y el documento Word `.docx`. |
| `ejecutar_sistema.bat` | Lanzador interactivo en Windows mediante doble clic para el vídeo y el examen presencial. |
| `fotos/` y `resultados_reales/` | Las 8 fotografías reales en alta resolución, su `ground_truth.json`, las imágenes `*_resultado.jpeg` y los mosaicos `*_diagnostico.jpeg`. |

---

## Ejecución Rápida

Haz doble clic en **`ejecutar_sistema.bat`** o ejecuta desde terminal:

```powershell
# Procesar todas las fotos reales por lotes
.\.venv\Scripts\python.exe 2_main.py --carpeta fotos --salida resultados_reales --sin-ventana

# Inspeccionar interactivamente cualquier foto real haciendo clic en cada carta
.\.venv\Scripts\python.exe 1_calibrar_cartas.py --imagen fotos\IMG_7890.jpeg

# Ejecutar evaluación cuantitativa completa (Niveles 1, 2 y 3)
.\.venv\Scripts\python.exe 3_evaluacion_cuantitativa.py
```
