# Sistema Experto de Visión Artificial Clásica — Baraja Española (Reto 07)

**Vídeo Demostrativo (YouTube):** [https://youtu.be/UpGTOjYVCVc](https://youtu.be/UpGTOjYVCVc)


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xImRaul/vision-artificial-baraja-espanola/blob/main/examen_google_colab.ipynb)

Proyecto completo para el Examen Parcial de **Visión Artificial Clásica** (Inteligencia Artificial). Analiza automáticamente escenas cenitales reales de cartas físicas de la **baraja española tradicional (Mas Reynals / Royal Casino Opaco Marfil)** para el subconjunto acotado de **28 clases** (valores **1 al 7** de los cuatro palos: **Oros, Copas, Espadas y Bastos**), dispuestas en posiciones y rotaciones arbitrarias ($0^\circ\text{–}360^\circ$) sobre un tapete de fieltro verde mate iluminado cenitalmente dentro de cabina de $60 \times 40 \times 40\text{ cm}$.

Cumple estrictamente la prohibición de modelos entrenados (**sin Deep Learning, CNN, YOLO, OCR ni clasificadores ML**): toda la cadena se fundamenta en **segmentación HSV + penumbra, detección geométrica con `cv2.minAreaRect`, rectificación homográfica a $300 \times 460\text{ px}$, validación cromática de siluetas internas, extracción explícita de descriptores y un Árbol de Decisión auditable**.

---


## ⚠️ Requerimientos de Uso y Colocación de las Cartas (Importante)

Para que el sistema clásico detecte y clasifique correctamente todos los naipes:
1. **Las cartas NO pueden estar una encima de otra (solapadas) ni tocándose:** Debe existir una separación mínima de **1–2 cm de tapete verde** alrededor de cada carta. Si una carta pisa a otra, `cv2.findContours(RETR_EXTERNAL)` fusiona ambas siluetas en un único contorno no rectangular y las descarta.
2. **Cartas completas y dentro del tapete verde:** Todas las cartas deben estar boca arriba e íntegramente dentro del tapete verde (sin cortarse en los bordes de la foto).
3. **Posición y rotación libres (`0°–360°`):** Se pueden colocar en cualquier ángulo o invertidas `180°` (la homografía las endereza automáticamente a `300 × 460 px`).
4. **Subconjunto del reto (28 cartas):** Naipes del **1 (As) al 7** de **Oros, Copas, Espadas y Bastos**.
5. **Cámara cenital e iluminación:** Fotografía perpendicular al tapete (`~40 cm`), funcionando igual con **flash del móvil o dos lámparas laterales**.

*(Ver documento completo en [`REQUERIMIENTOS_DE_USO_Y_ESCENA.md`](REQUERIMIENTOS_DE_USO_Y_ESCENA.md)).*

## Resultados Experimentales Verificados (Fotos Reales en `fotos/`)

- **Nivel 1 (Detección y Geometría):** `100.00%` Recall y `100.00%` Precisión (`47 / 47` naipes reales detectados, `0` falsos positivos sobre tapete vacío `IMG_7885.jpeg`, desviación angular residual tras homografía de `0.02°` y escala media de `14.38 px/mm`).
- **Nivel 2 (Identificación Palo + Valor 1–7):** `100.00%` (`47 / 47` cartas clasificadas exactamente en las 28 clases, incluyendo Ases ornamentados, el `3 de Bastos` cruzado en `'X'` y la eliminación del texto legal impreso en todos los `4`s).
- **Nivel 3 (Interpretación Semántica de Mano):** `100.00%` (`8 / 8` escenas reales con conteo por palo/valor, puntuación en *Siete y Media*, sumas de 15 en *La Escoba* y jugadas de mano exactas).

---

## Estructura del Repositorio

| Archivo / Carpeta | Descripción |
| :--- | :--- |
| `examen_google_colab.ipynb` | **Cuaderno para el examen presencial en Google Colab**: clona el repo, analiza al instante las fotos subidas del aula, muestra ruta del árbol + mosaico homográfico y permite ajustar el umbral HSV/sombra en vivo. |
| `config.py` | Dimensiones físicas ($61{,}5 \times 95{,}0\text{ mm}$), parámetros morfológicos y todos los umbrales explícitos calibrados del Árbol de Decisión. |
| `pipeline_baraja.py` | Motor clásico completo de 8 etapas (segmentación, detección por contornos y `minAreaRect`, homografía a $300 \times 460\text{ px}$, descriptores, árbol de decisión e interpretación de Nivel 3). |
| `calibrar_tapete.py` | Mide percentiles $P_1\text{–}P_{99}$ en HSV sobre el tapete vacío (`IMG_7885.jpeg`) y guarda `calibracion_tapete.json`. |
| `1-calibrar.py` | Herramienta interactiva: haz **clic izquierdo** sobre cualquier naipe para inspeccionar en vivo sus características, ruta en el árbol y máscara interna. |
| `2_main.py` | Programa principal: selector directo de fotografía, soporte CLI para `--imagen`, `--carpeta` y `--camara 0` en tiempo real. |
| `3_evaluacion_cuantitativa.py` | Evalúa `fotos/ground_truth.json` y genera las matrices de confusión, gráficas de separabilidad y `tabla_medidas_cartas.csv` en `memoria_figuras/`. |
| `ejecutar_sistema.bat` | Lanzador interactivo en Windows mediante doble clic. |
| `fotos/` | Las 8 fotografías reales en alta resolución, su `ground_truth.json` y las salidas procesadas `*_resultado.jpeg`. |

---

## Ejecución Rápida (Local y Google Colab)

- **En Google Colab (Examen presencial):** Pulsa en [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xImRaul/vision-artificial-baraja-espanola/blob/main/examen_google_colab.ipynb)
- **En local / Visual Studio Code:**

```powershell
# Ejecutar el programa principal (selector directo de foto o --camara 0)
python 2_main.py

# Inspeccionar interactivamente cualquier foto real haciendo clic en cada carta
python 1-calibrar.py --imagen fotos/IMG_7890.jpeg

# Ejecutar evaluación cuantitativa completa (Niveles 1, 2 y 3)
python 3_evaluacion_cuantitativa.py
```
