# REQUERIMIENTOS DE USO Y CONDICIONES DE LA ESCENA
**Proyecto:** Reto 07 — Examen Parcial de Visión Artificial Clásica (Baraja Española)  
**Autor:** Raúl García — Universidad Europea del Atlántico (`IYA051`)

Para que el sistema de visión artificial clásica (`100% OpenCV`, sin redes neuronales ni aprendizaje automático) detecte y clasifique las cartas con **100% de precisión**, la escena fotografiada debe cumplir los siguientes requerimientos:

---

## 1. Requerimientos Físicos y de Colocación de las Cartas (Obligatorios)

1. **PROHIBIDO SOLAPAR O SUPERPONER CARTAS (No poner una carta encima de otra):**
   * Las cartas **NO pueden estar una encima de otra (solapadas) ni tocándose por sus bordes blancos**.
   * Debe dejarse una separación mínima de **1 a 2 cm de tapete verde visible** entre carta y carta.
   * **Motivo técnico:** El algoritmo segmenta la escena mediante contornos externos (`cv2.findContours` con `cv2.RETR_EXTERNAL`) y ajusta un rectángulo de área mínima (`cv2.minAreaRect`). Si una carta se coloca encima de otra o se tocan, sus siluetas blancas se fusionan en un único polígono irregular que rompe la relación de aspecto (`[1.20, 1.85]`) y la rectangularidad (`>= 0.72`), por lo que el sistema no las detectará como cartas individuales.

2. **CARTAS COMPLETAMENTE DENTRO DEL TAPETE VERDE:**
   * Todas las cartas deben apoyarse **íntegramente dentro del tapete verde**, sin quedar cortadas por el borde de la foto ni ocultas parcialmente bajo las paredes de la caja de cartón.

3. **ORIENTACIÓN Y POSICIÓN LIBRES (`0° a 360°`, BOCA ARRIBA):**
   * Las cartas deben colocarse **boca arriba** (con los palos visibles).
   * Pueden colocarse en **cualquier posición del tapete y con cualquier ángulo de giro (`0° a 360°`, incluso boca abajo / invertidas `180°`)**, ya que el sistema las endereza automáticamente mediante transformación homográfica (`cv2.warpPerspective`) a `300 × 460 px`.

4. **SUBCONJUNTO ACOTADO DE LA BARAJA ESPAÑOLA (28 CARTAS: 1 AL 7):**
   * El sistema está diseñado y calibrado para las **28 cartas numéricas del 1 (As) al 7** de los cuatro palos (**Oros, Copas, Espadas y Bastos**), excluyendo figuras (Sota, Caballo, Rey) y cartas 8/9 según la acotación oficial del Reto 07.

5. **ILUMINACIÓN Y CÁMARA CENITAL:**
   * **Posición de la cámara:** Cenital (perpendicular al tapete verde, a unos `~40 cm` de altura, encuadrando el tapete).
   * **Iluminación admitida:** Funciona indistintamente con **flash del móvil** o con **dos lámparas laterales**, siempre que la luz no sature el barniz de la carta con un reflejo blanco puro directo (`S -> 0, V -> 255`) sobre los dibujos interiores.
   * **Tapete despejado:** No deben colocarse otros objetos blancos (papeles, cables, manos) encima del tapete verde en el momento de hacer la foto.

---

## 2. Requerimientos de Software (Entorno Python)

* **Python:** `>= 3.9` (probado en Python `3.10`, `3.11` y `3.12` y en Google Colab)
* **Librerías necesarias (`requirements.txt`):**
  ```bash
  pip install opencv-python>=4.8.0 numpy>=1.24.0 matplotlib>=3.7.0
  ```
