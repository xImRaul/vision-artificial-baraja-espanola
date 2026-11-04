"""
config.py — Parámetros físicos, geométricos y umbrales del Árbol de Decisión.
Reto 07: Visión Artificial Clásica — Baraja Española.

Calibrado experimentalmente sobre el montaje real (caja cenital 60x40x40 cm con
iluminación flash sobre tapete de fieltro verde mate y naipes españoles Mas Reynals,
capturas de 5712x4284 px / 5477x4118 px).
"""

import json
import os
import numpy as np

RUTA_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_CALIBRACION_TAPETE = os.path.join(RUTA_BASE, "calibracion_tapete.json")
RUTA_UMBRALES_JSON = os.path.join(RUTA_BASE, "umbrales_calibrados.json")

# =============================================================================
# 1. SUBCONJUNTO ACOTADO DEL RETO (Punto 7 del enunciado)
# =============================================================================
PALOS_VALIDOS = ("Oros", "Copas", "Espadas", "Bastos")
VALOR_MINIMO = 1
VALOR_MAXIMO = 7  # 28 clases: valores 1 (As) al 7 de los 4 palos

# =============================================================================
# 2. DIMENSIONES FÍSICAS DE REFERENCIA (Punto 2: Montaje experimental)
# =============================================================================
ANCHO_TAPETE_CM = 60.0
ALTO_TAPETE_CM = 40.0
ANCHO_CARTA_MM = 61.5   # Naipe español estándar (61.5 x 95.0 mm)
ALTO_CARTA_MM = 95.0
ASPECTO_TEORICO_CARTA = ALTO_CARTA_MM / ANCHO_CARTA_MM  # 1.545

# Tamaño canónico tras la rectificación homográfica (invariante a escala y giro)
CARTA_RECT_W = 300
CARTA_RECT_H = 460

# =============================================================================
# 3. SEGMENTACIÓN DEL TAPETE VERDE Y SOMBRAS EN HSV (Paso 1 del Pipeline)
# =============================================================================
# En captura cenital con flash dentro de caja, el tapete presenta zonas muy
# iluminadas (S moderada, V alta) y esquinas en penumbra (V baja), mientras que
# la cartulina blanca del naipe mantiene un perímetro continuo de alta luminancia
# (V >= 120) y tono fuera del verde del tapete.
VERDE_BAJO = np.array([35, 22, 5], dtype=np.uint8)
VERDE_ALTO = np.array([98, 255, 255], dtype=np.uint8)
UMBRAL_V_SOMBRA_FONDO = 120   # Píxeles oscuros (V < 120) son penumbra, nunca cartulina blanca
KERNEL_TAPETE_PX = 7          # Kernel elíptico 7x7 para apertura/cierre morfológico

# =============================================================================
# 4. DETECCIÓN GEOMÉTRICA DE CARTAS (Nivel 1: Filtrado de contornos externos)
# =============================================================================
AREA_MIN_FRAC_IMAGEN = 0.008    # 0.8% del área del fotograma (descarta motas/bordes)
AREA_MAX_FRAC_IMAGEN = 0.30     # 30% del fotograma
ASPECTO_CARTA_MIN = 1.20        # Relación lado_mayor / lado_menor mínima admitida
ASPECTO_CARTA_MAX = 1.85        # Relación lado_mayor / lado_menor máxima admitida
RECTANGULARIDAD_MIN = 0.80      # area_contorno / area_minAreaRect (naipe real ~0.95)
SOLIDEZ_EXTERNA_MIN = 0.88      # area_contorno / area_convexHull

# =============================================================================
# 5. SEGMENTACIÓN INTERNA DE SÍMBOLOS SOBRE CARTA RECTIFICADA (300 x 460 px)
# =============================================================================
# Recorte porcentual interior al recuadro negro ("pintas") del naipe
ROI_X_MIN_FRAC = 0.13
ROI_X_MAX_FRAC = 0.87
ROI_Y_MIN_FRAC = 0.11
ROI_Y_MAX_FRAC = 0.89

# Diferencia de intensidad respecto al percentil 75 del papel marfil de cada carta
CAIDA_GRIS_TINTA = 32
SATURACION_MIN_TINTA = 48

# Validación de símbolos reales frente a texto impreso del fabricante en los 4s
# ("MAS REYNALS NAIPES / OPACO MARFIL"): todo símbolo real del palo posee relleno
# cromático (fracción cromática >= 0.22), mientras que el texto legal es tinta negra (0.00).
AREA_MIN_SIMBOLO_REL = 0.012
FRACCION_CROMATICA_MIN_SIMBOLO = 0.22
SATURACION_MIN_CROMATICA = 55
VALOR_MIN_CROMATICO = 60

# =============================================================================
# 6. UMBRALES EXPLÍCITOS DEL ÁRBOL DE DECISIÓN (Calibrados sobre las 28 cartas)
# =============================================================================
# --- NODO P1: Separación de ESPADAS ---
# Cartas 2..7 de Espadas: hojas azul-acero con ratio_azul en [0.56, 0.59] (>= 0.25).
# As (1) de Espadas: espada azul (ratio_azul = 0.14 >= 0.11) con cinta verde/roja
# (ratio_verde = 0.22 >= 0.10) y área moderada (area_max_rel = 0.42 < 0.49).
UMBRAL_AZUL_ESPADAS_ALTO = 0.25
UMBRAL_AZUL_AS_ESPADAS = 0.11
UMBRAL_AREA_MAX_AS_ESPADAS = 0.49

# --- NODO P2: Separación de BASTOS ---
# Todos los Bastos (1..7) tienen verde vegetal significativo (ratio_verde en [0.11, 0.46])
# y proporción amarilla moderada (ratio_amarillo en [0.14, 0.44] < 0.52), a diferencia
# de los Oros (ratio_amarillo en [0.62, 0.82]) y las Copas (ratio_verde <= 0.01).
UMBRAL_VERDE_BASTOS = 0.10
UMBRAL_AMARILLO_MAX_BASTOS = 0.52

# --- NODO P3: Separación de OROS vs COPAS ---
# Todos los Oros (1..7) presentan fuerte predominio dorado (ratio_amarillo >= 0.59) y
# ausencia de azul celeste (ratio_azul < 0.055), mientras que todas las Copas (1..7)
# tienen rojo carmesí + cenefa azul celeste (ratio_azul en [0.08, 0.10], ratio_amarillo <= 0.57).
UMBRAL_AMARILLO_OROS = 0.59
UMBRAL_AZUL_MAX_OROS = 0.055

# --- NODO DE VALOR ESPECIAL EN BASTOS (As de Bastos vs 3 de Bastos en 'X') ---
# En el 3 de Bastos los tres garrotes se cruzan formando una única componente conexa
# de gran área (area_max_rel = 0.57 >= 0.50) atada con un lazo azul (ratio_azul = 0.12 >= 0.05),
# mientras que el As (1) de Bastos es un único garrote (area_max_rel = 0.43, ratio_azul = 0.00).
UMBRAL_AREA_3_BASTOS = 0.50
UMBRAL_AZUL_LAZO_3_BASTOS = 0.05


def cargar_calibraciones_locales():
    """Carga automáticamente umbrales personalizados si existen en JSON."""
    if os.path.exists(RUTA_UMBRALES_JSON):
        try:
            with open(RUTA_UMBRALES_JSON, "r", encoding="utf-8") as f:
                umbrales = json.load(f)
            g = globals()
            for clave, valor in umbrales.items():
                if clave in g and isinstance(g[clave], (int, float)):
                    g[clave] = type(g[clave])(valor)
        except (OSError, ValueError, KeyError):
            pass


cargar_calibraciones_locales()
