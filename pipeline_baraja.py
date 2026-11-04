"""
pipeline_baraja.py — Motor completo de Visión Artificial Clásica para Baraja Española.

Implementa estrictamente la cadena obligatoria del Reto 07 (sin Machine Learning ni Deep Learning):
  Imagen RGB -> Preprocesamiento -> Segmentación HSV + Sombras -> Detección y Homografía (300x460)
  -> Segmentación Interna de Siluetas Cromáticas -> Extracción de Características Explícitas
  -> Árbol de Decisión Auditable -> Interpretación de Escena (Nivel 3).
"""

from itertools import combinations
import cv2
import numpy as np
import config as cfg


# =============================================================================
# UTILIDADES DE ENTRADA/SALIDA (Compatibilidad con tildes y eñes en Windows)
# =============================================================================

def leer_imagen(ruta):
    """Lee una imagen soportando rutas con caracteres no ASCII en Windows."""
    try:
        datos = np.fromfile(ruta, dtype=np.uint8)
        if datos.size == 0:
            return None
        return cv2.imdecode(datos, cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        return None


def guardar_imagen(ruta, imagen_bgr):
    """Guarda una imagen soportando rutas con caracteres no ASCII en Windows."""
    ext = "." + ruta.rsplit(".", 1)[-1] if "." in ruta else ".jpg"
    exito, buffer = cv2.imencode(ext, imagen_bgr)
    if exito:
        buffer.tofile(ruta)
    return exito


def redimensionar_vista(imagen, max_ancho=1600, max_alto=1050):
    """Escala proporcionalmente una imagen para visualizarla cómodamente en pantalla."""
    h, w = imagen.shape[:2]
    escala = min(max_ancho / max(w, 1), max_alto / max(h, 1), 1.0)
    if escala >= 1.0:
        return imagen.copy(), 1.0
    nuevo_tam = (max(1, int(w * escala)), max(1, int(h * escala)))
    return cv2.resize(imagen, nuevo_tam, interpolation=cv2.INTER_AREA), escala


# =============================================================================
# ETAPAS 1 Y 2: PREPROCESAMIENTO Y SEGMENTACIÓN DEL TAPETE VERDE
# =============================================================================

def segmentar_tapete(imagen_bgr):
    """
    Suaviza ruido de fibra textil y segmenta el fondo (tapete verde + zonas en penumbra)
    en espacio HSV. Devuelve la máscara binaria de objetos (cartas = 255) y el % de fondo.
    """
    suavizada = cv2.GaussianBlur(imagen_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(suavizada, cv2.COLOR_BGR2HSV)

    mask_verde = cv2.inRange(hsv, cfg.VERDE_BAJO, cfg.VERDE_ALTO)
    mask_penumbra = (hsv[:, :, 2] < cfg.UMBRAL_V_SOMBRA_FONDO).astype(np.uint8) * 255
    mask_fondo = cv2.bitwise_or(mask_verde, mask_penumbra)
    porcentaje_tapete = 100.0 * cv2.countNonZero(mask_fondo) / max(mask_fondo.size, 1)

    # Todo lo que NO es tapete ni sombra oscura corresponde a la cartulina blanca de las cartas
    mask_objetos = cv2.bitwise_not(mask_fondo)

    k = cfg.KERNEL_TAPETE_PX
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask_objetos = cv2.morphologyEx(mask_objetos, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_objetos = cv2.morphologyEx(mask_objetos, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask_objetos, hsv, porcentaje_tapete


# =============================================================================
# ETAPA 3: DETECCIÓN GEOMÉTRICA, POSICIÓN, ORIENTACIÓN Y HOMOGRAFÍA (NIVEL 1)
# =============================================================================

def _ordenar_cuatro_vertices(pts):
    """
    Ordena 4 puntos en sentido horario [sup_izq, sup_der, inf_der, inf_izq]
    garantizando que (v0 -> v1) sea el LADO CORTO (300 px) y (v1 -> v2) el LADO LARGO (460 px).
    """
    centro = np.mean(pts, axis=0)
    angulos = np.arctan2(pts[:, 1] - centro[1], pts[:, 0] - centro[0])
    orden = np.argsort(angulos)
    pts_ord = pts[orden].astype(np.float32)

    d01 = np.linalg.norm(pts_ord[1] - pts_ord[0])
    d12 = np.linalg.norm(pts_ord[2] - pts_ord[1])

    if d01 > d12:
        pts_ord = np.roll(pts_ord, -1, axis=0)

    y_arriba = 0.5 * (pts_ord[0, 1] + pts_ord[1, 1])
    y_abajo = 0.5 * (pts_ord[2, 1] + pts_ord[3, 1])
    if y_arriba > y_abajo:
        pts_ord = np.roll(pts_ord, 2, axis=0)

    return pts_ord


def obtener_esquinas_carta(contorno, rect_min):
    """
    Obtiene los 4 vértices externos del naipe a partir del rectángulo mínimo
    orientado (minAreaRect), que se ajusta exactamente a los 4 lados rectos
    de la carta sin recortar hacia dentro por culpa de las esquinas redondeadas.
    """
    pts_box = cv2.boxPoints(rect_min).astype(np.float32)
    return _ordenar_cuatro_vertices(pts_box)


def rectificar_carta(imagen_bgr, vertices_ordenados):
    """Transformación proyectiva (homografía) del naipe al tamaño canónico 300x460 px."""
    w, h = cfg.CARTA_RECT_W, cfg.CARTA_RECT_H
    destino = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    )
    matriz_h = cv2.getPerspectiveTransform(vertices_ordenados, destino)
    return cv2.warpPerspective(imagen_bgr, matriz_h, (w, h), flags=cv2.INTER_LINEAR)


def _resolver_sentido_180(carta_rect_bgr):
    """
    Estima si una carta asimétrica (ej. 1, 3, 5 o 7) se encuentra invertida 180°
    evaluando el centro de gravedad cromático vertical en la región interior:
      - En los 7s y Ases de Copas/Bastos el peso visual está en la mitad superior;
      - En el As de Oros la cartela roja 'NAIPE ESPAÑOL' está en el tercio inferior,
        mientras el medallón circular amarillo domina la mitad superior.
    Devuelve (carta_orientada, fue_girada_180).
    """
    h, w = carta_rect_bgr.shape[:2]
    roi = carta_rect_bgr[int(0.13 * h):int(0.87 * h), int(0.15 * w):int(0.85 * w)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    crom = (hsv[:, :, 1] >= cfg.SATURACION_MIN_CROMATICA) & (hsv[:, :, 2] >= cfg.VALOR_MIN_CROMATICO)

    if np.count_nonzero(crom) < 200:
        return carta_rect_bgr, False

    # Para el As de Oros (medallón amarillo arriba vs cartela roja abajo) o 7s (4 figuras arriba vs 3 abajo)
    amarillo = crom & (hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 35)
    if np.count_nonzero(amarillo) > 0.55 * np.count_nonzero(crom):
        ys, _ = np.where(amarillo)
    else:
        ys, _ = np.where(crom)

    if len(ys) > 100:
        cy_norm = float(np.mean(ys)) / roi.shape[0]
        if cy_norm > 0.535:
            return cv2.rotate(carta_rect_bgr, cv2.ROTATE_180), True

    return carta_rect_bgr, False


def detectar_cartas(imagen_bgr, mask_objetos):
    """
    Nivel 1 — Localiza todas las cartas sobre el tapete, descarta ruido o bordes
    y extrae posición (px y cm), escala espacial (px/mm), orientación y homografía.
    """
    h_img, w_img = imagen_bgr.shape[:2]
    area_img = float(h_img * w_img)

    # RETR_EXTERNAL extrae solo el perímetro exterior blanco de cada carta,
    # ignorando cualquier hueco interior causado por dibujos verdes o sombras internas.
    contornos, _ = cv2.findContours(
        mask_objetos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    cartas_detectadas = []
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        frac_area = area / area_img
        if frac_area < cfg.AREA_MIN_FRAC_IMAGEN or frac_area > cfg.AREA_MAX_FRAC_IMAGEN:
            continue

        x, y, w_box, h_box = cv2.boundingRect(cnt)
        if x <= 2 or y <= 2 or (x + w_box) >= (w_img - 2) or (y + h_box) >= (h_img - 2):
            continue

        rect_min = cv2.minAreaRect(cnt)
        (_, _), (lado_a, lado_b), _ = rect_min
        lado_mayor = max(lado_a, lado_b)
        lado_menor = min(lado_a, lado_b)
        if lado_menor < 20:
            continue

        aspecto_orientado = lado_mayor / lado_menor
        area_rect = lado_mayor * lado_menor
        rectangularidad = area / area_rect if area_rect > 0 else 0.0

        hull = cv2.convexHull(cnt)
        area_hull = cv2.contourArea(hull)
        solidez_ext = area / area_hull if area_hull > 0 else 0.0

        if not (cfg.ASPECTO_CARTA_MIN <= aspecto_orientado <= cfg.ASPECTO_CARTA_MAX):
            continue
        if rectangularidad < cfg.RECTANGULARIDAD_MIN or solidez_ext < cfg.SOLIDEZ_EXTERNA_MIN:
            continue

        momentos = cv2.moments(cnt)
        if momentos["m00"] == 0:
            continue
        cx = momentos["m10"] / momentos["m00"]
        cy = momentos["m01"] / momentos["m00"]

        # Resolución espacial experimental medida directamente sobre el naipe (px/mm)
        px_por_mm = 0.5 * (
            (lado_mayor / cfg.ALTO_CARTA_MM) + (lado_menor / cfg.ANCHO_CARTA_MM)
        )
        pos_x_cm = (cx / max(px_por_mm, 1e-6)) / 10.0
        pos_y_cm = (cy / max(px_por_mm, 1e-6)) / 10.0

        vertices = obtener_esquinas_carta(cnt, rect_min)
        carta_rect = rectificar_carta(imagen_bgr, vertices)
        carta_rect, girada_180 = _resolver_sentido_180(carta_rect)
        if girada_180:
            vertices = np.roll(vertices, 2, axis=0)

        medio_sup = 0.5 * (vertices[0] + vertices[1])
        medio_inf = 0.5 * (vertices[2] + vertices[3])
        vec_arriba = medio_sup - medio_inf
        angulo_360 = (np.degrees(np.arctan2(vec_arriba[0], -vec_arriba[1])) + 360.0) % 360.0
        angulo_eje = angulo_360 % 180.0

        cartas_detectadas.append({
            "contorno": cnt,
            "bbox": (int(x), int(y), int(w_box), int(h_box)),
            "vertices": vertices,
            "centroide_px": (float(cx), float(cy)),
            "posicion_cm": (float(pos_x_cm), float(pos_y_cm)),
            "px_por_mm": float(px_por_mm),
            "area_carta_px2": float(area),
            "aspecto_carta": float(aspecto_orientado),
            "rectangularidad": float(rectangularidad),
            "solidez_externa": float(solidez_ext),
            "angulo_eje_deg": float(angulo_eje),
            "angulo_360_deg": float(angulo_360),
            "carta_rect": carta_rect,
        })

    # Ordenar por filas (superior/inferior) y de izquierda a derecha
    paso_fila = max(120.0, h_img * 0.22)
    cartas_detectadas.sort(key=lambda c: (int(c["centroide_px"][1] // paso_fila), c["centroide_px"][0]))
    for idx, c in enumerate(cartas_detectadas, start=1):
        c["id"] = idx

    return cartas_detectadas


# =============================================================================
# ETAPAS 4 Y 5: SEGMENTACIÓN INTERNA Y EXTRACCIÓN DE CARACTERÍSTICAS
# =============================================================================

def contar_cortes_pintas(carta_rect_bgr):
    """
    Cuenta las discontinuidades ("pintas": 0=Oros, 1=Copas, 2=Espadas, 3=Bastos)
    en la línea negra superior e inferior del recuadro interior de la carta.
    """
    gris = cv2.cvtColor(carta_rect_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(carta_rect_bgr, cv2.COLOR_BGR2HSV)
    h, w = gris.shape
    negro = ((gris < 135) & (hsv[:, :, 1] < 90)).astype(np.uint8)

    cortes = []
    for y1, y2 in [(int(0.03 * h), int(0.11 * h)), (int(0.89 * h), int(0.97 * h))]:
        sub = negro[y1:y2, int(0.18 * w):int(0.82 * w)]
        banda_occ = cv2.blur(sub.astype(np.float32), (1, 5))
        occ = np.sum(banda_occ > 0.15, axis=1)
        if len(occ) == 0 or np.max(occ) < 40:
            continue
        r0 = int(np.argmax(occ))
        band = np.max(sub[max(0, r0 - 3):min(sub.shape[0], r0 + 4), :], axis=0)
        band = cv2.morphologyEx(band.reshape(1, -1), cv2.MORPH_CLOSE, np.ones((1, 4), np.uint8)).ravel()
        idx = np.where(band > 0)[0]
        if len(idx) < 70:
            continue
        tramo = band[idx[0]:idx[-1] + 1]
        huecos = 1 - tramo
        cambios = np.diff(np.concatenate(([0], huecos, [0])))
        anchos = np.where(cambios == -1)[0] - np.where(cambios == 1)[0]
        cortes.append(int(np.sum((anchos >= 5) & (anchos <= 95))))

    if not cortes:
        return -1
    return int(round(float(np.median(cortes))))


def extraer_caracteristicas_carta(carta_rect_bgr):
    """
    Segmenta las siluetas completas de los símbolos en la ROI interior y descarta
    automáticamente textos negros acromáticos (como la leyenda del fabricante en los 4s)
    exigiendo relleno cromático interno en cada componente válida.
    """
    h, w = carta_rect_bgr.shape[:2]
    x1 = int(cfg.ROI_X_MIN_FRAC * w)
    x2 = int(cfg.ROI_X_MAX_FRAC * w)
    y1 = int(cfg.ROI_Y_MIN_FRAC * h)
    y2 = int(cfg.ROI_Y_MAX_FRAC * h)

    roi_bgr = carta_rect_bgr[y1:y2, x1:x2]
    roi_h, roi_w = roi_bgr.shape[:2]
    area_roi = float(roi_h * roi_w)

    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    roi_gris = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Umbral adaptativo respecto al nivel de blanco/marfil de ESTA carta
    blanco_ref = float(np.percentile(roi_gris, 75))
    mask_tinta = (
        (roi_gris < (blanco_ref - cfg.CAIDA_GRIS_TINTA))
        | (roi_hsv[:, :, 1] >= cfg.SATURACION_MIN_TINTA)
    ).astype(np.uint8) * 255

    # Apertura morfológica 3x3: elimina contactos puntuales de 1-2 px entre figuras vecinas
    # (ej. borde del 3 de Copas) conservando intactos los tallos (~10 px de grosor)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_limpia = cv2.morphologyEx(mask_tinta, cv2.MORPH_OPEN, k_open, iterations=1)

    # Mapa de píxeles cromáticos vivos (usado para validar símbolos y medir ratios HSV)
    mask_crom_total = (
        (roi_hsv[:, :, 1] >= cfg.SATURACION_MIN_CROMATICA)
        & (roi_hsv[:, :, 2] >= cfg.VALOR_MIN_CROMATICO)
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_limpia, connectivity=8)
    area_min_px = cfg.AREA_MIN_SIMBOLO_REL * area_roi
    mask_simbolos_rellena = np.zeros_like(mask_limpia)
    componentes = []

    for lbl in range(1, num_labels):
        if float(stats[lbl, cv2.CC_STAT_AREA]) < area_min_px:
            continue

        comp_bin = (labels == lbl).astype(np.uint8) * 255
        cnts_ext, _ = cv2.findContours(comp_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts_ext:
            continue

        cnt_sim = max(cnts_ext, key=cv2.contourArea)
        area_silueta = float(cv2.contourArea(cnt_sim))
        if area_silueta < area_min_px:
            continue

        # Máscara rellena individual para comprobar que tiene color interior real
        # (descarta al 100% textos legales negros de los 4s y posibles dígitos de esquina)
        m_ind = np.zeros_like(mask_limpia)
        cv2.drawContours(m_ind, [cnt_sim], -1, 255, cv2.FILLED)
        frac_crom = float(np.count_nonzero(mask_crom_total & (m_ind > 0))) / max(area_silueta, 1.0)
        if frac_crom < cfg.FRACCION_CROMATICA_MIN_SIMBOLO:
            continue

        cv2.drawContours(mask_simbolos_rellena, [cnt_sim], -1, 255, cv2.FILLED)

        perim_silueta = float(cv2.arcLength(cnt_sim, True))
        circ = (4.0 * np.pi * area_silueta) / (perim_silueta ** 2) if perim_silueta > 0 else 0.0

        (_, _), (la, lb), _ = cv2.minAreaRect(cnt_sim)
        l_max, l_min = max(la, lb), min(la, lb)
        ar_sim = l_max / l_min if l_min > 1.0 else 1.0

        hull_sim = cv2.convexHull(cnt_sim)
        area_hull_sim = float(cv2.contourArea(hull_sim))
        solidez_sim = area_silueta / area_hull_sim if area_hull_sim > 0 else 0.0

        componentes.append({
            "area_silueta": area_silueta,
            "area_rel": area_silueta / area_roi,
            "circularidad": circ,
            "aspecto": ar_sim,
            "solidez": solidez_sim,
        })

    componentes.sort(key=lambda c: c["area_silueta"], reverse=True)
    n_componentes = len(componentes)

    if n_componentes > 0:
        pesos = np.array([c["area_silueta"] for c in componentes], dtype=np.float64)
        suma_pesos = float(np.sum(pesos))
        circularidad_media = float(np.sum(pesos * [c["circularidad"] for c in componentes]) / suma_pesos)
        aspecto_simbolo_medio = float(np.sum(pesos * [c["aspecto"] for c in componentes]) / suma_pesos)
        area_max_rel = float(componentes[0]["area_rel"])
        area_min_rel = float(componentes[-1]["area_rel"])
        solidez_mayor = float(componentes[0]["solidez"])
        area_total_rel = suma_pesos / area_roi
    else:
        circularidad_media = 0.0
        aspecto_simbolo_medio = 1.0
        area_max_rel = 0.0
        area_min_rel = 0.0
        solidez_mayor = 1.0
        area_total_rel = 0.0

    # Descriptores cromáticos HSV y Lab evaluados sobre los píxeles cromáticos de los símbolos
    mask_crom_simbolos = mask_crom_total & (mask_simbolos_rellena > 0)
    if np.count_nonzero(mask_crom_simbolos) < 25:
        mask_crom_simbolos = mask_crom_total

    pix_hsv = roi_hsv[mask_crom_simbolos]
    pix_lab = roi_lab[mask_simbolos_rellena > 0] if np.count_nonzero(mask_simbolos_rellena) > 0 else roi_lab[mask_tinta > 0]

    if len(pix_hsv) > 0:
        tonos = pix_hsv[:, 0]
        sats = pix_hsv[:, 1]
        total_crom = float(len(tonos))
        ratio_rojo = float(np.sum((tonos <= 11) | (tonos >= 165))) / total_crom
        ratio_amarillo = float(np.sum((tonos >= 12) & (tonos <= 35))) / total_crom
        ratio_verde = float(np.sum((tonos >= 36) & (tonos <= 88))) / total_crom
        ratio_azul = float(np.sum((tonos >= 89) & (tonos <= 138))) / total_crom
        hue_mediana = float(np.median(tonos))
        hue_medio = float(np.mean(tonos))
        saturacion_media = float(np.mean(sats))
    else:
        ratio_rojo = ratio_amarillo = ratio_verde = ratio_azul = 0.0
        hue_mediana = hue_medio = saturacion_media = 0.0

    if len(pix_lab) > 0:
        lab_l_med = float(np.mean(pix_lab[:, 0]))
        lab_a_med = float(np.mean(pix_lab[:, 1]))
        lab_b_med = float(np.mean(pix_lab[:, 2]))
    else:
        lab_l_med, lab_a_med, lab_b_med = 200.0, 128.0, 128.0

    cortes_pintas = contar_cortes_pintas(carta_rect_bgr)

    caracteristicas = {
        "n_componentes": int(n_componentes),
        "hue_mediana": round(hue_mediana, 1),
        "hue_medio": round(hue_medio, 1),
        "saturacion_media": round(saturacion_media, 1),
        "area_max_rel": round(area_max_rel, 4),
        "area_min_rel": round(area_min_rel, 4),
        "area_total_rel": round(area_total_rel, 4),
        "circularidad_media": round(circularidad_media, 4),
        "aspecto_simbolo_medio": round(aspecto_simbolo_medio, 4),
        "solidez_mayor": round(solidez_mayor, 4),
        "ratio_rojo": round(ratio_rojo, 4),
        "ratio_amarillo": round(ratio_amarillo, 4),
        "ratio_verde": round(ratio_verde, 4),
        "ratio_azul": round(ratio_azul, 4),
        "lab_l_medio": round(lab_l_med, 2),
        "lab_a_medio": round(lab_a_med, 2),
        "lab_b_medio": round(lab_b_med, 2),
        "cortes_pintas": int(cortes_pintas),
    }

    return caracteristicas, roi_bgr, mask_simbolos_rellena


# =============================================================================
# ETAPA 6: ÁRBOL DE DECISIÓN EXPLÍCITO Y AUDITABLE (NIVEL 2)
# =============================================================================

def clasificar_carta_arbol(feat):
    """
    Árbol de decisión explícito calibrado experimentalmente sobre las 28 clases
    reales de la baraja española (valores 1 al 7 de Oros, Copas, Espadas y Bastos):
      - Nodo P1: Separa ESPADAS por acero azul dominante o espada con cinta (As).
      - Nodo P2: Separa BASTOS por pigmento verde vegetal con amarillo moderado.
      - Nodo P3: Separa OROS de COPAS por predominio dorado y ausencia de azul celeste.
      - Fase Valor: Conteo topológico N_componentes + distinción As vs 3 de Bastos en 'X'.
    """
    ruta = []
    r_rojo = feat["ratio_rojo"]
    r_amar = feat["ratio_amarillo"]
    r_verde = feat["ratio_verde"]
    r_azul = feat["ratio_azul"]
    n_comp = feat["n_componentes"]
    area_max = feat["area_max_rel"]

    if n_comp == 0:
        return "Desconocido", 0, "Carta no reconocida", ["N_comp == 0 (sin simbolos)"]

    # -------------------------------------------------------------------------
    # FASE A: CLASIFICACIÓN JERÁRQUICA DEL PALO (Nodos P1 -> P2 -> P3)
    # -------------------------------------------------------------------------
    es_as_espadas = (
        r_azul >= cfg.UMBRAL_AZUL_AS_ESPADAS
        and r_verde >= cfg.UMBRAL_VERDE_BASTOS
        and area_max < cfg.UMBRAL_AREA_MAX_AS_ESPADAS
    )

    # Nodo P1: Espadas (acero azul dominante >= 0.25, o As de Espadas con hoja azul y cinta)
    if r_azul >= cfg.UMBRAL_AZUL_ESPADAS_ALTO or es_as_espadas:
        palo = "Espadas"
        ruta.append(
            f"P1: Azul={r_azul:.2f} (>={cfg.UMBRAL_AZUL_ESPADAS_ALTO} o AsEsp={es_as_espadas}) -> ESPADAS"
        )

    # Nodo P2: Bastos (presencia de verde vegetal >= 0.10 y amarillo < 0.52)
    elif r_verde >= cfg.UMBRAL_VERDE_BASTOS and r_amar < cfg.UMBRAL_AMARILLO_MAX_BASTOS:
        palo = "Bastos"
        ruta.append(
            f"P2: Verde={r_verde:.2f}>={cfg.UMBRAL_VERDE_BASTOS} y Amar={r_amar:.2f}<{cfg.UMBRAL_AMARILLO_MAX_BASTOS} -> BASTOS"
        )

    # Nodo P3: Oros vs Copas (Oros: dorado >= 0.59 y sin cenefa azul celeste < 0.055)
    elif r_amar >= cfg.UMBRAL_AMARILLO_OROS and r_azul < cfg.UMBRAL_AZUL_MAX_OROS:
        palo = "Oros"
        ruta.append(
            f"P3: Amar={r_amar:.2f}>={cfg.UMBRAL_AMARILLO_OROS} y Azul={r_azul:.2f}<{cfg.UMBRAL_AZUL_MAX_OROS} -> OROS"
        )
    else:
        palo = "Copas"
        ruta.append(
            f"P3: Rojo={r_rojo:.2f}, Azul_cenefa={r_azul:.2f}>={cfg.UMBRAL_AZUL_MAX_OROS} -> COPAS"
        )

    # -------------------------------------------------------------------------
    # FASE B: CLASIFICACIÓN DEL VALOR NUMÉRICO (1..7)
    # -------------------------------------------------------------------------
    area_min = feat.get("area_min_rel", area_max)
    ratio_max_min = area_max / max(area_min, 1e-4)

    if palo == "Bastos" and n_comp == 1:
        # En el 3 de Bastos los 3 garrotes cruzados en 'X' con lazo azul ocupan mayor área
        # (area_max_rel >= 0.50 y ratio_azul >= 0.05) que el garrote único del As de Bastos.
        if area_max >= cfg.UMBRAL_AREA_3_BASTOS or r_azul >= cfg.UMBRAL_AZUL_LAZO_3_BASTOS:
            valor = 3
            ruta.append(
                f"V_Bastos: N_comp=1 con Area={area_max:.2f}>={cfg.UMBRAL_AREA_3_BASTOS} y LazoAzul={r_azul:.2f} -> Valor 3"
            )
        else:
            valor = 1
            ruta.append(
                f"V_Bastos: N_comp=1 con Area={area_max:.2f}<{cfg.UMBRAL_AREA_3_BASTOS} (Garrote unico) -> Valor 1"
            )
    elif palo in ("Copas", "Oros") and n_comp == 2 and ratio_max_min >= 1.65:
        # Si en el 3 de Copas/Oros dos figuras adyacentes se tocan por 1 px, forman una
        # componente doble (~2x el área de la figura suelta), separándose del 2 real (ratio ~1.0).
        valor = 3
        ruta.append(
            f"V_Contacto: N_comp=2 pero Area_max/Area_min={ratio_max_min:.2f}>=1.65 (2 figuras en contacto + 1) -> Valor 3"
        )
    else:
        valor = int(np.clip(n_comp, cfg.VALOR_MINIMO, cfg.VALOR_MAXIMO))
        ruta.append(f"V_Conteo: N_componentes={n_comp} -> Valor {valor}")

    nombre = f"As (1) de {palo}" if valor == 1 else f"{valor} de {palo}"
    return palo, valor, nombre, ruta


# =============================================================================
# ETAPA 7: INTERPRETACIÓN DE ESCENA DE NIVEL SUPERIOR (NIVEL 3)
# =============================================================================

def interpretar_mano(cartas_clasificadas):
    """
    Nivel 3 — Calcula conteos por palo y valor y propiedades semánticas de la mano:
      1. Suma de puntos y estado en 'Siete y Media'.
      2. Combinaciones que suman exactamente 15 ('La Escoba').
      3. Jugada estructural de la mano (Pareja, Doble Pareja, Trío, Full, Póker, Escalera, Color).
    """
    validas = [c for c in cartas_clasificadas if c["palo"] in cfg.PALOS_VALIDOS and c["valor"] > 0]
    conteo_palos = {p: 0 for p in cfg.PALOS_VALIDOS}
    conteo_valores = {}

    for c in validas:
        conteo_palos[c["palo"]] += 1
        conteo_valores[c["valor"]] = conteo_valores.get(c["valor"], 0) + 1

    valores_lista = sorted([c["valor"] for c in validas])
    suma_puntos = sum(valores_lista)

    if not validas:
        estado_7ymedia = "Sin cartas en mesa"
    elif abs(suma_puntos - 7.5) < 1e-6:
        estado_7ymedia = "SIETE Y MEDIA EXACTA"
    elif suma_puntos == 7:
        estado_7ymedia = f"{suma_puntos} pts (A medio punto de 7.5 - Plantarse)"
    elif suma_puntos < 7.5:
        estado_7ymedia = f"{suma_puntos} pts (Jugada valida < 7.5)"
    else:
        estado_7ymedia = f"{suma_puntos} pts (PASADO > 7.5)"

    escobas_15 = []
    if 1 <= len(validas) <= 12:
        for r in range(2, len(validas) + 1):
            for combo in combinations(validas, r):
                if sum(x["valor"] for x in combo) == 15:
                    etiqueta = " + ".join(f"{x['valor']}{x['palo'][0]}" for x in combo)
                    escobas_15.append(etiqueta)

    frecuencias = sorted(conteo_valores.values(), reverse=True)
    mismo_palo = len(validas) >= 3 and len({c["palo"] for c in validas}) == 1
    es_escalera = (
        len(validas) >= 3
        and len(set(valores_lista)) == len(validas)
        and (max(valores_lista) - min(valores_lista) == len(validas) - 1)
    )

    if not validas:
        jugada = "Ninguna"
    elif mismo_palo and es_escalera:
        jugada = f"ESCALERA DE COLOR ({validas[0]['palo']}, {valores_lista[0]} al {valores_lista[-1]})"
    elif frecuencias[0] >= 4:
        jugada = "POKER (4 iguales)"
    elif frecuencias[0] == 3 and len(frecuencias) > 1 and frecuencias[1] >= 2:
        jugada = "FULL (Trio + Pareja)"
    elif mismo_palo:
        jugada = f"COLOR / MISMO PALO ({validas[0]['palo']})"
    elif es_escalera:
        jugada = f"ESCALERA ({valores_lista[0]} al {valores_lista[-1]})"
    elif frecuencias[0] == 3:
        jugada = "TRIO / MEDIAS (3 iguales)"
    elif len(frecuencias) >= 2 and frecuencias[0] == 2 and frecuencias[1] == 2:
        jugada = "DOBLE PAREJA / DUPLES"
    elif frecuencias[0] == 2:
        jugada = "PAREJA"
    else:
        mayor = max(validas, key=lambda c: c["valor"])
        jugada = f"Carta alta ({mayor['nombre']})"

    return {
        "total_cartas": len(validas),
        "conteo_palos": conteo_palos,
        "conteo_valores": dict(sorted(conteo_valores.items())),
        "suma_puntos": suma_puntos,
        "estado_7ymedia": estado_7ymedia,
        "escobas_15": escobas_15,
        "jugada_mano": jugada,
    }


# =============================================================================
# PIPELINE INTEGRAL Y VISUALIZACIÓN GRÁFICA
# =============================================================================

COLORES_PALO_BGR = {
    "Oros": (0, 215, 255),      # Dorado brillante
    "Copas": (60, 60, 255),     # Rojo coral
    "Espadas": (255, 200, 60),  # Azul celeste
    "Bastos": (70, 230, 90),    # Verde claro
    "Desconocido": (180, 180, 180),
}


def procesar_escena(imagen_bgr):
    """Ejecuta de principio a fin las 7 etapas del pipeline sobre un fotograma BGR."""
    mask_objetos, hsv, porcentaje_tapete = segmentar_tapete(imagen_bgr)
    cartas = detectar_cartas(imagen_bgr, mask_objetos)

    for carta in cartas:
        feat, roi_bgr, mask_simbolos = extraer_caracteristicas_carta(carta["carta_rect"])
        palo, valor, nombre, ruta_arbol = clasificar_carta_arbol(feat)
        carta["caracteristicas"] = feat
        carta["palo"] = palo
        carta["valor"] = valor
        carta["nombre"] = nombre
        carta["ruta_arbol"] = ruta_arbol
        carta["roi_bgr"] = roi_bgr
        carta["mask_simbolos"] = mask_simbolos

    interpretacion = interpretar_mano(cartas)
    return {
        "mask_objetos": mask_objetos,
        "porcentaje_tapete": porcentaje_tapete,
        "cartas": cartas,
        "interpretacion": interpretacion,
    }


def dibujar_resultados(imagen_bgr, resultado):
    """
    Dibuja sobre una copia de la fotografía original al estilo del proyecto de referencia:
      - Contorno exacto de cada carta en verde brillante (0, 255, 0).
      - Línea superior en amarillo (0, 255, 255) con H med y S.
      - Línea inferior en blanco (255, 255, 255) con el nombre de la carta clasificada.
      - Recuadro superior negro con el resumen total en amarillo.
    """
    frame = imagen_bgr.copy()
    h_img, w_img = frame.shape[:2]

    interp = resultado["interpretacion"]
    rect_total = (20, 20, 1180, 110)

    def _solapa(r1, r2):
        return not (r1[2] < r2[0] or r1[0] > r2[2] or r1[3] < r2[1] or r1[1] > r2[3])

    cajas_cartas = [
        (c["bbox"][0], c["bbox"][1], c["bbox"][0] + c["bbox"][2], c["bbox"][1] + c["bbox"][3])
        for c in resultado["cartas"]
    ]

    for idx_c, c in enumerate(resultado["cartas"]):
        cnt = c["contorno"]
        x, y, w_box, h_box = c["bbox"]
        feat = c["caracteristicas"]

        # Dibujar el contorno real exacto de la carta en verde (igual que donde Martín)
        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 6)

        # Alinear la coordenada X con la esquina superior real del contorno cuando la carta está girada
        pts_cnt = cnt.reshape(-1, 2)
        pts_arriba = pts_cnt[pts_cnt[:, 1] <= (y + max(35, int(0.08 * h_box)))]
        x_sup = int(np.min(pts_arriba[:, 0])) if len(pts_arriba) > 0 else x
        x_txt = max(20, min(w_img - 560, x_sup))

        rc_arriba = (x_txt, y - 80, x_txt + 520, y - 5)
        choca_arriba = (
            y < 90
            or _solapa(rc_arriba, rect_total)
            or any(_solapa(rc_arriba, cc) for j, cc in enumerate(cajas_cartas) if j != idx_c)
        )

        if not choca_arriba:
            y_hue = y - 52
            y_clase = y - 12
        else:
            pts_abajo = pts_cnt[pts_cnt[:, 1] >= (y + h_box - max(35, int(0.08 * h_box)))]
            x_inf = int(np.min(pts_abajo[:, 0])) if len(pts_abajo) > 0 else x
            x_txt = max(20, min(w_img - 560, x_inf))
            y_hue = min(h_img - 60, y + h_box + 44)
            y_clase = min(h_img - 15, y + h_box + 86)

        texto_hue = f"H med: {feat['hue_mediana']:.1f} S: {feat['saturacion_media']:.0f}"
        texto_clase = f"{c['nombre']}"

        cv2.putText(
            frame, texto_hue, (x_txt, y_hue),
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6, cv2.LINE_AA
        )
        cv2.putText(
            frame, texto_hue, (x_txt, y_hue),
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3, cv2.LINE_AA
        )
        cv2.putText(
            frame, texto_clase, (x_txt, y_clase),
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6, cv2.LINE_AA
        )
        cv2.putText(
            frame, texto_clase, (x_txt, y_clase),
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA
        )

    # (Línea de resumen superior suprimida para mantener la imagen completamente limpia como en el proyecto de referencia)
    return frame


def crear_mosaico_diagnostico(resultado):
    """
    Construye una imagen resumen mostrando cada carta rectificada (300x460) junto a su
    máscara binaria interna de símbolos cromáticos validados.
    """
    cartas = resultado["cartas"]
    if not cartas:
        panel = np.zeros((260, 640, 3), dtype=np.uint8)
        cv2.putText(
            panel, "Tapete vacio (0 cartas detectadas)", (45, 135),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2, cv2.LINE_AA
        )
        return panel

    tarjetas_vis = []
    for c in cartas[:8]:
        rect_vis = cv2.resize(c["carta_rect"], (180, 276), interpolation=cv2.INTER_AREA)
        mask_bgr = cv2.cvtColor(c["mask_simbolos"], cv2.COLOR_GRAY2BGR)
        mask_vis = cv2.resize(mask_bgr, (180, 276), interpolation=cv2.INTER_NEAREST)

        par = np.hstack((rect_vis, mask_vis))
        banda = np.zeros((58, par.shape[1], 3), dtype=np.uint8)
        color = COLORES_PALO_BGR.get(c["palo"], (255, 255, 255))
        feat = c["caracteristicas"]
        cv2.putText(
            banda, f"#{c['id']} {c['nombre']}", (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA
        )
        cv2.putText(
            banda,
            f"N={feat['n_componentes']} Am={feat['ratio_amarillo']:.2f} R={feat['ratio_rojo']:.2f} V={feat['ratio_verde']:.2f} Az={feat['ratio_azul']:.2f}",
            (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (220, 220, 220), 1, cv2.LINE_AA
        )
        tarjetas_vis.append(np.vstack((banda, par)))

    filas = []
    cols_por_fila = min(4, len(tarjetas_vis))
    ancho_celda = tarjetas_vis[0].shape[1]
    alto_celda = tarjetas_vis[0].shape[0]

    for idx in range(0, len(tarjetas_vis), cols_por_fila):
        bloque = tarjetas_vis[idx:idx + cols_por_fila]
        while len(bloque) < cols_por_fila:
            bloque.append(np.zeros((alto_celda, ancho_celda, 3), dtype=np.uint8))
        filas.append(np.hstack(bloque))

    return np.vstack(filas)
