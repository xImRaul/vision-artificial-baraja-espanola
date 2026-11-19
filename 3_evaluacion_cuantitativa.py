"""
3_evaluacion_cuantitativa.py — Demostración cuantitativa y generación de figuras para la memoria.

Cumple los apartados 7, 8, 9 y 10 del enunciado:
  - Evalúa automáticamente el banco de escenas etiquetadas en 'fotos_prueba/ground_truth.json'.
  - Soporta etiquetas tanto con coordenadas (cx, cy) como listas simples de cartas por foto
    para que puedas etiquetar tus fotos reales en 2 minutos.
  - Calcula precisión en Nivel 1 (Detección), Nivel 2 (Palo, Valor e Identidad completa)
    y Nivel 3 (Conteo y suma de la mano).
  - Genera automáticamente en 'memoria_figuras/':
      * matriz_confusion_palos.png
      * matriz_confusion_valores.png
      * separabilidad_arbol_decision.png (nubes de puntos reales + umbrales del árbol)
      * resumen_cuantitativo.md (tablas listas para incluir en el trabajo escrito)
"""

import argparse
import csv
import json
import os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import config as cfg
from pipeline_baraja import leer_imagen, procesar_escena


def medir_error_angular_marco_rectificado(carta_rect_bgr):
    """
    Mide la desviación angular residual (en grados) de las líneas verticales impresas
    del marco interior ('pintas') tras la rectificación homográfica a 300x460 px.
    """
    gris = cv2.cvtColor(carta_rect_bgr, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gris, 60, 160)
    h, w = gris.shape
    desviaciones = []
    for x1, x2 in ((int(0.03 * w), int(0.15 * w)), (int(0.85 * w), int(0.97 * w))):
        franja = bordes[int(0.12 * h):int(0.88 * h), x1:x2]
        lineas = cv2.HoughLinesP(franja, 1.0, np.pi / 720.0, threshold=45, minLineLength=85, maxLineGap=14)
        if lineas is None or lineas.size == 0:
            continue
        for ln in lineas.reshape(-1, 4):
            dx = float(ln[2] - ln[0])
            dy = float(ln[3] - ln[1])
            ang_vert = abs(np.degrees(np.arctan2(abs(dx), max(abs(dy), 1e-6))))
            if ang_vert <= 8.0:
                desviaciones.append(ang_vert)
    return float(np.median(desviaciones)) if desviaciones else None


def emparejar_predicciones(cartas_pred, cartas_gt):
    """
    Empareja cartas detectadas con las reales (ground truth).
    Si el ground truth incluye 'centroide_px', empareja por distancia espacial mínima;
    si solo incluye {'palo', 'valor'} (formato manual rápido), empareja óptimamente por coincidencia.
    """
    usadas_pred = set()
    pares = []

    con_coords = all("centroide_px" in g for g in cartas_gt) and len(cartas_gt) > 0
    if con_coords:
        for gt in cartas_gt:
            gx, gy = gt["centroide_px"]
            mejor_idx = None
            mejor_dist = 1e9
            for idx, pr in enumerate(cartas_pred):
                if idx in usadas_pred:
                    continue
                px, py = pr["centroide_px"]
                d = float(np.hypot(px - gx, py - gy))
                if d < mejor_dist and d < 220.0:
                    mejor_dist = d
                    mejor_idx = idx
            if mejor_idx is not None:
                usadas_pred.add(mejor_idx)
                pares.append((gt, cartas_pred[mejor_idx]))
            else:
                pares.append((gt, None))
    else:
        # Emparejamiento voraz: primero coincidencias exactas, luego mismo palo, luego resto
        restantes_gt = list(cartas_gt)
        no_emparejadas_gt = []
        for gt in restantes_gt:
            elegido = None
            for idx, pr in enumerate(cartas_pred):
                if idx not in usadas_pred and pr["palo"] == gt["palo"] and pr["valor"] == gt["valor"]:
                    elegido = idx
                    break
            if elegido is None:
                for idx, pr in enumerate(cartas_pred):
                    if idx not in usadas_pred and pr["palo"] == gt["palo"]:
                        elegido = idx
                        break
            if elegido is None:
                for idx, pr in enumerate(cartas_pred):
                    if idx not in usadas_pred:
                        elegido = idx
                        break
            if elegido is not None:
                usadas_pred.add(elegido)
                pares.append((gt, cartas_pred[elegido]))
            else:
                no_emparejadas_gt.append(gt)
        for gt in no_emparejadas_gt:
            pares.append((gt, None))

    falsos_positivos = [pr for idx, pr in enumerate(cartas_pred) if idx not in usadas_pred]
    return pares, falsos_positivos


def graficar_matriz_confusion(matriz, etiquetas, titulo, ruta_salida):
    fig, ax = plt.subplots(figsize=(6.8, 5.6), dpi=160)
    im = ax.imshow(matriz, cmap="Blues")
    ax.set_xticks(range(len(etiquetas)))
    ax.set_yticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas, rotation=25, ha="right", fontsize=10)
    ax.set_yticklabels(etiquetas, fontsize=10)
    ax.set_xlabel("Predicción del Árbol de Decisión", fontsize=11, fontweight="bold")
    ax.set_ylabel("Clase Real (Ground Truth)", fontsize=11, fontweight="bold")
    ax.set_title(titulo, fontsize=12, fontweight="bold", pad=12)

    umbral_color = matriz.max() / 2.0 if matriz.size > 0 else 0
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            ax.text(
                j, i, str(int(matriz[i, j])),
                ha="center", va="center",
                color="white" if matriz[i, j] > umbral_color else "black",
                fontsize=10, fontweight="bold",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ruta_salida)
    plt.close(fig)


def graficar_separabilidad(muestras, ruta_salida):
    """
    Dibuja los planos de decisión experimentales con los umbrales explícitos del árbol,
    justificando visualmente cada nodo de decisión para la memoria escrita.
    """
    if not muestras:
        return

    colores_mpl = {
        "Oros": "#E6A800",
        "Copas": "#D9383A",
        "Espadas": "#2B7BBA",
        "Bastos": "#2E9E49",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), dpi=160)

    # Panel 1: Nodo P1 (Ratio Azul vs Ratio Verde — Separación de Espadas)
    ax1 = axes[0]
    for palo in cfg.PALOS_VALIDOS:
        sub = [m for m in muestras if m["palo_real"] == palo]
        if sub:
            ax1.scatter(
                [m["ratio_verde"] for m in sub],
                [m["ratio_azul"] for m in sub],
                label=palo, c=colores_mpl[palo], s=60, alpha=0.88, edgecolors="k", linewidths=0.5
            )
    ax1.axhline(cfg.UMBRAL_AZUL_ESPADAS_ALTO, color="royalblue", linestyle="--", linewidth=1.6, label=f"Nodo P1 Azul={cfg.UMBRAL_AZUL_ESPADAS_ALTO}")
    ax1.axhline(cfg.UMBRAL_AZUL_AS_ESPADAS, color="navy", linestyle=":", linewidth=1.3, label=f"As Espadas Azul={cfg.UMBRAL_AZUL_AS_ESPADAS}")
    ax1.set_xlabel("Ratio Verde en HSV (H en [36, 88])")
    ax1.set_ylabel("Ratio Azul en HSV (H en [89, 138])")
    ax1.set_title("Nodo P1: Separación de Espadas")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # Panel 2: Nodo P2 (Ratio Verde vs Ratio Amarillo — Separación de Bastos)
    ax2 = axes[1]
    for palo in ("Bastos", "Oros", "Copas"):
        sub = [m for m in muestras if m["palo_real"] == palo]
        if sub:
            ax2.scatter(
                [m["ratio_verde"] for m in sub],
                [m["ratio_amarillo"] for m in sub],
                label=palo, c=colores_mpl[palo], s=60, alpha=0.88, edgecolors="k", linewidths=0.5
            )
    ax2.axvline(cfg.UMBRAL_VERDE_BASTOS, color="forestgreen", linestyle="--", linewidth=1.6, label=f"Nodo P2 Verde={cfg.UMBRAL_VERDE_BASTOS}")
    ax2.axhline(cfg.UMBRAL_AMARILLO_MAX_BASTOS, color="darkgoldenrod", linestyle=":", linewidth=1.6, label=f"Max Amar Bastos={cfg.UMBRAL_AMARILLO_MAX_BASTOS}")
    ax2.set_xlabel("Ratio Verde en HSV")
    ax2.set_ylabel("Ratio Amarillo-Dorado en HSV")
    ax2.set_title("Nodo P2: Separación de Bastos")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    # Panel 3: Nodo P3 (Ratio Amarillo vs Ratio Azul — Separación Oros vs Copas)
    ax3 = axes[2]
    for palo in ("Oros", "Copas"):
        sub = [m for m in muestras if m["palo_real"] == palo]
        if sub:
            ax3.scatter(
                [m["ratio_amarillo"] for m in sub],
                [m["ratio_azul"] for m in sub],
                label=palo, c=colores_mpl[palo], s=62, alpha=0.88, edgecolors="k", linewidths=0.5
            )
    ax3.axvline(cfg.UMBRAL_AMARILLO_OROS, color="darkorange", linestyle="--", linewidth=1.6, label=f"Nodo P3 Amar={cfg.UMBRAL_AMARILLO_OROS}")
    ax3.axhline(cfg.UMBRAL_AZUL_MAX_OROS, color="crimson", linestyle=":", linewidth=1.6, label=f"Nodo P3 Azul={cfg.UMBRAL_AZUL_MAX_OROS}")
    ax3.set_xlabel("Ratio Amarillo-Dorado en HSV")
    ax3.set_ylabel("Ratio Azul Celeste en HSV")
    ax3.set_title("Nodo P3: Separación Oros vs Copas")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    fig.suptitle("Justificación Experimental de Umbrales del Árbol de Decisión", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ruta_salida)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluación cuantitativa del sistema.")
    parser.add_argument(
        "--gt",
        type=str,
        default=os.path.join(cfg.RUTA_BASE, "fotos", "ground_truth.json"),
        help="Ruta al archivo JSON de ground truth.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.gt):
        print(f"No existe '{args.gt}'. Ejecuta primero 4_generar_escenas_demo.py o crea tu ground_truth.json.")
        return

    with open(args.gt, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    carpeta_fotos = os.path.dirname(os.path.abspath(args.gt))
    carpeta_fig = os.path.join(cfg.RUTA_BASE, "memoria_figuras")
    os.makedirs(carpeta_fig, exist_ok=True)

    palos_idx = {p: i for i, p in enumerate(cfg.PALOS_VALIDOS)}
    valores_lista = list(range(cfg.VALOR_MINIMO, cfg.VALOR_MAXIMO + 1))
    valores_idx = {v: i for i, v in enumerate(valores_lista)}

    mat_palos = np.zeros((len(cfg.PALOS_VALIDOS), len(cfg.PALOS_VALIDOS)), dtype=int)
    mat_valores = np.zeros((len(valores_lista), len(valores_lista)), dtype=int)

    total_gt = 0
    tp_det = 0
    fp_det = 0
    fn_det = 0
    aciertos_palo = 0
    aciertos_valor = 0
    aciertos_carta_completa = 0
    escenas_nivel3_ok = 0
    errores_angulo = []
    errores_aspecto_pct = []
    escalas_px_mm = []
    muestras_feat = []
    filas_escenas = []

    escenas = dataset.get("escenas", [])
    for esc in escenas:
        ruta_img = os.path.join(carpeta_fotos, esc["archivo"])
        img = leer_imagen(ruta_img)
        if img is None:
            print(f"[AVISO] No se encontró {ruta_img}, se omite.")
            continue

        res = procesar_escena(img)
        cartas_gt = esc["cartas"]
        pares, fps = emparejar_predicciones(res["cartas"], cartas_gt)

        total_gt += len(cartas_gt)
        fp_det += len(fps)

        aciertos_escena = 0
        for gt, pr in pares:
            if pr is None:
                fn_det += 1
                continue

            tp_det += 1
            if "angulo_360_deg" in gt:
                dif = abs(pr["angulo_360_deg"] - gt["angulo_360_deg"]) % 360.0
                dif = min(dif, 360.0 - dif)
                dif_eje = min(dif, abs(dif - 180.0))
                errores_angulo.append(dif_eje)
            else:
                desv_marco = medir_error_angular_marco_rectificado(pr["carta_rect"])
                if desv_marco is not None:
                    errores_angulo.append(desv_marco)

            err_ar = 100.0 * abs(pr["aspecto_carta"] - cfg.ASPECTO_TEORICO_CARTA) / cfg.ASPECTO_TEORICO_CARTA
            errores_aspecto_pct.append(err_ar)
            escalas_px_mm.append(pr["px_por_mm"])

            p_ok = pr["palo"] == gt["palo"]
            v_ok = pr["valor"] == gt["valor"]
            if p_ok:
                aciertos_palo += 1
            if v_ok:
                aciertos_valor += 1
            if p_ok and v_ok:
                aciertos_carta_completa += 1
                aciertos_escena += 1

            if gt["palo"] in palos_idx and pr["palo"] in palos_idx:
                mat_palos[palos_idx[gt["palo"]], palos_idx[pr["palo"]]] += 1
            if gt["valor"] in valores_idx and pr["valor"] in valores_idx:
                mat_valores[valores_idx[gt["valor"]], valores_idx[pr["valor"]]] += 1

            m = dict(pr["caracteristicas"])
            m["escena"] = esc["archivo"]
            m["palo_real"] = gt["palo"]
            m["valor_real"] = gt["valor"]
            m["palo_pred"] = pr["palo"]
            m["valor_pred"] = pr["valor"]
            m["px_por_mm"] = round(pr["px_por_mm"], 2)
            m["aspecto_carta"] = round(pr["aspecto_carta"], 4)
            m["rectangularidad"] = round(pr["rectangularidad"], 4)
            m["angulo_360_deg"] = round(pr["angulo_360_deg"], 1)
            muestras_feat.append(m)

        suma_real = sum(c["valor"] for c in cartas_gt)
        n3_ok = (
            res["interpretacion"]["total_cartas"] == len(cartas_gt)
            and res["interpretacion"]["suma_puntos"] == suma_real
            and aciertos_escena == len(cartas_gt)
        )
        if n3_ok:
            escenas_nivel3_ok += 1

        filas_escenas.append({
            "archivo": esc["archivo"],
            "n_real": len(cartas_gt),
            "n_det": len(res["cartas"]),
            "aciertos_id": aciertos_escena,
            "suma_real": suma_real,
            "suma_pred": res["interpretacion"]["suma_puntos"],
            "jugada": res["interpretacion"]["jugada_mano"],
            "n3_ok": "SI" if n3_ok else "NO",
        })

    recall_det = 100.0 * tp_det / max(total_gt, 1)
    prec_det = 100.0 * tp_det / max(tp_det + fp_det, 1)
    acc_palo = 100.0 * aciertos_palo / max(tp_det, 1)
    acc_valor = 100.0 * aciertos_valor / max(tp_det, 1)
    acc_total = 100.0 * aciertos_carta_completa / max(total_gt, 1)
    acc_n3 = 100.0 * escenas_nivel3_ok / max(len(filas_escenas), 1)
    err_ang_med = float(np.mean(errores_angulo)) if errores_angulo else 0.0
    err_ar_med = float(np.mean(errores_aspecto_pct)) if errores_aspecto_pct else 0.0
    escala_med = float(np.mean(escalas_px_mm)) if escalas_px_mm else 0.0

    # Exportar CSV con las medidas experimentales completas de las cartas evaluadas
    if muestras_feat:
        ruta_csv = os.path.join(carpeta_fig, "tabla_medidas_cartas.csv")
        campos = [
            "escena", "palo_real", "valor_real", "palo_pred", "valor_pred",
            "n_componentes", "ratio_amarillo", "ratio_rojo", "ratio_verde", "ratio_azul",
            "area_max_rel", "area_min_rel", "circularidad_media", "aspecto_simbolo_medio",
            "px_por_mm", "aspecto_carta", "rectangularidad", "angulo_360_deg",
        ]
        with open(ruta_csv, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=campos, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(muestras_feat)

    # Generar figuras para la memoria
    ruta_mp = os.path.join(carpeta_fig, "matriz_confusion_palos.png")
    ruta_mv = os.path.join(carpeta_fig, "matriz_confusion_valores.png")
    ruta_sep = os.path.join(carpeta_fig, "separabilidad_arbol_decision.png")

    graficar_matriz_confusion(
        mat_palos, list(cfg.PALOS_VALIDOS),
        "Matriz de Confusión — Identificación de Palo", ruta_mp
    )
    graficar_matriz_confusion(
        mat_valores, [str(v) for v in valores_lista],
        "Matriz de Confusión — Identificación de Valor (1-7)", ruta_mv
    )
    graficar_separabilidad(muestras_feat, ruta_sep)

    # Guardar informe Markdown listo para la memoria
    ruta_md = os.path.join(carpeta_fig, "resumen_cuantitativo.md")
    lineas = [
        "# Evaluación Cuantitativa del Sistema de Visión Clásica (Baraja Española)\n",
        "## 1. Métricas Globales por Nivel del Reto\n",
        "| Nivel | Métrica | Resultado | Detalle |",
        "| :--- | :--- | :---: | :--- |",
        f"| **Nivel 1 (Detección)** | Sensibilidad (Recall) | **{recall_det:.2f}%** | {tp_det} / {total_gt} cartas localizadas ({fn_det} FN) |",
        f"| **Nivel 1 (Detección)** | Precisión (Precision) | **{prec_det:.2f}%** | {fp_det} falsos positivos (incl. tapete vacío) |",
        f"| **Nivel 1 (Geometría)** | Desviación angular residual tras homografía | **{err_ang_med:.2f}°** | Medida sobre el marco impreso rectificado ({escala_med:.2f} px/mm) |",
        f"| **Nivel 1 (Geometría)** | Error relativo medio de aspecto físico | **{err_ar_med:.2f}%** | Respecto a proporción teórica 95.0 / 61.5 = 1.545 |",
        f"| **Nivel 2 (Palo)** | Exactitud de Palo | **{acc_palo:.2f}%** | {aciertos_palo} / {tp_det} cartas |",
        f"| **Nivel 2 (Valor)** | Exactitud de Valor (1–7) | **{acc_valor:.2f}%** | {aciertos_valor} / {tp_det} cartas |",
        f"| **Nivel 2 (Global)** | Identificación exacta (Palo + Valor) | **{acc_total:.2f}%** | {aciertos_carta_completa} / {total_gt} sobre el total real |",
        f"| **Nivel 3 (Escena)** | Escenas 100% bien interpretadas | **{acc_n3:.2f}%** | {escenas_nivel3_ok} / {len(filas_escenas)} manos completas |\n",
        "## 2. Desglose por Escena de Prueba\n",
        "| Escena | Cartas Reales | Detectadas | Bien Clasificadas | Suma Real | Suma Pred. | Jugada Detectada | Nivel 3 OK |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |",
    ]
    for r in filas_escenas:
        lineas.append(
            f"| `{r['archivo']}` | {r['n_real']} | {r['n_det']} | {r['aciertos_id']}/{r['n_real']} | "
            f"{r['suma_real']} | {r['suma_pred']} | {r['jugada']} | **{r['n3_ok']}** |"
        )

    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")

    print("\n" + "=" * 72)
    print("RESULTADOS CUANTITATIVOS GLOBALES")
    print("=" * 72)
    print(f"  Escenas evaluadas                 : {len(filas_escenas)}")
    print(f"  Cartas totales (Ground Truth)     : {total_gt}")
    print(f"  Escala espacial media             : {escala_med:.2f} px/mm")
    print(f"  Nivel 1 - Recall / Precisión      : {recall_det:.2f}% / {prec_det:.2f}%")
    print(f"  Nivel 1 - Desviación angular / AR : {err_ang_med:.2f}° / {err_ar_med:.2f}%")
    print(f"  Nivel 2 - Acierto Palo            : {acc_palo:.2f}%")
    print(f"  Nivel 2 - Acierto Valor (1-7)     : {acc_valor:.2f}%")
    print(f"  Nivel 2 - Identidad Exacta Global : {acc_total:.2f}%")
    print(f"  Nivel 3 - Escenas Perfectas       : {acc_n3:.2f}%")
    print("-" * 72)
    print(f"  Informe Markdown : {ruta_md}")
    print(f"  Figuras guardadas: {carpeta_fig}")
    print("=" * 72)


if __name__ == "__main__":
    main()
