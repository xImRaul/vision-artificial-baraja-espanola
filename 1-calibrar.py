"""
1_calibrar_cartas.py — Herramienta interactiva de calibración y extracción de métricas.

Inspirado en 1-calibrar.py del proyecto de referencia:
  - Detecta todas las cartas presentes en una fotografía cenital.
  - Al hacer CLIC IZQUIERDO sobre cualquier carta muestra en consola todos sus
    descriptores geométricos, topológicos y cromáticos, la ruta tomada en el Árbol
    de Decisión y abre una ventana detalle con la carta rectificada y su máscara interna.
  - Si se hace clic en el fondo, muestra el valor HSV/Lab del píxel del tapete.
  - Exporta 'calibracion_cartas.json' y 'calibracion_cartas.csv' para justificar los
    umbrales del Árbol de Decisión en la memoria escrita.
"""

import argparse
import csv
import json
import os
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import config as cfg
from pipeline_baraja import (
    dibujar_resultados,
    leer_imagen,
    procesar_escena,
    redimensionar_vista,
)

punto_clic = None


def seleccionar_imagen_dialogo():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    carpeta_inicial = os.path.join(cfg.RUTA_BASE, "fotos")
    if not os.path.isdir(carpeta_inicial):
        carpeta_inicial = cfg.RUTA_BASE
    ruta = filedialog.askopenfilename(
        initialdir=carpeta_inicial,
        title="Selecciona una fotografía con cartas para calibrar",
        filetypes=[
            ("JPEG", "*.jpg *.jpeg"),
            ("Imágenes", "*.jpg *.jpeg *.png *.bmp"),
            ("Todos los archivos", "*.*"),
        ],
    )
    try:
        root.destroy()
    except tk.TclError:
        pass
    return ruta


def callback_raton(evento, x, y, flags, param):
    global punto_clic
    if evento == cv2.EVENT_LBUTTONDOWN:
        punto_clic = (x, y)


def exportar_datos_calibracion(ruta_imagen, tam_img, resultado):
    registros = []
    for c in resultado["cartas"]:
        fila = {
            "id": c["id"],
            "prediccion_palo": c["palo"],
            "prediccion_valor": c["valor"],
            "nombre": c["nombre"],
            "pos_x_cm": round(c["posicion_cm"][0], 2),
            "pos_y_cm": round(c["posicion_cm"][1], 2),
            "px_por_mm": round(c["px_por_mm"], 3),
            "angulo_360_deg": round(c["angulo_360_deg"], 1),
            "aspecto_carta": round(c["aspecto_carta"], 3),
            "rectangularidad": round(c["rectangularidad"], 3),
        }
        fila.update(c["caracteristicas"])
        registros.append(fila)

    ruta_json = os.path.join(cfg.RUTA_BASE, "calibracion_cartas.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ruta_imagen": os.path.abspath(ruta_imagen),
                "resolucion": list(tam_img),
                "porcentaje_tapete": round(resultado["porcentaje_tapete"], 2),
                "num_cartas": len(registros),
                "cartas": registros,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    ruta_csv = os.path.join(cfg.RUTA_BASE, "calibracion_cartas.csv")
    if registros:
        with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(registros[0].keys()))
            writer.writeheader()
            writer.writerows(registros)

    return ruta_json, ruta_csv


def main():
    global punto_clic
    parser = argparse.ArgumentParser(description="Calibración interactiva de cartas por clic.")
    parser.add_argument("--imagen", type=str, default="", help="Ruta de la imagen a calibrar.")
    parser.add_argument("--sin-ventana", action="store_true", help="Solo exportar JSON/CSV y salir.")
    args = parser.parse_args()

    ruta = args.imagen or seleccionar_imagen_dialogo()
    if not ruta:
        print("No se seleccionó ninguna imagen.")
        return

    imagen = leer_imagen(ruta)
    if imagen is None:
        print(f"Error: no se pudo cargar '{ruta}'.")
        return

    h_img, w_img = imagen.shape[:2]
    resultado = procesar_escena(imagen)
    hsv_full = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    print("\n" + "=" * 70)
    print("CALIBRACIÓN EXPERIMENTAL POR CARTA (INSPECCIÓN DE CARACTERÍSTICAS)")
    print("=" * 70)
    print(f"Imagen           : {ruta}")
    print(f"Resolución       : {w_img} x {h_img} px")
    print(f"Tapete detectado : {resultado['porcentaje_tapete']:.1f}% del encuadre")
    print(f"Cartas halladas  : {len(resultado['cartas'])}")
    if resultado["cartas"]:
        escala_media = float(np.mean([c["px_por_mm"] for c in resultado["cartas"]]))
        print(f"Escala espacial  : {escala_media:.2f} px/mm (medida experimentalmente sobre las cartas)")

    ruta_json, ruta_csv = exportar_datos_calibracion(ruta, (w_img, h_img), resultado)
    print(f"Datos guardados  : {ruta_json}")
    print(f"Tabla CSV        : {ruta_csv}")

    if args.sin_ventana:
        for c in resultado["cartas"]:
            f = c["caracteristicas"]
            print(
                f"  [#{c['id']}] {c['nombre']:16s} | N_comp={f['n_componentes']} "
                f"AR={f['aspecto_simbolo_medio']:.2f} Circ={f['circularidad_media']:.2f} "
                f"Sol={f['solidez_mayor']:.2f} | R={f['ratio_rojo']:.2f} "
                f"Am={f['ratio_amarillo']:.2f} V={f['ratio_verde']:.2f} Az={f['ratio_azul']:.2f}"
            )
        return

    anotada = dibujar_resultados(imagen, resultado)
    vista, factor_escala = redimensionar_vista(anotada)

    titulo_win = "1 - Calibracion Interactiva (Haz CLIC en cualquier carta | 'q' para salir)"
    cv2.namedWindow(titulo_win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(titulo_win, callback_raton)

    print("\nINSTRUCCIONES:")
    print("  - Haz CLIC IZQUIERDO sobre cualquier carta para inspeccionar todos sus parámetros.")
    print("  - Haz CLIC en el tapete para consultar el valor HSV exacto del fondo.")
    print("  - Pulsa 'q' o ESC para cerrar.")
    print("=" * 70)

    while True:
        cv2.imshow(titulo_win, vista)
        if punto_clic is not None:
            vx, vy = punto_clic
            ox = min(w_img - 1, max(0, int(vx / factor_escala)))
            oy = min(h_img - 1, max(0, int(vy / factor_escala)))
            px_hsv = hsv_full[oy, ox]

            encontrada = None
            for c in resultado["cartas"]:
                if cv2.pointPolygonTest(c["contorno"], (float(ox), float(oy)), False) >= 0:
                    encontrada = c
                    break

            if encontrada is not None:
                f = encontrada["caracteristicas"]
                print(f"\n--- CARTA #{encontrada['id']}: {encontrada['nombre']} ---")
                print(f"  Posición centroide : ({encontrada['centroide_px'][0]:.0f}, {encontrada['centroide_px'][1]:.0f}) px  |  ({encontrada['posicion_cm'][0]:.1f}, {encontrada['posicion_cm'][1]:.1f}) cm")
                print(f"  Orientación        : {encontrada['angulo_360_deg']:.1f}° (eje {encontrada['angulo_eje_deg']:.1f}°)")
                print(f"  Resolución local   : {encontrada['px_por_mm']:.2f} px/mm  |  Rectangularidad: {encontrada['rectangularidad']:.3f}")
                print(f"  Topología interna  : N_componentes={f['n_componentes']}, Area_max_rel={f['area_max_rel']:.4f}, Area_total_rel={f['area_total_rel']:.4f}")
                print(f"  Morfología símbolo : Aspecto_medio={f['aspecto_simbolo_medio']:.3f}, Circularidad={f['circularidad_media']:.3f}, Solidez_mayor={f['solidez_mayor']:.3f}")
                print(f"  Color HSV símbolos : Rojo={f['ratio_rojo']:.3f}, Amarillo={f['ratio_amarillo']:.3f}, Verde={f['ratio_verde']:.3f}, Azul={f['ratio_azul']:.3f}")
                print(f"  Color CIE Lab med  : L={f['lab_l_medio']:.1f}, a={f['lab_a_medio']:.1f}, b={f['lab_b_medio']:.1f}")
                print(f"  Cortes del marco   : {f['cortes_pintas']} pintas detectadas")
                print("  Ruta Árbol Decisión:")
                for paso in encontrada["ruta_arbol"]:
                    print(f"    -> {paso}")

                # Mostrar detalle de la carta rectificada + máscara de símbolos
                mask_vis = cv2.cvtColor(encontrada["mask_simbolos"], cv2.COLOR_GRAY2BGR)
                mask_vis_resized = cv2.resize(
                    mask_vis,
                    (encontrada["carta_rect"].shape[1], encontrada["carta_rect"].shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                detalle = np.hstack((encontrada["carta_rect"], mask_vis_resized))
                cv2.imshow("Detalle Carta Rectificada vs Mascara Interna", detalle)
            else:
                print(
                    f"\n[CLIC EN TAPETE ({ox}, {oy})] -> HSV = "
                    f"(H={int(px_hsv[0])}, S={int(px_hsv[1])}, V={int(px_hsv[2])})"
                )

            punto_clic = None

        tecla = cv2.waitKey(20) & 0xFF
        if tecla in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
