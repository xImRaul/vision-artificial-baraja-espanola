"""
0_calibrar_tapete.py — Calibración experimental del tapete verde en espacio HSV.

Selecciona una fotografía donde aparezca únicamente el tapete verde vacío bajo la
iluminación de trabajo, mide los percentiles 1 y 99 de H, S y V, añade un margen de
seguridad frente a sombras/compresión JPEG y guarda 'calibracion_tapete.json'.
"""

import argparse
import json
import os
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import config as cfg
from pipeline_baraja import leer_imagen, redimensionar_vista

PERCENTIL_INFERIOR = 1.0
PERCENTIL_SUPERIOR = 99.0
MARGEN_H = 4
MARGEN_S = 18
MARGEN_V = 20


def seleccionar_imagen_dialogo():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    carpeta_inicial = os.path.join(cfg.RUTA_BASE, "fotos")
    if not os.path.isdir(carpeta_inicial):
        carpeta_inicial = cfg.RUTA_BASE
    ruta = filedialog.askopenfilename(
        initialdir=carpeta_inicial,
        title="Selecciona una foto que contenga SOLO el tapete verde vacío",
        filetypes=[
            ("JPEG", "*.jpg *.jpeg"),
            ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("Todos los archivos", "*.*"),
        ],
    )
    try:
        root.destroy()
    except tk.TclError:
        pass
    return ruta


def calcular_limites_tapete(hsv):
    canales = ("H", "S", "V")
    margenes = (MARGEN_H, MARGEN_S, MARGEN_V)
    maximos = (179, 255, 255)
    inferiores, superiores = [], []
    estadisticas = {}

    for idx, (nombre, margen, lim_max) in enumerate(zip(canales, margenes, maximos)):
        canal = hsv[:, :, idx]
        p_inf, p_sup = np.percentile(canal, [PERCENTIL_INFERIOR, PERCENTIL_SUPERIOR])
        inf = max(0, int(np.floor(p_inf)) - margen)
        sup = min(lim_max, int(np.ceil(p_sup)) + margen)
        inferiores.append(inf)
        superiores.append(sup)
        estadisticas[nombre] = {
            "min": int(canal.min()),
            "max": int(canal.max()),
            "mediana": float(np.median(canal)),
            "p_inf": float(p_inf),
            "p_sup": float(p_sup),
        }

    return (
        np.array(inferiores, dtype=np.uint8),
        np.array(superiores, dtype=np.uint8),
        estadisticas,
    )


def main():
    parser = argparse.ArgumentParser(description="Calibrar límites HSV del tapete verde.")
    parser.add_argument("--imagen", type=str, default="", help="Ruta de la foto del tapete.")
    parser.add_argument("--sin-ventana", action="store_true", help="No abrir ventana gráfica.")
    args = parser.parse_args()

    ruta = args.imagen or seleccionar_imagen_dialogo()
    if not ruta:
        print("No se seleccionó ninguna imagen.")
        return

    imagen = leer_imagen(ruta)
    if imagen is None:
        print(f"Error: no se pudo abrir la imagen '{ruta}'.")
        return

    suavizada = cv2.GaussianBlur(imagen, (5, 5), 0)
    hsv = cv2.cvtColor(suavizada, cv2.COLOR_BGR2HSV)
    lim_inf, lim_sup, stats = calcular_limites_tapete(hsv)
    mascara = cv2.inRange(hsv, lim_inf, lim_sup)
    cobertura = 100.0 * cv2.countNonZero(mascara) / max(mascara.size, 1)

    alto, ancho = imagen.shape[:2]
    print("\n" + "=" * 65)
    print("CALIBRACIÓN EXPERIMENTAL DEL TAPETE VERDE (ESPACIO HSV)")
    print("=" * 65)
    print(f"Imagen analizada : {ruta}")
    print(f"Resolución       : {ancho} x {alto} px")
    print(f"\nEstadísticas medidas por canal (Percentiles {PERCENTIL_INFERIOR:.0f} - {PERCENTIL_SUPERIOR:.0f}):")
    for ch, d in stats.items():
        print(
            f"  Canal {ch}: min={d['min']:3d}, mediana={d['mediana']:5.1f}, "
            f"P1..P99=[{d['p_inf']:5.1f}, {d['p_sup']:5.1f}], max={d['max']:3d}"
        )

    print("\nLímites HSV calculados:")
    print(f"  VERDE_BAJO = np.array({lim_inf.tolist()}, dtype=np.uint8)")
    print(f"  VERDE_ALTO = np.array({lim_sup.tolist()}, dtype=np.uint8)")
    print(f"  Cobertura del tapete reconocida: {cobertura:.2f}%")

    salida_json = {
        "imagen_origen": os.path.abspath(ruta),
        "resolucion": [ancho, alto],
        "verde_bajo": lim_inf.tolist(),
        "verde_alto": lim_sup.tolist(),
        "cobertura_pct": round(cobertura, 2),
        "estadisticas": stats,
    }
    with open(cfg.RUTA_CALIBRACION_TAPETE, "w", encoding="utf-8") as f:
        json.dump(salida_json, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado automáticamente en: {cfg.RUTA_CALIBRACION_TAPETE}")

    if not args.sin_ventana:
        img_v, _ = redimensionar_vista(imagen, 750, 580)
        msk_v = cv2.resize(mascara, (img_v.shape[1], img_v.shape[0]), interpolation=cv2.INTER_NEAREST)
        msk_bgr = cv2.cvtColor(msk_v, cv2.COLOR_GRAY2BGR)
        comp = np.hstack((img_v, msk_bgr))
        cv2.putText(comp, "Tapete original", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        cv2.putText(
            comp, f"Mascara HSV ({cobertura:.1f}%)", (img_v.shape[1] + 15, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2
        )
        cv2.imshow("0 - Calibracion del Tapete Verde (Pulsa 'q' o ESC)", comp)
        while True:
            if (cv2.waitKey(20) & 0xFF) in (ord("q"), 27):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
