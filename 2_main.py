"""
2_main.py — Programa principal del Sistema Experto de Visión Artificial Clásica
aplicado a la Baraja Española (Reto 07 - Examen Parcial).

Modos de ejecución soportados:
  1. Fotografía individual (seleccionada por ventana o --imagen):
       Guarda '*_resultado.jpg' y '*_diagnostico.jpg' y muestra el análisis completo.
  2. Cámara en tiempo real (--camara, ideal para el Examen Parcial en vivo):
       Analiza vídeo en directo desde webcam o smartphone (DroidCam/Camo/USB).
       Teclas: [ESPACIO] guardar captura analizada, [d] ver mosaico diagnóstico, [q] salir.
  3. Procesamiento por lotes de una carpeta completa (--carpeta).
"""

import argparse
import glob
import os
import time
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import config as cfg
from pipeline_baraja import (
    crear_mosaico_diagnostico,
    dibujar_resultados,
    guardar_imagen,
    leer_imagen,
    procesar_escena,
    redimensionar_vista,
)


def seleccionar_imagen_dialogo():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    carpeta_inicial = os.path.join(cfg.RUTA_BASE, "fotos")
    if not os.path.isdir(carpeta_inicial):
        carpeta_inicial = cfg.RUTA_BASE
    ruta = filedialog.askopenfilename(
        initialdir=carpeta_inicial,
        title="Selecciona la foto a procesar",
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


def seleccionar_carpeta_dialogo():
    root = tk.Tk()
    root.withdraw()
    ruta = filedialog.askdirectory(title="Selecciona la carpeta con fotografías")
    try:
        root.destroy()
    except tk.TclError:
        pass
    return ruta


def imprimir_informe_consola(origen, tam_img, resultado):
    w_img, h_img = tam_img
    interp = resultado["interpretacion"]
    print("\n" + "=" * 74)
    print(f"ESCENA ANALIZADA : {origen} ({w_img}x{h_img} px)")
    print(f"COBERTURA TAPETE : {resultado['porcentaje_tapete']:.1f}%")
    if resultado["cartas"]:
        escala = float(np.mean([c["px_por_mm"] for c in resultado["cartas"]]))
        fov_x_cm = (w_img / max(escala, 1e-6)) / 10.0
        fov_y_cm = (h_img / max(escala, 1e-6)) / 10.0
        print(
            f"ESCALA ESPACIAL  : {escala:.2f} px/mm  "
            f"(Campo de visión estimado: {fov_x_cm:.1f} x {fov_y_cm:.1f} cm)"
        )
    print("-" * 74)
    print("NIVEL 1 Y 2 — DETECCIÓN E IDENTIFICACIÓN DE NAIPES:")
    if not resultado["cartas"]:
        print("  (No se detectaron cartas válidas sobre el tapete)")
    for c in resultado["cartas"]:
        print(
            f"  #{c['id']:02d} | {c['nombre']:17s} | "
            f"Pos: ({c['posicion_cm'][0]:5.1f}, {c['posicion_cm'][1]:5.1f}) cm | "
            f"Orientación: {c['angulo_360_deg']:5.1f}° | "
            f"{' ; '.join(c['ruta_arbol'])}"
        )

    print("-" * 74)
    print("NIVEL 3 — INTERPRETACIÓN DE LA ESCENA:")
    cp = interp["conteo_palos"]
    print(
        f"  Total cartas     : {interp['total_cartas']}  "
        f"(Oros: {cp['Oros']}, Copas: {cp['Copas']}, Espadas: {cp['Espadas']}, Bastos: {cp['Bastos']})"
    )
    print(f"  Conteo valores   : {interp['conteo_valores']}")
    print(f"  Suma de puntos   : {interp['suma_puntos']}")
    print(f"  Jugada de mano   : {interp['jugada_mano']}")
    print(f"  Siete y Media    : {interp['estado_7ymedia']}")
    escobas = ", ".join(interp["escobas_15"]) if interp["escobas_15"] else "Ninguna combinación suma 15"
    print(f"  La Escoba (15)   : {escobas}")
    print("=" * 74)


def analizar_y_guardar_foto(ruta_imagen, mostrar_ventana=True, carpeta_salida=""):
    frame = leer_imagen(ruta_imagen)
    if frame is None:
        print(f"Error: no se pudo cargar la imagen '{ruta_imagen}'.")
        return None

    h_img, w_img = frame.shape[:2]
    resultado = procesar_escena(frame)
    imprimir_informe_consola(ruta_imagen, (w_img, h_img), resultado)

    anotada = dibujar_resultados(frame, resultado)
    diagnostico = crear_mosaico_diagnostico(resultado)

    nombre_base, ext = os.path.splitext(os.path.basename(ruta_imagen))
    if not ext:
        ext = ".jpg"
    dir_destino = carpeta_salida if carpeta_salida else os.path.dirname(ruta_imagen)
    os.makedirs(dir_destino, exist_ok=True)

    ruta_res = os.path.join(dir_destino, f"{nombre_base}_resultado{ext}")
    guardar_imagen(ruta_res, anotada)
    print(f"Imagen resultado guardada en: {ruta_res}")
    if carpeta_salida:
        ruta_diag = os.path.join(dir_destino, f"{nombre_base}_diagnostico{ext}")
        guardar_imagen(ruta_diag, diagnostico)

    if mostrar_ventana:
        vista_res, _ = redimensionar_vista(anotada)
        vista_diag, _ = redimensionar_vista(diagnostico, 1400, 750)
        cv2.imshow("Sistema Experto - Baraja Espanola (Pulsa 'd' diagnostico | 'q' salir)", vista_res)
        mostrando_diag = False
        while True:
            tecla = cv2.waitKey(20) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if tecla == ord("d"):
                mostrando_diag = not mostrando_diag
                if mostrando_diag:
                    cv2.imshow("Mosaico Diagnostico (Cartas Rectificadas + Mascaras)", vista_diag)
                else:
                    cv2.destroyWindow("Mosaico Diagnostico (Cartas Rectificadas + Mascaras)")
        cv2.destroyAllWindows()

    return resultado


def ejecutar_modo_camara(indice_camara=0):
    print(f"\nAbriendo cámara índice {indice_camara} (Full HD solicitado)...")
    cap = cv2.VideoCapture(indice_camara)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir la cámara con índice {indice_camara}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    carpeta_capturas = os.path.join(cfg.RUTA_BASE, "resultados")
    os.makedirs(carpeta_capturas, exist_ok=True)

    ver_diagnostico = False
    print("\nMODO TIEMPO REAL ACTIVO:")
    print("  [ESPACIO] -> Congelar/Guardar captura e imprimir informe detallado")
    print("  [d]       -> Mostrar/Ocultar mosaico de cartas rectificadas y máscaras")
    print("  [q / ESC] -> Salir")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        resultado = procesar_escena(frame)
        anotada = dibujar_resultados(frame, resultado)
        vista, _ = redimensionar_vista(anotada, 1500, 950)
        cv2.imshow("Examen en Vivo - Baraja Espanola [ESPACIO=Guardar | d=Diagnostico | q=Salir]", vista)

        if ver_diagnostico:
            diag = crear_mosaico_diagnostico(resultado)
            diag_v, _ = redimensionar_vista(diag, 1200, 700)
            cv2.imshow("Diagnostico en Tiempo Real", diag_v)

        tecla = cv2.waitKey(15) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord("d"):
            ver_diagnostico = not ver_diagnostico
            if not ver_diagnostico:
                cv2.destroyWindow("Diagnostico en Tiempo Real")
        elif tecla == 32:  # Barra espaciadora
            marca = time.strftime("%Y%m%d_%H%M%S")
            ruta_cap = os.path.join(carpeta_capturas, f"captura_{marca}.jpg")
            guardar_imagen(ruta_cap, frame)
            analizar_y_guardar_foto(ruta_cap, mostrar_ventana=False, carpeta_salida=carpeta_capturas)

    cap.release()
    cv2.destroyAllWindows()


def procesar_carpeta(carpeta, carpeta_salida=""):
    patrones = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    archivos = []
    for p in patrones:
        archivos.extend(glob.glob(os.path.join(carpeta, p)))
    archivos = sorted(
        set(
            f for f in archivos
            if "_resultado." not in os.path.basename(f).lower()
            and "_diagnostico." not in os.path.basename(f).lower()
        )
    )
    if not archivos:
        print(f"No se encontraron imágenes de entrada en '{carpeta}'.")
        return

    destino = carpeta_salida or os.path.join(cfg.RUTA_BASE, "resultados")
    for ruta in archivos:
        analizar_y_guardar_foto(ruta, mostrar_ventana=False, carpeta_salida=destino)
    print(f"\nLote completado: {len(archivos)} escenas procesadas en '{destino}'.")


def main():
    parser = argparse.ArgumentParser(description="Sistema de Visión Clásica — Baraja Española.")
    parser.add_argument("--imagen", type=str, default="", help="Procesar una foto concreta.")
    parser.add_argument("--carpeta", type=str, default="", help="Procesar todas las fotos de una carpeta.")
    parser.add_argument("--salida", type=str, default="", help="Carpeta donde guardar resultados.")
    parser.add_argument("--camara", type=int, default=-1, help="Índice de cámara para modo en vivo (ej. 0).")
    parser.add_argument("--sin-ventana", action="store_true", help="No mostrar ventanas OpenCV.")
    args = parser.parse_args()

    if args.imagen:
        analizar_y_guardar_foto(
            args.imagen, mostrar_ventana=not args.sin_ventana, carpeta_salida=args.salida
        )
        return

    if args.carpeta:
        procesar_carpeta(args.carpeta, carpeta_salida=args.salida)
        return

    if args.camara >= 0:
        ejecutar_modo_camara(args.camara)
        return

    # Igual que el programa de Martín: al darle a Play abre directamente el selector de foto
    print("=" * 60)
    print("SISTEMA DE VISIÓN CLÁSICA - PROCESAMIENTO POR FOTO")
    print("=" * 60)
    ruta = seleccionar_imagen_dialogo()
    if not ruta:
        print("No se seleccionó ninguna imagen. Saliendo...")
        return
    analizar_y_guardar_foto(ruta, mostrar_ventana=not args.sin_ventana)


if __name__ == "__main__":
    main()
