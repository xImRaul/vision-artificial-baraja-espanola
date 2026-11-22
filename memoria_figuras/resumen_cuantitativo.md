# Evaluación Cuantitativa del Sistema de Visión Clásica (Baraja Española)

## 1. Métricas Globales por Nivel del Reto

| Nivel | Métrica | Resultado | Detalle |
| :--- | :--- | :---: | :--- |
| **Nivel 1 (Detección)** | Sensibilidad (Recall) | **100.00%** | 47 / 47 cartas localizadas (0 FN) |
| **Nivel 1 (Detección)** | Precisión (Precision) | **100.00%** | 0 falsos positivos (incl. tapete vacío) |
| **Nivel 1 (Geometría)** | Desviación angular residual tras homografía | **0.02°** | Medida sobre el marco impreso rectificado (14.38 px/mm) |
| **Nivel 1 (Geometría)** | Error relativo medio de aspecto físico | **1.23%** | Respecto a proporción teórica 95.0 / 61.5 = 1.545 |
| **Nivel 2 (Palo)** | Exactitud de Palo | **100.00%** | 47 / 47 cartas |
| **Nivel 2 (Valor)** | Exactitud de Valor (1–7) | **100.00%** | 47 / 47 cartas |
| **Nivel 2 (Global)** | Identificación exacta (Palo + Valor) | **100.00%** | 47 / 47 sobre el total real |
| **Nivel 3 (Escena)** | Escenas 100% bien interpretadas | **100.00%** | 8 / 8 manos completas |

## 2. Desglose por Escena de Prueba

| Escena | Cartas Reales | Detectadas | Bien Clasificadas | Suma Real | Suma Pred. | Jugada Detectada | Nivel 3 OK |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| `IMG_7880.jpeg` | 7 | 7 | 7/7 | 28 | 28 | ESCALERA DE COLOR (Oros, 1 al 7) | **SI** |
| `IMG_7881.jpeg` | 7 | 7 | 7/7 | 28 | 28 | ESCALERA DE COLOR (Espadas, 1 al 7) | **SI** |
| `IMG_7882.jpeg` | 7 | 7 | 7/7 | 28 | 28 | ESCALERA DE COLOR (Bastos, 1 al 7) | **SI** |
| `IMG_7885.jpeg` | 0 | 0 | 0/0 | 0 | 0 | Ninguna | **SI** |
| `IMG_7886.jpeg` | 7 | 7 | 7/7 | 28 | 28 | ESCALERA DE COLOR (Copas, 1 al 7) | **SI** |
| `IMG_7888.jpeg` | 7 | 7 | 7/7 | 26 | 26 | DOBLE PAREJA / DUPLES | **SI** |
| `IMG_7889.jpeg` | 6 | 6 | 6/6 | 21 | 21 | DOBLE PAREJA / DUPLES | **SI** |
| `IMG_7890.jpeg` | 6 | 6 | 6/6 | 22 | 22 | PAREJA | **SI** |
