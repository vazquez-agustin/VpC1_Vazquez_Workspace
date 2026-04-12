# TP3

## Estructura del repositorio

```
Trabajo_Practico_3/
├── tp3_punto1.py                  # Script para el punto 1
├── tp3_punto2.py                  # Script para el punto 2
├── tp3_punto1_reporte.md          # Reporte del punto 1
├── README.md                      # Este archivo
├── images/                        # Imágenes a procesar
│   ├── COCA-COLA-LOGO.jpg
│   ├── coca_logo_1.png
│   ├── coca_logo_2.png
│   ├── coca_multi.png
│   ├── coca_retro_1.png
│   ├── coca_retro_2.png
│   └── logo_1.png
├── template/                      # Template del logotipo
│   └── pattern.png
├── output/                        # Resultados de las detecciones
│   ├── tp3_punto1_resultado.png
│   └── tp3_punto2_resultado.png
```

## Objetivo
Encontrar el logotipo de la gaseosa dentro de las imágenes provistas en `images/` a partir del template `template/pattern.png`.

Visualizar los resultados con bounding boxes en cada imagen mostrando el nivel de confianza de la detección.

### Ejercicios

1. **(4 puntos)** Obtener una detección del logo en cada imagen sin falsos positivos.
2. **(4 puntos)** Plantear y validar un algoritmo para múltiples detecciones en la imagen `coca_multi.png` con el mismo template del ítem 1.
3. **(2 puntos)** Generalizar el algoritmo del ítem 2 para todas las imágenes.