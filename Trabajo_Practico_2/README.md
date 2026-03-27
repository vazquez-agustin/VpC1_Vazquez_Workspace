
# TP2

## Estructura del repositorio

```
Trabajo_Practico_2/
├── TP2_VisionComputadora.ipynb   # Notebook principal
├── README.md                     # Este archivo
├── focus_video.mov               # Video a procesar
├── focus_matrix/                 # Resultados de la matriz de enfoque
│   └── ...
├── focus_curves/                 # Resultados de las curvas de enfoque
│   └── ...
```

## Objetivo
Implementar un detector de máximo enfoque sobre un video aplicando técnicas de análisis espectral similar a las que utilizan las cámaras digitales modernas. El video a procesar será: `focus_video.mov`.

### Experimentos a realizar
1. Medición sobre todo el frame.
2. Medición sobre una ROI ubicada en el centro del frame. Área de la ROI = 5 o 10% del área total del frame.

Para cada experimento se debe presentar:
- Una curva o varias curvas que muestren la evolución de la métrica frame a frame donde se vea claramente cuando el algoritmo detectó el punto de máximo enfoque.

El algoritmo de detección a implementar debe detectar y devolver los puntos de máximo enfoque de manera automática.

## Ejemplo visual
- Matriz de enfoque superpuesta a uno de los frames del video
- Curva de la métrica de enfoque frame a frame

## Puntos extra
Aplicar unsharp masking para expandir la zona de enfoque y devolver.
