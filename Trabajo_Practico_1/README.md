# TP1
#
## Estructura del repositorio

```
Trabajo_Practico_1/
├── TP1_VisionCommputadora.ipynb   # Notebook principal
├── README.md                      # Este archivo
├── img1_tp.png                    # Imagen para Parte 2
├── img2_tp.png                    # Imagen para Parte 2
├── segmentacion.png               # Imagen extra
├── white_patch/                   # Imágenes originales para White Patch
│   ├── test_blue.png
│   ├── test_green.png
│   ├── test_red.png
│   ├── wp_blue.jpg
│   ├── wp_green.png
│   ├── wp_green2.jpg
│   ├── wp_red.png
│   ├── wp_red2.jpg
├── white_patch_results/           # Resultados del algoritmo White Patch
│   ├── wp_test_blue.png
│   ├── wp_test_green.png
│   ├── wp_test_red.png
│   ├── wp_wp_blue.jpg
│   ├── wp_wp_blue.png
│   ├── wp_wp_green.png
│   ├── wp_wp_green2.jpg
│   ├── wp_wp_green2.png
│   ├── wp_wp_red.png
│   ├── wp_wp_red2.jpg
│   ├── wp_wp_red2.png
└── VpC1_Vazquez_Workspace/        # Subrepo (clonado)
  ├── README.md
  └── Trabajo_Practico_1/
    ├── img1_tp.png
    ├── img2_tp.png
    ├── segmentacion.png
    └── white_patch/
      ├── test_blue.png
      ├── test_green.png
      ├── test_red.png
      ├── wp_blue.jpg
      ├── wp_green.png
      ├── wp_green2.jpg
      ├── wp_red.png
      ├── wp_red2.jpg
```


- **Parte 1 (imágenes en /white_patch):**
  1. Implementar el algoritmo White Patch para librarnos de las diferencias de color de iluminación.
  2. Mostrar los resultados obtenidos y analizar las posibles fallas (si es que las hay) en el caso de White patch.

- **Parte 2:**
  1. Para las imágenes `img1_tp.png` y `img2_tp.png` leerlas con OpenCV en escala de grisas y visualizarlas.
  2. Elija el numero de bins que crea conveniente y grafique su histograma, compare los histogramas entre si. Explicar lo que se observa, si tuviera que entrenar un modelo de clasificación/detección de imágenes, considera que puede ser de utilidad tomar como ‘features’ a los histogramas?
