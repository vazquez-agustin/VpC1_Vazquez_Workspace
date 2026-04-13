"""
TP3 - Punto 3: Generalización del algoritmo de múltiples detecciones
=====================================================================
Extiende el algoritmo del Punto 2 (iterative minMaxLoc + enmascarado)
a todas las imágenes, combinándolo con búsqueda multi-escala del Punto 1
para ser robusto a diferentes tamaños de logo.

Método:
    1. Para cada imagen, barrer múltiples escalas del template.
    2. En cada escala, calcular el mapa de correlación (gris + Canny).
    3. Combinar los mapas de todas las escalas quedándose con el mejor
       score por posición.
    4. Extraer múltiples detecciones con el loop minMaxLoc + enmascarado.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import math

# ── Rutas ──────────────────────────────────────────────────────────────────
BASE          = r"C:\Users\pandr\Documents\Especializacion IA\VPC\VpC1_Vazquez_Workspace\Trabajo_Practico_3"
TEMPLATE_PATH = os.path.join(BASE, "template", "pattern.png")
IMAGES_DIR    = os.path.join(BASE, "images")
OUTPUT_DIR    = os.path.join(BASE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_FILES = [
    
    "coca_logo_1.png",
    "coca_logo_2.png",
    "coca_multi.png",
    "coca_retro_1.png",
    "coca_retro_2.png",
    "logo_1.png",
]

# ── Parámetros ─────────────────────────────────────────────────────────────
THRESHOLD  = 0.25   # umbral mínimo de confianza para aceptar una detección
MIN_LOGO_W = 60     # ancho mínimo del logo en px
ESCALAS    = np.linspace(0.1, 3.0, 200)


# ── Cargar y preprocesar el template ──────────────────────────────────────
template_color = cv2.imread(TEMPLATE_PATH)
template_gray  = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)

# Recortar bordes blancos del template
_, mascara = cv2.threshold(template_gray, 200, 255, cv2.THRESH_BINARY_INV)
coords = cv2.findNonZero(mascara)
x0, y0, w0, h0 = cv2.boundingRect(coords)
template = template_gray[y0:y0+h0, x0:x0+w0]

# Versiones para matching
template_inv   = cv2.bitwise_not(template)
template_canny = cv2.Canny(template, 50, 150)

t_alto, t_ancho = template.shape[:2]
print(f"Template listo: {t_ancho}x{t_alto} px")


# ── Función de detección multi-escala + múltiples detecciones ─────────────

def detectar_logos(ruta_imagen):
    """
    Detecta múltiples logos en una imagen combinando:
    - Búsqueda multi-escala (como Punto 1)
    - Extracción iterativa de picos (como Punto 2)

    Para cada escala se prueban 3 representaciones del template:
    gris normal, gris invertido y Canny, para ser robusto a
    variaciones de color (logo rojo sobre blanco vs blanco sobre rojo).
    """
    imagen    = cv2.imread(ruta_imagen)
    img_gray  = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_gray, 50, 150)
    alto_img, ancho_img = img_gray.shape[:2]

    # Buscar la mejor escala global (como en punto 1)
    mejor_score_global = -1
    mejor_escala       = None

    for escala in ESCALAS:
        nw = int(t_ancho * escala)
        nh = int(t_alto  * escala)

        if nw >= ancho_img or nh >= alto_img or nw < MIN_LOGO_W:
            continue

        tpl_gray_s  = cv2.resize(template, (nw, nh))
        tpl_inv_s   = cv2.resize(template_inv, (nw, nh))
        tpl_canny_s = cv2.resize(template_canny, (nw, nh))

        # Probar las 3 representaciones y quedarse con el mejor score
        for tpl, img_repr in [(tpl_gray_s, img_gray),
                               (tpl_inv_s, img_gray),
                               (tpl_canny_s, img_canny)]:
            res = cv2.matchTemplate(img_repr, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > mejor_score_global:
                mejor_score_global = max_val
                mejor_escala       = escala

    if mejor_escala is None:
        return [], imagen

    # Con la mejor escala, hacer múltiples detecciones (como en punto 2)
    rw = int(t_ancho * mejor_escala)
    rh = int(t_alto  * mejor_escala)

    tpl_gray_best  = cv2.resize(template, (rw, rh))
    tpl_inv_best   = cv2.resize(template_inv, (rw, rh))
    tpl_canny_best = cv2.resize(template_canny, (rw, rh))

    # Calcular mapas para las 3 representaciones
    mapa_gray  = cv2.matchTemplate(img_gray,  tpl_gray_best,  cv2.TM_CCOEFF_NORMED)
    mapa_inv   = cv2.matchTemplate(img_gray,  tpl_inv_best,   cv2.TM_CCOEFF_NORMED)
    mapa_canny = cv2.matchTemplate(img_canny, tpl_canny_best, cv2.TM_CCOEFF_NORMED)

    # Combinar: tomar el máximo de las 3 representaciones por píxel
    mapa = np.maximum(mapa_gray, np.maximum(mapa_inv, mapa_canny))

    # Extraer detecciones iterativamente
    detecciones  = []
    mapa_trabajo = mapa.copy()

    while True:
        _, max_val, _, max_loc = cv2.minMaxLoc(mapa_trabajo)

        if max_val < THRESHOLD:
            break

        x, y = max_loc
        detecciones.append((x, y, rw, rh, max_val))

        # Enmascarar zona detectada
        pad_x = rw // 2
        pad_y = rh // 2
        y1 = max(0, y - pad_y)
        y2 = min(mapa_trabajo.shape[0], y + rh + pad_y)
        x1 = max(0, x - pad_x)
        x2 = min(mapa_trabajo.shape[1], x + rw + pad_x)
        mapa_trabajo[y1:y2, x1:x2] = 0

    return detecciones, imagen


# ── Procesar todas las imágenes ───────────────────────────────────────────

print("\n=== Resultados ===")
resultados = {}

for nombre in IMAGE_FILES:
    ruta = os.path.join(IMAGES_DIR, nombre)
    detecciones, imagen = detectar_logos(ruta)
    resultados[nombre] = (detecciones, imagen)
    print(f"\n{nombre}: {len(detecciones)} detección(es)")
    for i, (x, y, w, h, conf) in enumerate(detecciones):
        print(f"  [{i+1}] conf={conf:.3f}  bbox=({x},{y},{w}x{h})")


# ── Visualización ──────────────────────────────────────────────────────────

n = len(IMAGE_FILES)
cols = 3
rows = math.ceil(n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
fig.suptitle(
    "TP3 – Punto 3: Múltiples detecciones generalizadas a todas las imágenes\n"
    "Template Matching Multi-Escala + Detección Iterativa",
    fontsize=14, fontweight='bold'
)
axes = axes.flatten()

for idx, nombre in enumerate(IMAGE_FILES):
    ax = axes[idx]
    detecciones, imagen = resultados[nombre]
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    ax.imshow(imagen_rgb)
    ax.set_title(f"{nombre}  ({len(detecciones)} det.)", fontsize=9)
    ax.axis('off')

    for x, y, w, h, conf in detecciones:
        color = 'lime' if conf >= THRESHOLD else 'orange'
        rect = patches.Rectangle((x, y), w, h,
                                  linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x, max(y - 4, 0), f"{conf:.2f}",
                color=color, fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.65))

# Ocultar ejes sobrantes
for idx in range(n, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
salida = os.path.join(OUTPUT_DIR, "tp3_punto3_resultado.png")
plt.savefig(salida, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nImagen guardada en: {salida}")
