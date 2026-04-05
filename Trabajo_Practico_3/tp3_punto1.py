"""
TP3 - Punto 1: Detección del logo Coca-Cola
============================================
Idea: usar Template Matching para encontrar el logo en cada imagen.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# ── Rutas ──────────────────────────────────────
BASE          = r"C:\Users\pandr\Documents\Especializacion IA\VPC\VpC1_Vazquez_Workspace\Trabajo_Practico_3"
TEMPLATE_PATH = os.path.join(BASE, "template", "pattern.png")
IMAGES_DIR    = os.path.join(BASE, "images")
OUTPUT_DIR    = os.path.join(BASE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_FILES = [
    "coca_logo_1.png",
    "coca_logo_2.png",
    "coca_retro_1.png",
    "coca_retro_2.png",
    "logo_1.png",
]

THRESHOLD  = 0.27   # umbral mínimo de confianza para considerar detección válida
MIN_LOGO_W = 60     # ancho mínimo del logo en px, evita falsos positivos a escalas chicas


# ── Cargar y preprocesar el template ──────────────────────────────────────

template_color = cv2.imread(TEMPLATE_PATH)
template_gray  = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)

# Recortar bordes blancos del template: no aportan información
# y pueden generar falsos positivos en zonas blancas de la imagen
_, mascara = cv2.threshold(template_gray, 200, 255, cv2.THRESH_BINARY_INV)
coords = cv2.findNonZero(mascara)
x0, y0, w0, h0 = cv2.boundingRect(coords)
template = template_gray[y0:y0+h0, x0:x0+w0]

# Versión Canny del template: resalta los bordes del lettering.
# Los bordes del logo "Coca-Cola" son muy característicos y el matching
# sobre bordes es más robusto a cambios de color e iluminación
template_canny = cv2.Canny(template, 50, 150)

t_alto, t_ancho = template.shape[:2]
print(f"Template listo: {t_ancho}x{t_alto} px")


# ── Función de detección ───────────────────────────────────────────────────

def detectar_logo(ruta_imagen):
    """
    Detecta el logo usando template matching multi-escala con dos
    representaciones: escala de grises y bordes Canny.

    Para cada escala se calcula el score en ambas representaciones
    y se toma el mejor. Esto permite manejar distintos tipos de imagen:
    - Gris es mejor para imágenes B&W (coca_retro_1)
    - Canny es mejor para imágenes con variaciones de color (logos en color)
    """
    imagen    = cv2.imread(ruta_imagen)
    img_gray  = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_gray, 50, 150)
    alto_img, ancho_img = img_gray.shape[:2]

    mejor_score  = -1
    mejor_loc    = None
    mejor_escala = None

    for escala in np.linspace(0.1, 3.0, 200):
        nuevo_ancho = int(t_ancho * escala)
        nuevo_alto  = int(t_alto  * escala)

        # Descartar escalas donde el template no entra en la imagen
        if nuevo_ancho >= ancho_img or nuevo_alto >= alto_img:
            continue
        # Descartar escalas donde el logo quedaría demasiado chico
        if nuevo_ancho < MIN_LOGO_W:
            continue

        tpl_gray_s  = cv2.resize(template,       (nuevo_ancho, nuevo_alto))
        tpl_canny_s = cv2.resize(template_canny, (nuevo_ancho, nuevo_alto))

        # Matching en escala de grises
        res_gray  = cv2.matchTemplate(img_gray,  tpl_gray_s,  cv2.TM_CCOEFF_NORMED)
        _, v_gray, _, l_gray = cv2.minMaxLoc(res_gray)

        # Matching en espacio de bordes Canny
        res_canny = cv2.matchTemplate(img_canny, tpl_canny_s, cv2.TM_CCOEFF_NORMED)
        _, v_canny, _, l_canny = cv2.minMaxLoc(res_canny)

        # Nos quedamos con la representación que da mayor confianza
        if v_gray >= v_canny:
            v, loc = v_gray, l_gray
        else:
            v, loc = v_canny, l_canny

        if v > mejor_score:
            mejor_score  = v
            mejor_loc    = loc
            mejor_escala = escala

    bw   = int(t_ancho * mejor_escala)
    bh   = int(t_alto  * mejor_escala)
    bbox = (mejor_loc[0], mejor_loc[1], bw, bh)
    return mejor_score, bbox, imagen


# ── Detección en todas las imágenes ───────────────────────────────────────

print("\n=== Resultados ===")
resultados = {}

for nombre in IMAGE_FILES:
    ruta = os.path.join(IMAGES_DIR, nombre)
    score, bbox, imagen = detectar_logo(ruta)
    resultados[nombre] = (score, bbox, imagen)

    estado = "DETECTADO ✓" if score >= THRESHOLD else "no detectado ✗"
    x, y, w, h = bbox
    print(f"[{estado}]  {nombre:25s}  confianza={score:.3f}  bbox=({x},{y},{w}x{h})")


# ── Visualización ──────────────────────────────────────────────────────────

n = len(IMAGE_FILES)
fig, axes = plt.subplots(1, n, figsize=(5*n, 6))
fig.suptitle("TP3 – Punto 1: Detección del logo Coca-Cola\nTemplate Matching Multi-Escala + Canny",
             fontsize=13, fontweight='bold')

for ax, nombre in zip(axes, IMAGE_FILES):
    score, bbox, imagen = resultados[nombre]
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    ax.imshow(imagen_rgb)
    ax.set_title(nombre, fontsize=8)
    ax.axis('off')

    x, y, w, h = bbox
    color = 'lime' if score >= THRESHOLD else 'orange'

    rect = patches.Rectangle((x, y), w, h,
                              linewidth=2.5, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    ax.text(x, max(y - 5, 0), f"conf: {score:.2f}",
            color=color, fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

plt.tight_layout()
salida = os.path.join(OUTPUT_DIR, "tp3_punto1_resultado.png")
plt.savefig(salida, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nImagen guardada en: {salida}")