"""
TP3 - Punto 2: Múltiples detecciones del logo Coca-Cola en coca_multi.png
=========================================================================
El punto 1 usaba minMaxLoc() que solo devuelve UN máximo global.
Para detectar múltiples logos necesitamos encontrar TODOS los picos
del mapa de correlación que superen el umbral.

Observación importante:
    El template (pattern.png) tiene el logo en ROJO sobre fondo BLANCO.
    En la imagen real el logo es texto BLANCO sobre fondo ROJO.
    Para que el matching funcione bien, invertimos el template con
    cv2.bitwise_not() → texto blanco sobre negro, igual que aparece
    en la imagen en escala de grises.

Método (visto en clase):
    1. Calcular el mapa de correlación R(x,y) con matchTemplate.
    2. Encontrar el máximo con minMaxLoc().
    3. Guardar esa detección y ENMASCARAR esa zona del mapa (ponerla en 0).
    4. Repetir desde el paso 2 hasta que ningún pico supere el umbral.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# ── Rutas ──────────────────────────────────────────────────────────────────
BASE          = r"C:\Users\pandr\Documents\Especializacion IA\VPC\VpC1_Vazquez_Workspace\Trabajo_Practico_3"
TEMPLATE_PATH = os.path.join(BASE, "template", "pattern.png")
IMAGES_DIR    = os.path.join(BASE, "images")
OUTPUT_DIR    = os.path.join(BASE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Parámetros ─────────────────────────────────────────────────────────────
ESCALA    = 0.22   # escala del template (elegida por análisis exploratorio)
THRESHOLD = 0.25   # umbral de confianza mínimo para aceptar una detección


# ── Cargar y preprocesar el template ──────────────────────────────────────
template_color = cv2.imread(TEMPLATE_PATH)
template_gray  = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)

# Recortar bordes blancos
_, mascara = cv2.threshold(template_gray, 200, 255, cv2.THRESH_BINARY_INV)
coords = cv2.findNonZero(mascara)
x0, y0, w0, h0 = cv2.boundingRect(coords)
template = template_gray[y0:y0+h0, x0:x0+w0]

# Invertir el template: en la imagen el logo es texto BLANCO sobre ROJO.
# En escala de grises, las zonas rojas quedan oscuras y el texto blanco queda
# brillante. El template original tiene texto OSCURO sobre BLANCO, lo opuesto.
# Invertirlo lo hace coincidir con la representación en la imagen real.
template_inv = cv2.bitwise_not(template)

t_alto, t_ancho = template.shape[:2]
rw = int(t_ancho * ESCALA)
rh = int(t_alto  * ESCALA)
template_escalado = cv2.resize(template_inv, (rw, rh))
print(f"Template escalado (invertido): {rw}x{rh} px")


# ── Cargar imagen ──────────────────────────────────────────────────────────
img      = cv2.imread(os.path.join(IMAGES_DIR, "coca_multi.png"))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"Imagen: {img.shape[1]}x{img.shape[0]} px")


# ── Calcular mapa de correlación ───────────────────────────────────────────
mapa = cv2.matchTemplate(img_gray, template_escalado, cv2.TM_CCOEFF_NORMED)


# ── Múltiples detecciones: loop minMaxLoc + enmascarado ───────────────────
detecciones  = []
mapa_trabajo = mapa.copy()

while True:
    _, max_val, _, max_loc = cv2.minMaxLoc(mapa_trabajo)

    if max_val < THRESHOLD:
        break

    x, y = max_loc
    detecciones.append((x, y, rw, rh, max_val))

    # Enmascarar la zona detectada para buscar el siguiente logo
    y1 = max(0, y - rh)
    y2 = min(mapa_trabajo.shape[0], y + 2 * rh)
    x1 = max(0, x - rw)
    x2 = min(mapa_trabajo.shape[1], x + 2 * rw)
    mapa_trabajo[y1:y2, x1:x2] = 0

print(f"\nDetecciones encontradas: {len(detecciones)}")
for i, (x, y, w, h, conf) in enumerate(detecciones):
    print(f"  [{i+1}] conf={conf:.3f}  bbox=({x},{y},{w}x{h})")


# ── Visualización ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 9))
ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax.set_title(
    f"TP3 - Punto 2: Múltiples detecciones en coca_multi.png\n"
    f"Template Matching  |  escala={ESCALA}  |  umbral={THRESHOLD}  |  {len(detecciones)} detecciones",
    fontsize=12, fontweight='bold'
)
ax.axis('off')

for x, y, w, h, conf in detecciones:
    rect = patches.Rectangle((x, y), w, h,
                              linewidth=2, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)
    ax.text(x, max(y - 4, 0), f"{conf:.2f}",
            color='lime', fontsize=7, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.65))

plt.tight_layout()
salida = os.path.join(OUTPUT_DIR, "tp3_punto2_resultado.png")
plt.savefig(salida, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nImagen guardada en: {salida}")