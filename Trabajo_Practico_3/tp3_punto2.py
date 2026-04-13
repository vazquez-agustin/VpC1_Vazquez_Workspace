"""
TP3 - Punto 2: Múltiples detecciones del logo Coca-Cola en coca_multi.png
=========================================================================
El punto 1 usaba minMaxLoc() que solo devuelve UN máximo global.
Para detectar múltiples logos necesitamos encontrar todos los picos
del mapa de correlación que superen el umbral.

Observación:
    El template tiene el logo ROJO sobre fondo BLANCO.
    En la imagen real el logo es texto BLANCO sobre fondo ROJO.
    En escala de grises son representaciones opuestas, por eso
    invertimos el template con cv2.bitwise_not().

    Además combinamos tres representaciones del template tomando
    el máximo pixel a pixel de los tres mapas de correlación:
    - template invertido  → mejor para logos en color
    - template normal     → útil para variaciones de contraste
    - template Canny      → robusto a cambios de iluminación

Método:
    1. Calcular el mapa de correlación R(x,y) con matchTemplate.
    2. Encontrar el máximo con minMaxLoc().
    3. Guardar esa detección y enmascarar esa zona (ponerla en 0).
       La máscara se centra en el pico y tiene tamaño ~1x el template,
       suficiente para suprimir el plateau del pico actual sin eliminar
       logos vecinos cercanos.
    4. Repetir hasta que ningún pico supere el umbral.
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
ESCALA    = 0.22   # escala elegida por análisis exploratorio
THRESHOLD = 0.25   # umbral mínimo de confianza


# ── Cargar y preprocesar el template ──────────────────────────────────────
template_color = cv2.imread(TEMPLATE_PATH)
template_gray  = cv2.cvtColor(template_color, cv2.COLOR_BGR2GRAY)

_, mascara = cv2.threshold(template_gray, 200, 255, cv2.THRESH_BINARY_INV)
coords = cv2.findNonZero(mascara)
x0, y0, w0, h0 = cv2.boundingRect(coords)
template = template_gray[y0:y0+h0, x0:x0+w0]

# Tres versiones del template
template_inv   = cv2.bitwise_not(template)          # texto blanco sobre negro
template_canny = cv2.Canny(template, 50, 150)       # bordes

t_alto, t_ancho = template.shape[:2]
rw = int(t_ancho * ESCALA)
rh = int(t_alto  * ESCALA)
print(f"Template escalado: {rw}x{rh} px")


# ── Cargar imagen ──────────────────────────────────────────────────────────
img       = cv2.imread(os.path.join(IMAGES_DIR, "coca_multi.png"))
img_gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_canny = cv2.Canny(img_gray, 50, 150)
print(f"Imagen: {img.shape[1]}x{img.shape[0]} px")


# ── Calcular mapa de correlación combinado ────────────────────────────────
# Tomamos el máximo pixel a pixel de las tres representaciones.
# Así nos quedamos con el score más alto para cada posición,
# sin importar qué representación lo generó.
res_inv   = cv2.matchTemplate(img_gray,  cv2.resize(template_inv,   (rw,rh)), cv2.TM_CCOEFF_NORMED)
res_gray  = cv2.matchTemplate(img_gray,  cv2.resize(template,       (rw,rh)), cv2.TM_CCOEFF_NORMED)
res_canny = cv2.matchTemplate(img_canny, cv2.resize(template_canny, (rw,rh)), cv2.TM_CCOEFF_NORMED)
mapa = np.maximum(res_inv, np.maximum(res_gray, res_canny))


# ── Múltiples detecciones: loop minMaxLoc + enmascarado ───────────────────
detecciones  = []
mapa_trabajo = mapa.copy()

while True:
    _, max_val, _, max_loc = cv2.minMaxLoc(mapa_trabajo)
    if max_val < THRESHOLD:
        break

    x, y = max_loc
    detecciones.append((x, y, rw, rh, max_val))

    # Máscara centrada en el pico: rw//2 a cada lado en X, rh//2 en Y.
    # El tamaño es ~1x el template: suficiente para suprimir el plateau
    # del pico actual, sin eliminar logos vecinos cercanos.
    pad_x = rw // 2
    pad_y = rh // 2
    y1 = max(0, y - pad_y)
    y2 = min(mapa_trabajo.shape[0], y + rh + pad_y)
    x1 = max(0, x - pad_x)
    x2 = min(mapa_trabajo.shape[1], x + rw + pad_x)
    mapa_trabajo[y1:y2, x1:x2] = 0

print(f"\nDetecciones encontradas: {len(detecciones)}")
for i, (x, y, w, h, conf) in enumerate(detecciones):
    print(f"  [{i+1}] conf={conf:.3f}  bbox=({x},{y},{w}x{h})")


# ── Visualización ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 9))
ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax.set_title(
    f"TP3 – Punto 2: Múltiples detecciones en coca_multi.png\n"
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
