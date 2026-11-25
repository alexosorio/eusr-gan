# ═══════════════════════════════════════════════════════════════
# busi_baseline.py — EDSR-baseline o SwinIR-light en BUSI nativo
#
# Reutiliza busi_native.py SIN cambiarlo: mismo split, mismos parches de
# entrenamiento (guiados por YOLO, semilla 42), misma validación, mismas
# pérdidas (L1 + 0.1·SSIM + 0.05·FFT), mismo Adam 5e-5, misma parada
# temprana y la misma inferencia por mosaicos. Solo cambia la red.
# Al final compara imagen por imagen con EUSR-GAN (B1, semilla 42).
#
# Requiere en la misma carpeta: busi_native.py y baselines_sr.py
# Uso: ARCH=swinir python -u busi_baseline.py   (o ARCH=edsr)
# ═══════════════════════════════════════════════════════════════
import os
import json

ARCH = os.environ.get('ARCH', 'swinir')   # 'swinir' (SwinIR-light) | 'edsr' (EDSR-baseline)
SEED = int(os.environ.get('SEED', '42'))  # semilla de entrenamiento (42, 7, 123), como B1
_sfx = '' if SEED == 42 else f'_s{SEED}'
# EUSR-GAN B1 con la MISMA semilla, para la comparación pareada
EUSR_JSON = f'resultados_busi_native_scratch_yolo{_sfx}/resultados_busi.json'

import numpy as np
import tensorflow as tf
import busi_native as bn                      # no ejecuta nada al importar
from baselines_sr import BUILDERS, NAMES

if SEED != bn.TRAIN_SEED:                     # busi_native fija 42 al importarse
    bn.TRAIN_SEED = SEED
    tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)
print(f"Semilla de entrenamiento: {SEED}")

bn.OUT_DIR = f'resultados_busi_native_{ARCH}{_sfx}'
os.makedirs(bn.OUT_DIR, exist_ok=True)
bn.W_S1 = os.path.join(bn.OUT_DIR, f'{ARCH}_busi_best.weights.h5')
bn.RUN_STAGE2 = False                         # comparación de fidelidad (sin GAN)
bn.INIT, bn.INIT_WEIGHTS = 'scratch', None
bn.COMPARE_WITH = [EUSR_JSON]                 # pareada: esta red − EUSR-GAN
bn.build_generator = lambda *a, **k: BUILDERS[ARCH](lr_size=bn.LR_PATCH)

# ── Continuación opcional: RESUME=1 reanuda desde los mejores pesos de la
#    corrida anterior, con el lr en que quedó, hasta que deje de mejorar.
#    Conserva los pesos previos si la continuación no los supera en validación.
RESUME = os.environ.get('RESUME') == '1'
if RESUME:
    prev_dir = bn.OUT_DIR
    bn.INIT_WEIGHTS = os.path.join(prev_dir, f'{ARCH}_busi_best.weights.h5')
    bn.LR_S1 = float(os.environ.get('RESUME_LR', '2.5e-5'))
    bn.EPOCHS_S1 = int(os.environ.get('RESUME_EPOCHS', '30'))
    bn.OUT_DIR = f'{prev_dir}_cont'
    os.makedirs(bn.OUT_DIR, exist_ok=True)
    bn.W_S1 = os.path.join(bn.OUT_DIR, f'{ARCH}_busi_best.weights.h5')
    _orig_train = bn.train_stage1

    def _resumed_train(G, x_tr, y_tr, x_val, y_val, epochs=bn.EPOCHS_S1, **kw):
        p0, s0 = bn.evaluate(G, x_val, y_val)
        print(f"Continuación desde {bn.INIT_WEIGHTS} · val previa: PSNR {p0:.2f} dB  SSIM {s0:.4f} "
              f"· lr inicial {bn.LR_S1:.2e} · hasta {epochs} épocas más", flush=True)
        prev_copy = os.path.join(bn.OUT_DIR, 'pesos_previos.weights.h5')
        G.save_weights(prev_copy)
        G, hist, best = _orig_train(G, x_tr, y_tr, x_val, y_val, epochs=epochs, **kw)
        if s0 >= best:
            print(f"La continuación no superó la validación previa ({best:.4f} ≤ {s0:.4f}); "
                  "se conservan los pesos previos.")
            G.load_weights(prev_copy); G.save_weights(bn.W_S1); best = s0
        return G, hist, best

    bn.train_stage1 = _resumed_train

print(f"Método de comparación: {NAMES[ARCH]} → {bn.OUT_DIR}/")
print("(En el reporte, 'EUSR-GAN v6' se refiere a la red de esta corrida; "
      "la comparación pareada final es contra EUSR-GAN.)")
bn.main()

# anotar la arquitectura en el JSON de resultados
fp = os.path.join(bn.OUT_DIR, 'resultados_busi.json')
d = json.load(open(fp))
d['config']['arch'] = NAMES[ARCH] + (' (continuación)' if RESUME else '')
json.dump(d, open(fp, 'w'), indent=2, ensure_ascii=False)
print(f"Arquitectura anotada en {fp}")
