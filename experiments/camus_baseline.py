# ═══════════════════════════════════════════════════════════════
# camus_baseline.py — EUSR-GAN, EDSR-baseline o SwinIR-light en CAMUS nativo
#
# Reutiliza camus_eusr.py SIN cambiarlo, con HR_MODE='native': mismo
# split por paciente, mismas pérdidas, mismo Adam 5e-5, misma parada
# temprana e inferencia por mosaicos. Solo cambia la red (ARCH) y, si se
# pide, la semilla de ENTRENAMIENTO (SEED).
#
# SEED cambia: inicialización de pesos, orden de los lotes y parches de
# entrenamiento. NO cambia: partición de pacientes (fija en 42) ni los
# parches de validación (fijos), igual que las semillas de BUSI.
#
# Requiere en la misma carpeta: camus_eusr.py y baselines_sr.py
# Uso: ARCH=edsr SEED=7 python -u camus_baseline.py
#      ARCH=eusr | edsr | swinir      SEED=42 | 7 | 123      RESUME=1 (opcional)
# ═══════════════════════════════════════════════════════════════
import os
import json

import numpy as np
import tensorflow as tf
from scipy import stats

ARCH = os.environ.get('ARCH', 'swinir')   # 'eusr' | 'edsr' | 'swinir'
SEED = int(os.environ.get('SEED', '42'))
_sfx = '' if SEED == 42 else f'_s{SEED}'
EUSR42_JSON = 'resultados_camus_native/resultados_camus.json'          # EUSR-GAN nativo, semilla 42
EUSR_JSON = f'resultados_camus_native{_sfx}/resultados_camus.json'     # misma semilla, para la pareada

import camus_eusr as ce                       # no ejecuta nada al importar (fija semilla 42)
from baselines_sr import BUILDERS, NAMES

NAMES = dict(NAMES, eusr='EUSR-GAN')
SPLIT_SEED, VAL_SEED = ce.SEED, ce.SEED + 1   # 42 y 43: no cambian con SEED

ce.HR_MODE = 'native'
if ARCH == 'eusr':
    ce.OUT_DIR = f'resultados_camus_native{_sfx}'
    # misma configuración de Etapa 2 que la corrida nativa de semilla 42
    try:
        ce.RUN_STAGE2 = bool(json.load(open(EUSR42_JSON))['config'].get('stage2_used', False))
    except Exception:
        ce.RUN_STAGE2 = False
    print(f"EUSR-GAN · Etapa 2 {'activada' if ce.RUN_STAGE2 else 'desactivada'} "
          f"(como en la corrida nativa de semilla 42)")
else:
    ce.OUT_DIR = f'resultados_camus_native_{ARCH}{_sfx}'
    ce.RUN_STAGE2 = False                     # comparación de fidelidad (sin GAN)
    ce.build_generator = lambda *a, **k: BUILDERS[ARCH](lr_size=ce.LR_PATCH)
os.makedirs(ce.OUT_DIR, exist_ok=True)
ce.W_S1 = os.path.join(ce.OUT_DIR, f'{ARCH}_camus_best.weights.h5')
ce.W_S2 = os.path.join(ce.OUT_DIR, f'{ARCH}_camus_stage2.weights.h5')

# ── Semilla de entrenamiento ──────────────────────────────────
if SEED != 42:
    tf.keras.utils.set_random_seed(SEED)      # inicialización y orden de lotes
    np.random.seed(SEED)
    _orig_build = ce.build_patch_set

    def _seeded_build(hrs, masks, n_per, seed, do_aug):
        # parches de entrenamiento con la nueva semilla; validación intacta
        return _orig_build(hrs, masks, n_per, SEED if do_aug else VAL_SEED, do_aug)

    ce.build_patch_set = _seeded_build
print(f"Semilla de entrenamiento: {SEED} · partición de pacientes: {SPLIT_SEED} · "
      f"parches de validación: {VAL_SEED}")

# ── Continuación opcional: RESUME=1 reanuda desde los mejores pesos de la
#    corrida anterior, con el lr en que quedó, hasta que deje de mejorar.
#    Conserva los pesos previos si la continuación no los supera en validación.
RESUME = os.environ.get('RESUME') == '1'
if RESUME:
    prev_w = ce.W_S1
    if not os.path.exists(prev_w):
        raise SystemExit(f"No existe {prev_w}")
    ce.LR_S1 = float(os.environ.get('RESUME_LR', '2.5e-5'))
    ce.EPOCHS_S1 = int(os.environ.get('RESUME_EPOCHS', '30'))
    ce.OUT_DIR = f'{ce.OUT_DIR}_cont'
    os.makedirs(ce.OUT_DIR, exist_ok=True)
    ce.W_S1 = os.path.join(ce.OUT_DIR, f'{ARCH}_camus_best.weights.h5')
    _orig_train = ce.train_stage1

    def _resumed_train(G, x_tr, y_tr, x_val, y_val, epochs=ce.EPOCHS_S1, **kw):
        G.load_weights(prev_w)
        p0, s0 = ce.evaluate(G, x_val, y_val)
        print(f"Continuación desde {prev_w} · val previa: PSNR {p0:.2f} dB  SSIM {s0:.4f} "
              f"· lr inicial {ce.LR_S1:.2e} · hasta {epochs} épocas más", flush=True)
        G, hist, best = _orig_train(G, x_tr, y_tr, x_val, y_val, epochs=epochs, **kw)
        if s0 >= best:
            print(f"La continuación no superó la validación previa ({best:.4f} ≤ {s0:.4f}); "
                  "se conservan los pesos previos.")
            G.load_weights(prev_w); G.save_weights(ce.W_S1); best = s0
        return G, hist, best

    ce.train_stage1 = _resumed_train

print(f"Método: {NAMES[ARCH]} · CAMUS nativo → {ce.OUT_DIR}/")
if ARCH != 'eusr':
    print("(En el reporte, 'EUSR-GAN v6' se refiere a la red de esta corrida.)")
ce.main()

fp = os.path.join(ce.OUT_DIR, 'resultados_camus.json')
d = json.load(open(fp))
d['config']['arch'] = NAMES[ARCH] + (' (continuación)' if RESUME else '')
d['config']['train_seed'] = SEED
json.dump(d, open(fp, 'w'), indent=2, ensure_ascii=False)

# ── comparación pareada con EUSR-GAN nativo de la misma semilla ─────
if ARCH == 'eusr':
    pass
elif not os.path.exists(EUSR_JSON):
    print(f"\n(No existe {EUSR_JSON}; se omite la comparación con EUSR-GAN.)")
else:
    other = {(r['patient'], r['view'], r['phase']): r for r in json.load(open(EUSR_JSON))['por_imagen']}
    com = [(r, other[(r['patient'], r['view'], r['phase'])]) for r in d['por_imagen']
           if (r['patient'], r['view'], r['phase']) in other]
    print(f"\nComparación pareada con EUSR-GAN nativo, semilla {SEED} (n={len(com)}):")
    for k, nom in (('sr_psnr', 'PSNR completa'), ('sr_psnr_sector', 'PSNR sector'),
                   ('sr_ssim', 'SSIM completa'), ('sr_ssim_sector', 'SSIM sector')):
        a = np.array([r[k] for r, _ in com]); b = np.array([o[k] for _, o in com])
        p = stats.wilcoxon(a, b).pvalue if np.any(a != b) else float('nan')
        fmt = '+.3f' if 'psnr' in k else '+.4f'
        print(f"  {nom:14s}: {NAMES[ARCH]} − EUSR-GAN = {format((a - b).mean(), fmt)} "
              f"(mejor en {np.mean(a > b):.0%}, p={p:.1e})")
