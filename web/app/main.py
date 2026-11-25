# ═══════════════════════════════════════════════════════════════
# main.py — API y servidor web del prototipo EUSR-GAN
#
#   GET  /              interfaz web
#   GET  /api/estado    modelos disponibles, límites y velocidad medida
#   POST /api/mejorar   imagen + zona marcada → resultados
#
# Dos modos:
#   ampliar  La zona marcada se trata como imagen de entrada y se amplía
#            ×4 con EUSR-GAN y con interpolación bicúbica. Es el uso
#            clínico imaginado, pero el modelo NO se validó así: se
#            entrenó para reconstruir imágenes reducidas ×4.
#   validar  La zona se reduce ×4 (bicúbica, como en el entrenamiento) y
#            se reconstruye; como la zona original es la referencia, se
#            calculan PSNR y SSIM. Es el escenario del artículo.
#
# Configuración por variables de entorno (ver README.md):
#   MODELO_MAMA, MODELO_CORAZON   rutas de los pesos .weights.h5
#   MAX_LADO_AMPLIAR (512)        lado máximo de la zona en modo ampliar
#   MAX_MB (25)                   tamaño máximo del archivo subido
#   APP_USUARIO, APP_CLAVE        si se definen, exige usuario y clave
# ═══════════════════════════════════════════════════════════════
import base64
import logging
import os
import secrets
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import modelo as M
from .imagenes import ErrorImagen, a_data_url, leer_imagen, mapa_error

log = logging.getLogger('eusr_web')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

VERSION = '1.0.0'
BASE = Path(__file__).resolve().parent
RAIZ = BASE.parent

CATALOGO = {   # id → (nombre visible, variable de entorno, ruta por defecto, datos de entrenamiento)
    'mama':    ('Mama (BUSI)',      'MODELO_MAMA',    'pesos/eusr_busi.weights.h5',
                'Ecografía mamaria, BUSI, resolución nativa, semilla 42'),
    'corazon': ('Corazón (CAMUS)',  'MODELO_CORAZON', 'pesos/eusr_camus.weights.h5',
                'Ecocardiografía apical de cuatro cámaras, CAMUS, resolución nativa'),
}
MAX_LADO_AMPLIAR = int(os.environ.get('MAX_LADO_AMPLIAR', '512'))
MAX_MB = float(os.environ.get('MAX_MB', '25'))
MAX_LADO_IMAGEN = 4096
MIN_VALIDAR = M.HR_PATCH                       # 128: la entrada reducida debe medir ≥ 32
USUARIO = os.environ.get('APP_USUARIO', '')
CLAVE = os.environ.get('APP_CLAVE', '')

app = FastAPI(title='EUSR-GAN · prototipo', version=VERSION, docs_url=None, redoc_url=None)
MODELOS: dict = {}
VELOCIDAD = {'mosaicos_por_segundo': None}


# ── Arranque: carga de modelos y medición de velocidad ────────
@app.on_event('startup')
def cargar_modelos():
    for mid, (nombre, var, defecto, _) in CATALOGO.items():
        ruta = Path(os.environ.get(var, defecto))
        if not ruta.is_absolute():
            ruta = RAIZ / ruta
        if not ruta.exists():
            log.warning("Sin pesos para '%s' (%s): ese modelo no estará disponible.", mid, ruta)
            continue
        t0 = time.time()
        MODELOS[mid] = M.Modelo(str(ruta))
        log.info("Modelo '%s' cargado desde %s en %.1f s", mid, ruta, time.time() - t0)
    if MODELOS:
        m = next(iter(MODELOS.values()))
        x = np.random.default_rng(0).random((128, 128), dtype=np.float32)
        t0 = time.time()
        m.super_resolver(x)
        VELOCIDAD['mosaicos_por_segundo'] = M.n_mosaicos(128, 128) / max(time.time() - t0, 1e-3)
        log.info("Velocidad medida: %.0f mosaicos/s", VELOCIDAD['mosaicos_por_segundo'])
    else:
        log.error("No se encontró ningún archivo de pesos. Revise la carpeta pesos/ (README.md).")


# ── Autenticación básica opcional ─────────────────────────────
@app.middleware('http')
async def autenticar(request: Request, call_next):
    if CLAVE:
        auth = request.headers.get('authorization', '')
        ok = False
        if auth.lower().startswith('basic '):
            try:
                u, _, c = base64.b64decode(auth[6:]).decode('utf-8').partition(':')
                ok = secrets.compare_digest(u, USUARIO) and secrets.compare_digest(c, CLAVE)
            except Exception:
                ok = False
        if not ok:
            return Response(status_code=401, headers={'WWW-Authenticate': 'Basic realm="EUSR-GAN"'})
    resp = await call_next(request)
    resp.headers['Cache-Control'] = 'no-store'
    resp.headers['X-Content-Type-Options'] = 'nosniff'
    return resp


app.mount('/static', StaticFiles(directory=BASE / 'static'), name='static')


@app.get('/')
def inicio():
    return FileResponse(BASE / 'static' / 'index.html')


@app.get('/api/estado')
def estado():
    return {
        'version': VERSION,
        'escala': M.SCALE,
        'modelos': [{'id': k, 'nombre': CATALOGO[k][0], 'descripcion': CATALOGO[k][3]}
                    for k in CATALOGO if k in MODELOS],
        'limites': {'max_lado_ampliar': MAX_LADO_AMPLIAR, 'min_lado_validar': MIN_VALIDAR,
                    'min_lado_ampliar': M.LR_PATCH, 'max_mb': MAX_MB},
        'mosaicos_por_segundo': VELOCIDAD['mosaicos_por_segundo'],
    }


@app.post('/api/vista')
def vista(archivo: UploadFile = File(...)):
    """Devuelve la imagen como PNG en escala de grises (sirve también para DICOM),
    con las mismas dimensiones que usará /api/mejorar."""
    datos = archivo.file.read(int(MAX_MB * 1024 * 1024) + 1)
    if len(datos) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"El archivo supera el máximo de {MAX_MB:g} MB.")
    try:
        img = leer_imagen(datos, archivo.filename or '')
    except ErrorImagen as e:
        raise HTTPException(400, str(e))
    if max(img.shape) > MAX_LADO_IMAGEN:
        raise HTTPException(400, f"La imagen supera {MAX_LADO_IMAGEN} píxeles por lado.")
    return {'ancho': img.shape[1], 'alto': img.shape[0], 'png': a_data_url(img)}


# ── Utilidades de la zona ─────────────────────────────────────
def _zona(img, x, y, w, h):
    """Zona pedida, recortada a los límites de la imagen; None = imagen completa."""
    H, W = img.shape
    if None in (x, y, w, h) or w <= 0 or h <= 0:
        return 0, 0, W, H
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x) + int(w)), min(H, int(y) + int(h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        raise ErrorImagen("La zona marcada es demasiado pequeña o está fuera de la imagen.")
    return x0, y0, x1 - x0, y1 - y0


def _agrandar(x, y, w, h, minimo, W, H):
    """Agranda la zona alrededor de su centro hasta 'minimo' por lado, dentro de la imagen."""
    def eje(a, l, L):
        if l >= minimo:
            return a, l
        c = a + l / 2
        a = int(round(c - minimo / 2))
        a = max(0, min(a, L - minimo))
        return a, minimo
    x, w = eje(x, w, W)
    y, h = eje(y, h, H)
    return x, y, w, h


@app.post('/api/mejorar')
def mejorar(archivo: UploadFile = File(...),
            modelo: str = Form('mama'),
            modo: str = Form('ampliar'),
            x: int | None = Form(None), y: int | None = Form(None),
            w: int | None = Form(None), h: int | None = Form(None),
            autoensamble: bool = Form(False)):
    if modelo not in MODELOS:
        raise HTTPException(400, f"El modelo '{modelo}' no está disponible en este servidor.")
    if modo not in ('ampliar', 'validar'):
        raise HTTPException(400, "Modo desconocido.")
    datos = archivo.file.read(int(MAX_MB * 1024 * 1024) + 1)
    if len(datos) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"El archivo supera el máximo de {MAX_MB:g} MB.")
    try:
        img = leer_imagen(datos, archivo.filename or '')
        if max(img.shape) > MAX_LADO_IMAGEN:
            raise ErrorImagen(f"La imagen supera {MAX_LADO_IMAGEN} píxeles por lado.")
        x0, y0, w0, h0 = _zona(img, x, y, w, h)
        G = MODELOS[modelo]
        H, W = img.shape
        avisos = []
        t0 = time.time()

        if modo == 'ampliar':
            if max(w0, h0) > MAX_LADO_AMPLIAR:
                raise ErrorImagen(
                    f"En modo ampliar la zona puede medir hasta {MAX_LADO_AMPLIAR} píxeles por lado "
                    f"(la marcada mide {w0}×{h0}). Marque una zona más pequeña.")
            roi = img[y0:y0 + h0, x0:x0 + w0]
            ph, pw = max(0, M.LR_PATCH - h0), max(0, M.LR_PATCH - w0)
            entrada = np.pad(roi, ((0, ph), (0, pw)), mode='reflect') if (ph or pw) else roi
            sr = G.super_resolver(entrada, autoensamble)[:h0 * M.SCALE, :w0 * M.SCALE]
            bic = M.bicubica(roi, (h0 * M.SCALE, w0 * M.SCALE))
            nn = M.vecino(roi)
            mos = M.n_mosaicos(*entrada.shape)
            imagenes = {'entrada': a_data_url(nn), 'bicubica': a_data_url(bic), 'eusr': a_data_url(sr)}
            metricas = None
            salida = [w0 * M.SCALE, h0 * M.SCALE]
            avisos.append("Modo exploratorio: el modelo se entrenó para reconstruir imágenes reducidas ×4, "
                          "no para ampliar la imagen adquirida. No hay referencia para medir la mejora.")
        else:
            if W < MIN_VALIDAR or H < MIN_VALIDAR:
                raise ErrorImagen(f"Para validar, la imagen debe medir al menos {MIN_VALIDAR}×{MIN_VALIDAR}.")
            if w0 < MIN_VALIDAR or h0 < MIN_VALIDAR:
                x0, y0, w0, h0 = _agrandar(x0, y0, w0, h0, MIN_VALIDAR, W, H)
                avisos.append(f"La zona se agrandó al mínimo de {MIN_VALIDAR}×{MIN_VALIDAR} píxeles.")
            w0, h0 = w0 - w0 % M.SCALE, h0 - h0 % M.SCALE
            ref = img[y0:y0 + h0, x0:x0 + w0]
            lr = M.degradar(ref)
            sr = G.super_resolver(lr, autoensamble)
            bic = M.bicubica(lr, ref.shape)
            nn = M.vecino(lr)
            mos = M.n_mosaicos(*lr.shape)
            imagenes = {'entrada': a_data_url(nn), 'bicubica': a_data_url(bic), 'eusr': a_data_url(sr),
                        'referencia': a_data_url(ref),
                        'error_bicubica': a_data_url(mapa_error(bic, ref)),
                        'error_eusr': a_data_url(mapa_error(sr, ref))}
            pb, pe = M.psnr(bic, ref), M.psnr(sr, ref)
            sb, se = M.ssim(bic, ref), M.ssim(sr, ref)
            metricas = {'psnr_bicubica': pb, 'psnr_eusr': pe, 'ganancia_psnr': pe - pb,
                        'ssim_bicubica': sb, 'ssim_eusr': se, 'ganancia_ssim': se - sb}
            salida = [w0, h0]

        seg = time.time() - t0
        log.info("modo=%s modelo=%s zona=%dx%d mosaicos=%d autoensamble=%s %.1fs",
                 modo, modelo, w0, h0, mos, autoensamble, seg)
        return JSONResponse({
            'modo': modo, 'modelo': modelo, 'autoensamble': autoensamble,
            'imagen': [W, H], 'zona': [x0, y0, w0, h0], 'salida': salida,
            'mosaicos': mos * (8 if autoensamble else 1), 'segundos': seg,
            'imagenes': imagenes, 'metricas': metricas, 'avisos': avisos,
        })
    except ErrorImagen as e:
        raise HTTPException(400, str(e))
