# ═══════════════════════════════════════════════════════════════
# imagenes.py — lectura de imágenes (PNG, JPG, BMP, TIFF y DICOM) y
# codificación de resultados. Todo ocurre en memoria: nada se guarda
# en disco y de los DICOM solo se leen los píxeles.
# ═══════════════════════════════════════════════════════════════
import base64
import io

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 50_000_000          # evita imágenes descomunales


class ErrorImagen(ValueError):
    """Error con un mensaje apto para mostrarse al usuario."""


def _es_dicom(datos: bytes, nombre: str) -> bool:
    return nombre.lower().endswith(('.dcm', '.dicom')) or datos[128:132] == b'DICM'


def _normalizar(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.ndim == 3:                         # color → luminancia
        arr = arr[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= 255 and lo >= 0 and hi > 1:      # 8 bits
        return arr / 255.0
    if hi - lo < 1e-6:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _leer_dicom(datos: bytes) -> np.ndarray:
    try:
        import pydicom
    except ImportError as e:
        raise ErrorImagen("El servidor no tiene soporte DICOM (falta pydicom).") from e
    try:
        ds = pydicom.dcmread(io.BytesIO(datos), force=True)
        arr = ds.pixel_array
    except Exception as e:
        raise ErrorImagen(f"No se pudo leer el DICOM: {e}") from e
    spp = int(getattr(ds, 'SamplesPerPixel', 1))
    frames = int(getattr(ds, 'NumberOfFrames', 1) or 1)
    if frames > 1:                            # cine: se usa el primer cuadro
        arr = arr[0]
    if spp == 3:
        pi = str(getattr(ds, 'PhotometricInterpretation', 'RGB'))
        if pi.startswith('YBR'):
            from pydicom.pixel_data_handlers.util import convert_color_space
            arr = convert_color_space(arr, pi, 'RGB')
    return _normalizar(arr)


def leer_imagen(datos: bytes, nombre: str = '') -> np.ndarray:
    """Bytes de un archivo → imagen en escala de grises, float32 en [0, 1]."""
    if not datos:
        raise ErrorImagen("El archivo está vacío.")
    if _es_dicom(datos, nombre):
        img = _leer_dicom(datos)
    else:
        try:
            im = Image.open(io.BytesIO(datos))
            im.load()
        except Exception as e:
            raise ErrorImagen("Formato no reconocido. Use PNG, JPG, BMP, TIFF o DICOM.") from e
        if im.mode in ('I;16', 'I;16B', 'I', 'F'):
            img = _normalizar(np.asarray(im))
        else:
            img = np.asarray(im.convert('L'), dtype=np.float32) / 255.0
    if img.ndim != 2:
        raise ErrorImagen("La imagen debe tener un solo cuadro en dos dimensiones.")
    return np.clip(img, 0.0, 1.0)


def a_png(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)).save(buf, 'PNG')
    return buf.getvalue()


def a_data_url(img: np.ndarray) -> str:
    return 'data:image/png;base64,' + base64.b64encode(a_png(img)).decode('ascii')


def mapa_error(a: np.ndarray, ref: np.ndarray, ganancia: float = 4.0) -> np.ndarray:
    """|a − ref| realzado para que se vea (ganancia ×4, saturado en 1)."""
    return np.clip(np.abs(a - ref) * ganancia, 0.0, 1.0)
