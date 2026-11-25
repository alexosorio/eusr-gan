# ═══════════════════════════════════════════════════════════════
# zenodo_subir_resultados.py — sube a Zenodo los pesos entrenados y los
# resultados por imagen de EUSR-GAN, como un registro de datos con DOI.
#
# Se corre en la VM donde están los resultados (/home/datascience):
#
#   1. En Zenodo: menú del usuario → Applications → Personal access tokens
#      → New token, con los permisos deposit:write y deposit:actions.
#   2. En la terminal de JupyterLab:
#        export ZENODO_TOKEN=el_token
#        python zenodo_subir_resultados.py --listar     # muestra qué se subiría
#        python zenodo_subir_resultados.py              # empaqueta y sube
#
# Recorre las carpetas resultados_* y evaluacion_* y toma los pesos
# (*.weights.h5), los resultados (*.json, *.csv), los registros (*.txt,
# *.log) y los detectores YOLO (*.pt). Omite cachés (*.npz) y figuras.
#
# El registro queda como BORRADOR con el DOI ya reservado: se revisa en la
# web de Zenodo y se publica con el botón "Publish". Ese DOI es el que va
# en la declaración de disponibilidad de datos de PLOS.
#
# Opciones:
#   --sandbox    usa sandbox.zenodo.org (pruebas; requiere token del sandbox)
#   --raiz DIR   carpeta donde están los resultados (por defecto, la actual)
#   --listar     solo lista los archivos y el tamaño total
# ═══════════════════════════════════════════════════════════════
import argparse
import fnmatch
import json
import os
import sys
import zipfile
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Falta requests:  pip install requests")

# ── Metadatos del registro (revíselos antes de subir) ─────────
CREADORES = [
    # {"name": "Apellido, Nombre", "affiliation": "Institución", "orcid": "0000-0000-0000-0000"},
    {"name": "Osorio Barone, Alexander", "affiliation": "Universidad Tecnológica de Bolívar",
     "orcid": "0009-0003-7537-8294"},
    {"name": "Contreras, Sonia", "affiliation": "Universidad Tecnológica de Bolívar"},
    {"name": "Puertas, Edwin", "affiliation": "Universidad Tecnológica de Bolívar"},
]
REPOSITORIO = "https://github.com/alexosorio/eusr-gan"
METADATOS = {
    "upload_type": "dataset",
    "title": "EUSR-GAN: trained weights and per-image results for ultrasound "
             "super-resolution on CAMUS and BUSI",
    "description": (
        "<p>Trained weights and per-image results underlying the manuscript "
        "<i>EUSR-GAN: A lightweight multi-scale GAN for ultrasound super-resolution "
        "and the effect of interpolated references on its evaluation</i>.</p>"
        "<p>Contents: EUSR-GAN weights (CAMUS under the protocol of Ahmad et al. and "
        "at native resolution; BUSI at native resolution, seeds 42, 7 and 123), the "
        "re-implementation of Ahmad et al., EDSR-baseline and SwinIR-light baselines, "
        "ablations, and the YOLOv8 lesion detector; one JSON file per run with the "
        "configuration, summary metrics and per-image PSNR/SSIM by region (whole image, "
        "tissue or sector, lesion); task-based and wavelet analyses; training logs.</p>"
        "<p>Code: " + REPOSITORIO + ". The images are not included: CAMUS and BUSI "
        "are public datasets available from their original sources under their own "
        "terms of use.</p>"),
    "creators": CREADORES,
    "keywords": ["ultrasound", "super-resolution", "generative adversarial networks",
                 "echocardiography", "breast ultrasound", "evaluation protocol",
                 "CAMUS", "BUSI"],
    "license": "cc-by-4.0",
    "access_right": "open",
    "related_identifiers": [
        {"identifier": REPOSITORIO, "relation": "isSupplementTo", "resource_type": "software"},
    ],
    "prereserve_doi": True,
}

CARPETAS = ("resultados_*", "evaluacion_*")
INCLUIR = ("*.weights.h5", "*.json", "*.csv", "*.txt", "*.log", "*.pt")
EXCLUIR_DIRS = {"cache", "__pycache__", ".ipynb_checkpoints"}


def archivos(raiz: Path):
    for d in sorted(p for patron in CARPETAS for p in raiz.glob(patron) if p.is_dir()):
        for dirpath, dirnames, filenames in os.walk(d):
            dirnames[:] = [x for x in dirnames if x not in EXCLUIR_DIRS]
            for f in sorted(filenames):
                if any(fnmatch.fnmatch(f, p) for p in INCLUIR):
                    yield Path(dirpath) / f


def mb(n):
    return f"{n / 1e6:,.1f} MB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--raiz", default=".")
    ap.add_argument("--listar", action="store_true")
    a = ap.parse_args()

    raiz = Path(a.raiz).resolve()
    lista = list(archivos(raiz))
    if not lista:
        sys.exit(f"No se encontraron carpetas resultados_* ni evaluacion_* en {raiz}")
    total = sum(p.stat().st_size for p in lista)
    for p in lista:
        print(f"  {mb(p.stat().st_size):>10}  {p.relative_to(raiz)}")
    print(f"\n{len(lista)} archivos · {mb(total)} sin comprimir")
    if a.listar:
        return

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("Defina el token:  export ZENODO_TOKEN=...")
    base = "https://sandbox.zenodo.org/api" if a.sandbox else "https://zenodo.org/api"
    auth = {"Authorization": f"Bearer {token}"}

    zip_path = raiz / "eusr_gan_pesos_y_resultados.zip"
    print(f"\nEmpaquetando en {zip_path.name} …")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in lista:
            z.write(p, p.relative_to(raiz).as_posix())
    print(f"  {mb(zip_path.stat().st_size)} comprimido")

    r = requests.post(f"{base}/deposit/depositions", json={}, headers=auth, timeout=60)
    r.raise_for_status()
    dep = r.json()
    print(f"Borrador creado (id {dep['id']}). Subiendo el archivo …")
    with open(zip_path, "rb") as fh:
        r = requests.put(f"{dep['links']['bucket']}/{zip_path.name}", data=fh,
                         headers=auth, timeout=None)
    r.raise_for_status()

    r = requests.put(f"{base}/deposit/depositions/{dep['id']}",
                     data=json.dumps({"metadata": METADATOS}),
                     headers={**auth, "Content-Type": "application/json"}, timeout=60)
    if not r.ok:
        sys.exit(f"Zenodo rechazó los metadatos ({r.status_code}): {r.text}")
    final = r.json()
    doi = final.get("metadata", {}).get("prereserve_doi", {}).get("doi", "(sin reservar)")
    print("\n══════════════════════════════════════════════")
    print(f"DOI reservado: {doi}")
    print(f"Revise y publique el borrador en: {final['links']['html']}")
    print("══════════════════════════════════════════════")


if __name__ == "__main__":
    main()
