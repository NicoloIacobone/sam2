import os
from PIL import Image
import re

cartella = "/Users/nicoloiacobone/Desktop/nico/UNIVERSITA/MAGISTRALE/Tesi/Tommasi/Zurigo/git_clones/examples/kubric/test_2"

for filename in os.listdir(cartella):
    if filename.lower().endswith(".png"):
        percorso_png = os.path.join(cartella, filename)
        nome_senza_estensione = os.path.splitext(filename)[0]
        numeri = re.findall(r'\d+', nome_senza_estensione)
        nuovo_nome = numeri[0] if numeri else nome_senza_estensione
        percorso_jpg = os.path.join(cartella, nuovo_nome + ".jpg")
        with Image.open(percorso_png) as img:
            rgb_img = img.convert("RGB")
            rgb_img.save(percorso_jpg, "JPEG", quality=100, subsampling=0)
        os.remove(percorso_png)  # Elimina il file PNG originale
        print(f"Convertito: {filename} -> {nuovo_nome}.jpg ed eliminato il PNG originale")
