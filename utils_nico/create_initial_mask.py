import cv2

# Sostituisci con il percorso della tua immagine
image_path = '/Users/nicoloiacobone/Desktop/nico/UNIVERSITA/MAGISTRALE/Tesi/Tommasi/Zurigo/git_clones/examples/kubric/bouncing_balls/00000.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Impossibile aprire l'immagine: {image_path}")
    exit(1)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinate del click: ({x}, {y})")

cv2.namedWindow('Immagine')
cv2.setMouseCallback('Immagine', mouse_callback)

while True:
    cv2.imshow('Immagine', img)
    if cv2.waitKey(1) & 0xFF == 27:  # Premi ESC per uscire
        break

cv2.destroyAllWindows()