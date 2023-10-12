from PIL import Image

def load_image(path:str):
    img = Image.open(path).convert("L")
    return (img.size, list(img.getdata()))

