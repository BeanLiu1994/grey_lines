from PIL import Image, ImageOps

def load_image(path:str):
    img = Image.open(path).convert("L")
    img = ImageOps.flip(img)
    return (img.size, list(img.getdata()))

