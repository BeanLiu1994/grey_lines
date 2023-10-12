from PIL import Image, ImageOps

def load_image(path:str, max_w: int = 300):
    img = Image.open(path).convert("L")
    # img = ImageOps.flip(img)
    scale = 1.0
    if img.size[0] > max_w:
        scale = max_w/img.size[0]
        img = ImageOps.scale(img, scale)
    return (img.size, list(img.getdata()), scale)

