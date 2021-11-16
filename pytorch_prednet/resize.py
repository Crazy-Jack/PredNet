

from PIL import Image

new_width = 160
new_height = 128

with Image.open("snake-non.png") as img:
    img = img.convert('RGB')
    width, height = img.size 
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    print(f"im {img.size}")
    img.save("snake-non-resize.png")