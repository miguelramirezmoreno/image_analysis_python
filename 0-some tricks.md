# Image mirroring
```
from PIL import image
Image.open('image.jpg')
img = Imagen.open('image.jpg')
mirror_image =imp.transpose(Image.FLIP_LEFT_RIGHT)
mirror_image.save('image_mirror.png')
Image.open('image_mirror.png')
```

# Convert Docx to Pdf
```
from doc2pdf import convert
docx_file = 'clcoding.docx'
pdf_file = 'output.pdf'
convert(dpcx_file, pdf_file)
```

# Create gif with images (any type)
```
import imageio
filenames = ["1.png", "3.png", "4.png"]
images= []
for filename in filenames:
    images.append(imageio.impread(filename))
imageio.mimsave('name.gif', images, 'GIF' duration=1)
```
