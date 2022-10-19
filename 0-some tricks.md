# Choose directory
https://stackoverflow.com/questions/50860640/ask-a-user-to-select-folder-to-read-the-files-in-python


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

# Convert PDF to Tiff or any format (png, jpg, tiff, gif, bmp, docx)
```
pip install aspose-words
import aspose.words as aw
doc= aw.Document~("clcoding.pdf")
doc.save("clcoding.tiff")
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
