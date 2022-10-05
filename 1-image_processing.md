
# Processing with sci-kit

RGB images have 3 channels, grayscale only 1. Number of channels appears with sci-kit as the third dimension

Some useful commands:


```
from skimage import data, color
rocket_image = data.rocket()
```

```

from skimage import color
grayscale = color.rgb2gray(rocket)
rgb= color.gray2rgb(grayscale)
```
Representing images with matplotlib:
```
def show_image(image, title='Image', cmap_type='gray')
plt.imshow(image, cmap=cmap_type)
plt.title(title)
plt.axis('off')
plt.show()
```

# Numpy with images
```
madrid_image= plt.imread('/madrid.jpeg')
type(madrid_image)
```
How to obtain colour values of an RGB image:

```
red= image[:, :, 0]
green= image[:, :, 1]
blue= image[:, :, 2]
```
The default colormap is not grayscale, we need extra coding for that:

```
plt.imshow(red, cmap="gray")
plt.title('Red')
plt.axis('off')
plt.show()
```

Shapes and dimensions:

```
madrid_image.shape
```

(426, 640, 3)

```
madrid_image.size
```
817920


```

# Flip the image in up direction
vertically_flipped = np.flipud(madrid_image)
show_image(vertically_flipped, 'vertically flipped image')

# Flip the image in left direction
horizontally_flipped = np.fliplr(madrid_image)
show_image(horizontally_flipped, 'horizontally flipped image')

```

# Create histograms

Base for analysis, threshold, brightness/contrast and equalize.


```
red= image[:,:,0]
plt.hist(red.ravel(), bins= 256)
plt.title('Red Histogramn')
plt.show()

