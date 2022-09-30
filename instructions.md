# image_analysis_python
Learning and work in progress


## Requirements

- ImageIO (load images)
- NumPy 
- SciPy
- matplotlib

## Loading images

```
import imageio
im = imageio.imread('image.format')

```

Image is stored as matrix, values can be accessed, as well as features

```
im[0,0]
im[0:4,0:4]
print('Image type:', type(im))
print('Shape of image array:', im.shape)
```

Image comes with stored metadata if available.
```
im.meta
im.meta['Category']
print(im.meta.keys())
```

## Plotting images

```
import matplotlib.pyplot as plt
plt.imshow(im, cmap = 'gray')
plt.axis('off')
plt.show()
```
