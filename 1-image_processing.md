
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

#alternative:

from skimage.color import rgb2gray
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
```

# Thresholding

```
thres= 127

binary = image > thresh

show_image(image, 'original')
show_image(binary, 'thresholded')

inverted_binary = image <= thresh
show_image(image, 'original')
show_image(inverted_binary, inverted 'thresholded')
```

There are many ways of thresholding. Two big categories are global or histagram based, and local or adaptative (good for uneven illumination, but slower).
```
from skimage.filters import try_all_threshold
fig, ax = try_all_threshold(image, verbose=False)
show_plot(fig, ax)
```
How to calculate optimal threshold values:
```
# optimal global threshold
from skimage.filters import threshold_otsu
thresh = threshold_otsu(image)
binary_global = image > thresh

# optimal local threshold


from skimage.filters import threshold_local
block_size = 35
##this is local neighborhood
local_thresh = threshold_local(text_image, block_size, offset=10)
binary_local = text_image > local_thresh
```

# Filtering
 - Enhancing an image
 - Smoothening
 - Empathize/remove features
 - Sharpening
 - Edge detection (e.g. Sobel method)
 It is a neighborhood operation
 
 Edge detection with Sobel method:
 ```
 from skimage.filters import sobel
 ##sobel requires a grayscale image
 edge_sobel= sobel(image_coin)
 
 def plot_comparison(original, filterd, title_filtered):
   fig, (Ax1, ax2) = plt.subplots(ncols=2, figsize=(8,6), sharex=True, 
                                  sharey=True)
   ax1.imshow(original, cmap=plt.cm.gray)
   ax1.set_title('original')
   ax2.imshow(filtered, cmap=plt.cm.gray)
   ax2.set_title(title_filtered)
   ax2.axis('off')
   

 plot_comparison(image_coin, edge_sobel, "Edge with Sobel")
 
 ```
 Gaussian smoothing
 ```
 from skimage.filters import gaussian
 gaussian_image = (gaussian, original_pic, multichannel =True)
 plot_comparison(original_pic, gaussian_image, "Blurred witht Gaussian filter")
  ```
  
Contrast enhancement (histogram equalization)
Spreads out most common values

 ```
 from skimage import exposure
 
 ##histogram equalization
 image_eq = exposure.equalize_hist(image)
 show_image(image, 'Original')
 show_image(image_eq, 'Histogram equalized')
 
 ##adaptive equalization (contrastive limited adaptive histogram equalization, CLAHE)
 image_adapteq = exposure.equalize_adapthist(image, clip_limit=0.03)
 ```
 
  ```
 
 # Import the required module
from skimage import exposure

# Show original x-ray image and its histogram
show_image(chest_xray_image, 'Original x-ray')

plt.title('Histogram of image')
plt.hist(chest_xray_image.ravel(), bins=256)
plt.show()

# Use histogram equalization to improve the contrast
xray_image_eq =  exposure.equalize_hist(chest_xray_image)

# Show the resulting image
show_image(xray_image_eq, 'Resulting image')
 ```
