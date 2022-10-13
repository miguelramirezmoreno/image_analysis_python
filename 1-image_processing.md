
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
 Medical images:
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
Improve the quality of an aerial image:
```
from skimage import exposure
image_eq= exposure.equalize_hist(image_aerial)
show_image(image_aerial, 'Original')
show_image(image_eq, 'Resulting image')

```
Increase impact and contrast of an image:
 ```
 from skimage import data, exposure
 
 original_image = data.coffee()
 adapthist_eq_image = exposure.equalize_adapthist(original_image, clip_limit=0.03)
 
 show_image(original_image)
 show_image(adapthist_eq_image, '#ImageprocessingDatacamp')
```


# Transformations
 - Preparing images for classification ML models
 - Optimization/compression
 - Save images with same proportions

Rotating
 ```
from skimage.transform import rotate
image_rotated = rotate(image, -90)
show_image(image, 'original')
show_image(image_rotated, 'rotated 90 degrees anticlockwise')
##NOTE: negative values means clockwise, use positive numbers to turn left
 ```
 
 Rescaling
 ```
##Downgrading:
from skimage.transform import rescale
image_rescaled = rescale(image, 1/4, anti-aliasing= True, multichannel=True)
 ```
 
 Resizing is similar tu rescaling, but allows to specify dimensions
 
```

from skimage.transform import resize
##We need to give values
height = 400
width = 500
image_resized = resize(image, (height, width), anti_aliasing= True)
##this method can change the scale ratio, unless we resize proportionally:
height= image.shape[0]/4
width = image.shape[1]/4
 ```
 Exercise:
 ```
 # Import the module and the rotate and rescale functions
from skimage.transform import rotate, rescale

# Rotate the image 90 degrees clockwise 
rotated_cat_image = rotate(image_cat, -90)

# Rescale with anti aliasing
rescaled_with_aa = rescale(rotated_cat_image, 1/4, anti_aliasing=True, multichannel=True)

# Rescale without anti aliasing
rescaled_without_aa = rescale(rotated_cat_image, 1/4, anti_aliasing=False, multichannel=True)

# Show the resulting images
show_image(rescaled_with_aa, "Transformed with anti aliasing")
show_image(rescaled_without_aa, "Transformed without anti aliasing")


# Import the module and function to enlarge images
from skimage.transform import rescale

# Import the data module
from skimage import data

# Load the image from data
rocket_image = data.rocket()

# Enlarge the image so it is 3 times bigger
enlarged_rocket_image = rescale(rocket_image, 3, anti_aliasing=True, multichannel=True)

# Show original and resulting image
show_image(rocket_image)
show_image(enlarged_rocket_image, "3 times enlarged image")


# Import the module and function
from skimage.transform import resize

# Set proportional height so its half its size
height = int(dogs_banner.shape[0] / 2)
width = int(dogs_banner.shape[1] / 2)

# Resize using the calculated proportional height and width
image_resized = resize(dogs_banner, (height, width), anti_aliasing=True)

# Show the original and resized image
show_image(dogs_banner, 'Original')
show_image(image_resized, 'Resized image')
```

# Morphology
 - Filtering removes imperfections in the binary image but some also on grayscale images
 - Dilation and erosion are the most used.
 - The number pixels added or removed depends on the structuring element, a small image used to probe the input (in/fit, intersect/hit, or out of the object. The structuring element can have a square, diamond, cross... shape, depending.

Creating the shape (filled with 1s):

 ```
 from skimage import morphology
 square = morphology.square(4)
 rectangle = morphology.rectangle(4,2)
 ```
 
Applying Erosion

 ```
 from skimage import moprhology
 selem=rectangle(12,6)
 eroded_image=morphology.binary_erosion(image_horse, selem=selem)
 plot_comparison(image_horse, eroded_image, 'Erosion')
 ##By default, erosion uses a cross shape unless selem is specified. It can be better or worse depending on the shape and the image.
 ```
 Dilation:

 ```
 from skimage import morphology
 dilated_image = morphology.binary_dilation(image_horse)
 plot_comparison(image_horse, dilated_image, 'Dilation')
 ```
 
 Exercise with handwritten letters (very useful for OCR), world image

 ```
 # Import the morphology module
from skimage import morphology

# Obtain the eroded shape 
eroded_image_shape = morphology.binary_erosion(upper_r_image) 

# See results
show_image(upper_r_image, 'Original')
show_image(eroded_image_shape, 'Eroded image')

# Import the module
from skimage import morphology

# Obtain the dilated image 
dilated_image = morphology.binary_dilation(world_image)

# See results
show_image(world_image, 'Original')
show_image(dilated_image, 'Dilated image')
 ```
# Title
 - Preparing images for classification ML models
 - Optimization/compression
 - Save images with same proportions
```
```
