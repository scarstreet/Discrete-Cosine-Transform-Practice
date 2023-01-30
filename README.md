<h1>Discrete Cosine Transform Practice</h1>
This practice aimed to help understand how image formatting works through DCT and IDCT
<ol>
<li> <h3>Question</h3>

<ol>
<li> Topic：Image Transform ~ DCT & IDCT
<li> Description：Link【e-Learning】Download hw4\_data.rar，Decompress Lena gray image，resolution 512×512。Dong Hwa e-Learning: [http://www.elearn.ndhu.edu.tw](http://www.elearn.ndhu.edu.tw/)
  - Take a DCT transform (Block size: 8\*8)
  - Quantization, Dequantization
  - Take an IDCT
  - Calculate the PSNR of
<li> Program development tool：No limitation。(Please don't use "DCT/JPEG related" function call.)
<li> Content submission and Deadline Upload：Compress into a file(file name: ID no. + name)，Upload Dong Hwa e-Learning
  - Please upload a word file, content is as follows: List and explain the program code, and show DCT, IDCT, and error images. Calculate the PSNR of.
  - Deadline：2021/12/20 (一) 23:55（Delay within one week: 50% OFF，Delay after one week: 100% OFF，Copier: Deduction）Early submission : Within one week: "2" total points Within two week: "1" total point
</ol>
<li> <h3>Code</h3>
- Explanation
  - Libraries Used
```
import cv2
import numpy asnp
from math import cos,pi,sqrt
```
These are the libraries used for the code. Cv2 is for opencv operations which are imread and imshow to read and show the image. Numpy is to manipulate the read images' array using functions such as copy, subtract, add, etc. Math is to use some math operations and variables such as cos, pi and square root.

  - Global Variables
```
QUANTUM = np.array([ # source = https://en.wikipedia.org/wiki/Quantization\_(image\_processing)
  [16,11,10,16,24,40,51,61],
  [12,12,14,19,26,58,60,55],
  [14,13,16,24,40,57,69,56],
  [14,17,22,29,51,87,80,62],
  [18,22,37,56,68,109,103,77],
  [24,35,55,64,81,104,113,92],
  [49,64,78,87,103,121,120,101],
  [72,92,95,98,112,100,103,99],
])

SIZE = 8
```
There are 2 global variables which are QUANTUM and SIZE. QUANTUM is to quantize and dequantize the DCT coefficient, while SIZE is to determine the size of blocks we will turn into DCT coefficients at a time.

  - Reading The Image
```
jpg = "Lena.jpg"
img = cv2.imread(jpg)
```
Lena is the name of the image that we will read. We use the function imread to read the jpg image as a numpy array that we will manipulate. This numpy array is then saved in the variable "img".

  - Gray scaling The Image
```
r,g,b = img[:,:,0], img[:,:,1], img[:,:,2]
grayscale = r\*0.299 + g\*0.587 + b\*0.114
cv2.imshow('Original Grayscale',grayscale/255)
cv2.waitKey(0)
```

Creating a grayscale image from img. Although the original image is black and white but the formatting is still rgb so it consists of 3 2D arrays according to each colour channel. The image is then put through a formula to create a new grayscale image in the variable grayscale. We then show the grayscale image, dividing it by 255 because opencv's imshow function shows luminance by the range of 0~1 so we're formatting our image by 255 because our current format has a max value of 255.

  - DCT coefficient & Quantization
```
dct_koef = DCT(grayscale)
```
This piece of code is located in the execution part. It is the first code executed after showing the grayscale image. This function is defined as below:
```
defDCT(img):
  img = np.subtract(img,128) #in case it overflows...
  h,w = img.shape
  dct_koef = np.copy(img)
  block_y, block_x = int(h/8) , int(w/8)
  for y in range(block_y):
    for x in range(block_x):
      start_x,start_y = x*8,y*8
      end_x,end_y = (x+1)*8, (y+1)*8
      dct_koef[start_y:end_y,start_x:end_x] = DCTkoefiy(dct_koef[start_y:end_y,start_x:end_x],SIZE)
    print(f'{y+1}/{block_y} done dct')
  cv2.imshow('DCT koef',dct_koef/255)
  cv2.waitKey(0)
  
  for y in range(block_y):
    for x in range(block_x):
      start_x,start_y = x*8,y*8
      end_x,end_y = (x+1)*8, (y+1)*8
      dct_koef[start_y:end_y,start_x:end_x] =(dct_koef[start_y:end_y,start_x:end_x]/QUANTUM).astype(int)

    print(f'{y+1}/{block_y} done quantize')
  return dct_koef
```
Basically, this function takes in the numpy array of the grayscaled Image and first subtracts the values by 128 to avoid overflow during the calculation. Then we take the shape of the image, storing the height in variable h and width in variable w. then we create a new array to store our DCT coefficient which has the same dimensions as the grayscale image, so we just copy the grayscale image and put it in a new variable "dct\_koef". Then we need to divide the image to 8 by 8 blocks so we determine how many blocks we have for each x and y axis from dividing the width and height by 8 and store the values for each x and y in block\_y and block\_x.

Then we iterate the blocks by using the block\_y and block\_x variables as the upper bound of the loops. For each block, we need to determine it's starting and end point to pass to the function that would calculate it's DCT coefficient matrix for those 8 blocks. We save those starting and end points in the variables start\_x, start\_y, end\_x and end\_y. We use those variables to slice the dct\_koef array to pass it to another function called DCTkoefiy, which description can be found below.

After receiving the DCT coefficient result of all blocks, we show it since the question requires us to show the image after DCT transformation. Then we iterate the blocks again to quantize them by the QUANTUM array. This is done by dividing the values in each block by the QUANTUM array.
```
defC(u):
  if u == 0 :
    retur n1/sqrt(2)
  retur n1

defDCTkoefiy(img, SIZE):
  array = np.zeros((SIZE,SIZE))
  for y in range(SIZE):
    for x in range(SIZE):
      array[y][x] = 0
      for yy in range(SIZE):
        for xx in range(SIZE):
          array[y][x] += img[yy][xx] *
          cos(((2*yy+1)*y*pi)/(SIZE*2)) *
          cos(((2*xx+1)*x*pi)/(SIZE*2))
      array[y][x] *= C(y)*C(x)*(1/4)
  returnarray
```

Above is the DCTKoefiy function along with a C function to help calculate it's coefficient. The DCTKoefiy function takes in the parameters of 8x8 block to be transformed and the size of our blocks. First it creates an array full of 0s with the size of SIZE by SIZE. Then for each pixel of the DCT coefficient, we sum the values of all the pixel multiplied by a cosine formula for x and y/
For each DCT function we sum the value of the original image block's pixel and multiply it by the cosine functions which represent the DCT array pattern at u or x and v and y coordinate and how much it will contribute based on the current pixel which will be added by the other pixels to sum up the total contribution for that DCT pattern at that u or x and v or y coordinate.

C is defined by (source : https://cs.stanford.edu/people/eroberts/courses/soco/projects/data-compression/lossy/jpeg/Image15.jpg)

Which is implemented in the C function as written on the previous page.

  - dequantization & IDCT
```
idct_result = IDCT(dct_koef)
cv2.imshow('IDCT result',idct_result/255)
cv2.waitKey(0)
```
After the DCT has completed it's operation, we run the IDCT function to perform the inverse IDCT from the quantized image. Afterwards, we show the IDCT result. Below is the definition of the IDCT function.
```
defIDCT(img):
  h,w = img.shape
  idct_koef = np.copy(img)
  block_y, block_x = int(h/8) , int(w/8)
  for y in range(block_y):
    for x in range(block_x):
      start_x,start_y_ = x*8,y*8
      end_x,end_y = (x+1)*8, (y+1)*8
      idct\_koef[start_y:end_y,start_x:end_x] = IDCTkoefiy(idct_koef[start_y:end_y,start_x:end_x],SIZE)
    print(f'{y+1}/{block_y} done dct')
  idct_koef = np.add(idct_koef,128)
  return idct_koef
```
The IDCT function takes in the quantized image and iterates it by 8x8 block in the same manner as the DCT function, but then it takes each block and processes it within a function named IDCTKoefiy. Then it adds the whole array by 128 to revert the overflow prevention. Afterwards we return the transformed image.
```
defIDCTkoefiy(img, SIZE):
  img = img*QUANTUM
  array = np.zeros((SIZE,SIZE))
  for y in range(SIZE):
    for x in range(SIZE):
      array[y][x] = 0
      for yy in range(SIZE):
        for xx in range(SIZE):
          array[y][x] += img[yy][xx] * cos(((2\*y+1)\*yy\*pi)/(SIZE\*2)) * cos(((2\*x+1)\*xx\*pi)/(SIZE\*2)) *C(yy)*C(xx)
      array[y][x] *= (1/4)
  array = array.astype(int)

  return array
```
The IDCTKoefiy function works similarly to the DCTKoefiy function but before that, we need to dequantize the block by multiplying it by QUANTUM. Then we create an empty array by the size of the block to store the result of IDCT. We then perform IDCT according to the formula previously shown which uses the DCT coefficients on the input image and multiplies it by the coordinate of the DCT pattern array which is dictated by the cosine functions. Then we return the result decompressed image.

  - Error Image
```
error = np.absolute(grayscale - idct_result)
cv2.imshow('Error',error/255)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
After showing the IDCT image, we can get the error image by simply calculating the absolute differences between the original grayscale image and idct decompression result. Then we show it and destroy all windows.

- PSNR
```
print(f'psnr is {PSNR(grayscale,idct_result)}')
```
The psnr is called after showing the error image. This is done by calling the PSNR function while passing in the original grayscale image and the idct result. The PSNR function is as follows
```
defPSNR(ori,orint):
  mse = 0
  h,w = ori.shape

  forhhinrange(h):
    forwwinrange(w):
      mse += (ori[hh][ww]-orint[hh][ww])**2
  mse *= 1/(h*w)
  psnr = 10 * log(255**2/mse)
  returnpsnr
```
The function takes in the original image and the decompressed image. Then it first calculates it's mse by iterating the image to sum the squared difference of each pixel of the original grayscale image and the decompressed result. Then the mse is divided by the total number of pixels.
</ol>
