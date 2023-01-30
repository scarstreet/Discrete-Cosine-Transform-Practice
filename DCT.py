import cv2
import numpy as np
import sys
from math import cos,pi,sqrt,log

QUANTUM = np.array([ # source = https://en.wikipedia.org/wiki/Quantization_(image_processing)
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

# Read, make grayscale ================================================
jpg = "Lena.jpg"
img = cv2.imread(jpg)
r,g,b = img[:,:,0], img[:,:,1], img[:,:,2]
grayscale = r*0.299 + g*0.587 + b*0.114
cv2.imshow('Original Grayscale',grayscale/255)
cv2.waitKey(0)

#functions ============================================================
def C(u):
  if u == 0 :
    return 1/sqrt(2)
  return 1

def DCTkoefiy(img, SIZE):
  array = np.zeros((SIZE,SIZE))
  for y in range(SIZE):
    for x in range(SIZE):
      array[y][x] = 0
      for yy in range(SIZE):
        for xx in range(SIZE):
          array[y][x] += img[yy][xx] * cos(((2*yy+1)*y*pi)/(SIZE*2)) * cos(((2*xx+1)*x*pi)/(SIZE*2))
      array[y][x] *= C(y)*C(x)*(1/4)
  return array

def IDCTkoefiy(img, SIZE):
  img = img*QUANTUM
  array = np.zeros((SIZE,SIZE))
  for y in range(SIZE):
    for x in range(SIZE):
      array[y][x] = 0
      for yy in range(SIZE):
        for xx in range(SIZE):
          array[y][x] += img[yy][xx] * cos(((2*y+1)*yy*pi)/(SIZE*2)) * cos(((2*x+1)*xx*pi)/(SIZE*2)) *C(yy)*C(xx)
      array[y][x] *= (1/4)
  array = array.astype(int)
  return array

def DCT(img):
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
      dct_koef[start_y:end_y,start_x:end_x] = (dct_koef[start_y:end_y,start_x:end_x]/QUANTUM).astype(int)
    print(f'{y+1}/{block_y} done quantize')
  print(dct_koef)
  return dct_koef

def IDCT(img):
  h,w = img.shape
  idct_koef = np.copy(img)
  block_y, block_x = int(h/8) , int(w/8)
  for y in range(block_y):
    for x in range(block_x):
      start_x,start_y = x*8,y*8
      end_x,end_y = (x+1)*8, (y+1)*8
      idct_koef[start_y:end_y,start_x:end_x] = IDCTkoefiy(idct_koef[start_y:end_y,start_x:end_x],SIZE)
    print(f'{y+1}/{block_y} done idct')
  idct_koef = np.add(idct_koef,128)
  print('done quantize')
  return idct_koef

def PSNR(ori,orint):
  mse = 0
  h,w = ori.shape
  for hh in range(h):
    for ww in range(w):
      mse += (ori[hh][ww]-orint[hh][ww])**2
  mse *= 1/(h*w)
  psnr = 10 * log(255**2/mse)
  return psnr
# exectution ===================================================================================

dct_koef = DCT(grayscale)

idct_result = IDCT(dct_koef)
cv2.imshow('IDCT result',idct_result/255)
cv2.waitKey(0)

error = np.absolute(grayscale - idct_result)
cv2.imshow('Error',error/255)
cv2.waitKey(0)

print(f'psnr is {PSNR(grayscale,idct_result)}')
cv2.destroyAllWindows()