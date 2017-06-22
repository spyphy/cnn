# Third-party libraries
import math
from math import sqrt
import numpy as np
import theano
import theano.tensor as T
from PIL import Image, ImageDraw

def show_image(digit):
	size = int(math.sqrt(len(digit)))
	nx, ny = (size, size)
	img = Image.new("RGBA", (2*nx, 2*ny), (0,0,0,0))
	draw = ImageDraw.Draw(img)
	for x in range(nx):
		for y in range(ny):
			S = 256 - int(digit[y*nx + x]*256)
			draw.point((2*x, 2*y), (S, S, S))
			draw.point((2*x, 2*y+1), (S, S, S))
			draw.point((2*x+1, 2*y), (S, S, S))
			draw.point((2*x+1, 2*y+1), (S, S, S))
	del draw
	img.show()
	
def save_image_png(digit):
	size = int(math.sqrt(len(digit)))
	nx, ny = (size, size)
	img = Image.new("RGBA", (nx, ny), (0,0,0,0))
	draw = ImageDraw.Draw(img)
	for x in range(nx):
		for y in range(ny):
			S = int(digit[y*nx + x]*256)
			draw.point((x, y), (S, S, S))
	del draw
	img.save("out.png", "PNG")
