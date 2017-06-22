from PIL import Image
import numpy as np

def convert_to_linear_array(img):
	mas = []
	nx, ny = img.size
	p = img.load()
	for y in range(0, ny):
		for x in range(0, nx):
			val = sum(p[x,y])/(255*3.0)
			mas.append(val)
	return np.array(mas, dtype=np.float32)

