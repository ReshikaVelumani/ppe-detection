import requests
from PIL import Image
import numpy as np

#test from GCR
# resp = requests.post("https://ppe-detection-wx2bjo7cia-uc.a.run.app", files={'file': open('data/example-21.jpg', 'rb')})
#test locally on image built on external drive
resp = requests.post("http://0.0.0.0:5000", files={'file': open('data/example-21.jpg', 'rb')})

# print(resp.json())

# Convert the pixels into an array using numpy
array = np.array(resp.json(), dtype=np.uint8)
print(array.shape)

# Use PIL to create an image from the new array of pixels
new_image = Image.fromarray(array)
new_image.save('new.png')
