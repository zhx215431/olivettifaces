import loadData
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def show_tensor_to_image(tensor):
    plt.imshow(tensor)
    plt.show()

bulider = loadData.olivettifaces_bulider()
for i in range(len(bulider.train_data)):
    img = bulider.train_data[i].reshape(57,47)
    show_tensor_to_image(img)
