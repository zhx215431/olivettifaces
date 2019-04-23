import os
import sys
import time
import tensorflow as tf
import numpy
from PIL import Image
origin_picture_path = "E:/study/DL/HJFaceRecognition/olivettifaces/olivettifaces.gif"

#将数字转换成向量 如1->(0,1,0,....,0),4->(0,0,0,0,1,0,0,...,0)
def label_transformer(number, setrange):
    label_array = [];
    for i in range(setrange):
        if i == number:
            label_array.append(1)
        else:
            label_array.append(0)
    return label_array

class olivettifaces_bulider:
    '''
    @param train_data 测试集数据
    @param train_label 测试集标签

    @param valid_data 验证集数据
    @param valid_label 验证集标签

    @param test_data 测试集数据
    @param test_label 测试集标签
    '''
    def __init__(self):
        self.train_data = []
        self.train_label = []
        self.valid_data = []
        self.valid_label = []
        self.test_data = []
        self.test_label = []
        self.load_data(dataset_path=origin_picture_path)


    def load_data(self,dataset_path):
        img = Image.open(dataset_path)
        img_ndarray = numpy.asarray(img, dtype='float64')/256
        faces=numpy.empty((400,2679))
        for row in range(20):
            for column in range(20):
                faces[row*20+column]=numpy.ndarray.flatten(img_ndarray [row*57:(row+1)*57,column*47:(column+1)*47])

        label=numpy.empty(400)
        for i in range(40):
            label[i*10:i*10+10]=i
        label=label.astype(numpy.int)

        #分成训练集、验证集、测试集，大小如下
        train_data=numpy.empty((320,2679))
        train_label=numpy.empty(320)
        valid_data=numpy.empty((40,2679))
        valid_label=numpy.empty(40)
        test_data=numpy.empty((40,2679))
        test_label=numpy.empty(40)

        for i in range(40):
            train_data[i*8:i*8+8]=faces[i*10:i*10+8]
            train_label[i*8:i*8+8]=label[i*10:i*10+8]
            valid_data[i]=faces[i*10+8]
            valid_label[i]=label[i*10+8]
            test_data[i]=faces[i*10+9]
            test_label[i]=label[i*10+9]

        self.train_data = train_data
        self.train_label = train_label
        self.valid_data = valid_data
        self.valid_label = valid_label
        self.test_data = test_data
        self.test_label = test_label

    def next_batch_image(self, training_count):
        whileCount = 0
        random_list = []
        batch_image_list = []
        batch_label_list = []
        while (whileCount < training_count):
            rnd = numpy.random.randint(1,320)
            if rnd not in random_list:
                random_list.append(rnd)
                batch_image_list.append(self.train_data[rnd - 1])
                batch_label_list.append(label_transformer(number=self.train_label[rnd - 1],setrange=40))
                whileCount = whileCount + 1
        return batch_image_list, batch_label_list

    def test_batch_image(self,test_count):
        pass
