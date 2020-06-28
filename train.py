# Copyright 2019 Doyoung Gwak (tucan.dev@gmail.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================
#-*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import keras
# tensorflow에 있는 keras를 사용하면 coreml로 변환시에 Unknown initializer: GlorotUniform 에러가 발생한다
# import tensorflow as tf
# from tensorflow import keras
from keras.initializers import glorot_uniform

import numpy as np
# import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

#load_data() 함수를 호출하면 네 개의 넘파이(NumPy) 배열이 반환된다.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

len(train_labels)

train_images.shape

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images = train_images / 255.0
test_images = test_images / 255.0


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    keras.layers.Dense(128, activation='relu'), # 128개의 노드 or 뉴런을 가진다.
    # 10개의 소프트맥스 층인데 각 노드는 이미지가 10개 클래스 중 하나 속할 확률을 출력한다.
    # 10개의 확률을 반환하고 반환된 값의 전체 합은 1이다. (응축)
    keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
# 올바르게 분류된 이미지의 비율인 정확도를 사용한다.

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\n 테스트 정확도:', test_acc)

from keras.models import load_model

model.save('fashion-MNIST-model-hyun.h5')