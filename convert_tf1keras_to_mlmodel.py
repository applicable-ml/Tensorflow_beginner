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

from keras.models import load_model
import coremltools

model = load_model("fashion-MNIST-model-hyun.h5")
model.summary()

print(f"model.input.name: {model.input.name}")
print(f"model.output.name: {model.output.name}")


output_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
scale = 1/255
fashion_model = coremltools.converters.keras.convert("fashion-MNIST-model-hyun.h5",
                                                     input_names='flatten_input_0',
                                                     image_input_names=['flatten_input_0'],
                                                     output_names='dense_1',
                                                     class_labels=output_labels,
                                                     image_scale=scale)

fashion_model.author = 'hyunable'
# fashion_model.input_description['image'] = 'Grayscale image of hand written digit'
# fashion_model.output_description['output'] = 'Predicted digit'
fashion_model.save('fashion-MNIST-model-hyunable.mlmodel')

print(fashion_model)
