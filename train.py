

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Dropout,Flatten,Dense,Activation,BatchNormalization,add
from tensorflow.keras.models  import Model,Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
import os



base_dir = os.getcwd() 

target_shape = (224,224) 
train_dir = base_dir+"\\chest_xray\\train"  
val_dir = base_dir+"\\chest_xray\\val"     
test_dir = base_dir+"\\chest_xray\\test"   


vgg = VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))
for layer in vgg.layers:
    layer.trainable = False 
    
x = Flatten()(vgg.output) 
predictions = Dense(2,activation='softmax')(x) 
model = Model(inputs=vgg.input, outputs=predictions)
model.summary()



train_gen = ImageDataGenerator(rescale=1/255.0,
                               horizontal_flip=True,
                               zoom_range=0.2,
                               shear_range=0.2) 
test_gen = ImageDataGenerator(rescale=1/255.0) 

train_data_gen = train_gen.flow_from_directory(train_dir,
                                               target_shape,
                                               batch_size=16,
                                               class_mode='categorical') 
test_data_gen = train_gen.flow_from_directory(test_dir,
                                               target_shape,
                                               batch_size=16,
                                               class_mode='categorical') 
plot_model(model, to_file='model.png')

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = model.fit_generator(train_data_gen,
        steps_per_epoch=20,
        epochs=20,
        validation_data=test_data_gen,
        validation_steps=10)

plt.style.use("ggplot")
plt.figure()
plt.plot(hist.history["loss"], label="train_loss")
plt.plot(hist.history["val_loss"], label="val_loss")
plt.plot(hist.history["accuracy"], label="train_acc")
plt.plot(hist.history["val_accuracy"], label="val_acc")
plt.title("Model Training")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("epochs.png")

model.save('model.h5')
