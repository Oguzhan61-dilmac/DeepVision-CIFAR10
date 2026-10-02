#veri setini içeriye aktar ve preprocessing : normalizasyon , one hot encoding , train test split
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

#cfar10 veri setini yükle
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


#veri setini normalizasyon yap
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
#görselleştirme yap
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
#bazi görünütleri ve etiketleri görselleştir 
fig, axes = plt.subplots(1, 5, figsize=(15, 10))
for i in range(5):
    axes[i].imshow(x_train[i])
    ladel = class_names[y_train[i][0]]
    axes[i].set_title(ladel)
    axes[i].axis('off')
plt.show()
#one hot encoding yap
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


#veri arttimi (data augmentation) yap
datagen = ImageDataGenerator (
     rotation_range=20 ,  # 20 derreceye kadar döndürme
     width_shift_range=0.2,# 20% kadar yatay kaydırma
     height_shift_range=0.2 ,# 20% kadar dikey kaydırma
     shear_range=0.2,# 20% kadar kesme
     zoom_range=0.2,# 20% kadar yakınlaştırma
     horizontal_flip=True,# yatay çevirme
     fill_mode='nearest'# boş alanları doldurma yöntemi
)
datagen.fit(x_train)#data augmentation için fit etme
#create , compile and train the model

# cnn modeli oluştur (base model )
model= Sequential()
# feature extraction   conv =>relu =>conv =>relu =>pool =>dropout 
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# feature extraction   conv =>relu =>conv =>relu =>pool =>dropout 
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

#classification  flatten => dense => dropout => dense => softmax
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()
#modeli derle
model.compile(optimizer=RMSprop(learning_rate=0.001 , decay=1e-6),
               loss='categorical_crossentropy', metrics=['accuracy'])

#model eğitimi
model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=50 #eğitim dönem sayısı
          , validation_data=(x_test, y_test)#doğrulama verisi olarak test verisini kullan
          , verbose=1)
#test the model and evaluate the performance