import numpy as np
import os
import json
# import PIL
# import PIL.Image
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
# import tensorflow_datasets as tfds
import numpy as np
from datetime import datetime
from keras.utils import load_img, img_to_array, plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
dir = os.path.dirname(os.path.realpath(__file__))

class TensorFlowKerasHelper():
    def __init__(self, batchSize = 32, imgHeight = 256, imgWidth = 256):
    # def __init__(self, batchSize = 32, imgHeight = 224, imgWidth = 224):
    # def __init__(self, batchSize = 32, imgHeight = 299, imgWidth = 299):
          self.batchSize = batchSize
          self.imgHeight = imgHeight
          self.imgWidth = imgWidth
        
          self.epochs = None
          self.model = None
          self.valDataset = None
          self.trainDataset = None
          self.classNames = None
          self.numClass = None
          self.optimizer= None
          self.nameModel = None
          

    def datasetPath(self, dirDatasetTrain = r'D:\Projects_Python\Prototipo-Modelo-de-reconhecimento-de-gestos\bsl\TestCam_MEDIAPIPE_CUT\train',\
                     dirDatasetVal = r'D:\Projects_Python\Prototipo-Modelo-de-reconhecimento-de-gestos\bsl\TestCam_MEDIAPIPE_CUT\val' ):
        
      self.trainDataset = tf.keras.utils.image_dataset_from_directory(
        dirDatasetTrain,
        validation_split=0.2, # 80%
        subset="training",
        seed=123,
        image_size=(self.imgHeight, self.imgWidth),
        batch_size=self.batchSize)

      self.valDataset = tf.keras.utils.image_dataset_from_directory(
        dirDatasetTrain,
        validation_split=0.2, # 10 %
        subset="validation",
        seed=123,
        image_size=(self.imgHeight, self.imgWidth),
        batch_size=self.batchSize)

      self.classNames = self.valDataset.class_names
      print(self.classNames)
      self.numClass = len(self.classNames)

    def createModel(self, typeModel = 'Sequential', optimizer='adam'):
      self.typeModel = typeModel
      self.optimizer = optimizer
      #https://www.tensorflow.org/tutorials/images/classification?hl=pt-br
      if self.typeModel == 'Sequential':
          
          self.model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.numClass, activation='softmax')
          ])
          self.model.compile(
            optimizer=self.optimizer,
            # loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
            # loss='mse', # horrivel -> modelTensorFlow_TestCam_MEDIAPIPE_CUT-loss='mse'_23-4_epochs-1_size-(256,256)_typeModel-Sequential_optimizer-adam_classNames-21_Count-1
            metrics=['accuracy'])

      elif self.typeModel == 'MobileNetV2':
          # WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not in [96, 128, 160, 192, 224]. Weights for input shape (224, 224) will be loaded as the default.
          output_layers = [
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(self.numClass, activation='softmax')
          ]

          model = tf.keras.applications.mobilenet_v2.MobileNetV2(
              input_shape=(self.imgHeight, self.imgWidth, 3),
              alpha=1.0,
              include_top=False,
              weights='imagenet',
              pooling=None,
              classes=self.numClass,
              classifier_activation=self.optimizer
          )
          output = model.output
          for layer in output_layers:
              output = layer(output)

          self.model = tf.keras.Model(inputs=model.input, outputs=output)
         

          self.model.compile(
            optimizer=self.optimizer,
            # loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
            # loss='mse', # horrivel -> modelTensorFlow_TestCam_MEDIAPIPE_CUT-loss='mse'_23-4_epochs-1_size-(256,256)_typeModel-Sequential_optimizer-adam_classNames-21_Count-1
            metrics=['accuracy'])
          
      elif self.typeModel == 'InceptionV3':
        
            base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(self.imgHeight, self.imgWidth, 3))

            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(1024, activation='relu')(x)
            predictions = tf.keras.layers.Dense(self.numClass, activation='softmax')(x)

            self.model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

            self.model.compile(optimizer='adam', 
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
      
      return self.model
          
    def trainModel(self, epochs = 5):
        print(f'Método trainModel')
        self.epochs = epochs
        now = datetime.now()
        self.hourBegginTrain = f'{now.day}-{now.month}-{now.year}-{now.hour}-{now.minute}-{now.second}'
        print(f'Total de épocas {self.epochs}')
        # self.history = self.model.fit(
        #             self.trainDataset,
        #             validation_data=self.valDataset,
        #             epochs=self.epochs)

        self.history = self.model.fit(
            self.trainDataset,
            # steps_per_epoch = 8000,
            validation_data=self.valDataset, 
            validation_steps=len(self.trainDataset)// self.batchSize,
            shuffle=True,
            epochs=self.epochs,
            use_multiprocessing=True,
        )
        
        now = datetime.now()
        self.hourEndTrain = f'{now.day}-{now.month}-{now.year}-{now.hour}-{now.minute}-{now.second}'
        return self.history
    
    def reTrainModel(self, epochs = 5):
        print(f'Método reTrainModel')
        self.epochs = epochs
        now = datetime.now()
        self.hourBegginTrain = f'{now.day}-{now.month}-{now.year}-{now.hour}-{now.minute}-{now.second}'
        print(f'Total de épocas {self.epochs}')
        for layer in self.model.layers:
          layer.trainable = False

        self.model = tf.keras.Sequential([
            self.model,
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.numClass, activation='softmax')
        ])

        self.model.compile(
            optimizer=self.optimizer,
            # loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
            # loss='mse', # horrivel -> modelTensorFlow_TestCam_MEDIAPIPE_CUT-loss='mse'_23-4_epochs-1_size-(256,256)_typeModel-Sequential_optimizer-adam_classNames-21_Count-1
            metrics=['accuracy'])

        self.history = self.model.fit(
            self.trainDataset,
            validation_data=self.trainDataset, 
            validation_steps=len(self.trainDataset)// self.batchSize,
            shuffle=True,
            epochs=self.epochs,
            use_multiprocessing=True,
        )
        
        now = datetime.now()
        self.hourEndTrain = f'{now.day}-{now.month}-{now.year}-{now.hour}-{now.minute}-{now.second}'
        return self.history
    
    def saveModel(self, path = '.\\', datasetNameModelComplement = ''):
        now = datetime.now()
        self.nameModel = f'modelTensorFlow_{datasetNameModelComplement}_{now.day}-{now.month}_epochs-{self.epochs}_size-({self.imgHeight},{self.imgWidth})_typeModel-{self.typeModel}_optimizer-{self.optimizer}_classNames-{len(self.classNames)}'
        count =  1
        
       
        while True:
          pathSaveDataModel = f'{path}\\{ self.nameModel}_Count-{count}.h5'
          if os.path.isfile(pathSaveDataModel):
             count +=1
          else:
              break
        print(f'Salvando o modelo em {pathSaveDataModel}')
        self.model.save(f'{pathSaveDataModel}')

    def loadModel(self, path = f'{dir}', nameModel = 'Test_Model_TensorFlow', typeModel=None, optimizer=None):
        self.model = load_model(f'{path}\\{nameModel}.h5')
        self.typeModel = typeModel
        self.optimizer =  optimizer

    def saveDataModel(self, path):
        dataModel = dict()
        dataModel = { 'imgHeight' : self.imgHeight ,\
                      'imgWidth' : self.imgWidth ,\
                      'classNames' : self.classNames ,\
                      'lenTrainDataset' : len(self.trainDataset) ,\
                      'lenValDataset' : len(self.valDataset) ,\
                      'epochs' : self.epochs, \
                      'typeModel': self.typeModel,\
                      'optimizer' : self.optimizer ,\
                      'modelAccuracy': {'accuracy' : self.history.history['accuracy'], 'val_accuracy' : self.history.history['val_accuracy'], \
                                        'title' :'Precisão do Modelo', 'ylabel' : 'Perda',  'xlabel' : 'Épocas',  'legend' : '["Treino", "Teste"]'},\
                      'modelLoss': {'loss' : self.history.history['loss'], 'val_loss' : self.history.history['val_loss'], \
                                        'title' :'Perda do Modelo', 'ylabel' : 'Perda',  'xlabel' : 'Épocas',  'legend' : '["Treino", "Teste"]'},\
                      'self.hourBegginTrain' : self.hourBegginTrain, \
                      'self.hourEndTrain' : self.hourEndTrain}
        
        count =  1
        
        while True:
          pathSaveDataModel = f'{path}\\{self.nameModel}_Count-{count}_dataModel.json'
          if os.path.isfile(pathSaveDataModel):
             count +=1
          else:
             break
        print(f'Salvando o arquivo em {pathSaveDataModel}')
        with open(f'{pathSaveDataModel}', 'w') as json_file:
          json.dump(dataModel, json_file, indent=2)

    def predictor(self, imagePath, classNames=None, imgHeight=None, imgWidth=None): 
      if self.classNames == None:
          self.classNames = classNames
          self.imgHeight = imgHeight
          self.imgWidth =imgWidth
      image = imagePath      
      image = load_img(image, target_size = (self.imgHeight,  self.imgWidth))
      image = img_to_array(image)
      image = np.expand_dims(image, axis = 0)
      result = self.model.predict(image)
      y_classes = result.argmax(axis=-1)
    
      print(f'{self.classNames[y_classes[0]]}')
      print(f'{result.max()}')
      return [result, self.classNames[y_classes[0]]]
        



# classifier = load_model('../models/cnn_model_LIBRAS_20190531_0135.h5')
# classifier = load_model(f'../models/cnn_model_LIBRAS_20190606_0106.h5')
# model = load_model(f'{dir}\modeloTensorFlow-19-04_4_TestCam_MEDIAPIPE_CUT_ESPELHO_FULL_ALFABETO_256-256-epocas-3-num_classes_21.h5')


# https://jobu.com.br/2021/07/15/tutorial-do-tensorflow-2/
# configure early stopping
# es = EarlyStopping(monitor='val_loss', patience=5)
# fit the model
# history = model.fit(self.trainDataset, validation_data=self.valDataset, epochs=epocas, batchSize=32, verbose=1, callbacks=[es])

# Original
# history = model.fit(
#           self.trainDataset,
#           validation_data=self.valDataset,
#           epochs=epocas)


# #Gráficos de avaliação de resultado
# import matplotlib.pyplot as plt

# # Renderização de gráfico de acuracia
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Precisão do Modelo')
# plt.ylabel('Precisão')
# plt.xlabel('Época')
# plt.legend(['Treino', 'Teste'], loc='upper left')
# plt.show()

# # Renderização de gráfico de perda
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Perda do Modelo')
# plt.ylabel('Perda')
# plt.xlabel('Épocas')
# plt.legend(['Treino', 'Teste'], loc='upper left')
# plt.show()

# nameModel = f'TESTE{self.imgHeight}-{imgWidth}-epocas-{epocas}-num_classes_{num_classes}'
# dataModel = dict()
# dataModel = {'modelAccuracy': {'accuracy' : history.history['accuracy'], 'val_accuracy' : history.history['val_accuracy'], \
#                                 'title' :'Precisão do Modelo', 'ylabel' : 'Perda',  'xlabel' : 'Épocas',  'legend' : '["Treino", "Teste"]'},\
#             'modelLoss': {'loss' : history.history['loss'], 'val_loss' : history.history['val_loss'], \
#                                 'title' :'Perda do Modelo', 'ylabel' : 'Perda',  'xlabel' : 'Épocas',  'legend' : '["Treino", "Teste"]'}}

# with open(f"{nameModel}_dataModel.json", "w") as json_file:
#   json.dump(dataModel, json_file, indent=2)

# # Salvando o modelo
# model_json = model.to_json()
# with open(f"{nameModel}_model_json.json", "w") as json_file:
#     json_file.write(model_json)

# # model.save(f'modeloTensorFlow-19-04_4_TestCam_MEDIAPIPE_CUT_ESPELHO_FULL_ALFABETO_2_{self.imgHeight}-{imgWidth}-epocas-{epocas}-num_classes_{num_classes}.h5')
# nameModel =f'modeloTensorFlow-19-04_4_TestCam_MEDIAPIPE_CUT_ESPELHO_FULL_ALFABETO_2_{self.imgHeight}-{imgWidth}-epocas-{epocas}-num_classes_{num_classes}.h5'
# # model.save(f'{nameModel}.h5')
# model = load_model(nameModel)

# plot_model(model, 'model.png', show_shapes=True)

# # VALIDAÇÂO DO MODELO 
# loss = model.evaluate(self.valDataset, verbose=1)
# print(loss)

# # A = f'{dirDatasetTrain}/A/001.jpg'
# # B = f'{dirDatasetTrain}/B/001.jpg'
# # C = f'{dirDatasetTrain}/C/001.jpg'
# # D = f'{dirDatasetTrain}/D/001.jpg'
# # E = f'{dirDatasetTrain}/E/001.jpg'
# # # test =  f'{dirDatasetTrain}/001.jpg'

# # nomes_de_classificacoes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
# # nomes_de_classificacoes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']


# # from keras.utils import load_img, img_to_array

# # # Segunda Imagem
# # def test(test):
# #     test_image = load_img(test, target_size = (64, 64))
# #     test_image = img_to_array(test_image)
# #     test_image = np.expand_dims(test_image, axis = 0)
# #     result = model.predict(test_image)
# #     y_classes = result.argmax(axis=-1)
# #     print(nomes_de_classificacoes[y_classes[0]])

# print('Fim')

