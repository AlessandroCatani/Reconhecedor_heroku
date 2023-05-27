import cv2
from flask import Flask, render_template, Response, jsonify, session, redirect, request
text = ''
app = Flask(__name__, static_folder='static')

# import tensorflow as tf
# import tensorflow_datasets as tfds
# import numpy as np
# from keras.utils import load_img, img_to_array
# from tensorflow.keras.models import load_model

from utl.ImageHelper import ImageHelper
from utl.File_Helper import FileHelper
# from utl.TensorFlowKerasHelper import TensorFlowKerasHelper
# from utl.HandDetectorHelper import HandDetectorHelper
import os
import time
import random

dir = os.path.dirname(os.path.realpath(__file__))
camera = cv2.VideoCapture(0)
# detector = HandDetectorHelper()
# model = TensorFlowKerasHelper()

confiJson = FileHelper.readJson(f'{dir}\\config.json')
pathModel = f'{dir}\\{confiJson["PATHMODEL"]}'
nameModel = f'{confiJson["NAMEMODEL"]}'
mode =  f'{confiJson["MODE"]}'
pathImagePredictor = f'{confiJson["PATHIMAGEPREDICTOR"]}' 

confiModelJson = FileHelper.readJson(f'{confiJson["PATHCONFIGMODEL"]}\\{nameModel}_dataModel.json')
classNames = confiModelJson.get('classNames')
imgHeight = confiModelJson.get('imgHeight')
imgWidth = confiModelJson.get('imgWidth')
# model.loadModel(path = f'{dir}\\models\\', nameModel = 'modelTensorFlow_bsl-10_05_2023_New_Dataset_TestCamAlessandroMEDIAPIPE_CUT_10-5_epochs-10_size-(224,224)_typeModel-Sequential_optimizer-adam_classNames-21_Count-1')

countText = 0
countEspera = 0
count = 0
textoAux = ''
texto = ''

palavras = [eval(f'{confiJson["PALAVRAS"]}')]

def process_frame(frame, mode='frame'):
    global text, countText, countEspera, count, textoAux
    # Aplica um filtro na imagem
    frame.flags.writeable = True
    frameInfo = frame.copy()
    frameBlack = np.zeros(frame.shape, dtype=np.uint8)
    framePredictor = frameBlack.copy()
    
    points, coordinates_x_min, coordinates_y_min, coordinates_x_max, coordinates_y_max = detector.findPointsHands(frame)
    if coordinates_x_min:
        frameInfo = detector.markPointsHands(frameInfo, True, points)
        frameBlack = detector.markPointsHands(frameBlack, True, points)
        frameInfo = detector.markRoiHands(frameInfo, coordinates_x_min, coordinates_y_min, coordinates_x_max, coordinates_y_max)

        framePredictor = frameBlack[coordinates_y_min-5 : coordinates_y_max+5, coordinates_x_min-5 :coordinates_x_max+5]
        if framePredictor.size:
                img_name = f'{dir}\\efs\\tmp\\img.jpg'     
                cv2.imwrite(img_name, framePredictor)
                ret = model.predictor(img_name, classNames, imgHeight, imgWidth)
                texto = ret[-1]
                # frame = ImageHelper.textToImage(frame, texto)
                pontoCorte = 0.95
                if ret[0].max() < pontoCorte :
                    texto = ''
                    print(f' Reuslt {ret[0].max()} menor que {pontoCorte}')
                else :
                    print(f'texto {texto} - textoAux {textoAux} - counText{countText}')
                    if texto != textoAux:
                        textoAux = texto
                        countText = 1
                        texto = ''
                    elif texto == textoAux and countText < 5:
                        countText +=1
                        textoAux = texto
                        texto = ''
                    elif countText >= 5 and textoAux == texto:
                        textoAux = texto
        else:
            texto = ''
            countText = 0
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       
        # frameInfo = ImageHelper.textToImage(frameInfo, texto)
    else:
        texto = ''
    text = texto
    return frameInfo
 
def gen_frames():
    while True:
        success, frame = camera.read()  # Captura o frame da webcam
        if not success:
            # time.sleep(2)
            # continue
            break
        # processed_frame = process_frame(frame, mode)  # Processa o frame
        # print(f"{texto}")
    
        ret, buffer = cv2.imencode('.jpg', frame)  # Codifica o frame em formato jpeg
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: frame/jpeg\r\n\r\n' + frame + b'\r\n')
      
def letra_color(letra):
    if letra == 'A':
        return (235, 231, 39)
    else:
        return (139, 139, 139) 
    

def escolher_palavra(copyPalavras):
    word = random.choice(copyPalavras).upper()
    return word
    
def createDictWord(word):
    print(f'Limpando o dictWord {dictWord}')
    dictWord = {}
    for count, word in enumerate(word, 0):
        if count == 0 :
            status = 'wait'
        else:
            status = 'NOk'

        test = {count:{'word':{word},'status':{status}}}
        dictWord.update(test)
    return dictWord
   
def functionWord():
    global dictWord
    global palavras
    global copyPalavras
    global word
    if word == '':
        if len(copyPalavras) == 0: 
            copyPalavras = palavras.copy()
        word = escolher_palavra(copyPalavras)
        copyPalavras.remove(word)
        return word


def verify(dictWord, text):
    for key in dictWord:
        if dictWord.get(key).get('status') == 'wait':
            if dictWord.get(key).get('word') == text :
                dictWord[key]['status'] = 'Ok'
                if len(dictWord)< dictWord.get(key)+1:
                    dictWord[key]['status'] = 'wait'
                    break
                # Verificar quando acabar
def enumerate_func(iterable):
    return zip(range(len(iterable)), iterable)

palavras = ['gato','bala', 'bolo', 'agua', ]
letra_amarela = 0
beggin = 0
palavra_aleatoria = random.choice(palavras)
count = 0


@app.route('/index.html', methods=['GET', 'POST'])
def index():
    print(f'****************** METODO  index *********************')
    global letra_amarela
    global palavra_aleatoria
    global palavras

    # palavra_aleatoria = 'FELICIDADE'

    # letra_amarela = 0
    # app.jinja_env.globals['enumerate'] = enumerate_func
    # # Lógica para reconhecimento do gesto
    # if request.method == 'POST':
    #     gesto_reconhecido = request.form.get('gesto_reconhecido')
        
    #     if gesto_reconhecido == palavra_aleatoria[letra_amarela]:
    #         letra_amarela += 1

    #         # Verifique se todas as letras foram reconhecidas corretamente
    #         if letra_amarela >= len(palavra_aleatoria):
    #             # Todas as letras foram reconhecidas corretamente, reinicie o contador
    #             letra_amarela = 0
    
    # return render_template('index.html', palavra_aleatoria=palavra_aleatoria, letra_amarela=letra_amarela)
    return render_template('index.html')


@app.route('/jogoDePalavras.html', methods=['GET', 'POST'])
def jogoDePalavras():
    print(f'****************** METODO  jogoDePalavras *********************')
    global letra_amarela
    global palavra_aleatoria
    global palavras

    letra_amarela = 0
    # palavra_aleatoria = random.choice(palavras)
    
    app.jinja_env.globals['enumerate'] = enumerate_func

    print(f'palavra_aleatoria -> {palavra_aleatoria}')
    print(f'palavra_aleatoria[letra_amarela] -> {palavra_aleatoria[letra_amarela].upper()}')
    
    return render_template('jogoDePalavras.html', palavra_aleatoria=palavra_aleatoria, letra_amarela=letra_amarela)

@app.route('/atualizar_palavra', methods=['POST'])
def atualizar_palavra():
    print(f'****************** METODO  atualizar_palavra *********************')
    global palavra_aleatoria
    global letra_amarela

    # Atualize os valores da palavra aleatória e letra amarela
    palavra_aleatoria = random.choice(palavras)
    letra_amarela = 0

    print(f'palavra_aleatoria -> {palavra_aleatoria}')
    print(f'palavra_aleatoria[letra_amarela] -> {palavra_aleatoria[letra_amarela].upper()}')
    
    return jsonify(palavra_aleatoria=palavra_aleatoria, letra_amarela=letra_amarela, letraAtual=palavra_aleatoria[letra_amarela].upper())


# @app.route('/gesto_reconhecido')
# def gesto_reconhecido():
#     global letra_amarela
#     global palavra_aleatoria
   
#     gesto_reconhecido = text# Seu código para reconhecimento do gesto

#     if gesto_reconhecido == text:
            
#         # Atualizar a posição da letra em amarelo
#         letra_amarela += 1

#         # Verificar se todas as letras foram reconhecidas corretamente
#         if letra_amarela >= len(palavra_aleatoria):
#             # Todas as letras foram reconhecidas corretamente, reinicie o contador
#             letra_amarela = 0
#     print(f'GESTOOOOOOOOOOOO -> {letra_amarela}')
#     # Retornar a resposta no formato JSON
#     return jsonify(letra_amarela=letra_amarela)

@app.route('/atualizar_letra_reconhecida')
def atualizar_letra_reconhecida():
    print(f'****************** METODO  atualizar_letra_reconhecida *********************')
    # Código para obter o novo texto
    global letra_amarela
    global palavra_aleatoria
    global beggin
    global palavras
    
    print(f'texto -> {text}')
    return jsonify(texto=text)

@app.route('/atualizar_datilografia')
def atualizar_datilografia():
    print(f'****************** METODO  atualizar_datilografia *********************')
    # Código para obter o novo texto
    global letra_amarela
    global palavra_aleatoria
    global beggin
    global palavras
    global count

    if  beggin == 1:
        time.sleep(3)
        beggin = 0
        letra_amarela = 0
    gesto_reconhecido = text# Seu código para reconhecimento do gesto
   
    if gesto_reconhecido == palavra_aleatoria[letra_amarela].upper():
        print(f'****************** gesto_reconhecido == palavra_aleatoria[letra_amarela] *********************')
        letra_amarela += 1
        if letra_amarela >= len(palavra_aleatoria):
            beggin = 1
            letra_amarela -=1
    

    print(f'beggin -> {beggin}')
    print(f'GESTOOOOOOOOOOOO -> {letra_amarela}')
    print(f'palavra_aleatoria -> {palavra_aleatoria}')
    print(f'palavra_aleatoria[letra_amarela] -> {palavra_aleatoria[letra_amarela].upper()}')

    return jsonify(letra_amarela=letra_amarela, palavra_aleatoria=palavra_aleatoria, beggin=beggin,  letraAtual=palavra_aleatoria[letra_amarela].upper())

@app.route('/reiniciar_palavra', methods=['GET'])
def reiniciar_palavra():
    print(f'****************** METODO  def reiniciar_palavra *********************')
    global palavra_aleatoria
    global letra_amarela
    global beggin
    global palavras
    
    # Reatribui os valores iniciais
    palavra_aleatoria = random.choice(palavras)
    letra_amarela = 0
    beggin = 0
    # print(f'beggin -> {beggin}')
    print(f'palavra_aleatoria -> {palavra_aleatoria}')
    print(f'palavra_aleatoria[letra_amarela] -> {palavra_aleatoria[letra_amarela].upper()}')
    
    # Retorna os novos valores como uma resposta JSON
    return jsonify(palavra_aleatoria=palavra_aleatoria, letra_amarela=letra_amarela, beggin=beggin)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=False)
