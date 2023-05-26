'''file ImageHelper'''
import cv2
import os
import random
from utl.LoggerHelper import LoggerHelper
dirname = os.path.dirname(__file__)
import numpy as np

class ImageHelper():
    '''class ImageHelper'''

    @staticmethod
    def escolher_palavra(palavras='TESTE'):
        
        return random.choice(palavras)

    @staticmethod
    def textToImage(img, texto, largura = 100, altura = 100, fonte = cv2.QT_FONT_NORMAL, escala = 2, grossura = 2, color=(255, 0, 0)):
       # Preenche o fundo de amarelo
        cv2.rectangle(img, (0, 0), (largura, altura), (255, 255, 255), -1)
        # Desenha uma borda azul
        cv2.rectangle(img, (0, 0), (largura-5, altura-5), (0, 0, 0), 5)
        # Desenha o texto com a variavel em preto, no centro
        fonte = cv2.QT_FONT_NORMAL
        escala = 2
        grossura = 2

        # Pega o tamanho (altura e largura) do texto em pixels
        tamanho, _ = cv2.getTextSize(texto, fonte, escala, grossura)

        # Desenha o texto no centro
        cv2.putText(img, texto, (int(largura / 2 - tamanho[0] / 2), int(altura / 2 + tamanho[1] / 2)), fonte, escala, color, grossura)
        return img
    
    @staticmethod
    def textToImageTest(img, palavra, count, texto='', countEspera = 0):
        # Define as cores de cada letra
        # cores = [(0, 255, 255), (128, 128, 128), (128, 128, 128), (128, 128, 128), (128, 128, 128), (128, 128, 128), (128, 128, 128)]

        # # Preenche o fundo de amarelo
        # cv2.rectangle(img, (0, 0), (100, 100), (255, 255, 0), -1)

        # Desenha a primeira letra em amarelo
        # cv2.putText(img, palavra[0], (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, cores[0], 2)
        

        if countEspera == 0 :
            # Desenha as demais letras
            for i in range(0, len(palavra)):
                # cor = cores[i]
                if i == count and palavra[count] != texto:
                    cor = (0, 255, 255) # Destaca a letra atual em vermelho
                elif palavra[count] == texto and count<len(palavra):
                    cor = (0, 255, 0) # Destaca as letras já reconhecidas em verde
                    count +=1
                # elif i in listOk:
                elif i < count:
                    cor = (0, 255, 0) # Destaca as letras já reconhecidas em verde
                # else:
                elif i > count:
                    cor = (128,128,128)
                cv2.putText(img, palavra[i], (10 + i*60, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, cor, 2)

                if count >= len(palavra):
                    count = 0
                    # palavra = None
                    countEspera = 20
                    break
        else:
            print(f'Espera {countEspera}')
            if countEspera %2 ==0 :
                for i in range(0, len(palavra)):
                    cor = (0, 255, 0)
                    cv2.putText(img, palavra[i], (10 + i*60, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, cor, 2)
            countEspera -=1
            if countEspera == 0:
                palavra = None

        return img, count, palavra, countEspera
    
    @staticmethod
    def imageShow( image, imageInfo, imageBlack, imagemAlfabeto, mode ='DEBUG', largura_tela=1920, altura_tela=1080):
      
        # tela1 = [int((altura_tela/100)), int((largura_tela/100))]
        # tela2 = [int(((altura_tela)/2)+10), int((largura_tela/100))]
        # tela3 = [int((altura_tela/100)), int((largura_tela/2)+10)]
        # tela4 = [int((altura_tela/2)+10), int((largura_tela/2)+10)]
        # x,y
        tela1 = [10, 20]
        tela2 = [int((image.shape[1])+10),  20]
        # tela2 = [int((largura_tela/100)), int(((altura_tela)/2)+10)]
        tela3 = [10, int((image.shape[0])+60)]
        tela4 = [int((imagemAlfabeto.shape[1])+10), int((imagemAlfabeto.shape[0])+60)]
       
        if mode == 'INFO':
            cv2.imshow('Image', image)
            cv2.moveWindow("Image", tela1[0], tela1[1])
            cv2.imshow('imagemAlfabeto', imagemAlfabeto)
            cv2.moveWindow("imagemAlfabeto", tela2[0], tela2[1])
            cv2.imshow('MediaPipe Hands',imageInfo)
            cv2.moveWindow("MediaPipe Hands", tela3[0], tela3[1])
        elif mode == 'DEBUG':
            cv2.imshow('Image', image)
            cv2.moveWindow("Image", tela1[0], tela1[1])
            cv2.imshow('imagemAlfabeto', imagemAlfabeto)
            cv2.moveWindow("imagemAlfabeto", tela2[0], tela2[1])
            cv2.imshow('MediaPipe Hands', imageInfo)
            cv2.moveWindow("MediaPipe Hands", tela3[0], tela3[1])
            cv2.imshow('imageBlack',imageBlack)
            cv2.moveWindow("imagemAlfabeto", tela2[0], tela2[1])
            cv2.moveWindow("imageBlack", tela4[0], tela4[1])
        elif mode == 'PROD':
            cv2.imshow('Image', image)
            cv2.moveWindow("Image", tela1[0], tela1[1])
            cv2.imshow('imagemAlfabeto', imagemAlfabeto)
            cv2.moveWindow("imagemAlfabeto", tela2[0], tela2[1])
            # cv2.imshow('Imagens', np.hstack((image)))

        if cv2.waitKey(1) & 0xFF == ord('i'):
            mode = 'INFO'
            cv2.destroyAllWindows()
        elif cv2.waitKey(1) & 0xFF == ord('d'):
            mode = 'DEBUG'
            cv2.destroyAllWindows()
        elif cv2.waitKey(1) & 0xFF == ord('p'):
            mode = 'PROD'
            cv2.destroyAllWindows()
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return mode

