'''file FileHelper'''
import json
import os
import time
from os.path import isfile
from textwrap import wrap
from utl.LoggerHelper import LoggerHelper
dirname = os.path.dirname(__file__)

class FileHelper():
    '''class FileHelper'''

    @staticmethod
    def readJson(file_path=dirname+'/../settings.json'):
        with open(file_path, "r") as f:
            return json.load(f)

    @staticmethod
    def saveJson(dict, file_path, file_name):
        with open(f'{file_path}/{file_name}', "w") as f:
            json.dump(dict, f, indent = 6)
            return True

    @staticmethod
    def saveTxt(path, fileName, msg=None):
        fileDir = os.path.join(path, fileName, "")
        os.makedirs(fileDir, exist_ok = True)
        with open(fileDir+ f'{fileName}.txt', 'a') as file:
            for line in wrap(msg, width=100):
                file.write(line)
                file.write('\r\n')

    @staticmethod
    def convertKeyJson2ListTxt(inputPath, key, outputPath, fileName):
        file_json = FileHelper.readJson(inputPath)
        list = []
        for i, marca in enumerate(file_json):
            FileHelper.saveTxt(outputPath, fileName,marca.get(key))

    @staticmethod
    def listPath(Path):
        list_paths = []
        if os.path.isdir(os.path.join(Path)) :
            for count, file in enumerate(os.listdir(os.path.join(Path)),1):
                if not os.path.isdir(os.path.join(f'{Path}\{file}')):
                     list_paths.append(f'{Path}')
                     break
                list_paths.append(f'{Path}\{file}')
                LoggerHelper.write(f'Path {count} -> {Path}\{file}','DEBUG')
        else :
            list_paths.append(f'{Path}')
            LoggerHelper.write(f'Path 1 -> {Path}','DEBUG')
        return  list_paths

    @staticmethod
    def readFileTxt(path, type='readlines'):
        with open(path) as f:
            if type=='read':
                lines = f.read()
            elif type=='readlines':
                lines = [l.lower() for l in (line.strip() for line in f) if l]
            elif type=='readlist':
                lines = [f.read()]
            else:
                LoggerHelper.write(f'Type não encontrado -> {type}','DEBUG')

        return lines

    @staticmethod
    def createDir(path):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except OSError:
            LoggerHelper.write(f'ERROR: creating directory with name {path}', 'ERROR')

    @staticmethod
    def separar_train_teste(path,quantTrain=0.8, quantTest=0.2):
        try:
            list_files = []
            if os.path.exists(path):
                for file in os.listdir(path):
                    if isfile(file):
                        list_files.append(file)
                LoggerHelper.write(f'Len Files {len(list_files)}','DEBUG')
            else:
                LoggerHelper.write(f'Path {path} não encontrado','DEBUG')
        except OSError:
            LoggerHelper.write(f'ERROR: creating directory with name {path}', 'ERROR')

    @staticmethod
    def renameFileOrder(path, newPath, extension='.jpg'):


        # Obter uma lista de todos os arquivos no diretório
        arquivos = os.listdir(path)
        FileHelper.createDir(newPath)

        # Filtrar apenas os arquivos que terminam com a extensão desejada
        arquivos = [arquivo for arquivo in arquivos if arquivo.endswith(extension)]

        # Ordenar a lista de arquivos pelo nome, que deve estar em ordem numérica crescente
        arquivos.sort(key=lambda x: x.split(".")[0])

        # Renomear cada arquivo na lista em ordem numérica crescente
        for i, arquivo in enumerate(arquivos,10000):
            novo_nome = f"{i}{extension}"
            caminho_antigo = os.path.join(path, arquivo)
            newPath = os.path.join(path, novo_nome)
            os.rename(caminho_antigo, newPath)

        time.sleep(2)
        # Filtrar apenas os arquivos que terminam com a extensão desejada
        arquivos = [arquivo for arquivo in arquivos if arquivo.endswith(extension)]

        # Ordenar a lista de arquivos pelo nome, que deve estar em ordem numérica crescente
        arquivos.sort(key=lambda x: x.split(".")[0])
        time.sleep(2)
        # Renomear cada arquivo na lista em ordem numérica crescente
        for i, arquivo in enumerate(arquivos, 1):
            novo_nome = f"{i}{extension}"
            caminho_antigo = os.path.join(path, arquivo)
            newPath = os.path.join(path, novo_nome)
            os.rename(caminho_antigo, newPath)
            time.sleep(0.3)

