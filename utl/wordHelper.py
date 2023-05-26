import random


class WordHelper():
    '''class WordHelper'''
    def __init__(self, word=None,  words=[], dictWord={}):
        self.word = word
        self.words = words
        self.dictWord = dictWord
        

    def escolher_palavra(self):
        self.word = random.choice(self.words).upper()
    
    def createDictWord(self):
        print(f'Limpando o dictWord {self.dictWord}')
        self.dictWord = {}
        for count, word in enumerate(self.word, 0):
            if count == 0 :
                status = 'wait'
            else:
                status = 'NOk'

            test = {count:{'word':{word},'status':{status}}}
            self.dictWord.update(test)
