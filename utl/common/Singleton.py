'''file Singleton'''

class Singleton(type):
    '''class Singleton'''

    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        #else:
            #print(cls.__qualname__, cls._instances[cls].Level)
        return cls._instances[cls]