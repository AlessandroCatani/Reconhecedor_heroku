import sys
import logging
from colorama import init, Fore, Back

from bsl.enum.LoggerLevelEnum import LoggerLevelEnum
from .common.Singleton import Singleton

init(autoreset=True)

class ColorFormatter(logging.Formatter):

    COLORS = {
        "WARNING": Fore.LIGHTYELLOW_EX,
        "ERROR": Fore.LIGHTRED_EX,
        "DEBUG": Fore.LIGHTBLUE_EX,
        "INFO": Fore.LIGHTGREEN_EX,
        "CRITICAL": Fore.WHITE + Back.LIGHTRED_EX
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        if color:
            record.name = color + record.name
            record.levelname = color + record.levelname
            record.msg = color + record.msg
        return logging.Formatter.format(self, record)

class LoggerHelper(metaclass=Singleton):

    @property
    def Level(self):
        return self.__level

    @classmethod
    def __init__(self, level=None):
        self.__level = level

        # Disable logging by default
        self.__logger = logging.getLogger(__name__)
        self.__logger.addHandler(logging.NullHandler())

        if self.__level is not None:
            self.__enum = {
                LoggerLevelEnum.NOTSET.name: logging.NOTSET,
                LoggerLevelEnum.WARNING.name: logging.WARNING,
                LoggerLevelEnum.INFO.name: logging.INFO,
                LoggerLevelEnum.DEBUG.name: logging.DEBUG,
                LoggerLevelEnum.ERROR.name: logging.ERROR,
                LoggerLevelEnum.CRITICAL.name: logging.CRITICAL
            }
            self.__enable(self.__enum[self.__level])

    @classmethod
    def __enable(self, level):
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = ColorFormatter('%(levelname)s: [%(process)d] %(message)s')
        handler.setFormatter(formatter)

        self.__logger.addHandler(handler)
        self.__logger.setLevel(level)

    @classmethod
    def write(self, msg, level=None):
        if level is None:
            level = self.__level

        # https://gist.github.com/JettJones/c236494013f22723c1822126df944b12
        frame = sys._getframe()
        while True:
            if 'LoggerHelper' in frame.f_globals["__name__"]:
                frame = frame.f_back
                caller = frame.f_globals["__name__"]
                break
            frame = frame.f_back

        try:
            callerReduced = caller.split(".")[-1]
            if callerReduced == '__main__':
                callerReduced = caller.split(".")[-2]
        except:
            callerReduced = caller
        text = f'[{callerReduced}] {msg}'

        if level == LoggerLevelEnum.WARNING.name:
            self.__logger.warning(text)
        elif level == LoggerLevelEnum.INFO.name:
            self.__logger.info(text)
        elif level == LoggerLevelEnum.DEBUG.name:
            self.__logger.debug(text)
        elif level == LoggerLevelEnum.ERROR.name:
            self.__logger.error(text)
        elif level == LoggerLevelEnum.CRITICAL.name:
            self.__logger.critical(text)
        else:
            print("Level not found: " + level)
