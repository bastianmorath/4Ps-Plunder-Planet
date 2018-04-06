'''Defines the interface that all the models have to implement
'''

from abc import ABC, abstractmethod


class AbstractMLModelClass(ABC):

    @abstractmethod
    def predict(self, x_test):
        pass