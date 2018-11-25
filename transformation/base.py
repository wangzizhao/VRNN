from abc import ABC, abstractmethod

# base class for transformation


class transformation(object):
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def transform(self, X_prev):
        pass
