import ctypes
import os
import sys

_dir = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(_dir, "minimax.so"))
_lib.choose_action.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
_lib.choose_action.restype = ctypes.c_int


class MinimaxAgent:
    def __init__(self, depth=5):
        self.depth = depth

    def pyAction(self, board, piece):
        flat = []
        for row in board:
            flat.extend(row)
        arr = (ctypes.c_int * 42)(*flat)
        return _lib.pyAction(arr, self.depth, piece)
