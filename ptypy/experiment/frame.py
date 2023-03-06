"""Stores all information about a frame, to be used by pre-processing to encode and processing to decode"""

import numpy as np
class Frame():
    
    def __init__(self, posx, posy, data, dtype, shape):
        self.posx = posx
        self.posy = posy
        self.data = data
        self.dtype = dtype
        self.shape = shape #Assume always square shape

        
    #should add checks to see if already encoded
    def encode(self):
        self.posx = str(self.posx).encode()
        self.posy = str(self.posy).encode()
        self.dtype = str(self.data.dtype).encode()
        self.shape = str(self.data.shape).encode()
          
    #Not the inverse of encode() since data isn't encoded there, (numpy arrays already have buffer interface)
    def decode(self):
        self.posx = float(self.posx.decode())
        self.posy = float(self.posy.decode())
        self.dtype = self.dtype.decode()
        self.shape = int(self.shape.decode()[1:].split(",")[0])
        self.data = np.frombuffer(self.data,dtype=self.dtype)
        self.data = self.data.reshape(self.shape,self.shape)
        