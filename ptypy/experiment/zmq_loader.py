"""\
Description here

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import logging
import frame
import zmq
from ptypy.core.data import PtyScan
from ptypy.experiment import register
from time import time
from time import sleep
logging.basicConfig(level=logging.DEBUG)

@register()
class ZMQLoader(PtyScan):
    """
    New params here

    """
    
    def __init__(self, pars=None, **kwargs):
        self.p = self.DEFAULT.copy(99)
        self.p.update(pars, in_place_depth=99)
        super().__init__(self.p, **kwargs)
        
        # Initialise other variables...
        
        self._scantype = None
        self._ismapped = None
        self.intensities = None
        self.intensities_dtype = None
        self.slow_axis = None
        self.fast_axis = None
        self.data_shape = None
        self.positions_fast_shape = None
        self.positions_slow_shape = None
        self.darkfield = None
        self.flatfield = None
        self.mask = None
        self.normalisation = None
        self.normalisation_laid_out_like_positions = None
        self.darkfield_laid_out_like_data = None
        self.flatfield_field_laid_out_like_data = None
        self.mask_laid_out_like_data = None
        self.preview_indices = None
        self.framefilter = None
        self._is_spectro_scan = False
        self.fhandle_intensities = None
        self.fhandle_positions_fast = None
        self.fhandle_positions_slow = None
        self.fhandle_darkfield = None
        self.fhandle_flatfield = None
        self.fhandle_normalisation = None
        self.fhandle_mask = None
        
        self.context = zmq.Context()
        #Create socket to request some information and ask for metadata
        self.info_socket = self.context.socket(zmq.REQ) 
        self.info_socket.connect("tcp://127.0.0.1:5556")
        self.info_socket.send(b"m")
        
        #Socket to pull main data
        self.main_pull = self.context.socket(zmq.PULL)
        self.main_pull.connect("tcp://127.0.0.1:5555")
        
        #Socket to recieve heartbeats
        self.heartbeat_socket = self.context.socket(zmq.SUB)
        self.heartbeat_socket.setsockopt(zmq.SUBSCRIBE, b'hb')
        self.heartbeat_socket.connect("tcp://127.0.0.1:5557")
        heartbeat_timer = time() #Do something if no heartbeats are detected after some time
        
        #Create poller for event loop
        self.poller = zmq.Poller()
        self.poller.register(self.main_pull, zmq.POLLIN)
        self.poller.register(self.heartbeat_socket, zmq.POLLIN)
        self.poller.register(self.info_socket, zmq.POLLIN)
        
        logging.debug('Waiting for metadata...')
        #Wait to recieve metadata
        self.metadata = self.info_socket.recv()
        self.frames_before_load = 60 #Wait for this many frames to be preprocessed before loading in, N
        self.frames_loaded = 0
        self.final_send = False #True when preprocessing has sent over all data
        logging.debug("Metadata recieved")
        
        
        
    def check(self, frames=None, start=None):
        """
        Use info socket to query how many frames are available.
    
        This method checks how many frames the preparation routine may
        process, starting from frame `start` at a request of `frames`.

        This method is supposed to return the number of accessible frames
        for preparation and should determine if data PREPROCESSING for this
        scan is finished. Its main purpose is to allow for a pre-processing to occur
        separately, and where the number of frames is not known
        when :py:class:`PtyScan` is constructed, i.e. a data stream or an
        on-the-fly reconstructions.

        Note
        ----
        If :py:data:`num_frames` is set on ``__init__()`` of the subclass,
        this method can be left as it is.

        Parameters
        ----------
        frames : int or None
            Number of frames requested.
        start : int or None
            Scanpoint index to start checking from.

        Returns
        -------
        frames_accessible : int
            Number of frames readable.

        end_of_scan : int or None
            is one of the following,
            - 0, end of the scan is not reached
            - 1, end of scan will be reached or is
            - None, can't say
        """
        
        self.info_socket.send(b'f') #Asks how many frames are ready to load
        
        reply = self.info_socket.recv().decode()
        if reply[-1] == ('f'): #Final chunk of data has been sent
            end_of_scan = 1
            frames_accessible = int(reply[:-1]) - self.frames_loaded
            logging.debug(f"Final frames ready to load, {frames_accessible} frames")
        else:
            end_of_scan = 0
            frames_accessible = int(reply) - self.frames_loaded
            logging.debug(f"Next {frames_accessible} frames are ready to load")
        
        return frames_accessible, end_of_scan
        
        
    def load(self, indices):
        """
        Loads data according to node specific scanpoint indices that have
        been determined by :py:class:`LoadManager` or otherwise.

        Returns
        -------
        raw, positions, weight : dict
            Dictionaries whose keys are the given scan point `indices`
            and whose values are the respective frame / position according
            to the scan point index. `weight` and `positions` may be empty
        """
        if self.ready_frames >= self.frames_before_load or self.final_send:
            self.frame_list=[] #Store loaded frames
            frames_loaded = 0 #Frames loaded local to this function call
            logging.info(f"Pulling {self.ready_frames} frames...")
            
            #Repeat until all the ready frames have been loaded
            while frames_loaded < self.ready_frames: #Not '<=' since frames_loaded starts on 0
                
                encoded_data = self.main_pull.recv_multipart() 
                chunk_size = int(encoded_data[0].decode()) #The size of the chunk sent from preprocessing is given in the 0th index
                print(f"Pulled chunk of length {chunk_size} frames")

                #Reconstruct data: First (chunk_size=20) values is the actual data, then posx,posy,dtype,shape
                for i in range(1,chunk_size+1):
                    new_frame = Frame(data=encoded_data[i], posx=encoded_data[i+chunk_size],
                                    posy=encoded_data[i+(chunk_size*2)],dtype=encoded_data[i+(chunk_size*3)],
                                    shape=encoded_data[i+(chunk_size*4)])
                    new_frame.decode()        
                    self.frame_list.append(new_frame) 
                frames_loaded += chunk_size   
            self.frames_loaded += frames_loaded #Update the overall frames loaded
                
            #Debugging
            try:
                logging.info(f"{self.ready_frames} frames have been loaded, 1st frame contains data {self.frame_list[0].data} and shape \
                    {self.frame_list[0].shape}")   
            except:
                IndexError 
                        
            if self.final_send:
                logging.info("All data has been loaded")
                return True
        return False
    
    def event_loop(self):
        """
        Main loop
        """
        finished = False
        while finished == False:
            self.check()
            finished = self.load()