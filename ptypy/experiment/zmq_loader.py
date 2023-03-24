"""\
Description here

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import zmq
import pickle
from ptypy import utils as u
from ptypy.core.data import PtyScan
from ptypy.experiment import register
from ptypy.experiment import frame
from time import time
from time import sleep
import numpy as np
from ptypy.utils.verbose import log
#logging.basicConfig(level=logging.DEBUG)

@register()
class ZMQLoader(PtyScan):
    """
    First attempt to make a generalised hdf5 loader for data. Please raise a ticket in github if changes are required
    so we can coordinate. There will be a Nexus and CXI subclass to this in the future.

    Defaults:

    [name]
    default = 'ZMQLoader'
    type = str
    help =

    [intensities]
    default =
    type = Param
    help = Parameters for the diffraction data.
    doc = Data shapes can be either (A, B, frame_size_m, frame_size_n) or (C, frame_size_m, frame_size_n).
          It is assumed in this latter case that the fast axis in the scan corresponds
          the fast axis on disc (i.e. C-ordered layout).

    [positions]
    default =
    type = Param
    help = Parameters for the position information data. 
    doc = Shapes for each axis that are currently covered and tested corresponding 
          to the intensity shapes are:
            * axis_data.shape (A, B) for data.shape (A, B, frame_size_m, frame_size_n),
            * axis_data.shape (k,) for data.shape (k, frame_size_m, frame_size_n),
            * axis_data.shape (C, D) for data.shape (C*D, frame_size_m, frame_size_n) ,
            * axis_data.shape (C,) for data.shape (C, D, frame_size_m, frame_size_n) where D is the
              size of the other axis, and 
            * axis_data.shape (C,) for data.shape (C*D, frame_size_m, frame_size_n) where D is the
              size of the other axis.


    [positions.slow_multiplier]
    default = 1.0
    type = float
    help = Multiplicative factor that converts motor positions to metres.

    [positions.fast_multiplier]
    default = 1.0
    type = float
    help = Multiplicative factor that converts motor positions to metres.

    [positions.bounding_box]
    default =
    type = Param
    help = Bounding box (in array indices) to reconstruct a restricted area

    [positions.bounding_box.fast_axis_bounds]
    default = None
    type = None, int, tuple, list
    help = If an int, this is the lower bound only, if a tuple is (min, max)

    [positions.bounding_box.slow_axis_bounds]
    default =
    type = None, int, tuple, list
    help = If an int, this is the lower bound only, if a tuple is (min, max)

    [positions.skip]
    default = 1
    type = int
    help = Skip a given number of positions (in each direction)

    [recorded_energy]
    default =
    type = Param
    help = This parameter contains information if we are use the recorded energy rather than as a parameter.
            It should be a scalar value.

    [recorded_energy.multiplier]
    default = 1.0
    type = float
    help = This is the multiplier for the recorded energy.

    [recorded_energy.offset]
    default = 0.0
    type = float
    help = This is an optional offset for the recorded energy in keV.

    [recorded_distance]
    default =
    type = Param
    help = This parameter contains information if we are use the recorded distance to the detector rather than as a parameter,
            It should be a scalar value.

    [recorded_distance.multiplier]
    default = 1.0
    type = float
    help = This is the multiplier for the recorded distance.

    [recorded_psize]
    default =
    type = Param
    help = This parameter contains information if we are use the recorded psize to the detector rather than as a parameter,
            It should be a scalar value.

    [recorded_psize.multiplier]
    default = 1.0
    type = float
    help = This is the multiplier for the recorded detector psize.

    [shape]
    type = int, tuple
    default = None
    help = Shape of the region of interest cropped from the raw data.
    doc = Cropping dimension of the diffraction frame
      Can be None, (dimx, dimy), or dim. In the latter case shape will be (dim, dim).
    userlevel = 1

    [padding]
    type = int, tuple, list
    default = None
    help = Option to pad the detector frames on all sides
    doc = A tuple of list with padding given as ( top, bottom, left, right)

    [electron_data]
    type = bool
    default = False
    help = Switch for loading data from electron ptychography experiments.
    doc = If True, the energy provided in keV will be considered as electron energy 
          and converted to electron wavelengths.   
           
    """
    
    def __init__(self, pars=None, **kwargs):
        self.p = self.DEFAULT.copy(99)
        self.p.update(pars, in_place_depth=99)
        super().__init__(self.p, **kwargs)
        
        # Initialise other variables...
        self._scantype = None
        self.intensities_dtype = None
        self.data_shape = None
        self.positions_fast_shape = None
        self.positions_slow_shape = None
        self._is_swmr = True
        
        #------------------Stuff specific to ZMQ logic---------------------:
        self.context = zmq.Context()
        #Create socket to request some information and ask for metadata
        self.info_socket = self.context.socket(zmq.REQ) 
        self.info_socket.connect("tcp://172.23.166.25:7556")
        self.info_socket.send(b"MetadataRequest")
        
        #Socket to pull main data
        self.main_pull = self.context.socket(zmq.PULL)
        self.main_pull.connect("tcp://172.23.166.25:7555")
        
        #Socket to recieve heartbeats
        self.heartbeat_socket = self.context.socket(zmq.SUB)
        self.heartbeat_socket.setsockopt(zmq.SUBSCRIBE, b'hb')
        self.heartbeat_socket.connect("tcp://172.23.166.25:7557")
        heartbeat_timer = time() #Do something if no heartbeats are detected after some time
        
        log(4, 'Waiting for metadata...')
        #Wait to recieve metadata
        self.metadata = pickle.loads(self.info_socket.recv())
        self.frames_before_load = 50 #Wait for this many frames to be preprocessed before loading in, N
        self.frames_loaded = 0
        self.final_send = False #True when preprocessing has sent over all data
        log(4, "Metadata recieved")

        self._prepare_intensity_and_positions()
        self._prepare_center()
        self._prepare_meta_info()
        self.close_sockets = False

        
    # Set meta info here, i.e energy, distance, psize etc.
    def _prepare_meta_info(self):
        """
        Prep for meta info (energy, distance, psize)
        """
        
        #multiplier and offset is set in run_ptypy
        self.p.energy = self.metadata['energy']
        self.p.energy = self.p.energy * self.p.recorded_energy.multiplier + self.p.recorded_energy.offset
        self.meta.energy  = self.p.energy
        log(3, "loading energy={} from file".format(self.p.energy))

        #Padding, psize and distance also set in run_ptypy for now
        self.meta.distance = self.p.distance
        log(3, "loading distance={} from file".format(self.p.distance))
        
        self.info.psize = self.p.psize
        log(3, "loading psize={} from file".format(self.p.psize))

        if self.p.padding is None:
            self.pad = np.array([0,0,0,0])
            log(3, "No padding will be applied.")
        else:
            self.pad = np.array(self.p.padding, dtype=int)
            assert self.pad.size == 4, "self.p.padding needs to of size 4"
            log(3, "Padding the detector frames by {}".format(self.p.padding))
        
        
    def _prepare_intensity_and_positions(self):
        """
        Uses metadata sent over ZMQ to set intensity dtype, data shapes, total number of frames.
        and energy. Also sets parameters for distance, psize and padding based off settings in
        run_ptypy
        """

        self.intensities_dtype = np.array([]).astype('uint16').dtype
        self.data_shape = self.metadata["shape"]
        self.p.shape = self.data_shape[1:]
        self.info.shape = self.p.shape
        self.positions_fast_shape = self.metadata["positions fast shape"]
        self.positions_slow_shape = self.metadata["positions slow shape"]
        
        #TODO: Metadata should send num_frames instead of also working it out here
        if len(self.data_shape) == 3:
            self.num_frames = self.data_shape[0]
        elif(len(self.data_shape) == 4):
            self.num_frames = self.data_shape[0]*self.data_shape[1]
        else:
            #Should make this a proper error
            print("Other dimensions not yet implemented")
            quit()
            
        
        log(3, "The shape of the \n\tdiffraction intensities is: {}\n\tslow axis data:{}\n\tfast axis data:{}".format(self.data_shape,
                                                                                                                      self.positions_slow_shape,
                                                                                                                      self.positions_fast_shape))

    def _prepare_center(self):
        """
        define how data should be loaded (center, cropping). Copied from hdf5 loader
        """
        
        self.frame_slices = (slice(None, None, 1), slice(None, None, 1))
        self.frame_shape = self.data_shape[-2:]
        self.info.center = None
        self.info.auto_center = self.p.auto_center
        log(3, "center is %s, auto_center: %s" % (self.info.center, self.info.auto_center))
        log(3, "The loader will not do any cropping.")
                
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

        if start is None:
            start = self.framestart
        
        if frames is None:
            frames = self.min_frames
            
        if not self.close_sockets:
            self.info_socket.send(b'FrameNumberRequest') #Asks how many frames are ready to load
            reply = int(self.info_socket.recv().decode())        
            self.available = min(int(reply), self.num_frames)
            new_frames = self.available - start
            # not reached expected nr. of frames
            if new_frames <= frames:
                # but its last chunk of scan so load it anyway
                if self.available == self.num_frames:
                    frames_accessible = new_frames    
                    end_of_scan = 1
                    self.close_sockets = True
                # otherwise, do nothing
                else:
                    end_of_scan = 0
                    frames_accessible = 0
            # reached expected nr. of frames
            else:
                end_of_scan = 0
                frames_accessible = frames

            log(3, f"frames = {frames}, start = {start}, self.available = {self.available}, frames_accessible = {frames_accessible}, end_of_scan = {end_of_scan}, reply = {reply}, new_frames = {new_frames}")
            return frames_accessible, end_of_scan
        
        #Data has been loaded and sockets have been closed, so don't send another request
        else:
            frames_accessible = 0
            end_of_scan = 1
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
        intensities = {}
        positions = {}
        weights = {}

        log(4, "Loading...")
        log(4, f"indices = {indices}")
        read_chunk = True
        chunk_size = 0
        
        for ind in indices:
            log(4, f"Reading index = {ind}, read_chunk = {read_chunk}")
            
            if read_chunk:
                encoded_data = self.main_pull.recv_multipart()
                # The size of the chunk sent from preprocessing is given in the 0th index
                chunk_size = int(encoded_data[0].decode()) 
                print(f"Pulled chunk of length {chunk_size} frames")

            read_chunk = not ((ind + 1) % chunk_size)

            i = (ind % chunk_size) + 1
            
            new_frame = frame.Frame(data=encoded_data[i], posx=encoded_data[i+chunk_size],
                                    posy=encoded_data[i+(chunk_size*2)],dtype=encoded_data[i+(chunk_size*3)],
                                    shape=encoded_data[i+(chunk_size*4)])
            new_frame.decode()

            intensities[ind] = new_frame.data
            positions[ind] = np.array([new_frame.posy * self.p.positions.slow_multiplier,
                                       new_frame.posx * self.p.positions.fast_multiplier]) 
            weights[ind] = np.ones(len(intensities[ind]))
            
        #Final data has been loaded, so sockets are now safe to close and preprocessing script can quit
        if self.close_sockets:
            self.info_socket.send(b'FinalFrameRecieved')
            self.info_socket.recv() #Ensure message is recieved before closing
            self.context.destroy() #End all ZMQ communications
            
            
        return intensities, positions, weights
    

