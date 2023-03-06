"""\
Description here

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import logging
import zmq
from ptypy import utils as u
from ptypy.core.data import PtyScan
from ptypy.experiment import register
from ptypy.experiment import frame
from time import time
from time import sleep
import numpy as np
import h5py as h5
from swmr_tools import KeyFollower
from ptypy.utils.verbose import log
logging.basicConfig(level=logging.DEBUG)

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

    [intensities.file]
    default = None
    type = str
    help = Path to the file containing the diffraction intensities.

    [intensities.key]
    default = None
    type = str
    help = Key to the intensities entry in the hdf5 file.

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

    [positions.file]
    default = None
    type = str
    help = Path to the file containing the position information. If None use "intensities.file".

    [positions.slow_key]
    default = None
    type = str
    help = Key to the slow-axis positions entry in the hdf5 file.

    [positions.slow_multiplier]
    default = 1.0
    type = float
    help = Multiplicative factor that converts motor positions to metres.

    [positions.fast_key]
    default = None
    type = str
    help = Key to the fast-axis positions entry in the hdf5 file.

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

    [mask]
    default =
    type = Param
    help = Parameters for mask data. 
    doc = The shape of the loaded data is assumed to be (frame_size_m, frame_size_n) or the same
          shape of the full intensities data.

    [mask.file]
    default = None
    type = str
    help = Path to the file containing the diffraction mask.

    [mask.key]
    default = None
    type = str
    help = Key to the mask entry in the hdf5 file.

    [mask.invert]
    default = False
    type = bool
    help = Inverting the mask

    [flatfield]
    default =
    type = Param
    help = Parameters for flatfield data.
    doc = The shape of the loaded data is assumed to be (frame_size_m, frame_size_n) or the same
            shape of the full intensities data.

    [flatfield.file]
    default = None
    type = str
    help = Path to the file containing the diffraction flatfield.

    [flatfield.key]
    default = None
    type = str
    help = Key to the flatfield entry in the hdf5 file.

    [darkfield]
    default =
    type = Param
    help = Parameters for darkfield data. 
    doc = The shape is assumed to be (frame_size_m, frame_size_n) or the same
          shape of the full intensities data.

    [darkfield.file]
    default = None
    type = str
    help = Path to the file containing the diffraction darkfield.

    [darkfield.key]
    default = None
    type = str
    help = Key to the darkfield entry in the hdf5 file.

    [normalisation]
    default =
    type = Param
    help = Parameters for per-point normalisation (i.e. ion chamber reading).
    doc = The shape of loaded data is assumed to have the same dimensionality as data.shape[:-2]

    [normalisation.file]
    default = None
    type = str
    help = This is the path to the file containing the normalisation information. If None then we try to find the information
            in the "intensities.file" location.

    [normalisation.key]
    default = None
    type = str
    help = This is the key to the normalisation entry in the hdf5 file.

    [normalisation.sigma]
    default = 3
    type = int
    help = Sigma value applied for automatic detection of outliers in the normalisation dataset.

    [framefilter]
    default = 
    type = Param
    help = Parameters for the filtering of frames
    doc = The shape of loaded data is assumed to hvae the same dimensionality as data.shape[:-2]

    [framefilter.file]
    default = None
    type = str
    help = This is the path to the file containing the filter information. 

    [framefilter.key]
    default = None
    type = str
    help = This is the key to the frame filter entry in the hdf5 file.

    [recorded_energy]
    default =
    type = Param
    help = This parameter contains information if we are use the recorded energy rather than as a parameter.
            It should be a scalar value.
    
    [recorded_energy.file]
    default = None
    type = str
    help = This is the path to the file containing the recorded_energy.

    [recorded_energy.key]
    default = None
    type = str
    help = This is the key to the recorded_energy entry in the hdf5 file.

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
    
    [recorded_distance.file]
    default = None
    type = str
    help = This is the path to the file containing the recorded_distance between sample and detector.

    [recorded_distance.key]
    default = None
    type = str
    help = This is the key to the recorded_distance entry in the hdf5 file.

    [recorded_distance.multiplier]
    default = 1.0
    type = float
    help = This is the multiplier for the recorded distance.

    [recorded_psize]
    default =
    type = Param
    help = This parameter contains information if we are use the recorded psize to the detector rather than as a parameter,
            It should be a scalar value.
    
    [recorded_psize.file]
    default = None
    type = str
    help = This is the path to the file containing the recorded detector psize.

    [recorded_psize.key]
    default = None
    type = str
    help = This is the key to the recorded_psize entry in the hdf5 file.

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

    [outer_index]
    type = int
    default = None
    help = Index for outer dimension (e.g. tomography, spectro scans), default is None.

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
          
          
    [intensities.live_key]
    default = None
    type = str
    help = Key to live keys inside the intensities file
    doc = Live_keys indicate where the data collection has progressed to.
          They are zero at the scan start, but non-zero when the position
          is complete.

    [positions.live_fast_key]
    default = None
    type = str
    help = Key to live key for fast axis inside the positions file
    doc = Live_keys indicate where the data collection has progressed to.
          They are zero at the scan start, but non-zero when the position
          is complete.

    [positions.live_slow_key]
    default = None
    type = str
    help = Key to live key for slow axis inside the positions file
    doc = Live_keys indicate where the data collection has progressed to.
          They are zero at the scan start, but non-zero when the position
          is complete.
           
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
        self.ready_frames = 0
        #self.darkfield = None
        #self.flatfield = None
        #self.mask = None
        #self.normalisation = None
        #self.normalisation_laid_out_like_positions = None
        #self.darkfield_laid_out_like_data = None
        #self.flatfield_field_laid_out_like_data = None
        #self.mask_laid_out_like_data = None
        self.preview_indices = None
        #self.framefilter = None
        self._is_spectro_scan = False
        self.fhandle_intensities = None
        self.fhandle_positions_fast = None
        self.fhandle_positions_slow = None
        #self.fhandle_darkfield = None
        #self.fhandle_flatfield = None
        #self.fhandle_normalisation = None
        #self.fhandle_mask = None
        
        #Stuff specific to ZMQ logic:
        
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
        self.frames_before_load = 50 #Wait for this many frames to be preprocessed before loading in, N
        self.frames_loaded = 0
        self.final_send = False #True when preprocessing has sent over all data
        logging.debug("Metadata recieved")
        self._is_swmr = True
        self._prepare_intensity_and_positions()
        self._prepare_center()
        
        
    def _prepare_intensity_and_positions(self):
        """
        Prep for loading intensity and position data and keyfollower. Copied from hdf5 and swmr loaders
        """
        self.fhandle_intensities = h5.File(self.p.intensities.file, 'r', swmr=self._is_swmr)
        self.intensities = self.fhandle_intensities[self.p.intensities.key]
        self.intensities_dtype = self.intensities.dtype
        self.data_shape = self.intensities.shape
        if self._is_spectro_scan and self.p.outer_index is not None:
            self.data_shape = tuple(np.array(self.data_shape)[1:])

        self.fhandle_positions_fast = h5.File(self.p.positions.file, 'r', swmr=self._is_swmr)
        self.fast_axis = self.fhandle_positions_fast[self.p.positions.fast_key]
        if self._is_spectro_scan and self.p.outer_index is not None:
            self.fast_axis = self.fast_axis[self.p.outer_index]
        self.positions_fast_shape = np.squeeze(self.fast_axis).shape if self.fast_axis.ndim > 2 else self.fast_axis.shape

        self.fhandle_positions_slow = h5.File(self.p.positions.file, 'r', swmr=self._is_swmr)
        self.slow_axis = self.fhandle_positions_slow[self.p.positions.slow_key]
        if self._is_spectro_scan and self.p.outer_index is not None:
            self.slow_axis = self.slow_axis[self.p.outer_index]
        self.positions_slow_shape = np.squeeze(self.slow_axis).shape if self.slow_axis.ndim > 2 else self.slow_axis.shape
        
        log(3, "The shape of the \n\tdiffraction intensities is: {}\n\tslow axis data:{}\n\tfast axis data:{}".format(self.data_shape,
                                                                                                                      self.positions_slow_shape,
                                                                                                                      self.positions_fast_shape))
        if self.p.positions.skip > 1:
            log(3, "Skipping every {:d} positions".format(self.p.positions.skip))
            
        self.kf = KeyFollower((self.fhandle_intensities[self.p.intensities.live_key],
                               self.fhandle_positions_slow[self.p.positions.live_slow_key],
                               self.fhandle_positions_fast[self.p.positions.live_fast_key]),
                               timeout=5)
        
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
        
        
    def get_data_chunk(self, *args, **kwargs):
        self.kf.refresh()
        self.intensities.refresh()
        self.slow_axis.refresh()
        self.fast_axis.refresh()
        # refreshing here to update before Ptyscan.get_data_chunk calls check and load
        return super().get_data_chunk(*args, **kwargs)
        
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
        
        self.info_socket.send(b'f') #Asks how many frames are ready to load
        
        reply = self.info_socket.recv().decode()
        if reply[-1] == ('f'): #Final chunk of data has been sent
            self.final_send = 0
            end_of_scan = 1
            frames_accessible = int(reply[:-1]) - self.frames_loaded
            self.ready_frames += frames_accessible
            #logging.debug(f"Final frames ready to load, {frames_accessible} frames")
        else:
            end_of_scan = 0
            frames_accessible = int(reply) - self.frames_loaded
            self.ready_frames = self.ready_frames + frames_accessible
            
        #Minimum frames ready to load
        if self.ready_frames >= self.frames_before_load or end_of_scan:
            logging.debug(f"Next {self.ready_frames} frames are ready to load")
            if end_of_scan:
                logging.debug(f"This is the final set of frames: {self.ready_frames}")
            frames_accessible = self.ready_frames #Variable to return
            self.ready_frames = 0 #Reset to 0 for next check
            
        #Need more frames before loading
        else:
            frames_accessible = 0
   
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
        #For now, ignore indexing logic
        
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
                new_frame = frame.Frame(data=encoded_data[i], posx=encoded_data[i+chunk_size],
                                posy=encoded_data[i+(chunk_size*2)],dtype=encoded_data[i+(chunk_size*3)],
                                shape=encoded_data[i+(chunk_size*4)])
                new_frame.decode()        
                self.frame_list.append(new_frame) 
                
                intensities[i-1] = new_frame.data
                positions[i-1] = np.array([new_frame.posx, new_frame.posy])
                weights = np.ones(len(positions[i-1]))
                
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
                      
        return intensities, positions, weights
    
    def event_loop(self):
        """
        Main loop
        """
        finished = False
        while finished == False:
            self.check()
            finished = self.load()