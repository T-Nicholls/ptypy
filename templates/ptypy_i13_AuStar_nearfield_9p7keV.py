
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
import numpy as np
p = u.Param()

### PTYCHO PARAMETERS
p.verbose_level = 3   

p.data_type = "single"      
p.run = None
p.io = u.Param()
p.io.home = "/tmp/ptypy/"     
p.io.run = None

p.io.autoplot =  u.Param()
p.io.autoplot.layout ='nearfield'

p.scan = u.Param()
p.scan.source = None           
p.scan.geometry = u.Param()
p.scan.geometry.energy = 9.7      
p.scan.geometry.lam = None      
p.scan.geometry.distance = 8.46e-2    
p.scan.geometry.psize = 100e-9         
p.scan.geometry.shape = 1024                    
p.scan.geometry.propagation = "nearfield"      

p.scan.geometry.precedence = 'meta'

sim = u.Param()
sim.xy = u.Param()
sim.xy.override = u.parallel.MPIrand_uniform(0.0,10e-6,(20,2))          
#sim.xy.positions = np.random.normal(0.0,3e-6,(20,2))
sim.verbose_level = 1

sim.illumination = u.Param()
sim.illumination.model = None
sim.illumination.photons = 1e11              
sim.illumination.aperture = u.Param()
sim.illumination.aperture.diffuser = (8.0, 10.0)
sim.illumination.aperture.form = "circ"
sim.illumination.aperture.size = 90e-6
sim.illumination.aperture.central_stop = 0.15
sim.illumination.propagation = u.Param()
sim.illumination.propagation.focussed = None#0.08 
sim.illumination.propagation.parallel = 0.005
sim.illumination.propagation.spot_size = None

sim.sample = u.Param()
sim.sample.model = u.scripts.xradia_star((1200,1200),minfeature=3,contrast=0.8)
sim.sample.process = u.Param()
sim.sample.process.offset = (0,0)        
sim.sample.process.zoom = 1.0            
sim.sample.process.formula = "Au"              
sim.sample.process.density = 19.3                  
sim.sample.process.thickness = 700e-9       
sim.sample.process.ref_index = None           
sim.sample.process.smoothing = None            
sim.sample.fill = 1.0+0.j      

sim.detector = 'GenericCCD32bit' 
sim.plot = False

p.scan.update(sim.copy(depth=4))
p.scan.coherence = u.Param()
p.scan.coherence.Nprobe_modes = 1     
p.scan.coherence.Nobject_modes = 1     
p.scan.coherence.energies = [1.0]     

# p.scan.sharing = u.Param()
# p.scan.sharing.EP_sharing = False
# p.scan.sharing.object_shared_with = None    
# p.scan.sharing.object_share_power = 1      
# p.scan.sharing.probe_shared_with = None    
# p.scan.sharing.probe_share_power = 1    

p.scan.sample = u.Param()
#p.scan.sample.model = 'stxm'
#p.scan.sample.process = None
#p.scan.sample.diversity = None
p.scan.xy = u.Param()
p.scan.xy.model=None
#p.scan.illumination = sim.illumination.copy()
p.scan.illumination.model = 'stxm'

p.scans = u.Param()
p.scans.sim = u.Param()
p.scans.sim.data=u.Param()
p.scans.sim.data.source = 'sim' 
p.scans.sim.data.recipe = sim.copy(depth=4) 
p.scans.sim.data.save = None #'append'
p.scans.sim.data.shape = None
p.scans.sim.data.num_frames = None

p.engine = u.Param()
p.engine.common = u.Param()
p.engine.common.numiter = 100           
p.engine.common.numiter_contiguous = 1   
p.engine.common.probe_support = None          
p.engine.common.probe_inertia = 0.001        
p.engine.common.object_inertia = 0.1            
p.engine.common.obj_smooth_std = 10          
p.engine.common.clip_object = None            

p.engine.DM = u.Param()
p.engine.DM.name = "DM"                       
p.engine.DM.alpha = 1                        
p.engine.DM.probe_update_start = 2          
p.engine.DM.update_object_first = True       
p.engine.DM.overlap_converge_factor = 0.5    
p.engine.DM.overlap_max_iterations = 100       
p.engine.DM.fourier_relax_factor = 0.05   

p.engine.ML = u.Param()

p.engines = u.Param()                 
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 100
p.engines.engine00.object_inertia = 1.
p.engines.engine00.fourier_relax_factor = 0.1
#p.engines.engine01 = u.Param()
#p.engines.engine01.name = 'ML'
#p.engines.engine01.numiter = 50

P = Ptycho(p,level=5)

