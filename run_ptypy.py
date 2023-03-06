import ptypy
from ptypy.core import Ptycho
# from mycho import Mycho as Ptycho
from ptypy import utils as u
from ptypy import parallel
from ptypy import io
import argparse
import socket
import sys, os, h5py
import numpy as np
import time
import subprocess
import json

# Load experiment loader
ptypy.load_ptyscan_module("zmq_loader")
ptypy.load_ptyscan_module("swmr_loader")

# PYCUDA engines
ptypy.load_gpu_engines("cuda")

# Current hostname
hn = socket.gethostbyname(socket.gethostname())
print(hn)

# Parsing arguments
parser = argparse.ArgumentParser(description="Reconstruction of something")
parser.add_argument("--input", type=str, help="path to the live streaming data file")
parser.add_argument("--output",   type=str, default="./output/", help="Output directory")
parser.add_argument("--frames-per-block",   type=int, default=None, help="Output directory")
parser.add_argument("--output-file", type=str, default=None)
parser.add_argument("-j", "--json-file", type=str, default=None)
parser.add_argument("-H", "--hdf5",  action='store_true')
parser.add_argument("-N", "--no-benchmarking",  action='store_true')


args = parser.parse_args()

if args.input:
    DATFILE = args.input
else:
    DATFILE = "/dls/p99/data/2022/cm31335-4/processing/ptycho-streaming-project/example-pipeline/streaming/data/stream.h5"

start_time = time.time()



p = u.Param()
p.verbose_level = 3
p.data_type = "single"
p.run = "test" + str(int(start_time))
p.dry_run = False
p.io = u.Param()
p.io.home = "./"
# p.io.autosave = u.Param(active=False)
p.io.autosave = u.Param(active=False, interval=100)

if not args.output_file:
    p.io.rfile = args.output + "./stream_%(run)s_%(engine)s_%(iterations)04d.ptyr"
else:
    p.io.rfile = args.output + args.output_file + '.ptyr'
    print(p.io.rfile)
p.io.benchmark = "all"

# Turn off plotclient
p.io.autoplot = u.Param()
p.io.autoplot.active = False
p.io.interaction = u.Param()
p.io.interaction.active = False
p.io.interaction.server = u.Param()
p.io.interaction.server.active = False

p.scans = u.Param()
p.scans.scan_00 = u.Param()
# The scan_00 parameters should be set within zmq_loader, and this metadata be acquired through ZMQ REQ/REP?
p.scans.scan_00.name = 'BlockFull'

p.scans.scan_00.illumination = u.Param()
p.scans.scan_00.illumination.model = None
p.scans.scan_00.illumination.photons = None
p.scans.scan_00.illumination.aperture = u.Param()
p.scans.scan_00.illumination.aperture.form = "circ"
p.scans.scan_00.illumination.aperture.size = 333e-6
p.scans.scan_00.illumination.propagation = u.Param()
p.scans.scan_00.illumination.propagation.focussed = 13.725e-3
p.scans.scan_00.illumination.propagation.parallel = 100e-6
p.scans.scan_00.illumination.propagation.antialiasing = 1
p.scans.scan_00.illumination.diversity = u.Param()
p.scans.scan_00.illumination.diversity.power = 0.1
p.scans.scan_00.illumination.diversity.noise = [0.5, 1.0]

p.scans.scan_00.sample = u.Param()
p.scans.scan_00.sample.model = None
p.scans.scan_00.sample.diversity = None
p.scans.scan_00.sample.process = None
p.scans.scan_00.sample.fill = 1

p.scans.scan_00.coherence = u.Param()
p.scans.scan_00.coherence.num_probe_modes = 1
p.scans.scan_00.coherence.num_object_modes = 1

p.scans.scan_00.data = u.Param()

p.scans.scan_00.data = u.Param()

p.scans.scan_00.data.name = 'ZMQLoader'
p.scans.scan_00.data.orientation = 2

#These bits might also need to be set within zmq_loader ?

p.scans.scan_00.data.intensities = u.Param()
p.scans.scan_00.data.intensities.file = DATFILE
p.scans.scan_00.data.intensities.key = "data"


p.scans.scan_00.data.positions = u.Param()
p.scans.scan_00.data.positions.file = DATFILE
p.scans.scan_00.data.positions.slow_key = "posy"
p.scans.scan_00.data.positions.slow_multiplier = 1e-3
p.scans.scan_00.data.positions.fast_key = "posx"
p.scans.scan_00.data.positions.fast_multiplier = 1e-3
p.scans.scan_00.data.positions.live_fast_key = "keys/posx"
p.scans.scan_00.data.positions.live_slow_key = "keys/posx"
p.scans.scan_00.data.intensities.live_key = "keys/data"


p.scans.scan_00.data.recorded_energy = u.Param()
p.scans.scan_00.data.recorded_energy.file = DATFILE
p.scans.scan_00.data.recorded_energy.key = "energyFocus/value"
p.scans.scan_00.data.recorded_energy.multiplier = 1e-3
p.scans.scan_00.data.recorded_energy.offset = -0.008

p.scans.scan_00.data.distance = 0.072
p.scans.scan_00.data.auto_center = False  # change back to True
p.scans.scan_00.data.center = [1038, 1018]
p.scans.scan_00.data.psize = 11e-6
p.scans.scan_00.data.shape = (1024, 1024)
p.scans.scan_00.data.rebin = 2
p.scans.scan_00.data.shape = (512, 512)
p.scans.scan_00.data.psize = 22e-6

p.scans.scan_00.data.darkfield = u.Param()
p.scans.scan_00.data.darkfield.file = DATFILE
p.scans.scan_00.data.darkfield.key = "darkField"

# p.scans.scan_00.data.finished_dataset = u.Param()
# p.scans.scan_00.data.finished_dataset.file = DATFILE
# p.scans.scan_00.data.finished_dataset.key = "finished"

# p.scans.scan_00.data.framefilter = u.Param()
# p.scans.scan_00.data.framefilter.file = DATFILE
# p.scans.scan_00.data.framefilter.key = "filter/data"

if args.frames_per_block:
    p.frames_per_block = args.frames_per_block
else:
    p.frames_per_block = 200 # --> mpirun -n 4: 4*10 = 40 frames per pod
p.min_frames_for_recon = 200
# if not args.hdf5:
#     p.scans.scan_00.data.max_frames = 3000
#p.scans.scan_00.data.min_frames = 50
# min num of frames to prepare with call of auto. (* num processes of course)

# p.scans.scan_00.data.checkpoints = [1600,]+[2000*i for i in range(1,30)]

# p.scans.scan_00.data.checkpoints = [10*i for i in range(9,200)]

p.engines = u.Param()
p.engines.engine = u.Param()
p.engines.engine.name = "DM_pycuda"
p.engines.engine.numiter = 100
p.engines.engine.numiter_contiguous = 50
p.engines.engine.probe_support = None
p.engines.engine.probe_update_start = 0
p.engines.engine.probe_fourier_support = None
p.engines.engine.record_local_error = False
# p.engines.engine.fft_lib = "cuda"
p.engines.engine.alpha = 0.95
p.engines.engine.fourier_power_bound = 0.25
p.engines.engine.overlap_converge_factor = 0.001
p.engines.engine.overlap_max_iterations = 20
p.engines.engine.update_object_first = False
p.engines.engine.obj_smooth_std = 20
p.engines.engine.object_inertia = 0.001
p.engines.engine.probe_inertia = 0.001
p.engines.engine.clip_object = [0, 1]
print('remember to add json file and start time when benchmarking')

if not args.no_benchmarking:
    if args.json_file:
        p.json_file = f"benchmarkjson/{args.json_file}.json"
    else:
        p.json_file = f"benchmarkjson/{int(start_time)}.json"
    p.start_time = float(start_time)
print('starting..')
P = Ptycho(p, level=5)
# print(P.debug_dict)
if not args.no_benchmarking:
    with open(p.json_file, "w") as f:
        to_save = {'start': start_time, 'end': time.time(),
                'readings': P.debug_dict,
                'recon_file': P.paths.recon_file(P.runtime)}
        json.dump(to_save, f)
