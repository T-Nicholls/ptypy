# -*- coding: utf-8 -*-
"""
Scan management.

The main task of this module is to prepare the data structure for
reconstruction, taking a data feed and connecting individual diffraction
measurements to the other containers. The way this connection is done
is defined by the user through a model definition. The connections are
described by the POD objects. This module also takes care of initializing
containers according to user-defined rules.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
import illumination
import sample
import geometry
import model
import xy
import data

from collections import OrderedDict
from .. import utils as u
from ..utils.verbose import logger, headerline, log
from classes import *
from classes import DEFAULT_ACCESSRULE
from classes import MODEL_PREFIX
from ..utils import parallel
from ..utils.descriptor import defaults_tree

# Please set these globally later
FType = np.float64
CType = np.complex128

__all__ = ['ModelManager', 'ScanModel']

NO_DATA_FLAG = 'No data'

@defaults_tree.parse_doc('scanmodel.Full')
class ScanModel(object):
    """
    Manage a single scan model (sharing, coherence, propagation, ...)

    Defaults:

    [name]
    default = Full
    type = str
    help =
    doc =

    [tags]
    default = ['dummy']
    help = Comma seperated string tags describing the data input
    doc = [deprecated?]
    type = list
    userlevel = 2

    [sharing]
    default = 
    help = Scan sharing options
    doc = 
    type = Param
    userlevel = 

    [sharing.object_share_with]
    default = None
    help = Label or index of scan to share object with.
    doc = Possible values:
       - ``None``: Do not share
       - *(string)*: Label of the scan to share with
       - *(int)*: Index of scan to share with
    type = str
    userlevel = 1

    [sharing.object_share_power]
    default = 1
    help = Relative power for object sharing
    doc = 
    type = float
    userlevel = 1
    lowlim = 0

    [sharing.probe_share_with]
    default = None
    help = Label or index of scan to share probe with.
    doc = Possible values:
       - ``None``: Do not share
       - *(string)*: Label of the scan to share with
       - *(int)*: Index of scan to share with
    type = str
    userlevel = 1

    [sharing.probe_share_power]
    default = 1
    help = Relative power for probe sharing
    doc = 
    type = float
    userlevel = 1
    lowlim = 0

    [sharing.EP_sharing]
    type = bool
    default = False
    help = Empty probe sharing switch
    doc =

    [coherence]
    default = 
    help = Coherence parameters
    doc = 
    type = Param
    userlevel = 
    lowlim = 0

    [coherence.num_probe_modes]
    default = 1
    help = Number of probe modes
    doc = 
    type = int
    userlevel = 0
    lowlim = 0

    [coherence.num_object_modes]
    default = 1
    help = Number of object modes
    doc = 
    type = int
    userlevel = 0
    lowlim = 0

    [coherence.energies]
    default = [1.0]
    type = list
    help = ?
    doc = ?

    [coherence.spectrum]
    default = [1.0]
    help = Amplitude of relative energy bins if the probe modes have a different energy
    doc = 
    type = list
    userlevel = 2
    lowlim = 0

    [coherence.object_dispersion]
    default = None
    help = Energy dispersive response of the object
    doc = One of:
       - ``None`` or ``'achromatic'``: no dispersion
       - ``'linear'``: linear response model
       - ``'irregular'``: no assumption
      **[not implemented]**
    type = str
    userlevel = 2

    [coherence.probe_dispersion]
    default = None
    help = Energy dispersive response of the probe
    doc = One of:
       - ``None`` or ``'achromatic'``: no dispersion
       - ``'linear'``: linear response model
       - ``'irregular'``: no assumption
      **[not implemented]**
    type = str
    userlevel = 2

    [propagation]
    type = str
    default = farfield
    help = Propagation type
    doc = Either "farfield" or "nearfield"
    userlevel = 1

    """

    def __init__(self, ptycho=None, specific_pars=None, generic_pars=None, label=None):
        """
        Create ScanModel object.

        Parameters
        ----------
        specific_pars : dict or Param
            Input parameters specific to the given scan.

        generic_pars : dict or Param
            Input parameters (see :py:attr:`DEFAULT`)
            If None uses defaults
        """
        from .. import experiment

        # Update parameter structure
        # Load default parameter structure
        p = self.DEFAULT.copy(99)
        p.update(generic_pars, in_place_depth=4)
        p.update(specific_pars, in_place_depth=4)
        self.p = p
        self.label = label
        self.ptycho = ptycho

        # Manage stand-alone cases
        if self.ptycho is None:
            self.Cdiff = Container(ptycho=self, ID='Cdiff', data_type='real')
            self.Cmask = Container(ptycho=self, ID='Cmask', data_type='bool')
            self.CType = CType
            self.FType = FType
        else:
            self.Cdiff = ptycho.diff
            self.Cmask = ptycho.mask

        # Create Associated PtyScan object
        self.ptyscan = experiment.makePtyScan(self.p.data)

        # Initialize instance attributes
        self.mask = None
        self.diff = None
        self.positions = []
        self.mask_views = []
        self.diff_views = []
        self.new_positions = None
        self.new_diff_views = None
        self.new_mask_views = None

        self.geometries = []
        self.shape = None
        self.psize = None

        self.data_available = True

        self.frames_per_call = 100000

    def new_data(self):
        """
        Feed data from ptyscan object.
        :return: None if no data is available, True otherwise.
        """

        # Initialize if that has not been done yet
        if not self.ptyscan.is_initialized:
            self.ptyscan.initialize()

        # Get data
        dp = self.ptyscan.auto(self.frames_per_call)

        self.data_available = (dp != data.EOS)
        logger.debug(u.verbose.report(dp))

        if dp == data.WAIT or not self.data_available:
            return None

        label = self.label
        logger.info('Importing data from scan %s.' % label)

        # Prepare the scan geometry if not already done.
        if not self.geometries:
            self.geometries = []

            # extract necessary info from the received data package
            get_keys = ['distance', 'center', 'energy', 'psize', 'shape']
            geo_pars = u.Param({key: dp['common'][key] for key in get_keys})

            # add propagation info from this scan model
            geo_pars.propagation = self.p.propagation

            # The multispectral case will have multiple geometries
            for ii, fac in enumerate(self.p.coherence.energies):
                geoID = geometry.Geo._PREFIX + '%02d' % ii + label
                g = geometry.Geo(self.ptycho, geoID, pars=geo_pars)
                # now we fix the sample pixel size, This will make the frame size adapt
                g.p.resolution_is_fix = True
                # save old energy value:
                g.p.energy_orig = g.energy
                # change energy
                g.energy *= fac
                # append the geometry
                self.geometries.append(g)

            # Store frame shape
            self.shape = np.array(dp['common'].get('shape', self.geometries[0].shape))
            self.psize = self.geometries[0].psize

        sh = self.shape

        # Storage generation if not already existing
        if self.diff is None:
            # This scan is brand new so we create storages for it
            self.diff = self.Cdiff.new_storage(shape=(1, sh[-2], sh[-1]), psize=self.psize, padonly=True,
                                                     layermap=None)
            old_diff_views = []
            old_diff_layers = []
        else:
            # ok storage exists already. Views most likely also. Let's do some analysis and deactivate the old views
            old_diff_views = self.Cdiff.views_in_storage(self.diff, active=False)
            old_diff_layers = []
            for v in old_diff_views:
                old_diff_layers.append(v.layer)

        # Same for mask
        if self.mask is None:
            self.mask = self.Cmask.new_storage(shape=(1, sh[-2], sh[-1]), psize=self.psize, padonly=True,
                                                     layermap=None)
            old_mask_views = []
            old_mask_layers = []
        else:
            old_mask_views = self.Cmask.views_in_storage(self.mask, active=False)
            old_mask_layers = []
            for v in old_mask_views:
                old_mask_layers.append(v.layer)

        # Prepare for View generation
        AR_diff_base = DEFAULT_ACCESSRULE.copy()
        AR_diff_base.shape = self.shape
        AR_diff_base.coord = 0.0
        AR_diff_base.psize = self.psize
        AR_mask_base = AR_diff_base.copy()
        AR_diff_base.storageID = self.diff.ID
        AR_mask_base.storageID = self.mask.ID

        diff_views = []
        mask_views = []
        positions = []

        # First pass: create or update views and reformat corresponding storage
        for dct in dp['iterable']:

            index = dct['index']
            active = dct['data'] is not None

            pos = dct.get('position')

            if pos is None:
                logger.warning('No position set to scan point %d of scan %s' % (index, label))

            AR_diff = AR_diff_base
            AR_mask = AR_mask_base
            AR_diff.layer = index
            AR_mask.layer = index
            AR_diff.active = active
            AR_mask.active = active

            # check here: is there already a view to this layer? Is it active?
            try:
                old_view = old_diff_views[old_diff_layers.index(index)]
                old_active = old_view.active
                old_view.active = active

                logger.debug(
                    'Diff view with layer/index %s of scan %s exists. \nSetting view active state from %s to %s' % (
                        index, label, old_active, active))
            except ValueError:
                v = View(self.Cdiff, accessrule=AR_diff)
                diff_views.append(v)
                logger.debug(
                    'Diff view with layer/index %s of scan %s does not exist. \nCreating view with ID %s and set active state to %s' % (
                        index, label, v.ID, active))
                # append position also
                positions.append(pos)

            try:
                old_view = old_mask_views[old_mask_layers.index(index)]
                old_view.active = active
            except ValueError:
                v = View(self.Cmask, accessrule=AR_mask)
                mask_views.append(v)

        # so now we should have the right views to this storages. Let them reformat()
        # that will create the right sizes and the datalist access
        self.diff.reformat()
        self.mask.reformat()

        # Second pass: copy the data
        for dct in dp['iterable']:
            parallel.barrier()
            if dct['data'] is None:
                continue
            diff_data = dct['data']
            idx = dct['index']

            # FIXME: Find a more transparent way than this.
            self.diff.data[self.diff.layermap.index(idx)][:] = diff_data
            self.mask.data[self.mask.layermap.index(idx)][:] = dct.get('mask', np.ones_like(diff_data))

        self.diff.nlayers = parallel.MPImax(self.diff.layermap) + 1
        self.mask.nlayers = parallel.MPImax(self.mask.layermap) + 1

        self.new_positions = positions
        self.new_diff_views = diff_views
        self.new_mask_views = mask_views
        self.positions += positions
        self.diff_views += diff_views
        self.mask_views += mask_views

        self._update_stats()

        return True

    def _update_stats(self):
        """
        (Re)compute the statistics for the data stored in the scan.
        These statistics are:
         * Itotal: The integrated power per frame
         * max/min/mean_frame: pixel-by-pixel maximum, minimum and
           average among all frames.
        """
        mask_views = self.mask_views
        diff_views = self.diff_views

        # Nothing to do if no view exist
        if not self.diff: return

        # Reinitialize containers
        Itotal = []
        max_frame = np.zeros(self.diff_views[0].shape)
        min_frame = np.zeros_like(max_frame)
        mean_frame = np.zeros_like(max_frame)
        norm = np.zeros_like(max_frame)

        for maview, diview in zip(mask_views, diff_views):
            if not diview.active:
                continue
            dv = diview.data
            m = maview.data
            v = m * dv
            Itotal.append(np.sum(v))
            max_frame[max_frame < v] = v[max_frame < v]
            min_frame[min_frame > v] = v[min_frame > v]
            mean_frame += v
            norm += m

        parallel.allreduce(mean_frame)
        parallel.allreduce(norm)
        parallel.allreduce(max_frame, parallel.MPI.MAX)
        parallel.allreduce(max_frame, parallel.MPI.MIN)
        mean_frame /= (norm + (norm == 0))

        self.diff.norm = norm
        self.diff.max_power = parallel.MPImax(Itotal)
        self.diff.tot_power = parallel.MPIsum(Itotal)
        self.diff.pbound_stub = self.diff.max_power / mean_frame.shape[-1]**2
        self.diff.mean = mean_frame
        self.diff.max = max_frame
        self.diff.min = min_frame

        info = {'label': self.label, 'max': self.diff.max_power, 'tot': self.diff.tot_power, 'mean': mean_frame.sum()}
        logger.info(
            '\n--- Scan %(label)s photon report ---\nTotal photons   : %(tot).2e \nAverage photons : %(mean).2e\nMaximum photons : %(max).2e\n' % info + '-' * 29)


@defaults_tree.parse_doc('scanmodel.Vanilla')
class ScanModel2(object):
    """
    Dummy for testing, there must be more than one for validate to react
    to invalid names.
    """
    pass


class ModelManager(object):
    """
    Manages ptypy objects creation and update.

    The main task of ModelManager is to follow the rules for a given
    reconstruction model and create:

     - the probe, object, exit, diff and mask containers
     - the views
     - the PODs

    A ptychographic problem is defined by the combination of one or
    multiple scans. ModelManager uses encapsulate
    scan-specific elements in .scans und .scans_pars

    Note
    ----
    This class is densely connected to :any:`Ptycho` the separation
    in two classes is more history than reason and these classes may get
    merged in future releases
    """
    DEFAULT = ScanModel.DEFAULT
    """ Default scan parameters. See :py:data:`.scan`
        and a short listing below """

    _PREFIX = MODEL_PREFIX

    _BASE_MODEL = OrderedDict(
        index = 0,
        energy = 0.0,
        pmode = 0,
        x = 0.0,
        y = 0.0,
    )

    def __init__(self, ptycho, pars):
        """

        Parameters
        ----------
        ptycho: Ptycho
            The parent Ptycho object

        pars : dict or Param
            Input parameters (see :py:attr:`DEFAULT`)
        """
        # Initialize the input parameters
        # REDESIGN: ModelManager will soon disappear. Here we fix things temporarily.
        # this means: no DEFAULT for this class.
        p = pars.copy(depth=99)
        self.p = p

        self.ptycho = ptycho

        # abort if ptycho is None:
        # FIXME: PT Is this the expected behavior?
        if self.ptycho is None:
            return

        self.scans = OrderedDict()

        # Create scan objects from information already available
        for label, scan_pars in self.p.iteritems():
            self.scans[label] = ScanModel(ptycho=self.ptycho, pars=scan_pars, label=label)

        # Sharing dictionary that stores sharing behavior
        self.sharing = {'probe_ids': {}, 'object_ids': {}}

        # REDESIGN: this will be replaced
        # Initialize sharing rules for POD creations
        sharing_pars = u.Param({'model_type': 'basic',
                                'scan_per_probe': 1,
                                'scan_per_object': 1,
                                'npts': None})
        self.sharing_rules = model.parse_model(sharing_pars, self.sharing)

    def _to_dict(self):
        # Delete the model class. We do not really need to store it.
        del self.sharing_rules
        return self.__dict__.copy()

    @classmethod
    def _from_dict(cls, dct):
        # create instance
        inst = cls(None, None)
        # overwrite internal dictionary
        inst.__dict__ = dct
        return inst

    @property
    def data_available(self):
        return any(s.data_available for s in self.scans.values())

    def new_data(self):
        """
        Get all new diffraction patterns and create all views and pods
        accordingly.
        """
        parallel.barrier()

        # Nothing to do if there are no new data.
        if not self.data_available:
            return 'No data'

        logger.info('Processing new data.')
        used_scans = []

        # Attempt to get new data
        for label, scan in self.scans.iteritems():
            new_data = scan.new_data()
            if new_data:
                used_scans.append(label)

        if not used_scans:
            return None

        # Create PODs
        new_pods, new_probe_ids, new_object_ids = self._create_pods(used_scans)
        logger.info('Process %d created %d new PODs, %d new probes and %d new objects.' % (
            parallel.rank, len(new_pods), len(new_probe_ids), len(new_object_ids)), extra={'allprocesses': True})

        # Adjust storages
        self.ptycho.probe.reformat(True)
        self.ptycho.obj.reformat(True)
        self.ptycho.exit.reformat()

        self._initialize_probe(new_probe_ids)
        self._initialize_object(new_object_ids)
        self._initialize_exit(new_pods)

    def _initialize_probe(self, probe_ids):
        """
        Initialize the probe storages referred to by the probe_ids
        """
        logger.info('\n'+headerline('Probe initialization', 'l'))

        # Loop through probe ids
        for pid, labels in probe_ids.items():

            # Pick first scan - this should not matter.
            scan = self.scans[labels[0]]
            illu_pars = scan.p.illumination

            # pick storage from container
            s = self.ptycho.probe.S.get(pid)

            if s is None:
                continue
            else:
                logger.info('Initializing probe storage %s using scan %s.'
                            % (pid, scan.label))


            # if photon count is None, assign a number from the stats.
            phot = illu_pars.get('photons')
            phot_max = scan.diff.max_power

            if phot is None:
                logger.info('Found no photon count for probe in parameters.\nUsing photon count %.2e from photon report' % phot_max)
                illu_pars['photons'] = phot_max
            elif np.abs(np.log10(phot)-np.log10(phot_max)) > 1:
                logger.warn('Photon count from input parameters (%.2e) differs from statistics (%.2e) by more than a magnitude' % (phot, phot_max))

            illumination.init_storage(s, illu_pars)

            s.reformat()  # Maybe not needed
            s.model_initialized = True

    def _initialize_object(self, object_ids):
        """
        Initializes the probe storages referred to by the object_ids.
        """

        logger.info('\n'+headerline('Object initialization', 'l'))

        # Loop through object IDs
        for oid, labels in object_ids.items():

            # Pick first scan - this should not matter.
            scan = self.scans[labels[0]]
            sample_pars = scan.p.sample

            # pick storage from container
            s = self.ptycho.obj.S.get(oid)

            if s is None or s.model_initialized:
                continue
            else:
                logger.info('Initializing object storage %s using scan %s.'
                            % (oid, scan.label))

            sample_pars = scan.p.sample

            if type(sample_pars) is u.Param:
                # Deep copy
                sample_pars = sample_pars.copy(depth=10)

                # Quickfix spectral contribution.
                if (scan.p.coherence.object_dispersion
                        not in [None, 'achromatic']
                        and scan.p.coherence.probe_dispersion
                        in [None, 'achromatic']):
                    logger.info(
                        'Applying spectral distribution input to object fill.')
                    sample_pars['fill'] *= s.views[0].pod.geometry.p.spectral


            sample.init_storage(s, sample_pars)
            s.reformat()  # maybe not needed

            s.model_initialized = True

    @staticmethod
    def _initialize_exit(pods):
        """
        Initializes exit waves using the pods.
        """
        logger.info('\n' + headerline('Creating exit waves', 'l'))
        for pod in pods:
            if not pod.active:
                continue
            pod.exit = pod.probe * pod.object

    def _create_pods(self, new_scans):
        """
        Create all pods associated with the scan labels in 'scans'.

        Return the list of new pods, probe and object ids (to allow for
        initialization).
        """
        logger.info('\n' + headerline('Creating PODS', 'l'))
        new_pods = []
        new_probe_ids = {}
        new_object_ids = {}

        # Get a list of probe and object that already exist
        existing_probes = self.ptycho.probe.storages.keys()
        # SC: delete? self.sharing_rules.probe_ids.keys()
        existing_objects = self.ptycho.obj.storages.keys()
        # SC: delete? self.sharing_rules.object_ids.keys()
        logger.info('Found these probes : ' + ', '.join(existing_probes))
        logger.info('Found these objects: ' + ', '.join(existing_objects))

        # Loop through scans
        for label in new_scans:
            scan = self.scans[label]

            positions = scan.new_positions
            di_views = scan.new_diff_views
            ma_views = scan.new_mask_views

            # Compute sharing rules
            share = scan.p.sharing
            alt_obj = share.object_share_with if share is not None else None
            alt_pr = share.probe_share_with if share is not None else None

            obj_label = label if alt_obj is None else alt_obj
            pr_label = label if alt_pr is None else alt_pr

            # Loop through diffraction patterns
            for i in range(len(di_views)):
                dv, mv = di_views.pop(0), ma_views.pop(0)

                index = dv.layer

                # Object and probe position
                pos_pr = u.expect2(0.0)
                pos_obj = positions[i] if 'empty' not in scan.p.tags else 0.0

                t, object_id = self.sharing_rules(obj_label, index)
                probe_id, t = self.sharing_rules(pr_label, index)

                # For multiwavelength reconstructions: loop here over
                # geometries, and modify probe_id and object_id.
                for ii, geometry in enumerate(scan.geometries):
                    # Make new IDs and keep them in record
                    # sharing_rules is not aware of IDs with suffix
                    
                    pdis = scan.p.coherence.probe_dispersion

                    if pdis is None or str(pdis) == 'achromatic':
                        gind = 0
                    else:
                        gind = ii

                    probe_id_suf = probe_id + 'G%02d' % gind
                    if (probe_id_suf not in new_probe_ids.keys()
                            and probe_id_suf not in existing_probes):
                        new_probe_ids[probe_id_suf] = (
                            self.sharing_rules.probe_ids[probe_id])

                    odis = scan.p.coherence.object_dispersion

                    if odis is None or str(odis) == 'achromatic':
                        gind = 0
                    else:
                        gind = ii

                    object_id_suf = object_id + 'G%02d' % gind
                    if (object_id_suf not in new_object_ids.keys()
                            and object_id_suf not in existing_objects):
                        new_object_ids[object_id_suf] = (
                            self.sharing_rules.object_ids[object_id])

                    # Loop through modes
                    for pm in range(scan.p.coherence.num_probe_modes):
                        for om in range(scan.p.coherence.num_object_modes):
                            # Make a unique layer index for exit view
                            # The actual number does not matter due to the
                            # layermap access
                            exit_index = index * 10000 + pm * 100 + om

                            # Create views
                            # Please note that mostly references are passed,
                            # i.e. the views do mostly not own the accessrule
                            # contents
                            pv = View(container=self.ptycho.probe,
                                      accessrule={'shape': geometry.shape,
                                                  'psize': geometry.resolution,
                                                  'coord': pos_pr,
                                                  'storageID': probe_id_suf,
                                                  'layer': pm,
                                                  'active': True})

                            ov = View(container=self.ptycho.obj,
                                      accessrule={'shape': geometry.shape,
                                                  'psize': geometry.resolution,
                                                  'coord': pos_obj,
                                                  'storageID': object_id_suf,
                                                  'layer': om,
                                                  'active': True})

                            ev = View(container=self.ptycho.exit,
                                      accessrule={'shape': geometry.shape,
                                                  'psize': geometry.resolution,
                                                  'coord': pos_pr,
                                                  'storageID': (probe_id +
                                                                object_id[1:] +
                                                                'G%02d' % ii),
                                                  'layer': exit_index,
                                                  'active': dv.active})

                            views = {'probe': pv,
                                     'obj': ov,
                                     'diff': dv,
                                     'mask': mv,
                                     'exit': ev}

                            pod = POD(ptycho=self.ptycho,
                                      ID=None,
                                      views=views,
                                      geometry=geometry)  # , meta=meta)

                            new_pods.append(pod)

                            # If Empty Probe sharing is enabled,
                            # adjust POD accordingly.
                            if share is not None:
                                pod.probe_weight = share.probe_share_power
                                pod.object_weight = share.object_share_power
                                if share.EP_sharing:
                                    pod.is_empty = True
                                else:
                                    pod.is_empty = False
                            else:
                                pod.probe_weight = 1
                                pod.object_weight = 1


        return new_pods, new_probe_ids, new_object_ids
