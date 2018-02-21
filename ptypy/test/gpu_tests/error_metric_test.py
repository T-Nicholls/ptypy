'''
A test for the module of the relevant error metrics
'''

import unittest
import numpy as np
import utils as tu
from ptypy.gpu import data_utils as du
import ptypy.utils as u
from collections import OrderedDict
from ptypy.gpu.error_metrics import log_likelihood, far_field_error


class ErrorMetricTest(unittest.TestCase):

    def test_loglikelihood_numpy(self):
        '''
        Test that it runs
        '''
        PtychoInstance = tu.get_ptycho_instance('log_likelihood_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        propagator = PtychoInstance.diff.V[first_view_id].pod.geometry.propagator
        addr = vectorised_scan['meta']['addr']  # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        mask = vectorised_scan['mask']
        exit_wave = vectorised_scan['exit wave']
        diffraction = vectorised_scan['diffraction']
        log_likelihood(probe, obj, mask, exit_wave, diffraction, propagator.pre_fft, propagator.post_fft, addr)


    def test_loglikelihood_numpy_UNITY(self):
        '''
        Check that it gives the same result as the ptypy original

        '''
        error_metric = {}
        PodPtychoInstance = tu.get_ptycho_instance('log_likelihood_test')
        ptypy_error_metric =self.get_ptypy_loglikelihood(PodPtychoInstance)
        PtychoInstance = tu.get_ptycho_instance('log_likelihood_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        propagator = PtychoInstance.diff.V[first_view_id].pod.geometry.propagator
        addr = vectorised_scan['meta']['addr']  # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        mask = vectorised_scan['mask']
        exit_wave = vectorised_scan['exit wave']
        diffraction = vectorised_scan['diffraction']
        vals = log_likelihood(probe, obj, mask, exit_wave, diffraction, propagator.pre_fft, propagator.post_fft, addr)
        k = 0
        for name, view in PtychoInstance.diff.V.iteritems():
            error_metric[name] = vals[k]
            k += 1


        for name, view in PodPtychoInstance.diff.V.iteritems():
            ptypy_error = ptypy_error_metric[name]
            numpy_error = error_metric[name]
            np.testing.assert_array_equal(ptypy_error, numpy_error)

    def test_far_field_error(self):
        PtychoInstance = tu.get_ptycho_instance('log_likelihood_test')
        af, fmag, mask = self.get_current_and_measured_solution(PtychoInstance)
        far_field_error(af, fmag, mask)

    def test_far_field_error_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('log_likelihood_test')
        PodPtychoInstance = tu.get_ptycho_instance('log_likelihood_test')
        af, fmag, mask = self.get_current_and_measured_solution(PtychoInstance)
        fmag_npy = far_field_error(af, fmag, mask)
        fmag_ptypy = self.get_ptypy_far_field_error(PodPtychoInstance)
        np.testing.assert_array_equal(fmag_ptypy, fmag_npy)


    def get_current_and_measured_solution(self, a_ptycho_instance):
        alpha = 1.0

        fmag = []
        af = []
        mask = []
        for dname, diff_view in a_ptycho_instance.diff.views.iteritems():
            fmag.append(np.sqrt(np.abs(diff_view.data)))
            af2 = np.zeros_like(diff_view.data)
            f = OrderedDict()
            for name, pod in diff_view.pods.iteritems():
                if not pod.active:
                    continue
                f[name] = pod.fw((1 + alpha) * pod.probe * pod.object
                                 - alpha * pod.exit)
                af2 += u.abs2(f[name])
            mask.append(diff_view.pod.mask)
            af.append(np.sqrt(af2))
        return np.array(af), np.array(fmag), np.array(mask)


    def get_ptypy_far_field_error(self, a_ptycho_instance):

        err_fmag = []
        af, fmag, mask = self.get_current_and_measured_solution(a_ptycho_instance)
        for i in range(af.shape[0]):
            fdev = af[i] - fmag[i]
            err_fmag.append(np.sum(mask[i] * fdev ** 2) / mask[i].sum())
        return np.array(err_fmag)


    def get_ptypy_loglikelihood(self, a_ptycho_instance):
        error_dct = {}
        for dname, diff_view in a_ptycho_instance.diff.views.iteritems():
            I = diff_view.data
            fmask = diff_view.pod.mask
            LL = np.zeros_like(diff_view.data)
            for name, pod in diff_view.pods.iteritems():
                LL += u.abs2(pod.fw(pod.probe * pod.object))

            error_dct[dname] = (np.sum(fmask * (LL - I) ** 2 / (I + 1.))
                            / np.prod(LL.shape))
        return error_dct


if __name__ == '__main__':
    unittest.main()
