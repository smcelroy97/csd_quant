"""Run this file directly in an editor to execute meaningful regression checks."""
import unittest
import tempfile
from pathlib import Path
import h5py
import numpy as np
from scipy.signal import resample_poly
from qc_templates import (Settings, align_laminar, anchors_to_csd, csd_from_lfp,
                          repair_channels, trial_qc, channel_qc, pca_fit, load_continuous)
from validate_wd import ShapeWD


class QCTests(unittest.TestCase):
    def test_block_downsampling_matches_whole_recording(self):
        rng = np.random.default_rng(11)
        raw = rng.normal(size=(50000, 5))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"sample.mat"
            with h5py.File(path, "w") as f:
                f.create_dataset("craw/cnt", data=raw)
                f.create_dataset("craw/adrate", data=[[2000.]])
                triggers = f.create_dataset("times", data=[6000, 12000, 18000])
                refs = f.create_dataset("anatrig", shape=(1, 1), dtype=h5py.ref_dtype)
                refs[0, 0] = triggers.ref
            result = load_continuous(path, Path(tmp)/"cache")
            expected = resample_poly(raw*.001, 1, 2, axis=0)
            np.testing.assert_allclose(result["lfp"], expected, atol=1e-9)
            cached = load_continuous(path, Path(tmp)/"cache")
            np.testing.assert_array_equal(result["lfp"], cached["lfp"])

    def test_csd_center_and_sign(self):
        depth = np.arange(7)*.1
        lfp = (depth**2)[:, None]*np.ones((1, 20))
        np.testing.assert_allclose(csd_from_lfp(lfp), -2, atol=1e-12)
        np.testing.assert_equal(anchors_to_csd([3, 5, 7], "raw-one"), [1, 3, 5])
        np.testing.assert_equal(anchors_to_csd([2, 4, 6], "raw-zero"), [1, 3, 5])

    def test_warp_hits_anchors(self):
        x = np.arange(21)[:, None]*np.ones((1, 10))
        y = align_laminar(x, [3, 8, 16])
        np.testing.assert_allclose(y[[7, 15, 22], 0], [3, 8, 16])
        with self.assertRaises(ValueError):
            align_laminar(x, [0, 8, 16])

    def test_repair_linear_lfp_not_csd(self):
        x = np.tile(np.arange(9)[None, :, None], (3, 1, 20)).astype(float)
        bad = np.zeros(9, bool); bad[4] = True
        original = x.copy(); x[:, 4] = 1000
        fixed = repair_channels(x, bad)
        np.testing.assert_allclose(fixed, original)
        np.testing.assert_allclose(csd_from_lfp(fixed), 0)
        bad[0] = True
        with self.assertRaises(ValueError):
            repair_channels(x, bad)

    def test_noise_and_artifact_do_not_reject_common_response(self):
        rng = np.random.default_rng(5)
        t = np.arange(-200, 300)
        n, c = 90, 23
        raw = rng.normal(0, .003, (n, c, len(t)))
        raw += .08*np.exp(-((t-30)/20)**2)[None, None, :]
        filtered = raw.copy()
        # A noisy contact with large independent high-frequency residual.
        raw[:, 10] += rng.normal(0, 2, (n, len(t)))
        filtered[:, 10] += rng.normal(0, .3, (n, len(t)))
        bad, _ = channel_qc(raw, filtered, t, np.zeros(c), np.zeros(c), Settings())
        self.assertTrue(bad[10])
        self.assertLessEqual(bad.sum(), 2)
        raw[7, :, 250] += 5; filtered[7, :, 250] += 3
        rejected, _ = trial_qc(raw, filtered, t, ~bad, np.zeros(n, bool), Settings())
        self.assertTrue(rejected[7])
        self.assertLess(rejected.sum(), 5)

    def test_pca_matches_svd(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(13, 30, 20))
        fit = pca_fit(X)
        _, s, v = np.linalg.svd(X.reshape(13, -1)-X.reshape(13, -1).mean(0), full_matrices=False)
        np.testing.assert_allclose(fit["evr"], (s*s/(s@s))[:12], atol=1e-12)
        self.assertAlmostEqual(abs(np.dot(v[0], fit["components"][0].ravel())), 1)

    def test_wd_scale_invariance_polarity_and_shift(self):
        metric = ShapeWD(time_bin_ms=20, n_depth=4)
        a = np.zeros((4, 200)); a[1, 20:40] = -1; a[2, 20:40] = 1
        self.assertAlmostEqual(metric(a, a*10)["wd"], 0, places=10)
        self.assertGreater(metric(a, -a)["wd"], 0)
        self.assertGreater(metric(a, np.roll(a, 40, axis=1))["wd"], 0)
        self.assertAlmostEqual(metric(a, np.zeros_like(a))["wd"], 2)


if __name__ == "__main__":
    unittest.main(argv=["test_qc_templates"], verbosity=2)
