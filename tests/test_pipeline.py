"""
Comprehensive pipeline test for Siamese-rPPG Deepfake Detection.
Verifies:
1. Loss function behavior & Pearson correlation correctness
2. SiameseRPPGNet tensor shapes, parameter sharing, and forward pass
3. ROI Extraction logic
4. Biological signal analyzer on simulated Real vs AI-Generated video signals
"""

import sys
import os
import unittest
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.losses import NegativePearsonLoss, pearson_correlation
from src.models.siamese_rppg import SiameseRPPGNet
from src.preprocessing.roi_extractor import ROIExtractor
from src.classifier.fake_detector import BiologicalSignalAnalyzer


class TestRPPGPipeline(unittest.TestCase):

    def test_01_negative_pearson_loss(self):
        """Test Pearson loss on known correlation values."""
        criterion = NegativePearsonLoss()

        # Perfect positive correlation (r = 1.0 -> Loss = 0.0)
        t = torch.linspace(0, 10, 300)
        sig1 = torch.sin(t).unsqueeze(0)
        sig2 = torch.sin(t).unsqueeze(0)
        loss_perfect = criterion(sig1, sig2)
        self.assertAlmostEqual(loss_perfect.item(), 0.0, places=4)

        # Perfect negative correlation (r = -1.0 -> Loss = 2.0)
        sig_neg = -torch.sin(t).unsqueeze(0)
        loss_neg = criterion(sig1, sig_neg)
        self.assertAlmostEqual(loss_neg.item(), 2.0, places=4)

        # Gradient flow test
        sig_param = torch.nn.Parameter(torch.randn(2, 100))
        target = torch.randn(2, 100)
        loss = criterion(sig_param, target)
        loss.backward()
        self.assertIsNotNone(sig_param.grad)
        self.assertTrue(torch.all(torch.isfinite(sig_param.grad)))
        print("✓ Test 1: Pearson Loss and gradients passed.")

    def test_02_siamese_architecture_and_weights_sharing(self):
        """Test model forward pass, tensor output shapes, and shared weights mechanism."""
        model = SiameseRPPGNet()
        model.eval()

        batch_size = 2
        time_steps = 30  # short window for quick unit testing
        h, w = 40, 140

        v_forehead = torch.randn(batch_size, 3, time_steps, h, w)
        v_cheek = torch.randn(batch_size, 3, time_steps, h, w)

        with torch.no_grad():
            fused, x_f, x_c = model(v_forehead, v_cheek, return_branches=True)

        # Output shape must be (Batch, Time)
        self.assertEqual(fused.shape, (batch_size, time_steps))
        self.assertEqual(x_f.shape, (batch_size, time_steps))
        self.assertEqual(x_c.shape, (batch_size, time_steps))

        # Check fusion: addition
        diff = torch.max(torch.abs(fused - (x_f + x_c)))
        self.assertLess(diff.item(), 1e-5)

        # Weight sharing check: passing identical input through forehead and cheek must yield identical outputs
        with torch.no_grad():
            _, x_f_same, x_c_same = model(v_forehead, v_forehead, return_branches=True)
            self.assertTrue(torch.allclose(x_f_same, x_c_same, atol=1e-5))

            # Keyword argument backward-compatibility check
            fused_kw = model(v_forehead=v_forehead, v_cheek=v_cheek, return_branches=False)
            self.assertTrue(torch.allclose(fused, fused_kw, atol=1e-5))

        print(f"✓ Test 2: Model forward pass shape {tuple(fused.shape)} and weight sharing verified.")

    def test_03_roi_extractor_mock(self):
        """Test ROI cropping ratios from face bounding box."""
        extractor = ROIExtractor(target_size=(140, 40))

        # Synthetic 480x640 frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Face bbox: x=100, y=100, w=200, h=300
        bbox = (100, 100, 200, 300)

        forehead_roi, cheek_roi = extractor.get_rois_from_face(frame, bbox)

        self.assertEqual(forehead_roi.shape, (40, 140, 3))
        self.assertEqual(cheek_roi.shape, (40, 140, 3))
        print("✓ Test 3: ROI extraction output dimensions verified (40, 140, 3).")

    def test_04_biological_signal_analyzer(self):
        """Test classification between real human periodic pulse vs AI jitter noise."""
        analyzer = BiologicalSignalAnalyzer(fps=30.0)

        t = np.linspace(0, 10, 300)  # 10 seconds at 30 fps
        # Real human: 1.2 Hz pulse (72 bpm) with slight noise, in-phase on forehead and cheek
        pulse_base = np.sin(2 * np.pi * 1.2 * t)
        forehead_real = pulse_base + 0.1 * np.random.randn(len(t))
        cheek_real = pulse_base + 0.1 * np.random.randn(len(t))
        fused_real = forehead_real + cheek_real

        res_real = analyzer.analyze(fused_real, forehead_real, cheek_real)
        self.assertTrue(res_real["is_real"], f"Real signal should be classified as Real. Got: {res_real}")
        self.assertGreater(res_real["authenticity_score"], 0.6)
        self.assertAlmostEqual(res_real["heart_rate_bpm"], 72.0, delta=5.0)

        # AI-Generated / Deepfake: No coherent pulse, uncoordinated noise between forehead and cheek
        forehead_fake = np.random.randn(len(t)) * 0.5
        cheek_fake = np.random.randn(len(t)) * 0.5
        fused_fake = forehead_fake + cheek_fake

        res_fake = analyzer.analyze(fused_fake, forehead_fake, cheek_fake)
        self.assertFalse(res_fake["is_real"], f"Noise signal should be classified as Fake. Got: {res_fake}")
        self.assertLess(res_fake["authenticity_score"], 0.5)

        print(f"✓ Test 4: Real signal score={res_real['authenticity_score']:.2f} ({res_real['prediction']}), "
              f"Fake signal score={res_fake['authenticity_score']:.2f} ({res_fake['prediction']}) verified.")

    def test_05_weights_loading_and_bi_arterial_net(self):
        """Test loading pretrained weights and forward pass of BiArterialDeepfakeNet without rPPG-Toolbox."""
        from src.models.bi_arterial_net import BiArterialDeepfakeNet
        from src.models.siamese_rppg import SiameseRPPG

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        weights_rppg = os.path.join(base_dir, "models", "pretrained_siamese_rppg.pth")
        weights_bi = os.path.join(base_dir, "models", "bi_arterial_deepfake_net.pth")

        # 1. SiameseRPPG loading
        bio_model = SiameseRPPG()
        if os.path.exists(weights_rppg):
            state_dict = torch.load(weights_rppg, map_location="cpu", weights_only=True)
            bio_model.load_state_dict(state_dict)
            bio_model.eval()

        # 2. BiArterialDeepfakeNet loading
        net_model = BiArterialDeepfakeNet()
        if os.path.exists(weights_bi):
            ckpt = torch.load(weights_bi, map_location="cpu", weights_only=True)
            if "model_state_dict" in ckpt:
                net_model.load_state_dict(ckpt["model_state_dict"])
            else:
                net_model.load_state_dict(ckpt)
            net_model.eval()

        # 3. Forward pass verification
        dummy_f = torch.randn(1, 3, 30, 40, 140)
        dummy_c = torch.randn(1, 3, 30, 40, 140)
        with torch.no_grad():
            sig_f = bio_model.shared_branch(dummy_f)
            logits, out_f, out_c = net_model(dummy_f, dummy_c)

        self.assertEqual(sig_f.shape, (1, 30))
        self.assertEqual(logits.shape, (1, 2))
        self.assertEqual(out_f.shape, (1, 30))
        self.assertEqual(out_c.shape, (1, 30))
        print("✓ Test 5: Standalone weights loading and BiArterialDeepfakeNet forward pass verified.")


if __name__ == "__main__":
    unittest.main()
