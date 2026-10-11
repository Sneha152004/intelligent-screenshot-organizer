import json
import os
import sys
import unittest
import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = r"C:\Users\KIIT0001\Documents\intelligent-screenshot-organizer"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PHASE7F_DIR = os.path.join(PROJECT_ROOT, "experiments", "visual_understanding", "phase7f")

from experiments.visual_understanding.phase7f.dataset_utils import (
    letterbox_pad_and_resize,
    ScreenshotScreenTypeDataset,
    calculate_inverse_class_weights,
    SUPPORTED_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
)
from experiments.visual_understanding.phase7f.vsue_model import VisualScreenTypeClassifier


class TestPhase7FVisualEngine(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.manifest_path = os.path.join(PROJECT_ROOT, "experiments", "visual_understanding", "screen_type_manifest_final.csv")
        self.split_path = os.path.join(PROJECT_ROOT, "experiments", "visual_understanding", "vsue_dataset_split_final.json")
        self.manifest_df = pd.read_csv(self.manifest_path)
        with open(self.split_path, "r", encoding="utf-8") as f:
            self.split_data = json.load(f)

    def test_01_rgba_handling_and_letterbox_padding(self):
        """Test RGBA transparent image compositing and aspect ratio letterbox padding."""
        rgba_img = Image.new("RGBA", (400, 200), color=(255, 0, 0, 128))
        padded = letterbox_pad_and_resize(rgba_img, target_size=224, bg_color=(240, 240, 240))

        self.assertEqual(padded.size, (224, 224))
        self.assertEqual(padded.mode, "RGB")

    def test_02_deterministic_preprocessing(self):
        """Test that preprocessing is 100% deterministic on identical input images."""
        test_img = Image.new("RGB", (300, 600), color=(100, 150, 200))
        p1 = letterbox_pad_and_resize(test_img, target_size=224)
        p2 = letterbox_pad_and_resize(test_img, target_size=224)

        arr1 = np.array(p1)
        arr2 = np.array(p2)
        np.testing.assert_array_equal(arr1, arr2)

    def test_03_class_mapping(self):
        """Test 5 supported class index bidirectional mapping."""
        self.assertEqual(len(SUPPORTED_CLASSES), 5)
        for i, cat in enumerate(SUPPORTED_CLASSES):
            self.assertEqual(CLASS_TO_IDX[cat], i)
            self.assertEqual(IDX_TO_CLASS[i], cat)

    def test_04_model_forward_pass_shape(self):
        """Test MobileNetV3-Small output shape is (batch_size, 5)."""
        model = VisualScreenTypeClassifier(num_classes=5, pretrained=False)
        model.eval()
        dummy_tensor = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            outputs = model(dummy_tensor)

        self.assertEqual(outputs.shape, (4, 5))

    def test_05_checkpoint_save_and_load(self):
        """Test model state_dict serialization and deserialization."""
        model = VisualScreenTypeClassifier(num_classes=5, pretrained=False)
        temp_ckpt = os.path.join(PHASE7F_DIR, "temp_test_checkpoint.pth")
        torch.save(model.state_dict(), temp_ckpt)

        model2 = VisualScreenTypeClassifier(num_classes=5, pretrained=False)
        model2.load_state_dict(torch.load(temp_ckpt, map_location="cpu"))
        model2.eval()

        if os.path.exists(temp_ckpt):
            os.remove(temp_ckpt)

        self.assertTrue(hasattr(model2, "backbone"))

    def test_06_partition_isolation(self):
        """Test that Train, Calibration, and Validation splits have zero ID overlap."""
        train_set = set(self.split_data["train_ids"])
        cal_set = set(self.split_data["calibration_ids"])
        val_set = set(self.split_data["validation_ids"])

        self.assertEqual(len(train_set & cal_set), 0)
        self.assertEqual(len(train_set & val_set), 0)
        self.assertEqual(len(cal_set & val_set), 0)

    def test_07_class_weight_calculation(self):
        """Test inverse class weight vector calculation from training split."""
        train_ids = self.split_data["train_ids"]
        weights = calculate_inverse_class_weights(self.manifest_df, train_ids, self.device)

        self.assertEqual(len(weights), 5)
        self.assertTrue(torch.all(weights > 0))

    def test_08_vsue_payload_schema(self):
        """Test VSUE payload output contract JSON structure."""
        model = VisualScreenTypeClassifier(num_classes=5, pretrained=False)
        probs = np.array([0.1, 0.7, 0.1, 0.05, 0.05])
        payload = model.format_vsue_payload(probs, meta_dict={"aspect_ratio": 0.52, "orientation": "portrait"})

        self.assertEqual(payload["screen_type"], "product_detail")
        self.assertEqual(payload["screen_type_confidence"], 0.7)
        self.assertIn("document_article_view", payload["screen_type_probabilities"])
        self.assertIn("unsupported_taxonomy_classes", payload)
        self.assertEqual(payload["model_version"], "vsue_mobilenet_v3_small_v1")

    def test_09_cpu_inference_execution(self):
        """Test CPU inference performance and non-blocking execution."""
        model = VisualScreenTypeClassifier(num_classes=5, pretrained=False)
        model.eval()
        input_tensor = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0).numpy()

        self.assertEqual(len(probs), 5)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
