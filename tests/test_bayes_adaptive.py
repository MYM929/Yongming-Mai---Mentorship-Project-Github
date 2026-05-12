import copy
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_next_adaptive_config.py"
CONFIG = ROOT / "configs" / "dataset_1_bedroom.json"

spec = importlib.util.spec_from_file_location("adaptive", SCRIPT)
adaptive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adaptive)


def load_config():
    import json

    with CONFIG.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def good_metrics(**overrides):
    metrics = {
        "accepted_frame_ratio": 0.98,
        "num_segments": 1,
        "head_tail_translation_drift_m": 0.05,
        "runtime_per_accepted_frame_sec": 0.6,
        "final_memory_mb": 2500,
        "total_global_lc_added": 25,
        "local_lc_accept_rate": 0.85,
        "orb_glc_accept_rate": 0.7,
        "fpfh_glc_accept_rate": 0.35,
    }
    metrics.update(overrides)
    return metrics


class BayesianAdaptiveTests(unittest.TestCase):
    def test_quality_first_objective_prefers_good_run(self):
        cfg = load_config()
        good = adaptive.evaluate_bayes_objective(good_metrics(), cfg["pose_tracking"])
        weak = adaptive.evaluate_bayes_objective(
            good_metrics(accepted_frame_ratio=0.6),
            cfg["pose_tracking"],
        )

        self.assertEqual(good["constraint_status"], "ok")
        self.assertGreater(good["objective_score"], 0)
        self.assertIn("weak_tracking", weak["constraint_status"])
        self.assertLess(weak["objective_score"], good["objective_score"])

    def test_quality_first_penalizes_segments_drift_and_memory(self):
        cfg = load_config()
        result = adaptive.evaluate_bayes_objective(
            good_metrics(
                num_segments=3,
                head_tail_translation_drift_m=0.8,
                final_memory_mb=4500,
            ),
            cfg["pose_tracking"],
        )

        self.assertIn("multiple_segments", result["constraint_status"])
        self.assertIn("high_drift", result["constraint_status"])
        self.assertIn("high_memory", result["constraint_status"])
        self.assertLess(result["objective_score"], 0)

    def test_early_stopping_requires_enough_trials(self):
        records = [
            {"sequence_number": 1, "objective_score": 0.5},
            {"sequence_number": 2, "objective_score": 0.51},
            {"sequence_number": 3, "objective_score": 0.515},
        ]

        decision = adaptive.should_stop_bayes_early(records)

        self.assertFalse(decision["should_stop"])
        self.assertIn("need at least", decision["reason"])

    def test_early_stopping_detects_plateau(self):
        scores = [0.5, 0.7, 0.8, 0.9]
        scores.extend([0.9005, 0.901, 0.9015, 0.902])
        scores.extend([0.9025, 0.903, 0.9035, 0.904])
        records = [
            {"sequence_number": index, "objective_score": score}
            for index, score in enumerate(scores, start=1)
        ]

        decision = adaptive.should_stop_bayes_early(records)

        self.assertTrue(decision["should_stop"])
        self.assertEqual(decision["best_sequence_number"], 12)

    def test_apply_bayes_params_stays_inside_expected_bounds(self):
        if importlib.util.find_spec("optuna") is None:
            self.skipTest("optuna is not installed in this environment")

        cfg = load_config()
        params = adaptive.extract_bayes_params_from_config(cfg)
        params.update({
            "odometry_scale": 99.0,
            "local_stride": -5,
            "orb_features_per_frame": 12345,
            "fpfh_voxel_size": 0.01,
        })

        next_cfg = adaptive.apply_bayes_params_to_config(copy.deepcopy(cfg), params)
        pt_cfg = next_cfg["pose_tracking"]

        self.assertLessEqual(pt_cfg["odometry"]["scale"], 0.4)
        self.assertGreaterEqual(pt_cfg["local_loop_closure"]["stride"], 3)
        self.assertLessEqual(
            pt_cfg["orb_global_loop_closure"]["features_per_frame"],
            1000,
        )
        self.assertGreaterEqual(pt_cfg["fpfh_global_loop_closure"]["voxel_size"], 0.06)
        self.assertEqual(pt_cfg["caches"], cfg["pose_tracking"]["caches"])

    def test_metric_conservative_still_generates_metadata(self):
        cfg = load_config()
        next_cfg = adaptive.generate_next_config(
            cfg,
            good_metrics(accepted_frame_ratio=0.4),
            adaptive_index=1,
            mode=adaptive.CONSERVATIVE_MODE,
        )

        self.assertEqual(
            next_cfg["adaptive_metadata"]["mode"],
            adaptive.CONSERVATIVE_MODE,
        )
        self.assertTrue(next_cfg["adaptive_metadata"]["reasons"])

    def test_bayes_generation_smoke(self):
        if importlib.util.find_spec("optuna") is None:
            self.skipTest("optuna is not installed in this environment")

        cfg = load_config()
        history = []
        for index, score in enumerate([0.4, 0.6, 0.61, 0.615, 0.616, 0.617], start=1):
            params = adaptive.extract_bayes_params_from_config(cfg)
            params["odometry_scale"] = min(0.4, 0.2 + index * 0.01)
            history.append({
                "sequence_number": index,
                "params": params,
                "objective_score": score,
                "constraint_status": "ok",
                "quality_score": score,
                "speed_score": 0.5,
            })

        next_cfg = adaptive.generate_next_config(
            cfg,
            good_metrics(),
            adaptive_index=6,
            mode=adaptive.BAYES_MODE,
            trial_history=history,
            dataset_name="dataset_1_bedroom",
            batch_run_id="unit-test",
        )

        self.assertEqual(next_cfg["adaptive_metadata"]["mode"], adaptive.BAYES_MODE)
        self.assertEqual(next_cfg["adaptive_metadata"]["history_trials"], 6)
        self.assertIn("suggested_params", next_cfg["adaptive_metadata"])
        self.assertIn("duplicate_suggestion_retries", next_cfg["adaptive_metadata"])

    def test_bayes_generation_avoids_duplicate_params(self):
        if importlib.util.find_spec("optuna") is None:
            self.skipTest("optuna is not installed in this environment")

        cfg = load_config()
        history = []
        for index in range(1, 10):
            params = adaptive.extract_bayes_params_from_config(cfg)
            params["odometry_scale"] = min(0.4, 0.2 + index * 0.01)
            history.append({
                "sequence_number": index,
                "params": params,
                "objective_score": 0.5 + index * 0.01,
                "constraint_status": "ok",
                "quality_score": 0.5 + index * 0.01,
                "speed_score": 0.5,
            })

        next_cfg = adaptive.generate_next_config(
            cfg,
            good_metrics(),
            adaptive_index=9,
            mode=adaptive.BAYES_MODE,
            trial_history=history,
            dataset_name="dataset_1_bedroom",
            batch_run_id="unit-test",
        )
        suggested = next_cfg["adaptive_metadata"]["suggested_params"]
        tried = {adaptive._bayes_param_key(record["params"]) for record in history}

        self.assertNotIn(adaptive._bayes_param_key(suggested), tried)


if __name__ == "__main__":
    unittest.main()
