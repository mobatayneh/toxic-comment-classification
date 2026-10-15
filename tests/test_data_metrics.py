from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from toxic_comments.data import LABELS, load_dataset, split_dataset
from toxic_comments.metrics import evaluate_predictions, select_thresholds


def write_rows(path: Path, rows: list[dict], *, columns=None) -> Path:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def row(text: str, value: int = 0, **extra) -> dict:
    return {"comment_text": text, **{label: value for label in LABELS}, **extra}


class DatasetInvariants(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "fixture.csv"

    def test_alias_and_mirror_filter_exclude_unlabeled_rows_before_validation(self):
        valid = row("Synthetic labeled input", 1, id="synthetic-row", set="train")
        valid["severe_tox"] = valid.pop("severe_toxic")
        excluded = {**valid, "comment_text": "", "set": "test"}
        excluded.update({label: -1 for label in LABELS if label != "severe_toxic"})
        excluded["severe_tox"] = -1
        dataset = load_dataset(write_rows(self.path, [valid, excluded]))
        self.assertEqual(len(dataset.frame), 1)
        self.assertEqual(dataset.targets.tolist(), [[1] * 6])
        self.assertEqual(list(dataset.frame.columns), ["comment_text", *LABELS])
        self.assertEqual(dataset.summary["excluded_nontrain_rows"], 1)

    def test_nonbinary_labels_are_rejected(self):
        for invalid in (2, -1, "", "not-a-label"):
            with self.subTest(value=invalid):
                candidate = row("Synthetic invalid-label example")
                candidate["threat"] = invalid
                with self.assertRaises(ValueError):
                    load_dataset(write_rows(self.path, [candidate]))

    def test_missing_text_or_required_column_is_rejected(self):
        for text in ("", " \t\n "):
            with self.subTest(text=repr(text)):
                with self.assertRaises(ValueError):
                    load_dataset(write_rows(self.path, [row(text)]))
        candidate = row("Synthetic incomplete-schema example")
        del candidate["toxic"]
        with self.assertRaises(ValueError):
            load_dataset(write_rows(self.path, [candidate]))

    def test_explicit_empty_text_drop_preserves_row_accounting(self):
        rows = [row("Synthetic retained", 1, set="train"), row(" \t\n", set="train"),
                row("", set="test"), row("Synthetic excluded", set="validation")]
        path = write_rows(self.path, rows)
        with self.assertRaisesRegex(ValueError, "empty or whitespace-only"):
            load_dataset(path)
        dataset = load_dataset(path, empty_text_policy="drop")
        self.assertEqual(dataset.texts.tolist(), ["Synthetic retained"])
        self.assertEqual(dataset.targets.tolist(), [[1] * len(LABELS)])
        self.assertEqual(dataset.summary["excluded_nontrain_rows"], 2)
        self.assertEqual(dataset.summary["excluded_empty_text_rows"], 1)
        self.assertEqual(dataset.summary["empty_text_policy"], "drop")
        self.assertEqual(dataset.summary["input_rows"], dataset.summary["excluded_nontrain_rows"]
                         + dataset.summary["excluded_empty_text_rows"] + dataset.summary["retained_rows"])
        with self.assertRaises(ValueError):
            load_dataset(path, empty_text_policy="impute")

    def test_disagreeing_alias_columns_and_missing_mirror_train_are_rejected(self):
        with self.assertRaises(ValueError):
            load_dataset(write_rows(self.path, [row("Synthetic alias conflict", severe_tox=1)]))
        with self.assertRaises(ValueError):
            load_dataset(write_rows(self.path, [row("Synthetic unlabeled mirror", set="test")]))

    def test_duplicate_headers_and_missing_text_column_are_rejected(self):
        self.path.write_text("comment_text,toxic,toxic\nSynthetic comment,0,1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unique column"):
            load_dataset(self.path)
        candidate = row("Synthetic missing-text-column example")
        del candidate["comment_text"]
        with self.assertRaisesRegex(ValueError, "comment_text"):
            load_dataset(write_rows(self.path, [candidate]))

    def test_identifier_audit_rejects_different_text_even_with_keep_policy(self):
        rows = [row("Synthetic original", id="synthetic-shared"), row("Synthetic other", id="synthetic-shared")]
        for policy in ("error", "keep"):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "different normalized texts"):
                load_dataset(write_rows(self.path, rows), conflict_policy=policy)
        rows = [row("NA", id="synthetic-shared"), row(" na ", id="synthetic-shared"), row("00012", id="")]
        dataset = load_dataset(write_rows(self.path, rows))
        self.assertEqual(dataset.texts.tolist(), ["NA", " na ", "00012"])
        self.assertEqual(dataset.summary["missing_source_ids"], 1)
        self.assertEqual(dataset.summary["duplicate_source_ids"], 1)
        self.assertEqual(dataset.summary["duplicate_source_id_groups"], 1)
        self.assertNotIn("id", dataset.frame.columns)

    def test_normalized_duplicate_groups_never_cross_partitions(self):
        rows = [row(f"Synthetic distinct group {index}", index % 2) for index in range(24)]
        rows.extend([row("Ａlpha\u00a0Beta", 1), row("  alpha\n beta  ", 1)])
        dataset = load_dataset(write_rows(self.path, rows))
        split = split_dataset(dataset, seed=2718)
        arrays = [split.train, split.validation, split.test]
        np.testing.assert_array_equal(np.sort(np.concatenate(arrays)), np.arange(len(rows)))
        groups = [set(dataset.groups[indices]) for indices in arrays]
        for first in range(3):
            for second in range(first + 1, 3):
                self.assertFalse(groups[first] & groups[second])
        self.assertEqual(dataset.groups[-2], dataset.groups[-1])
        self.assertEqual(dataset.texts[-2], "Ａlpha\u00a0Beta", "Model text must retain original content")
        repeated = split_dataset(dataset, seed=2718)
        for name in ("train", "validation", "test"):
            np.testing.assert_array_equal(getattr(split, name), getattr(repeated, name))
        # Assignment is by the normalized group, not the CSV's row order.
        reversed_ds = load_dataset(write_rows(self.path, list(reversed(rows))))
        reversed_split = split_dataset(reversed_ds, seed=2718)
        for name in ("train", "validation", "test"):
            self.assertEqual(set(dataset.groups[getattr(split, name)]), set(reversed_ds.groups[getattr(reversed_split, name)]))

    def test_conflicting_duplicate_labels_require_explicit_policy(self):
        path = write_rows(self.path, [row("Synthetic repeated text", 0), row(" synthetic REPEATED text ", 1)])
        with self.assertRaises(ValueError):
            load_dataset(path)
        accepted = load_dataset(path, conflict_policy="keep")
        self.assertEqual(len(accepted.frame), 2)
        self.assertEqual(accepted.groups[0], accepted.groups[1])
        self.assertEqual(accepted.summary["conflicting_label_groups"], 1)


class MetricInvariants(unittest.TestCase):
    def test_average_precision_matches_hand_calculation_and_not_accuracy(self):
        targets = np.tile(np.array([0, 1, 1, 0])[:, None], (1, 6))
        probabilities = np.tile(np.array([0.9, 0.8, 0.7, 0.1])[:, None], (1, 6))
        report = evaluate_predictions(targets, probabilities, np.full(6, 0.5))
        # Positives occur at ranks 2 and 3: mean precision = (1/2 + 2/3)/2.
        self.assertAlmostEqual(report["macro_average_precision"], 7 / 12)
        # Trapezoidal PR integration for this ranking is 5/12, a different statistic.
        self.assertNotAlmostEqual(report["macro_average_precision"], 5 / 12)
        self.assertAlmostEqual(report["macro_roc_auc"], 0.5)
        self.assertAlmostEqual(report["macro_f1"], 0.8)
        self.assertEqual(report["per_label"]["threat"]["confusion"], {
            "true_positive": 2, "false_positive": 1, "false_negative": 0, "true_negative": 1,
        })

    def test_single_class_labels_do_not_fabricate_full_macro_ranking(self):
        targets = np.tile(np.array([0, 1, 1, 0])[:, None], (1, 6))
        targets[:, 0] = 0
        targets[:, 1] = 1
        scores = np.tile(np.array([0.9, 0.8, 0.7, 0.1])[:, None], (1, 6))
        report = evaluate_predictions(targets, scores, np.full(6, 0.5))
        self.assertIsNone(report["macro_average_precision"])
        self.assertIsNone(report["macro_roc_auc"])
        self.assertIsNone(report["macro_f1"])
        self.assertAlmostEqual(report["macro_average_precision_eligible_labels"], 7 / 12)
        self.assertEqual(report["coverage"]["ranking_eligible_count"], 4)
        self.assertEqual(report["coverage"]["f1_eligible_count"], 5)
        self.assertIsNone(report["per_label"][LABELS[0]]["f1"])
        self.assertIsNone(report["per_label"][LABELS[1]]["average_precision"])
        self.assertEqual(report["per_label"][LABELS[1]]["ranking_status"], "no_negatives")

    def test_empty_positive_coverage_stays_undefined(self):
        report = evaluate_predictions(np.zeros((4, 6)), np.full((4, 6), 0.1), np.full(6, 0.5))
        for key in ("macro_average_precision", "macro_average_precision_eligible_labels", "macro_roc_auc", "macro_f1", "micro_f1"):
            self.assertIsNone(report[key], key)
        self.assertEqual(report["coverage"]["ranking_eligible_count"], 0)

    def test_validation_thresholds_follow_known_f1_optima_and_degenerate_default(self):
        targets = np.tile(np.array([0, 0, 1, 1])[:, None], (1, 6))
        low = np.array([0.1, 0.2, 0.46, 0.49])
        high = np.array([0.6, 0.65, 0.8, 0.9])
        scores = np.column_stack([low, high, low, high, low, high])
        thresholds = select_thresholds(targets, scores, grid=[0.3, 0.5, 0.7])
        np.testing.assert_allclose(thresholds, [0.3, 0.7, 0.3, 0.7, 0.3, 0.7])
        targets[:, 0] = 0
        targets[:, 1] = 1
        thresholds = select_thresholds(targets, scores, grid=[0.3, 0.5, 0.7])
        np.testing.assert_allclose(thresholds[:2], [0.5, 0.5])

    def test_nonfinite_scores_and_invalid_shapes_are_rejected(self):
        targets = np.tile(np.array([0, 1])[:, None], (1, 6))
        for invalid in (float("nan"), float("inf"), 1.01, -0.01):
            with self.subTest(score=invalid):
                scores = np.full((2, 6), 0.5)
                scores[0, 0] = invalid
                with self.assertRaises(ValueError):
                    evaluate_predictions(targets, scores, np.full(6, 0.5))
        with self.assertRaises(ValueError):
            evaluate_predictions(targets, np.ones((2, 5)), np.full(6, 0.5))


if __name__ == "__main__":
    unittest.main()
