import unittest
import numpy as np
import pandas as pd

from sampler_core import ARMS, ControlledBatches, build_draw_indices, largest_remainder, sampling_ledger


def fixture():
    rows = []
    counts = {"BRSET": [30, 3, 8, 5], "mBRSET": [20, 2, 7, 5]}
    for domain, group_counts in counts.items():
        for key, count in zip(("00", "01", "10", "11"), group_counts):
            for i in range(count):
                rows.append({"domain": domain, "diabetic_retinopathy": int(key[0]),
                             "macular_edema": int(key[1]), "file": f"{domain}_{key}_{i}", "role": "fit"})
    return pd.DataFrame(rows)


class SamplerTests(unittest.TestCase):
    def test_largest_remainder(self):
        self.assertEqual(largest_remainder(10, [1, 1, 1]).tolist(), [4, 3, 3])

    def test_deterministic_and_in_range(self):
        rows = fixture()
        for arm in ARMS:
            first, _ = build_draw_indices(rows, arm, 640, 17)
            second, _ = build_draw_indices(rows, arm, 640, 17)
            self.assertTrue(np.array_equal(first, second))
            self.assertTrue((first >= 0).all() and (first < len(rows)).all())

    def test_resume_suffix_matches(self):
        rows = fixture()
        for arm in ARMS:
            full = [x for batch in ControlledBatches(rows, arm, 16, 0, 640, 9) for x in batch]
            suffix = [x for batch in ControlledBatches(rows, arm, 16, 320, 640, 9) for x in batch]
            self.assertEqual(full[320:], suffix)

    def test_equal_domain_each_microbatch(self):
        rows = fixture()
        for arm in ("C1_equal_domain", "C3_equal_domain_target_label"):
            for batch in ControlledBatches(rows, arm, 16, 0, 640, 4):
                domains = rows.iloc[[item[0] for item in batch]].domain.value_counts().to_dict()
                self.assertEqual(domains, {"BRSET": 8, "mBRSET": 8})

    def test_exact_total_ledger(self):
        rows = fixture()
        for arm in ARMS:
            ledger = sampling_ledger(rows, arm, 640, 8)
            self.assertEqual(sum(x["draws"] for x in ledger["records"]), 640)
            self.assertTrue(ledger["all_indices_in_range"])


if __name__ == "__main__":
    unittest.main()
