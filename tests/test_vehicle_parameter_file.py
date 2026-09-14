from __future__ import annotations

from pathlib import Path

import yaml


def test_kit_parameter_file_matches_uploaded_parameter_m():
    path = Path(__file__).parents[1] / "configs" / "vehicles" / "kit_ioniq5.yaml"
    cfg = yaml.safe_load(path.read_text())
    assert cfg["mass_kg"] == 2254.0
    assert cfg["yaw_inertia_kgm2"] == 2193.0
    assert cfg["lf_m"] == 1.563
    assert cfg["lr_m"] == 1.437
    assert cfg["cornering_stiffness_front_nprad"] == 234008.0
    assert cfg["cornering_stiffness_rear_nprad"] == 234008.0
