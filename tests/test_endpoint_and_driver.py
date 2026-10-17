import random

import pytest

from process_ai.drivers import DriverError, parse_status
from process_ai.endpoint import DryEndpointDetector


def test_detector_waits_for_full_window_then_detects_plateau():
    rng = random.Random(0)
    det = DryEndpointDetector(window_s=30, band_pct=0.5)
    for t in range(0, 29, 2):
        det.update(t, 850 * (1 + rng.gauss(0, 0.001)))
        assert not det.is_stable()
    for t in range(30, 60, 2):
        det.update(t, 850 * (1 + rng.gauss(0, 0.001)))
    assert det.is_stable()
    assert 30 <= det.span_s() < 32


def test_detector_rejects_drifting_signal():
    det = DryEndpointDetector(window_s=30, band_pct=0.5)
    for t in range(0, 120, 2):
        det.update(t, 700 + t * 0.5)
    assert not det.is_stable()


def test_parse_status_line():
    tel = parse_status("T=80.25 P=5.10 PD=812 DOOR=CLOSED HEAT=1 PUMP=1 VENT=0", t=12.0)
    assert (tel.temp_c, tel.pressure_kpa, tel.photodiode, tel.door) == (80.25, 5.1, 812.0, "closed")
    assert tel.heater_on and tel.pump_on and not tel.vent_open
    assert parse_status("T=1 P=1 PD=1 DOOR=JAMMED HEAT=0 PUMP=0 VENT=0", 0).door == "unknown"
    with pytest.raises(DriverError):
        parse_status("T=80.25 P=abc", t=0)
