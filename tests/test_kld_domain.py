import math

import llama_bench.kld_domain as kld_domain


def test_pixel_to_y_and_y_to_pixel_linear_round_trip() -> None:
    y = kld_domain.pixel_to_y(150.0, 100.0, 200.0, 0.0, 10.0, "linear")

    assert y == 5.0
    assert kld_domain.y_to_pixel(5.0, 100.0, 200.0, 0.0, 10.0, "linear") == 150.0


def test_pixel_to_y_and_y_to_pixel_log_round_trip() -> None:
    y = kld_domain.pixel_to_y(150.0, 100.0, 200.0, 1e-2, 1e-4, "log")

    assert y is not None
    assert math.isclose(y, 1e-3)

    px = kld_domain.y_to_pixel(y, 100.0, 200.0, 1e-2, 1e-4, "log")
    assert px is not None
    assert math.isclose(px, 150.0)


def test_calibration_conversion_rejects_degenerate_inputs() -> None:
    assert kld_domain.pixel_to_y(150.0, 100.0, 100.0, 0.0, 10.0, "linear") is None
    assert kld_domain.y_to_pixel(5.0, 100.0, 200.0, 10.0, 10.0, "linear") is None
    assert kld_domain.pixel_to_y(150.0, 100.0, 200.0, -1.0, 10.0, "log") is None
    assert kld_domain.y_to_pixel(-1.0, 100.0, 200.0, 1.0, 10.0, "log") is None


def test_major_ticks_match_existing_log_exponent_padding() -> None:
    assert kld_domain.major_ticks(1e-3, 1e-1, "log") == [
        (1e-4, "10^-4"),
        (1e-3, "10^-3"),
        (1e-2, "10^-2"),
        (1e-1, "10^-1"),
        (1.0, "10^0"),
    ]


def test_minor_log_ticks_match_existing_exponent_padding() -> None:
    ticks = kld_domain.minor_log_ticks(1e-3, 1e-1)

    assert len(ticks) == 40
    assert all(math.isclose(actual, expected) for actual, expected in zip(ticks[:3], [2e-4, 3e-4, 4e-4]))
    assert all(math.isclose(actual, expected) for actual, expected in zip(ticks[-3:], [7.0, 8.0, 9.0]))


def test_measurement_csv_row_rounds_value_and_formats_scientific() -> None:
    assert kld_domain.measurement_csv_row("Q4_K_M", 0.001234) == {
        "label": "Q4_K_M",
        "y_value": 0.00123,
        "y_value_scientific": "1.23e-3",
    }
