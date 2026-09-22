import importlib

import geodesiq

about_module = importlib.import_module("geodesiq.about")
about = geodesiq.about


def test_about_prints_expected_sections(capsys):
    about()
    captured = capsys.readouterr()

    expected_labels = ["geodesiq: geometric optimal control", "geodesiq Version:", "Numpy Version:", "Scipy Version:",
                       "QuTiP Version:", "Matplotlib Version:", "Python Version:", "Number of CPUs:",
                       "Platform Info:", ]

    for label in expected_labels:
        assert label in captured.out


def test_module_version_returns_none_for_missing_modules():
    assert about_module._module_version("geodesiq_definitely_missing_module") == "None"
