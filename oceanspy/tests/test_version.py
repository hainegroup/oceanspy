import oceanspy


def test_version() -> None:
    assert oceanspy.__version__ != "999"
