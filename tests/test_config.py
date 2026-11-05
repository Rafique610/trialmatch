from src.config import settings


def test_split_matches_brief():
    assert settings.split_seed == 42
    assert settings.split_sizes == (37, 8, 8)
