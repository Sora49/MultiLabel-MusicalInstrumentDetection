from app import segments


def test_segments_merges_consecutive_windows():
    result = segments([0, 3, 6], [True, True, False], duration=10)
    assert result == [(0, 6.0)]


def test_segments_caps_end_at_duration():
    result = segments([9], [True], duration=10)
    assert result == [(9, 10)]


def test_segments_returns_empty_when_none_present():
    result = segments([0, 3], [False, False], duration=10)
    assert result == []
