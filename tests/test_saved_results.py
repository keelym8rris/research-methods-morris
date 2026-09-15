"""Check recorded results and detect changes to the reported summary."""

import hashlib

import pandas as pd
import pytest

from verify_results import main, matches_data_hash


def test_data_hash_allows_line_endings_but_rejects_changed_values():
    content = b'age,lpsa\n63,2.4\n'
    recorded = hashlib.sha256(content).hexdigest()
    assert matches_data_hash(content.replace(b'\n', b'\r\n'), recorded)
    assert not matches_data_hash(content.replace(b'2.4', b'2.5'), recorded)


def test_saved_results():
    main()


def test_changed_summary_is_rejected(monkeypatch):
    read_csv = pd.read_csv

    def changed_summary(path, *args, **kwargs):
        frame = read_csv(path, *args, **kwargs)
        if str(path).endswith('model_cv_summary.csv'):
            frame.loc[0, 'rmse_mean'] += 0.1
        return frame

    monkeypatch.setattr(pd, 'read_csv', changed_summary)
    with pytest.raises(AssertionError):
        main()
