# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the cohort statistics and fail-closed eligibility."""
from __future__ import annotations

import json

import pytest

from research.joint_time_oracle.cohort_statistics import (
    analyze, html_report, qualified, read_records,
)


def entry(t, *, scaled=True, physical=True):
    return dict(duration_s=t, solver_converged=True,
                verification={"physical":physical, "scaled95":scaled,
                              "endpoint":True, "dense":True,
                              "collision_independent":True,
                              "dynamics_independent":True})


def records():
    return [
        dict(case_id="pilot_free_0",cohort="pilot",population="free",
             methods={"A10":entry(.5),"A40":entry(.48),
                      "B":entry(.4),"C":entry(.42),"D":entry(.39)}),
        dict(case_id="pilot_free_1",cohort="pilot",population="free",
             methods={"A10":entry(.6),"B":entry(.5,scaled=False),
                      "C":entry(.49),"D":entry(.51)}),
        dict(case_id="confirm_clutter_0",cohort="confirm",population="clutter",
             methods={"A10":entry(.7),"B":entry(.65,physical=False),
                      "C":entry(.6),"D":entry(.61)}),
    ]


def test_qualifications_fail_closed():
    assert qualified(entry(.4),"physical")
    assert not qualified(entry(.4,scaled=False),"scaled95")
    missing=entry(.4)
    del missing["verification"]["collision_independent"]
    assert not qualified(missing,"physical")
    assert not qualified(None,"physical")
    assert not qualified({"duration_s":.4},"physical")


def test_stratified_pairwise_counts_and_signed_improvement():
    summary=analyze(records())
    assert summary["total_tasks"] == 3
    assert summary["groups"]["pilot"]["tasks"] == 2
    assert summary["groups"]["confirm"]["tasks"] == 1
    bc=summary["groups"]["all"]["comparisons"]["physical:B->C"]
    assert bc["eligible_pairs"] == 2
    assert bc["unpaired_or_unqualified"] == 1
    assert bc["improved_count"] == 1
    assert bc["slower_count"] == 1
    assert bc["median_improvement_pct"] == pytest.approx(-1.5)
    bc95=summary["groups"]["all"]["comparisons"]["scaled95:B->C"]
    assert bc95["eligible_pairs"] == 1
    assert bc95["slower_count"] == 1
    bd=summary["groups"]["all"]["comparisons"]["physical:B->D"]
    assert bd["eligible_pairs"] == 2
    assert bd["improved_count"] == 1


def test_io_and_report(tmp_path):
    inp=tmp_path/"cases.jsonl"
    inp.write_text("\n".join(json.dumps(x) for x in records()))
    rows=read_records(inp)
    assert len(rows)==3
    content=html_report(rows,analyze(rows))
    assert "pilot_free_0" in content
    assert "B-&gt;C" in content
    assert "Physical and 95%-scaled" in content
    assert "0.420000" in content


def test_duplicate_rejected(tmp_path):
    inp=tmp_path/"bad.json"
    bad=[records()[0],records()[0]]
    inp.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match="duplicate task"):
        read_records(inp)


def test_invalid_duration_rejected(tmp_path):
    inp=tmp_path/"bad.jsonl"
    bad=records()[0]
    bad["methods"]["B"]["duration_s"]=-1
    inp.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match="invalid B duration"):
        read_records(inp)


def test_direct_analysis_rejects_invalid_duration_and_counts_missing() -> None:
    """Unknown or invalid records cannot enter paired summaries."""
    from research.joint_time_oracle.cohort_statistics import analyze, qualified
    flags = {k: True for k in ('physical', 'scaled95', 'endpoint', 'dense',
                                 'collision_independent', 'dynamics_independent')}
    assert not qualified({'duration_s': float('nan'), 'verification': flags}, 'physical')
    rows = [{'case_id': 'a', 'cohort': 'pilot', 'population': 'free',
             'methods': {'A10': {'duration_s': 1., 'verification': flags}}}]
    pair = analyze(rows)['groups']['pilot']['comparisons']['physical:A10->A40']
    assert pair['eligible_pairs'] == 0
    assert pair['exclusion_reasons'] == {'A40:missing_method': 1}
    assert pair['improved_fraction'] is None


def test_absolute_duration_difference_preserves_negative_savings() -> None:
    """Absolute duration difference is distinct from signed time saved."""
    from research.joint_time_oracle.cohort_statistics import analyze
    flags = {k: True for k in ('physical', 'scaled95', 'endpoint', 'dense',
                                 'collision_independent', 'dynamics_independent')}
    rows = [{'case_id': 'a', 'cohort': 'pilot', 'population': 'free', 'methods': {
        'B': {'duration_s': 1., 'verification': flags},
        'D': {'duration_s': 1.2, 'verification': flags}}}]
    pair = analyze(rows)['groups']['pilot']['comparisons']['physical:B->D']
    assert abs(pair['median_signed_saving_ms'] + 200.) < 1e-8
    assert abs(pair['median_absolute_difference_ms'] - 200.) < 1e-8
    assert pair['slower_fraction'] == 1.
