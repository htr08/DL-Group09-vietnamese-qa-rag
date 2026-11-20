import sys

import pytest

import scripts.build_index as build_index


class FakeRun:
    def __init__(self):
        self.summary = {}
        self.finished = False

    def finish(self):
        self.finished = True


def test_log_index_run_records_chunks_and_encode_time(monkeypatch):
    calls = {}
    run = FakeRun()

    def fake_init_run(name, job_type, config=None):
        calls.update(name=name, job_type=job_type, config=config)
        return run

    monkeypatch.setattr(build_index, "init_run", fake_init_run)
    build_index.log_index_run("domain", 214, 12.3456)

    assert calls["name"] == "B_bge-m3_index-domain"
    assert calls["job_type"] == "build_index"
    assert calls["config"]["sliding_max_words"] == build_index.SLIDING_MAX_WORDS
    assert run.summary == {"n_chunks": 214, "encode_time_s": 12.35}
    assert run.finished


@pytest.mark.parametrize("extra_args, expected", [([], False), (["--wandb"], True)])
def test_wandb_flag_is_forwarded_to_builder(monkeypatch, extra_args, expected):
    seen = {}
    monkeypatch.setattr(build_index, "build_domain_index", lambda log_wandb=False: seen.update(flag=log_wandb))
    monkeypatch.setattr(sys, "argv", ["build_index.py", "--corpus", "domain", *extra_args])

    build_index.main()

    assert seen["flag"] is expected
