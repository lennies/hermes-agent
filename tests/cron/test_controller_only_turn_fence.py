import pytest

from cron.scheduler import run_job
from hermes_cli.profiles import ProfileDispatchDeniedError


def test_agent_cron_rechecks_active_profile_before_agent_import(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name", lambda: "governed"
    )
    monkeypatch.setattr(
        "hermes_cli.profiles.require_unclaimed_profile_turn",
        lambda _name: (_ for _ in ()).throw(ProfileDispatchDeniedError("denied")),
    )

    with pytest.raises(ProfileDispatchDeniedError, match="denied"):
        run_job({"id": "job", "name": "job", "prompt": "do work"})
