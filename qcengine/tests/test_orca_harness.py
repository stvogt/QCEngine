"""Tests for ORCA harness program detection.

Many Linux distributions ship the GNOME screen reader as ``orca`` in /usr/bin, so
a bare ``which("orca")`` hit says nothing about whether the quantum-chemistry
program is present. Reporting found()=True for it aborted QCFractal compute
manager startup entirely.
"""
import pytest


def _fake_orca(tmp_path, stdout_text):
    # `echo` is a shell builtin, so the stub still works when PATH is trimmed to
    # just tmp_path (an external `cat` would not be found and would emit nothing,
    # making every stub look like "not ORCA" for the wrong reason).
    exe = tmp_path / "orca"
    exe.write_text('#!/bin/sh\necho "' + stdout_text + '"\n')
    exe.chmod(0o755)
    return exe


def _clear_orca_caches():
    from qcengine.programs.orca import OrcaHarness
    OrcaHarness._identity_cache.clear()
    OrcaHarness.version_cache.clear()


def test_orca_found_false_for_gnome_screen_reader(tmp_path, monkeypatch):
    """Most Linux distros ship the GNOME screen reader as /usr/bin/orca. Reporting
    found()=True for it aborted QCFractal manager startup entirely, because the
    manager builds its program list by calling get_version() on every *available*
    program and that raised."""
    from qcengine.programs.orca import OrcaHarness
    _clear_orca_caches()
    _fake_orca(tmp_path, "orca 46.2")          # screen-reader style output
    monkeypatch.setenv("PATH", str(tmp_path), prepend=False)
    assert OrcaHarness.found() is False


def test_orca_found_true_for_real_orca(tmp_path, monkeypatch):
    from qcengine.programs.orca import OrcaHarness
    _clear_orca_caches()
    _fake_orca(tmp_path, "Program Version 6.1.1  - RELEASE -")
    monkeypatch.setenv("PATH", str(tmp_path), prepend=False)
    assert OrcaHarness.found() is True
    assert OrcaHarness().get_version() == "6.1.1"


def test_program_enumeration_survives_a_bogus_orca(tmp_path, monkeypatch):
    """The exact call QCFractal's compute manager makes at startup."""
    import qcengine
    from qcengine.programs.orca import OrcaHarness
    _clear_orca_caches()
    _fake_orca(tmp_path, "orca 46.2")
    monkeypatch.setenv("PATH", str(tmp_path), prepend=False)
    assert "orca" not in qcengine.list_available_programs()
