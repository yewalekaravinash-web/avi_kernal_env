"""
conftest.py — pytest root configuration for DataCleaner OpenEnv.

This file must be in the project root so pytest loads it BEFORE it tries to
import any package __init__.py during test collection.

What it does
------------
1.  Injects a minimal openenv stub into sys.modules so the root __init__.py
    (which imports from models.py → openenv) can be imported without openenv-core
    being installed.  This keeps the Phase 1 + Phase 2 test suites fully offline.

2.  Adds the project root to sys.path so `from server.tasks import ...` resolves
    without PYTHONPATH being set externally (pytest invoked from any directory).

Industry standard pattern
-------------------------
Injecting stubs via conftest.py is the canonical pytest approach for testing code
that has optional heavyweight dependencies (like openenv-core) that are unavailable
in a lightweight CI environment.  The stubs expose only the attributes consumed by
the modules under test.
"""
from __future__ import annotations

import sys
import os
import types

# ── 1. Ensure project root is on sys.path ────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── 2. Stub out openenv.core before anything tries to import it ───────────────
# Build a minimal fake module tree that satisfies every import statement in:
#   models.py, server/environment.py, client.py, inference.py
# Only the class/attribute *names* need to exist — the stubs are never exercised
# during test runs because the tests import server.tasks / server.judge directly.

def _make_mod(name: str, **attrs) -> types.ModuleType:
    """Create and register a stub module with the given attributes."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


class _StubBase:
    """Base for stub Pydantic-like model classes used by models.py."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _StubState(_StubBase):
    episode_id: str = ""
    step_count: int = 0


class _StubObservation(_StubBase):
    pass


class _StubAction(_StubBase):
    pass


class _StubStepResult(_StubBase):
    pass


class _StubEnvClient:
    pass


class _StubSyncEnvClient:
    pass


# openenv.core.env_server.types
_types_mod = _make_mod(
    "openenv.core.env_server.types",
    State=_StubState,
    Observation=_StubObservation,
    Action=_StubAction,
)

# openenv.core.env_server.interfaces
_iface_mod = _make_mod(
    "openenv.core.env_server.interfaces",
    Environment=object,
)

# openenv.core.env_server.http_server
def _stub_create_app(*a, **kw):
    """Return a trivial object; never called in tests."""
    try:
        from fastapi import FastAPI
        return FastAPI()
    except ImportError:
        return object()

_make_mod("openenv.core.env_server.http_server", create_app=_stub_create_app)

# openenv.core.env_server
_make_mod("openenv.core.env_server")

# openenv.core.client_types
_make_mod("openenv.core.client_types", StepResult=_StubStepResult)

# openenv.core
_make_mod(
    "openenv.core",
    EnvClient=_StubEnvClient,
    SyncEnvClient=_StubSyncEnvClient,
    StepResult=_StubStepResult,
)

# openenv
_make_mod("openenv")

# ── 3. Ensure pydantic Field / field_validator stubs if pydantic unavailable ─
try:
    import pydantic  # noqa: F401
except ImportError:
    def _field(*a, **kw):
        return None
    _make_mod("pydantic", Field=_field, field_validator=lambda *a, **kw: (lambda f: f))
    _make_mod("pydantic.fields")
