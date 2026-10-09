"""Build and test each supported wheel extra in an isolated, disposable venv."""

import json
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRAS = ("", "postgres", "s3", "postgres,s3")
PROBE = r"""
import importlib
import importlib.metadata as metadata
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

import vv_agent
from vv_agent import Agent, RunConfig, Runner, ScriptedModelProvider, SQLiteStore
from vv_agent.constants import get_default_tool_schemas
from vv_agent.session.surfaces import SessionDriver
from vv_agent.types import LLMResponse

def packages():
    return {d.metadata['Name'].lower().replace('_', '-'): d.version for d in metadata.distributions()}

def resolve(path):
    if path.startswith('list[') and path.endswith(']'):
        return list[resolve(path[5:-1])]
    parts = path.split('.')
    for count in range(len(parts), 0, -1):
        name = '.'.join(parts[:count])
        try:
            value = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name != name and not name.startswith(exc.name + '.'):
                raise
            continue
        for attribute in parts[count:]:
            value = getattr(value, attribute)
        return value
    raise AssertionError(path)

extras = sys.argv[2].split(',') if sys.argv[2] else []
fixture = json.loads(Path(sys.argv[1]).read_text())
exports = [resolve(c['python']) for d in fixture['domains'] for c in d['capabilities']]
for name in vv_agent.__all__:
    assert getattr(vv_agent, name) is not None, name
for surface in fixture['surfaces']:
    resolve(surface['python_target'])
    for group in ('members', 'protocol_operations', 'supporting_operations'):
        for member in surface.get(group, []):
            mapping = member['python']
            if mapping['kind'] in ('method', 'function'):
                assert callable(getattr(resolve(mapping.get('target', surface['python_target'])), mapping['name']))
assert get_default_tool_schemas(), 'builtin schemas absent'
installed = packages()
for extra, package in (('postgres', 'psycopg'), ('s3', 'boto3')):
    assert (package in installed) == (extra in extras), installed
for package in ('redis', 'celery'):
    assert package not in installed and package not in sys.modules
if not extras:
    assert not any(p in installed for p in ('psycopg', 'psycopg-binary', 'boto3', 'redis', 'celery'))
    assert not any(p in sys.modules for p in ('psycopg', 'boto3', 'redis', 'celery'))

def turn(workspace, store=None):
    config = RunConfig(model_provider=ScriptedModelProvider.new('test', 'm', [LLMResponse('done')]),
                       workspace=workspace, session_memory_enabled=False)
    driver = SessionDriver(store=store) if store is not None else None
    # Runner fixes SQLite memory internally; only replace its store-constructor binding.
    # The real Runner, handle, driver, model and SQL operations all run unchanged.
    try:
        with patch('vv_agent.session.surfaces.SessionDriver', return_value=driver) if driver else nullcontext():
            result = Runner.run_sync(Agent('install', 'Finish.', model='m'), 'go', run_config=config)
        assert result.final_output == 'done' and result.status.value == 'completed'
    finally:
        if driver:
            driver.close()

with TemporaryDirectory(prefix='vvsk-install-turn-') as directory:
    workspace = Path(directory)
    turn(workspace)
    with SQLiteStore.standalone(str(workspace / 'session.sqlite')) as store:
        store.install_schema()
        turn(workspace, store)
    pg = 'not applicable'
    if 'postgres' in extras:
        dsn = os.environ.get('VV_AGENT_TEST_POSTGRES_DSN')
        pg = 'skipped: VV_AGENT_TEST_POSTGRES_DSN is not configured'
        if dsn:
            import psycopg
            from psycopg import sql
            from psycopg.conninfo import make_conninfo
            from vv_agent import PostgresStore
            name = 'vvsk_test_install_' + uuid4().hex
            with psycopg.connect(dsn, autocommit=True) as admin:
                admin.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(name)))
                try:
                    with PostgresStore.standalone(make_conninfo(dsn, dbname=name)) as store:
                        store.install_schema()
                        turn(workspace, store)
                    pg = 'passed'
                finally:
                    admin.execute(sql.SQL('DROP DATABASE {}').format(sql.Identifier(name)))
print(json.dumps({'extra': sys.argv[2] or 'base', 'version': metadata.version('vv-agent'),
                  'api_capabilities': len(exports), 'sqlite_memory': 'passed', 'sqlite_file': 'passed',
                  'postgres': pg, 'packages': installed}, sort_keys=True))
"""


def command(args, *, cwd, env):
    result = subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    return result.stdout


def check_artifacts(sdist, wheel):
    with tarfile.open(sdist) as archive:
        members = archive.getnames()
        forbidden = {"tests", "fixtures", "tmp", ".venv", "__pycache__", ".pytest_cache", ".git", "local_settings.py"}
        assert not [name for name in members if forbidden.intersection(Path(name).parts)], "sdist contains development junk"
    with zipfile.ZipFile(wheel) as archive:
        files = [p for p in (ROOT / "src/vv_agent").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
        for path in files:
            name = path.relative_to(ROOT / "src").as_posix()
            assert archive.read(name) == path.read_bytes(), f"wheel missing or changed package data: {name}"
    return {"sdist_members": len(members), "wheel_package_files": len(files), "package_data": "passed"}


def main():
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "VIRTUAL_ENV"}}
    env["PYTHONNOUSERSITE"] = "1"
    with tempfile.TemporaryDirectory(prefix="vv-agent-install-matrix-") as directory:
        temp = Path(directory)
        command(["uv", "build", "--out-dir", str(temp / "dist")], cwd=ROOT, env=env)
        wheel = next((temp / "dist").glob("*.whl"))
        artifacts = check_artifacts(next((temp / "dist").glob("*.tar.gz")), wheel)
        probe = temp / "probe.py"
        probe.write_text(PROBE)
        results = []
        for index, extra in enumerate(EXTRAS):
            venv = temp / f"venv-{index}"
            command(["uv", "venv", "--python", sys.executable, str(venv)], cwd=temp, env=env)
            python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            requirement = str(wheel) + (f"[{extra}]" if extra else "")
            command(["uv", "pip", "install", "--python", str(python), requirement], cwd=temp, env=env)
            args = [str(python), str(probe), str(ROOT / "tests/fixtures/parity/public_api.json"), extra]
            result = json.loads(command(args, cwd=temp, env=env))
            before = result.pop("packages")
            for retired in ("redis", "celery"):
                command(["uv", "pip", "install", "--python", str(python), f"{wheel}[{retired}]"], cwd=temp, env=env)
                after = json.loads(
                    command(
                        [
                            str(python),
                            "-c",
                            "import importlib.metadata as m,json; print(json.dumps({"
                            "d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}))",
                        ],
                        cwd=temp,
                        env=env,
                    )
                )
                assert before == after, f"removed extra {retired} installed additional packages"
            result["removed_extras"] = "both install no additional packages"
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
        print(json.dumps({"artifacts": artifacts, "matrix": results}, sort_keys=True))


if __name__ == "__main__":
    main()
