# PostgreSQL tenant RLS

The base PostgreSQL schema is intentionally unisolated. `install_schema()`
creates the S1 tables, but it does not guess which login belongs to which
tenant. A deployment that claims credential-level tenant isolation installs it
explicitly after creating the schema:

```python
from vv_agent.session.postgres import PostgresStore

with PostgresStore.standalone(migration_dsn) as migration:
    migration.install_schema()
    installed = migration.install_tenant_rls("tenant-a", "vv_tenant_a", password)

assert installed.credential_isolation is True
```

Use `install_tenant_rls()` once for each tenant login and
`install_worker_rls()` once for the separate cross-tenant worker login. The
methods create dedicated `LOGIN` roles on first use and enable plus force RLS
on every tenant-bearing table. The caller supplies each password; no password
is returned, persisted in store records, or included in an installation
result.

## Role boundaries

| Role | Permitted use | RLS behavior |
| --- | --- | --- |
| Migration/owner | Schema DDL and role installation only | Never a runtime tenant credential |
| Tenant login | One tenant, through an explicit transaction context | Rows require the fixed tenant, that `session_user`, and matching `vv.tenant_id` |
| Worker login | Explicit cross-tenant maintenance and execution | Cross-tenant policy; rejected by tenant startup |
| Shared gateway login | Explicit weaker application trust mode | Context-scoped rows, but credentials can impersonate served tenants |

Tenant policies require all of these values on reads and writes:

```sql
session_user = 'vv_tenant_a'
AND tenant_id = 'tenant-a'
AND current_setting('vv.tenant_id', true) = 'tenant-a'
```

The `true` argument to `current_setting` is required so readback fails closed
when the setting is absent. `WITH CHECK` carries the same predicate. The login
cannot switch to another bound tenant, worker, or migration role because it has
no role membership. Grants cover the required `SELECT`, `INSERT`, `UPDATE`, and
`DELETE` operations only; they exclude `TRUNCATE` and schema DDL. The installer
uses psycopg identifier and literal quoting and installs no `SECURITY DEFINER`
function.

## Tenant startup

Use the tenant startup path rather than an arbitrary pooled connection:

```python
with PostgresStore.tenant_standalone(tenant_dsn, tenant_id="tenant-a") as store:
    with store.connection.transaction():
        store.connection.execute(
            "SELECT set_config('vv.tenant_id', %s, true)", ("tenant-a",)
        )
        page = store.read("session-id")
```

Startup rejects a connection whose `session_user` differs from `current_user`,
whose role is a superuser, has `BYPASSRLS`, owns the schema or a session table,
has role memberships, can create schema objects, or lacks the complete bound
policy set. Reusing a tenant A physical connection with a forged tenant B
context exposes no B rows and cannot insert B rows. Reconnection and rerouting
must use the B login.

## Weak modes

`install_worker_rls()` installs a distinct cross-tenant role. It stays outside
the tenant credential boundary and cannot pass tenant startup.

`install_shared_gateway_rls()` is available only for a trusted gateway. Its
schema still enforces context-scoped reads and writes, but the shared credential
can set any served tenant context. Both worker and shared-gateway installation
results report `credential_isolation=False`. They do not satisfy the
credential-isolation claim for a tenant login.

The real A/B acceptance suite starts a disposable TCP/password PostgreSQL
cluster:

```bash
uv run pytest -q tests/test_s1_postgres_rls.py
```

The test skips only when the PostgreSQL client/server programs or psycopg are
unavailable.
