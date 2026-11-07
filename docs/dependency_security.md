# Dependency security review — 2026-11-06

The release audit covers the locked optional dependencies as well as the base
package. Advisory exceptions are explicit IDs in `security.yml`; an unrelated
advisory still fails the job.

## Cryptography: remediated

Tidy3D 2.12.2 still requires `cryptography==48.0.1` on Intel macOS. The tidy3d
extra now requires `cryptography>=50.0.2` and selects Tidy3D 2.11.2 on Intel
macOS (`>=2.11.2,<2.12`), with `>=2.12.2,<3` elsewhere. Universal resolution
selects cryptography 50.0.2 throughout. The three old advisory exceptions
(CVE-2026-69247/69248/69249) are removed.

The lock also upgrades Authlib, joserfc, MCP, FastMCP, Tornado, Pillow,
OpenTelemetry, and dnspython. These updates incorporate the compatible open
Dependabot proposals and remediate dnspython CVE-2026-114605.

## Filelock: upstream constraint remains

Tidy3D 2.11.2 and 2.12.2 both require `filelock>=3.20.1,<4.0.0`. The lock
selects 3.29.6. The current fixes require filelock 4.2.1 or later, making an
upgrade incompatible with the supported Tidy3D dependency metadata. This is a
separate blocker from the resolved cryptography pin.

The alternative Tidy3D 2.10.2 conflicts with our h5py >=3.16 development floor
(it requires h5py <3.15). A downgrade is not part of this release.

| Advisory | Affected API |
|---|---|
| CVE-2026-112940 | SQLite-backed ReadWriteLock on POSIX |
| CVE-2026-112913 | SQLite-backed ReadWriteLock on Windows |
| CVE-2026-112907 | Windows soft-lock oplock timeout |
| CVE-2026-112936 | ReadWriteLock follows a symbolic link |
| CVE-2026-112910 | Windows lock files inherit parent ACLs |
| CVE-2026-112909 | UnixFileLock can truncate a substituted file |
| CVE-2026-112939 | SoftFileLock accepts a forged process identifier |
| CVE-2026-112915 | ReadWriteLock accepts a crafted SQLite database |
| CVE-2026-112935 | SoftFileLock can remove a successor's live lock |

Reachability was checked in the installed source. gds_fdtd does not import
filelock. Tidy3D's `web/cache.py` and JAX's `_src/lru_cache.py` use `FileLock`,
which resolves to UnixFileLock or WindowsFileLock. JAX can also select
SoftFileLock for a bounded cache on a remote filesystem. Neither caller uses
ReadWriteLock. The Unix and Windows FileLock issues are relevant when an
untrusted local account can alter the cache/lock
directory or its ACLs. Use cache directories writable only by their owning
account; shared writable cache directories are outside this exception's
assumptions. This is a qualified exception, not a claim that filelock is fixed.

All nine IDs are temporarily scoped exceptions. Exit criteria: a supported
Tidy3D release permits patched filelock, then update the extra and lock, run
the offline suite and audit, and remove these exceptions. The weekly audit
continues to fail for every other advisory.

Primary references: [Tidy3D metadata](https://pypi.org/pypi/tidy3d/2.12.2/json),
[Filelock security advisories](https://github.com/tox-dev/filelock/security/advisories).
