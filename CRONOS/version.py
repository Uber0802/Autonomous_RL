"""Single source of truth for the CRONOS version string.

`main.py::_dump_run_config` stamps this into every run's `run_config.json`, so it
is the provenance marker that says which code produced a set of results. It used
to be a literal `"V0.3"` inlined at the write site, which stopped tracking
reality around the V0.9x series — every run since was labelled with a version it
was not.

**Bump this in the same commit that moves the git tag.** Nothing derives it
automatically: `git describe` is unavailable in a tarball export, and reading it
at import time would make the stamp depend on whether the run happened inside a
checkout.

Deliberately dependency-free so anything can import it — including the offline
plot tools, which run without a GPU stack.
"""

__version__ = "V0.93"

# Config-file format version. Tracked separately because the YAML schema changes
# far less often than the code. NOTE: the `cronos_version` key inside a config is
# accepted by `envs/config.py` but never read or validated — it is a human
# annotation only, not a compatibility gate.
CONFIG_FORMAT_VERSION = "V0.4"
