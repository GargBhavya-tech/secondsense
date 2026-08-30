"""
`room3d.simple` is now merged into `room3d.app` — the accumulating coloured point-cloud output
IS the default of `app.py`. This shim is kept so old commands still work.

    python -m room3d.app --source phone:10.156.105.9 --seconds 180     # same thing
"""

from __future__ import annotations

from .app import main                       # noqa: F401  (re-export: `python -m room3d.simple`)
from .depth import RelDepth, backproject    # noqa: F401  (back-compat re-export)

if __name__ == "__main__":
    main()
