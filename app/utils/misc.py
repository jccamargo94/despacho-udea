"""Back-compat shim. The download logic now lives in app.data.download."""

from app.data.download import (  # noqa: F401
    PARAMS,
    XM_DOWNLOAD_URL,
    save_file,
    ensure_data_for_date,
)
