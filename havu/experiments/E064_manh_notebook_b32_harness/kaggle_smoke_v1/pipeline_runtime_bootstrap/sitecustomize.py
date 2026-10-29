"""Activate a flat target in every fresh worker, including EngineCore spawn."""

import os

if os.environ.get("ARC_PIPELINE_ROLE"):
    try:
        from pipeline_runtime import activate_runtime

        activate_runtime()
    except BaseException:
        # Python normally prints and ignores sitecustomize exceptions. Never
        # continue with the wrong Torch stack after an activation failure.
        import traceback

        traceback.print_exc()
        os._exit(78)
