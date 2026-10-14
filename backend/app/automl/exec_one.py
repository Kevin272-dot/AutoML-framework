"""Subprocess entry point for AutoML run execution.

`python -m app.automl.exec_one <run_id> <job_id>` runs in a DETACHED subprocess,
so a uvicorn restart (--reload, crash, deploy) does not kill an in-flight run.
Progress is written to the DB by the runner; the web process tracks the pid.
"""

import sys


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: python -m app.automl.exec_one <run_id> <job_id>", file=sys.stderr)
        sys.exit(2)

    run_id, job_id = sys.argv[1], sys.argv[2]

    from app.automl.runner import execute_run
    from app.db import SessionLocal

    execute_run(run_id, job_id, SessionLocal)


if __name__ == "__main__":
    main()
