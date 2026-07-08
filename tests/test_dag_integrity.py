import os
import tempfile

# Configure Airflow to use a local SQLite database for testing,
# preventing it from loading other database drivers (like MySQL or Postgres) from the host machine's airflow.cfg.
temp_db_path = os.path.join(tempfile.gettempdir(), "airflow_test.db")
os.environ["AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"] = f"sqlite:////{temp_db_path}"
os.environ["AIRFLOW__CORE__UNIT_TEST_MODE"] = "True"

import unittest  # noqa: E402
from airflow.models import DagBag  # noqa: E402


class TestDagIntegrity(unittest.TestCase):
    def test_dagbag_clean_import(self):
        """
        Verify that all DAGs in the airflow/dags directory can be imported
        without syntax errors, cyclic dependencies, or import issues.
        """
        # Ensure we point to the correct DAGs directory relative to this file
        dag_dir = os.path.join(os.path.dirname(__file__), "..", "airflow", "dags")

        # Load the DAGs (handling Airflow 2.x and 3.x parameter changes dynamically)
        import inspect

        sig = inspect.signature(DagBag.__init__)
        kwargs = {"dag_folder": dag_dir}
        if "include_examples" in sig.parameters:
            kwargs["include_examples"] = False
        dagbag = DagBag(**kwargs)

        # Check for import errors
        import_errors = dagbag.import_errors

        # Format error messages if any
        error_msg = ""
        if len(import_errors) > 0:
            error_msg = "\n".join(
                [
                    f"File: {filename}\nError: {error}"
                    for filename, error in import_errors.items()
                ]
            )

        self.assertEqual(
            len(import_errors), 0, f"DAG import errors found:\n{error_msg}"
        )


if __name__ == "__main__":
    unittest.main()
