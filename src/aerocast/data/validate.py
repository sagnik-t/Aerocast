"""
Great Expectations validation for raw and processed DataFrames.

Uses an ephemeral (in-memory) GE context so no gx/ directory is required.
Returns a structured result dict that callers can log or act on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# Allowed parameter values in raw measurements
_VALID_PARAMETERS = {"pm25", "pm10", "o3", "no2", "so2", "co"}


@dataclass
class ValidationResult:
    suite_name: str
    success: bool
    failed_expectations: list[str] = field(default_factory=list)
    statistics: dict = field(default_factory=dict)

    def raise_on_failure(self) -> None:
        if not self.success:
            raise ValueError(
                f"Validation suite '{self.suite_name}' failed. "
                f"Failed expectations: {self.failed_expectations}"
            )


def validate_raw(df: pd.DataFrame) -> ValidationResult:
    """
    Validate the raw long-form OpenAQ DataFrame.

    Checks:
    - Required columns exist
    - No nulls in critical columns
    - Value is non-negative
    - Parameter is in the known set
    - datetime is parseable
    """
    try:
        import great_expectations as gx
        import great_expectations.expectations as gxe
        from great_expectations.core import ExpectationSuite
    except ImportError as exc:
        raise ImportError("Install great-expectations to use validation.") from exc

    context = gx.get_context(mode="ephemeral")

    datasource = context.data_sources.add_pandas("openaq_raw_source")
    asset = datasource.add_dataframe_asset("raw_measurements")
    batch_def = asset.add_batch_definition_whole_dataframe("raw_batch")

    suite = context.suites.add(ExpectationSuite(name="raw_suite"))

    required_cols = ["location_id", "parameter", "datetime", "value", "unit"]
    for col in required_cols:
        suite.add_expectation(gxe.ExpectColumnToExist(column=col))

    for col in ["location_id", "parameter", "datetime", "value"]:
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=col))

    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="value", min_value=0, max_value=10_000)
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="parameter", value_set=list(_VALID_PARAMETERS)
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToMatchRegex(
            column="datetime",
            regex=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
        )
    )

    vd = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="raw_validation",
            data=batch_def,
            suite=suite,
        )
    )
    result = vd.run(batch_parameters={"dataframe": df})

    failed = [str(r.expectation_config.type) for r in result.results if not r.success]
    stats = {
        "evaluated": len(result.results),
        "successful": sum(1 for r in result.results if r.success),
        "failed": len(failed),
    }
    vr = ValidationResult(
        suite_name="raw_suite",
        success=bool(result.success),
        failed_expectations=failed,
        statistics=stats,
    )
    _log_result(vr)
    return vr


def validate_processed(df: pd.DataFrame) -> ValidationResult:
    """
    Validate the wide, hourly, AQI-enriched DataFrame.

    Checks:
    - Required columns exist
    - aqi is between 0 and 500
    - datetime and location_id are non-null
    - No duplicate (datetime, location_id) pairs
    """
    try:
        import great_expectations as gx
        import great_expectations.expectations as gxe
        from great_expectations.core import ExpectationSuite
    except ImportError as exc:
        raise ImportError("Install great-expectations to use validation.") from exc

    context = gx.get_context(mode="ephemeral")

    datasource = context.data_sources.add_pandas("openaq_processed_source")
    asset = datasource.add_dataframe_asset("processed_measurements")
    batch_def = asset.add_batch_definition_whole_dataframe("processed_batch")

    suite = context.suites.add(ExpectationSuite(name="processed_suite"))

    for col in ["datetime", "location_id", "aqi"]:
        suite.add_expectation(gxe.ExpectColumnToExist(column=col))
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=col))

    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="aqi", min_value=0, max_value=500)
    )
    suite.add_expectation(
        gxe.ExpectCompoundColumnsToBeUnique(column_list=["datetime", "location_id"])
    )

    vd = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="processed_validation",
            data=batch_def,
            suite=suite,
        )
    )
    result = vd.run(batch_parameters={"dataframe": df})

    failed = [str(r.expectation_config.type) for r in result.results if not r.success]
    stats = {
        "evaluated": len(result.results),
        "successful": sum(1 for r in result.results if r.success),
        "failed": len(failed),
    }
    vr = ValidationResult(
        suite_name="processed_suite",
        success=bool(result.success),
        failed_expectations=failed,
        statistics=stats,
    )
    _log_result(vr)
    return vr


def _log_result(vr: ValidationResult) -> None:
    if vr.success:
        logger.info(
            "✓ Validation '%s' passed (%d/%d expectations).",
            vr.suite_name,
            vr.statistics["successful"],
            vr.statistics["evaluated"],
        )
    else:
        logger.warning(
            "✗ Validation '%s' FAILED. Failed: %s",
            vr.suite_name,
            vr.failed_expectations,
        )
