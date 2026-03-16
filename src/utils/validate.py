import great_expectations as gx
import pandas as pd

# Load dataset
# df = pd.read_csv("mlops\\EDA\\lung_cancer_dataset.csv")

def validate_data(df):
    context = gx.get_context()

    datasource_name = "lung_cancer_datasource"

    # Add datasource if it doesn't exist
    try:
        datasource = context.sources.get(datasource_name)
    except:
        datasource = context.sources.add_pandas(datasource_name)

    # Add dataframe asset (THIS WAS MISSING)
    asset_name = "lung_cancer_data"

    try:
        data_asset = datasource.get_asset(asset_name)
    except:
        data_asset = datasource.add_dataframe_asset(name=asset_name)

    # Build batch request directly from dataframe
    batch_request = data_asset.build_batch_request(dataframe=df)

    # Create expectation suite
    suite_name = "lung_cancer_expectation_suite"

    context.add_or_update_expectation_suite(
        expectation_suite_name=suite_name
    )

    # Get validator
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name,
    )

    # ==================================================
    # Add Expectations
    # ==================================================

    validator.expect_table_columns_to_match_ordered_list([
        'patient_id', 'age', 'gender', 'pack_years',
        'radon_exposure', 'asbestos_exposure',
        'secondhand_smoke_exposure', 'copd_diagnosis',
        'alcohol_consumption', 'family_history', 'lung_cancer'
    ])

    validator.expect_table_row_count_to_equal(50000)

    # validator.expect_column_values_not_to_be_null("patient_id")

    validator.expect_column_values_to_not_be_null("patient_id")

    validator.expect_column_values_to_be_unique("patient_id")

    validator.expect_column_values_to_be_between("age", 0, 120)

    validator.expect_column_values_to_be_between("pack_years", 0, 100)

    # gender

    validator.expect_column_values_to_be_in_set("gender",["Male","Female"])

    binary_columns = [
        "asbestos_exposure",
        "secondhand_smoke_exposure",
        "copd_diagnosis",
        "family_history",
        "lung_cancer"
    ]

    for cols in binary_columns:
        validator.expect_column_values_to_be_in_set(cols,["Yes","No"])

    # "radon_exposure"
    validator.expect_column_values_to_be_in_set("radon_exposure",["High","Low","Medium"])

    # 
    validator.expect_column_values_to_be_in_set("alcohol_consumption",['Heavy', 'Moderate', 'Unknown'])


    # Save
    validator.save_expectation_suite()

    # Validate
    results = validator.validate()

    # print(results)
    failed_expectations = []

    for r in results["results"]:
        if not r["success"]:
            expectation_type = r["expectation_config"]["expectation_type"]
            failed_expectations.append(expectation_type)

        total_checks = len(results["results"])
        passed_checks = sum(1 for r in results["results"] if r["success"])
        failed_checks = total_checks - passed_checks

        if results["success"]:
            print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
        else:
            print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
            print(f"   Failed expectations: {failed_expectations}")

        # print(results["success"], failed_expectations)
        return results["success"], failed_expectations