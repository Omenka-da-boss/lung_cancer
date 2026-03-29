import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset , DataQualityPreset , TargetDriftPreset,ClassificationPreset
from evidently import ColumnMapping
from evidently.metrics import ClassificationQualityMetric,ClassificationClassBalance,ClassificationConfusionMatrix
from evidently.tests import TestAccuracyScore
from evidently.test_preset import BinaryClassificationTestPreset,NoTargetPerformanceTestPreset
# from evidently.options import DataDriftOptions
from evidently.test_suite import TestSuite
import json
import warnings
import pickle
import joblib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore")
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


bin_map = {
    "gender": {"Male":1,"Female":0},
    "asbestos_exposure":{"Yes":1,"No":0},
    "secondhand_smoke_exposure": {"Yes":1,"No":0},
    "family_history": {"Yes":1,"No":0},
    "copd_diagnosis": {"Yes":1,"No":0},
    "lung_cancer": {"Yes":1,"No":0}
    }

multi_map = {
        "radon_exposure" : {"High":2,"Medium":1,"Low":0},
        "alcohol_consumption": {"Heavy":2,"Moderate":1,"Unknown":0}
    }

num_cols = ["pack_years","age"]

def load_artifacts(model_path):
        with open(model_path,"rb") as f:
            model = pickle.load(f)
            if model:
                print("Successfully Loaded")
            
            with open("artifacts\\features_columns.json","rb") as f:
                features = json.load(f)
                features.append("lung_cancer")
                # print(features)
        
        return model,features
    
model,features = load_artifacts("models\\model.pkl")
        
def clean_input(df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()
        df.columns = df.columns.str.strip()
        
        # Numeric columns
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        

        for c, mapping in bin_map.items():
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip().map(mapping).fillna(0).astype(int)
        
        
        for c, mapping in multi_map.items():
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip().map(mapping).fillna(0).astype(int)
        

        bool_cols = df.select_dtypes(include=["bool"]).columns
        if len(bool_cols) > 0:
            df[bool_cols] = df[bool_cols].astype(int)
        
        return df

# Reference/Training Dataset
ref_df = pd.read_csv("data\\lung_cancer_dataset.csv")
# Current/Production Data
current_df = pd.read_csv("monitoring\\current_data.csv")

# Configuring Column Maps

columns = ref_df.columns.tolist()

column_maps = ColumnMapping()

column_maps.id = "patient_id"

# Numerical Data Column
column_maps.numerical_features =  ["age","pack_years"]
# Categorical Data Column
column_maps.categorical_features = ["gender","radon_exposure","asbestos_exposure","secondhand_smoke_exposure","copd_diagnosis","alcohol_consumption","family_history"]
column_maps.target = "lung_cancer"

print("✅ Column mapping configured:")
print(f"Numerical features: {column_maps.numerical_features}")
print(f"Categorical features: {column_maps.categorical_features}")
print(f"Target: {column_maps.target}")

# Data Drift Detection
def data_drift_detection(ref_df,current_df,column_maps,output_path=None):
    data_drift = Report(metrics=[DataDriftPreset()])
    
    data_drift.run(reference_data=ref_df,current_data=current_df,column_mapping=column_maps)
    
    if output_path:
        data_drift.save_html(f"{output_path}/monitoring_html.html")
        print(f"Report is saved at: {output_path}/monitoring_html.html")
        
    report_dict = data_drift.as_dict()
    # data_drift.save_json("mlops\\monitoring\\report.json")
    
    
    # print(report_dict["metrics"][0]['result'])
    
    genral_rep = report_dict["metrics"][0]["result"]
    
    metrics = report_dict["metrics"][1]
    
    # print(metrics["result"])
    

        
    results = metrics["result"]
    
    # {'drift_share': 0.5, 'number_of_columns': 10, 'number_of_drifted_columns': 2, 'share_of_drifted_columns': 0.2, 'dataset_drift': False}
    
    drifted_results = {
        "dataset_drift": genral_rep["dataset_drift"],
        "drift_share": genral_rep["drift_share"],
        'number_of_drifted_columns': genral_rep['number_of_drifted_columns'],
        "drifted_cols": []
    }
    
    # print(drifted_results)
            
            
    # Details for each column
            
    for col_name,col_info in results["drift_by_columns"].items():
           if col_info["drift_detected"]:
                drifted_results["drifted_cols"].append({
                        "col_name":col_name,
                        "drift_score": col_info["drift_score"],
                        "test": col_info["stattest_name"],
                        'stattest_threshold': col_info['stattest_threshold'],
                        "drift_detected": col_info['drift_detected']
                    })
    
    # print(drifted_results)

            
        #Summary
    print(f"\n📈 DRIFT DETECTION SUMMARY:")
    print(f"Dataset drift detected: {'✅ YES' if drifted_results['dataset_drift'] else '✅ NO'}")
    # print(f"Drift score: {drifted_results['drift_score']:.4f}")
    print(f"Number of drifted columns: {drifted_results['number_of_drifted_columns']}") 
            
    if drifted_results["drifted_cols"]:
            print("\n Drifted Columns information")
            for cols in drifted_results["drifted_cols"]:
                print(f"Column Name: {cols['col_name']}")
                print(f"Drift_score: {cols["drift_score"]}")
                print(f"Drift Detected: {cols["drift_detected"]}")
                print(f"Stattest Threshold: {cols['stattest_threshold']}")
                print(f"Test: {cols["test"]}")
    return data_drift, drifted_results

        
# Advanced Data Drift with customs
def advanced_drift_detection(ref_df,current_df,column_maps):
    # data_drift_options = DataDriftOptions()
# patient_id,age,gender,pack_years,radon_exposure,asbestos_exposure,secondhand_smoke_exposure,copd_diagnosis,alcohol_consumption,family_history,lung_cancer
    drift_options = {
        "age": 0.1,
        "gender": 0.03,
        "pack_years": 0.1,
        "radon_exposure": 0.03,
        "asbestos_exposure": 0.03,
        "secondhand_smoke": 0.03,
        "copd_diagnosis": 0.03,
        "alcohol_consumption": 0.03,
        "family_history": 0.03,
    }
    
    column_stattest={
        'age': 'ks',                     # Kolmogorov-Smirnov
        'pack_years': 'wasserstein',     # Skewed (heavy smokers vs light)
        'radon_exposure': 'chisquare', # Likely skewed (most low, some high)
        'asbestos_exposure': 'chisquare', # Typically skewed exposure data
        'alcohol_consumption': 'chisquare', # Often right-skewed distribution
        
        # Binary categorical columns
        'gender': 'z',                   # Z-test for binary (male/female)
        'copd_diagnosis': 'z',           # Binary diagnosis (yes/no)
        'family_history': 'z',           # Binary family history
        'lung_cancer': 'z',              # Binary target (yes/no)
        
        # Categorical columns with >2 categories
        'secondhand_smoke_exposure': 'chisquare',  # Exposure levels (none/light/moderate/heavy)
        }
    metrics = DataDriftPreset(per_column_stattest_threshold=drift_options,per_column_stattest=column_stattest)
    advanced_report =  Report(metrics=[metrics])
    
    advanced_report.run(reference_data=ref_df,current_data=current_df,column_mapping=column_maps)
    
    metrics = advanced_report.as_dict()
    
    results = metrics["metrics"][1]["result"]
    general = metrics["metrics"][0]["result"]
    
    print(f"\n📊 ADVANCED DRIFT RESULTS:")
    print(f"Dataset drift: {general['dataset_drift']}")
    print(f"Drift score: {general['drift_share']:.4f}")
    print(f"Number of Drifted Columns: {general['number_of_drifted_columns']:.4f}")
    
    print("\n Statistics Used on each column")
    
    for cols in results["drift_by_columns"]:
        test_used = results["drift_by_columns"][cols]["stattest_name"]
        values = (f"{cols}:{test_used}")
        print(values)
        
    
    return advanced_report

# Data Quality Monitoring
def data_quality_monitoring(ref_df,current_df,column_maps):
    
    report = Report(metrics=[DataQualityPreset()])
    
    report.run(reference_data=ref_df,current_data=current_df,column_mapping=column_maps)
    
    metrics = report.as_dict()
    
    report.save_html("monitoring\\quality.html")
    
    quality_report = {
        "Report per column": {
            "missing values per columns": [],
            "unique values in each column": []},
        "General Report": {
            "total missing values": [],
            "total columns in dataset": [],
            "duplicate rows": [],
            "duplicate columns": [],
            "No. of numeric columns": [],
            "No. of category columns": [],
            "Total Number of rows": []
        }
    }
    results = metrics["metrics"][0]["result"]
    for col_name,col_info in results.items():
        if col_name == "current":
            quality_report["Report per column"]["missing values per columns"].append({"Reference Data":results["reference"]["nans_by_columns"],"Current Data": results["current"]["nans_by_columns"]})
            
            quality_report["Report per column"]["unique values in each column"].append({"Reference Data":results["reference"]["number_uniques_by_columns"],"Current Data: ": results["current"]["number_uniques_by_columns"]})
            
            quality_report["General Report"]["No. of category columns"].append(results["reference"]["number_of_categorical_columns"])
            quality_report["General Report"]["No. of numeric columns"].append(results["reference"]["number_of_numeric_columns"])
            
            quality_report["General Report"]["duplicate rows"].append({"Reference Dataset: ":results["reference"]["number_of_duplicated_rows"],"Current Dataset: ": results["current"]["number_of_duplicated_rows"]})
            
            quality_report["General Report"]["Total Number of rows"].append({"Reference Dataset: ":results["reference"]["number_of_rows"],"Current Dataset: ": results["current"]["number_of_rows"]})
            
            quality_report["General Report"]["total missing values"].append({"Reference Dataset: ":results["reference"]["number_of_missing_values"],"Current Dataset: ":results["current"]["number_of_missing_values"]})
            
            quality_report["General Report"]["duplicate columns"].append({"Refernce Dataset:":results["reference"]["number_of_duplicated_columns"],"Current Dataset: ":results["current"]["number_of_duplicated_columns"]})
            
            quality_report["General Report"]["total columns in dataset"].append({"Reference Dataset: ":results["reference"]["number_of_columns"],"Current Dataset: ":results["current"]["number_of_columns"]})
            
    # print(quality_report)
    print(f"\nGeneral Dataset Quality Analysis:")
    for cols in quality_report["General Report"]["Total Number of rows"]:
        for col,values in cols.items():
            print(f"Total Number Of Rows: {col}:{values}")
    
    print(f"\nMissing Values Analysis:")
    for cols in quality_report["Report per column"]["missing values per columns"]:
        for col,values in cols.items():
            print(f"{col}: {values}",sep="-")
                
    return report ,quality_report

# Model Performance 

def model_performance(ref_df,current_df,column_maps):
    
    
    ref_X = clean_input(ref_df[features])
    current_X = clean_input(current_df[features])
    
    current_X["prediction"] = model.predict(current_X[features])
    ref_X["prediction"] = model.predict(ref_X[features])
    
    
    current_X.rename(columns={"lung_cancer":"actual"},inplace=True)
    ref_X.rename(columns={"lung_cancer":"actual"},inplace=True)
    
    
        
    report = Report(metrics=[
        ClassificationPreset(),
        ClassificationQualityMetric(),
        ClassificationClassBalance(),
        ClassificationConfusionMatrix()
    ])
    
    # report.run(reference_data=ref_X,current_data=current_X,column_mapping=)

    report.run(
        reference_data=ref_X,
        current_data=current_X,
        column_mapping=ColumnMapping(target="actual",prediction="prediction")
    )    
    report.save_html("monitoring\\model.html")
    metrics = report.as_dict()
    results = metrics["metrics"][0]["result"]
    general = metrics["metrics"][2]["result"]
    # wow = metrics["metrics"][2]
    
    performance_report = {
        "accuracy score": {},
        "precision score": {},
        "recall score": {},
        "f1 score": {},
        "confusion matrix": {},
        "fpr":{},
        "fnr":{},
        "tpr": {},
        "tnr": {}        
    }
    
    performance_report["accuracy score"].update({"Reference Accuracy": results["reference"]["accuracy"],"Current Accuracy": results["current"]["accuracy"]})
    performance_report["precision score"].update({"Reference Precision":results["reference"]["precision"],"Current Precision":results["current"]["precision"]})
    performance_report["recall score"].update({"Reference Recall": results["reference"]["recall"],"Current Recall":results["current"]["recall"]})
    performance_report["f1 score"].update({"Reference F1 Score": results["reference"]["f1"],"Current F1 Score":results["current"]["f1"]})
    performance_report["tpr"].update({"Reference tpr":results["reference"]["tpr"],"Current tpr": results["current"]["tpr"]})
    performance_report["tnr"].update({"Reference tnr":results["reference"]["tnr"],"Current tnr": results["current"]["tnr"]})
    performance_report["fpr"].update({"Reference fpr":results["reference"]["fpr"],"Current fpr": results["current"]["fpr"]})
    performance_report["fnr"].update({"Reference fnr":results["reference"]["fnr"],"Current fnr": results["current"]["fnr"]})
    performance_report["confusion matrix"].update({"Reference Confusion matrix": {"labels":general["reference_matrix"]["labels"],"values":general["reference_matrix"]["values"]},"Current Confusion matrix":{"labels":general["current_matrix"]["labels"],"values":general["current_matrix"]["values"]}})
    
    reference_accuracy = performance_report['accuracy score']["Reference Accuracy"]
    current_accuracy = performance_report['accuracy score']["Current Accuracy"]
    
    degradation = reference_accuracy- current_accuracy
    
    if degradation > 0.05:  # More than 5% degradation
        print(f"\n🚨 PERFORMANCE ALERT: Accuracy degraded by {degradation*100:.2f}%")
    else:
        print(f"\n✅ Performance within acceptable range: {degradation*100:.2f}% degradation")

    return performance_report

report,drift_results = data_drift_detection(ref_df,current_df,column_maps,output_path="monitoring")

# Automated Test for CI\CD

def automated_test(ref_df,current_df,column_maps,output_path):
    from evidently.test_preset import DataDriftTestPreset, DataQualityTestPreset, DataStabilityTestPreset

    test_suite = TestSuite(tests=[
        DataDriftTestPreset(),    # Includes your TestNumberOfDriftedColumns
        DataQualityTestPreset(),  # Includes your TestNumberOfColumnsWithMissingValues
        DataStabilityTestPreset() # Adds row/column count and type checks
    ])
    
    
    test_suite.run(reference_data=ref_df,current_data=current_df,column_mapping=column_maps)
    
    test_suite.save_html(f"{output_path}\\test.html")
    metrics = test_suite.as_dict()
    with open(f"{output_path}\\test.json","w") as f:
        json.dump(metrics,f,indent=3)
    
    ref_X = clean_input(ref_df[features])
    current_X = clean_input(current_df[features])
    
    current_X["prediction"] = model.predict(current_X[features])
    ref_X["prediction"] = model.predict(ref_X[features])
    
    
    current_X.rename(columns={"lung_cancer":"actual"},inplace=True)
    ref_X.rename(columns={"lung_cancer":"actual"},inplace=True)
    
    binary_suite = TestSuite(tests=[BinaryClassificationTestPreset()])
    
    binary_suite.run(reference_data=ref_X,current_data=current_X,column_mapping=ColumnMapping(target="actual",prediction="prediction"))
    
    target_col = "actual"
    
    ref_no_actual = ref_X.drop([target_col],axis=1)
    current_no_actual = current_X.drop([target_col],axis=1)
    
    no_targets = TestSuite(tests=[NoTargetPerformanceTestPreset()])
    
    no_targets.run(reference_data=ref_no_actual,current_data=current_no_actual,column_mapping=ColumnMapping(target="actual",prediction="prediction"))
    
    no_targets.save_html(f"{output_path}\\binary.html")
    
    tests = metrics.get('tests', [])

    summary = metrics.get("summary", {})
    
    with open(f"{output_path}/summary.json","w") as f:
        json.dump({"summary": summary, "tests": tests},f,indent=3)
        
    with open(f"{output_path}/test_output.log","w",encoding="utf-8") as f:
        
        all_passed = summary.get("all_passed", False)
        if metrics:
            if all_passed:
                passed_tests = [ t for t in metrics.get("tests",[]) if t.get("status") == "SUCCESS"]
                f.write("✅✅ PASSED TESTS\n")
                for test in passed_tests:
                    f.write(f" ✅ {test["name"]}: {test["description"]}\n")
                f.write("✅ All tests passed.\n\n")
            else:
                f.write("❌ Some tests failed.\n")
                passed_count = summary.get("success_tests", 1)
                failed_count = summary.get("failed_tests", 1)
                total_tests = summary.get("total_tests", 1)
                f.write(f"No. of passed tests: {passed_count}\n\n")
                f.write(f"Failed {failed_count} out of {total_tests} tests.\n\n")
                failed_tests = [
                t for t in metrics.get('tests', []) if t.get('status') == 'FAIL']
                for test in failed_tests:
                    f.write(f"  ❌ {test['name']}: {test.get('description', 'No description')}\n\n")
    return metrics
            
def classify_failure(test):
    name = test.get('name')
    status = test.get('status')
    if status != 'FAIL':
            return None
    if name in ["Number of Rows",'Number of Columns','Column Types',"Share of Drifted Columns","Drift per Column","The Share of Missing Values in a Column","Share of the Most Common Value"]:
            return 'critical'
    if name == 'Share of Out-of-Range Values':
            share = test.get('parameters', {}).get('value', 0)
            return 'critical' if share > 0.1 else 'warning'
    if name == "Share of Out-of-List Values":
        share = test.get('parameters', {}).get('value', 0)
        return 'critical' if share > 0.05 else 'warning'
    return 'warning'

    has_critical = False
    critical_list = []
    warning_list = []
    for suite_dict in [suite1_dict, binary_dict, no_target_dict]:
        for test in suite_dict.get('tests', []):
            severity = classify_failure(test)
            if severity == 'critical':
                has_critical = True
                critical_list.append(f"{test['name']} : {test.get('description')}")
            elif severity == 'warning':
                warning_list.append(f"{test['name']}: {test.get('description')}")

    # Print summary (optional)
    if critical_list:
        print("❌ Critical failures detected. Pipeline will fail.")
        return 1
    elif warning_list:
        print("⚠️ Warnings detected but pipeline will continue.")
        # Optionally send alert here
        return 0
    else:
        print("✅ All tests passed.")
        return 0    


def is_critical_failure(failure):
    """Classify if a failure is critical based on test name and parameters"""
    name = failure.get('name', '')
    
    # Critical by test type
    critical_tests = ["Number of Rows",'Number of Columns','Column Types',"Share of Drifted Columns","Drift per Column","The Share of Missing Values in a Column","Share of the Most Common Value"]
    if name in critical_tests:
        return True
    
    # Critical based on severity thresholds
    if name == 'Share of Out-of-Range Values':
        share = failure.get('parameters', {}).get('value', 0)
        return share > 0.1  # >10% out-of-range is critical
    
    if name == 'Share of Out-of-List Values':
        share = failure.get('parameters', {}).get('value', 0)
        return share > 0.05  # >5% new categories is critical
    
    # Default to warning for other failures
    return False
def alert_system(suite_report):
    
    summary = suite_report.get("summary",{})
    tests = suite_report.get("tests",[])
    
    all_passed = summary.get("all_passed",0)
    passed_tests = summary.get("success_tests",0)
    failed_tests = summary.get("failed_tests",0)
    total_tests = summary.get("total_tests")
    
    # failed tests
    failures = []
    for test in tests:
        if test.get("status") == "FAIL":
            failures.append({
                "name": test.get("name","Unknown"),
                "description": test.get("description","No description"),
                "parameters": test.get("parameters",{})
            })
        
    critical_failures = []
    warning_failures = []
    
    for failure in failures:
        if is_critical_failure(failure):
            critical_failures.append(failure)
        
        else:
            warning_failures.append(failure)
    
        return {
        'timestamp': datetime.now().isoformat(),
        'all_passed': all_passed,
        'should_alert': not all_passed,  # Alert if any test failed
        'summary': {
            'total': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'critical_count': len(critical_failures),
            'warning_count': len(warning_failures)
        },
        'critical_failures': critical_failures,
        'warning_failures': warning_failures
    }
        
def send_console_alert(alert_data):
    print("\n" + "="*60)
    print("🔔 EVIDENTLY MONITORING ALERT")
    print("="*60)
    print(f"Timestamp: {alert_data['timestamp']}")
    print(f"Status: {'❌ FAILED' if alert_data['should_alert'] else '✅ PASSED'}")
    print(f"Test Pass Rate: {alert_data["summary"]["passed"]/alert_data["summary"]["total"]}%")
    
    if alert_data["critical_failures"]:
        print(f"\n The number of critical failures: {len(alert_data["critical_failures"])}")
        for f in alert_data["critical_failures"]:
            print(f"{f['name']}: {f['description']}")
    
    if alert_data["warning_failures"]:
        print(f"\n The number of warning failures: {len(alert_data["warning_failures"])}")
        for f in alert_data["warning_failures"]:
            print(f"{f["name"]}: { f["description"]}")

def send_email_alert(alert_data,smtp_server,smtp_port,sender_email,sender_password,reciever_email):
    
    if not alert_data["should_alert"]:
        print("No failures found skip the email alert")
        return 
    
    subject =f"Monitoring alert: {alert_data["summary"]["failed"]} tests failed "
    
    html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .critical {{ color: #d9534f; background-color: #f2dede; padding: 10px; border-radius: 5px; }}
                .warning {{ color: #f0ad4e; background-color: #fcf8e7; padding: 10px; border-radius: 5px; }}
                .summary {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h2>📊 Evidently Monitoring Alert</h2>
            <p><strong>Timestamp:</strong> {alert_data['timestamp']}</p>
            
            <div class="summary">
                <h3>Summary</h3>
                <p>✅ Passed: {alert_data['summary']['passed']}/{alert_data['summary']['total']}</p>
                <p>❌ Failed: {alert_data['summary']['failed']}/{alert_data['summary']['total']}</p>
                <p>🚨 Critical: {alert_data['summary']['critical_count']}</p>
                <p>⚠️ Warnings: {alert_data['summary']['warning_count']}</p>
            </div>
    """
    if alert_data["critical_failures"]:
        html_content += "<div class='critical'>"
        html_content += "<h3>🚨 Critcal Failure </h3><ul>"
        for f in alert_data["critical_failures"]:
            html_content += f"<li><strong>{f['name']}</strong>: {f['description']}</li>"
            html_content += "</ul></div"
    
    if alert_data["warning_failures"]:
        html_content += "<div class='warning'>"
        html_content += "<h3>⚠️ Warining Failure </h3><ul>"
        for f in alert_data["warning_failures"]:
            html_content += f"<li><strong>{f['name']}</strong>: {f['description']}</li>"
            html_content += "</ul></div>"
    html_content += """
        <hr>
        <p><small>This alert was generated automatically by Evidently monitoring.</small></p>
    </body>
    </html>
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = ", ".join(reciever_email)
    
    msg.attach(MIMEText(html_content,"html"))
    
    try:
        with smtplib.SMTP_SSL(smtp_server,smtp_port) as server:
            server.login(sender_email,sender_password)
            server.send_message(msg)
            print(f"✅ Email alert sent to {len(reciever_email)} recipients")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

def slack_message_alert(alert_data,webhook_url):
    
    if not alert_data['should_alert']:
        print("No failures detected - skipping Slack alert")
        return

    if alert_data["critical_failures"]:
        color = "##d9534f"
        emoji = "🚨"
    else:
        color = "#f0ad4e"  
        emoji = "⚠️"
    
    # build slack report
    passed = alert_data['summary']['passed']
    total = alert_data['summary']['total']
    percentage = (passed / total) * 100
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} Evidently Alert For Slack"
            }
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn" , "text": f"Status: *\n {'❌ FAILED' if alert_data['should_alert'] else '✅ PASSED'}"},
                {"type": "mrkdwn", "text": f"Passed Percentage: {percentage}%" },
                {"type": "mrkdwn","text": f"Failed Tests: {alert_data['summary']['failed']}"},
                {"type": "mrkdwn","text": f"Time stamp: {alert_data['timestamp']}"}
            ]
        }
    ]
    
    if alert_data['critical_failures']:
        critical_text = "\n".join([f"*{f['name']}*: {f['description'][:100]}" for f in alert_data['critical_failures'][:5]])
        
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn","text": f"🚨 Crtical Failures : {critical_text}"
            }
        })
    
    if alert_data["warning_failures"]:
        warning_text = "\n".join([
            f" * {f['name']}*: {f['description'][:100]}" for f in alert_data['warning_failures'][:5] 
        ])
        
        blocks.append({
            "type": "section",
            "text":{
                "type": "mrkdwn",
                "text": f"⚠️ Warning Failures : {warning_text}"
            }
        })
        
    payload = {"text": "Tests From Ndubuisi Evidently","blocks":blocks}
    import requests
    try:
       response = requests.post(webhook_url,json=payload)
       response.raise_for_status()
       print("✅ Slack alert sent")
    except Exception as e:
        print(f"❌ Failed to send Slack alert: {e}")
        
def save_slack_message(alert_data,history_file="monitoring\\history_alert.json"):
    
    history = []
    
    if os.path.exists(history_file):
        try:
            with open(history_file,"r",encoding="utf-8") as f:
                json.load(f)
        except (json.JSONDecodeError or IOError):
            history = []
        
    history.append(alert_data)
        
    if len(history) > 1000:
        history = history[-1000:]
        
    with open(history_file,"w",encoding="utf-8") as f:
        json.dump(history,f,indent=3,ensure_ascii=False)
    print(f"✅ Alert history stored ({len(history)} total entries)")


def run_alerts(suite_dict):
    
    alert_data = alert_system(suite_dict)
    
    send_console_alert(alert_data)
    
    if alert_data['should_alert']:
        # Configure these from environment variables
        SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
        SMTP_PORT = int(os.environ.get('SMTP_PORT', 587))
        SENDER_EMAIL = os.environ.get('ALERT_EMAIL_SENDER')
        SENDER_PASSWORD = os.environ.get('ALERT_EMAIL_PASSWORD')
        RECIPIENTS = os.environ.get('ALERT_EMAIL_RECIPIENTS', '').split(',')
        
        if SENDER_EMAIL and SENDER_PASSWORD and RECIPIENTS:
            send_email_alert(alert_data, SMTP_SERVER, SMTP_PORT, 
                           SENDER_EMAIL, SENDER_PASSWORD, RECIPIENTS)
    
    SLACK_WEBHOOK = os.environ.get('SLACK_WEBHOOK_URL')
    if SLACK_WEBHOOK and alert_data['should_alert']:
        slack_message_alert(alert_data, SLACK_WEBHOOK)
        
    critical_count = alert_data['summary']['critical_count']
    return 1 if critical_count > 0 else 0  
    
    
  
    
    
    
    

# with open("monitoring\\drift_results.json", 'w') as f:

#         json_results = {
#             'dataset_drift': bool(drift_results['dataset_drift']),
#             'drift_score': float(drift_results['drift_share']),
#             'number_of_drifted_columns': int(drift_results['number_of_drifted_columns']),
#             'drifted_columns': [
#                 {k: (float(v) if isinstance(v, (np.float32, np.float64)) else v) 
#                 for k, v in col.items()}
#                 for col in drift_results['drifted_cols']
#             ]
#         }
#         json.dump(json_results, f, indent=2)
       
reports = automated_test(ref_df,current_df,column_maps,"monitoring")

run_alerts(reports)

print("\n✅ Drift detection complete. Results saved to JSON.")
                    
# advanced_reports = advanced_drift_detection(ref_df,current_df,column_maps)
# advanced_reports.save_html("monitoring\\advanced.html")          
      
# quality,quality_report = data_quality_monitoring(ref_df,current_df,column_maps)   

# with open("monitoring\\quality.json","w") as f:
#     json.dump(quality_report,f,indent=3)       

# model_report = model_performance(ref_df,current_df,column_maps)

# with open("monitoring\\model.json","w") as f:
#     json.dump(model_report,f,indent=3)
