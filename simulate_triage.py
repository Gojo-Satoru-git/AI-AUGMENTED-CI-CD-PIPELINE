import os
import json
import time
import urllib.request
import urllib.error

API_URL = "http://localhost:8000/api/v1/triage"

def extract_sample_trace(csv_path: str):
    """Extracts a single TaskID and Trace Sequence from the given CSV path (skips header)."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        f.readline() # Skip header
        line = f.readline().strip()
        
    parts = line.split(',', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "UNKNOWN_TASK", line

def simulate():
    print("==================================================")
    print("STARTING AI-DRIVEN TRIAGE POLICY SIMULATION")
    print("==================================================\n")
    
    # 1. Load Data
    print("Loading sample traces from raw HDFS data...")
    fail_task_id, fail_trace = extract_sample_trace('data/raw/failure_trace.csv')
    norm_task_id, norm_trace = extract_sample_trace('data/raw/normal_trace.csv')
    
    print(f"Loaded Anomaly Trace [TaskID: {fail_task_id}]")
    print(f"Loaded Normal Trace  [TaskID: {norm_task_id}]\n")
    
    # Define our test scenarios
    scenarios = [
        {
            "name": "Scenario 1: Code Bug (Normal Trace) by Junior Dev",
            "payload": {
                "test_id": norm_task_id,
                "trace_sequence": norm_trace,
                "developer_id": "dev_junior_1"
            }
        },
        {
            "name": "Scenario 2: Flaky Infrastructure (Anomaly Trace) by Junior Dev",
            "payload": {
                "test_id": fail_task_id,
                "trace_sequence": fail_trace,
                "developer_id": "dev_junior_1"
            }
        },
        {
            "name": "Scenario 3: Flaky Infrastructure (Anomaly Trace) by Senior Dev",
            "payload": {
                "test_id": fail_task_id,
                "trace_sequence": fail_trace,
                "developer_id": "dev_senior_3"
            }
        }
    ]
    
    # Run the simulation
    for idx, scenario in enumerate(scenarios, 1):
        print(f"--- {scenario['name']} ---")
        print(f"Sending webhook for Test ID: {scenario['payload']['test_id']}")
        
        try:
            req = urllib.request.Request(
                API_URL, 
                data=json.dumps(scenario['payload']).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    print(f"AI Classification:   {data['classification']} (Prob: {data['anomaly_probability']})")
                    print(f"Dev Trust Tier:     Tier {data['trust_tier']} ({data['developer_id']})")
                    print(f"Policy Action:      {data['action_recommended']}")
                    if data.get('root_cause_graph'):
                        print(f"Graph Generated at: {data['root_cause_graph']}")
                else:
                    print(f"Error {response.status}: {response.read().decode('utf-8')}")
                
        except urllib.error.URLError as e:
            print("ERROR: Connection failed! Make sure the FastAPI server is running (python run_api.py).")
            print(f"Details: {e}")
            break
            
        print("\n" + "-"*50 + "\n")
        time.sleep(1) # Add a small delay for dramatic effect

if __name__ == "__main__":
    simulate()
