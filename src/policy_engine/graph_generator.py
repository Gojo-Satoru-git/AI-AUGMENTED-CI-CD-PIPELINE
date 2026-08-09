import os
import uuid
import pandas as pd
import networkx as nx
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# The root directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")
FALLBACK_DATA_DIR = os.path.join(
    BASE_DIR, 
    "data", "raw", "tracebench", 
    "AN_Data_corruptBlk_r_00FDN_30C_1to19B_0to120INT_15RT_5WT"
)
EVENT_COLS = ['TaskID', 'TID', 'OpName', 'StartTime', 'EndTime', 'HostAddress', 'HostName', 'Agent', 'Description']
EDGE_COLS = ['TaskID', 'FatherTID', 'FatherStartTime', 'ChildTID']

def generate_root_cause_graph(task_id: str) -> str:
    """
    Generates a NetworkX DAG for a specific task_id and saves it as a PNG file.
    Returns the file path to the saved graph.
    """
    event_file = os.path.join(FALLBACK_DATA_DIR, 'event.csv')
    edge_file = os.path.join(FALLBACK_DATA_DIR, 'edge.csv')
    
    # Ensure graphs directory exists
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    
    # If fallback data doesn't exist, we can't generate the graph
    if not os.path.exists(event_file) or not os.path.exists(edge_file):
        return None

    # Load data
    try:
        df_events = pd.read_csv(event_file, header=0, names=EVENT_COLS, index_col=False, low_memory=False)
        df_edges = pd.read_csv(edge_file, header=0, names=EDGE_COLS, index_col=False, low_memory=False)
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return None

    # Clean hidden spaces
    df_events['TaskID'] = df_events['TaskID'].astype(str).str.strip()
    df_edges['TaskID'] = df_edges['TaskID'].astype(str).str.strip()

    # Isolate Task
    events_filtered = df_events[df_events['TaskID'] == task_id]
    edges_filtered = df_edges[df_edges['TaskID'] == task_id]
    
    # Check if we have data for this task
    if len(events_filtered) == 0 or len(edges_filtered) == 0:
        return None
        
    G = nx.DiGraph()
    
    # Add Nodes
    for index, row in events_filtered.iterrows():
        G.add_node(str(row['TID']).strip(), label=str(row['OpName']).strip())
        
    # Add Edges
    for index, row in edges_filtered.iterrows():
        G.add_edge(str(row['FatherTID']).strip(), str(row['ChildTID']).strip())
        
    # Draw Graph
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.8, iterations=100)
    nx.draw(G, pos, 
            labels=nx.get_node_attributes(G, 'label'), 
            with_labels=True, 
            node_color='lightblue', 
            edge_color='gray', 
            node_size=2500, 
            font_size=9, 
            font_weight='bold')
    
    plt.title(f"Root Cause Graph for Task: {task_id}")
    
    # Save Graph
    filename = f"graph_{task_id}_{uuid.uuid4().hex[:6]}.png"
    filepath = os.path.join(GRAPHS_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close() # Close figure to avoid memory leak
    
    return os.path.relpath(filepath, BASE_DIR)
