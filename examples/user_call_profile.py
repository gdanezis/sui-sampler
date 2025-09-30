#!/usr/bin/env python3
"""
User Call Profile Clustering for Sui Blockchain Data.

This script analyzes checkpoint data to:
1. Extract all move calls made by each sender
2. Cluster senders based on the similarity of their call patterns using DBSCAN
3. Print the resulting clusters

The distance metric used is the Jaccard distance (1 - |A ∩ B| / |A ∪ B|) where
A and B are the sets of calls made by two different senders.
"""

import json
import sys
import csv
import os
import numpy as np
from collections import defaultdict
from typing import Dict, Set, List, Tuple
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances


# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def load_package_names(csv_path: str = 'examples/sui_package_name.csv') -> Dict[str, Tuple[str, str]]:
    """Load package ID to (name, vertical) mapping from CSV file."""
    package_names = {}
    
    if not os.path.exists(csv_path):
        print(f"Warning: Package names CSV not found at {csv_path}", file=sys.stderr)
        return package_names
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                contract_address = row.get('CONTRACT_ADDRESS', '').strip()
                app_name = row.get('APP_NAME', '').strip()
                vertical = row.get('VERTICAL', '').strip()
                
                if contract_address and app_name:
                    package_names[contract_address] = (app_name, vertical)
    except Exception as e:
        print(f"Warning: Error loading package names CSV: {e}", file=sys.stderr)
    
    return package_names


def extract_programmable_transaction(transaction):
    """Extract ProgrammableTransaction from transaction data."""
    if 'data' in transaction and isinstance(transaction['data'], list):
        for data_item in transaction['data']:
            if 'intent_message' in data_item:
                intent_message = data_item['intent_message']
                if 'value' in intent_message:
                    value = intent_message['value']
                    if 'V1' in value and 'kind' in value['V1']:
                        kind = value['V1']['kind']
                        if 'ProgrammableTransaction' in kind:
                            return kind['ProgrammableTransaction']
    return None


def extract_move_call_names(programmable_tx) -> List[str]:
    """Extract package::module::function names from MoveCall commands."""
    names = []
    if 'commands' in programmable_tx:
        for command in programmable_tx['commands']:
            if 'MoveCall' in command:
                move_call = command['MoveCall']
                package = move_call.get('package', '')
                module = move_call.get('module', '')
                function = move_call.get('function', '')
                if package and module and function:
                    # Normalize package ID to include 0x prefix
                    if not package.startswith('0x'):
                        package = f'0x{package}'
                    names.append(f"{package}::{module}::{function}")
                    names.append(f"{package}::{module}::-")
    return names


def extract_event_type_names(events) -> List[str]:
    """Extract package::module::type names from event data."""
    names = []
    if events and 'data' in events:
        for event in events['data']:
            if 'type_' in event:
                type_info = event['type_']
                # Events have address, module, name directly in type_
                package = type_info.get('address', '')
                module = type_info.get('module', '')
                name = type_info.get('name', '')
                if package and module and name:
                    # Normalize package ID to include 0x prefix
                    if not package.startswith('0x'):
                        package = f'0x{package}'
                    names.append(f"{package}::{module}::{name}")
    return names


def is_system_package(package_id: str) -> bool:
    """Check if a package ID is a system package (has at least 10 leading zeros)."""
    if not package_id.startswith('0x'):
        return False
    
    # Remove the '0x' prefix and count leading zeros
    hex_part = package_id[2:]
    leading_zeros = 0
    for char in hex_part:
        if char == '0':
            leading_zeros += 1
        else:
            break
    
    return leading_zeros >= 10


def filter_system_calls(calls: List[str]) -> List[str]:
    """Filter out calls/events from system packages."""
    filtered = []
    for call in calls:
        # Extract package ID from call (first part before ::)
        if '::' in call:
            package_id = call.split('::', 1)[0]
            if not is_system_package(package_id):
                filtered.append(call)
        else:
            # If no :: found, assume it's not a system call
            filtered.append(call)
    return filtered


def extract_sender_call_profiles(data) -> Dict[str, Set[str]]:
    """Extract all move calls and event types made by each sender, filtering out system packages."""
    sender_calls = defaultdict(set)
    
    checkpoints = data.get('checkpoints', [])
    
    for checkpoint in checkpoints:
        transactions = checkpoint.get('transactions', [])
        
        for tx_data in transactions:
            transaction = tx_data.get('transaction', {})
            
            # Extract sender address
            sender = ''
            if 'data' in transaction and isinstance(transaction['data'], list) and transaction['data']:
                intent_message = transaction['data'][0].get('intent_message', {})
                value = intent_message.get('value', {})
                v1_data = value.get('V1', {})
                sender = v1_data.get('sender', '')
            
            if not sender:
                continue
                
            # Extract MoveCall names from ProgrammableTransaction
            programmable_tx = extract_programmable_transaction(transaction)
            if programmable_tx:
                call_names = extract_move_call_names(programmable_tx)
                # Filter out system package calls
                filtered_calls = filter_system_calls(call_names)
                sender_calls[sender].update(filtered_calls)
            
            # Extract event type names from events
            events = tx_data.get('events')
            if events:
                event_type_names = extract_event_type_names(events)
                # Filter out system package events and add "event:" prefix to distinguish from calls
                filtered_events = filter_system_calls(event_type_names)
                # Add "event:" prefix to distinguish from calls
                prefixed_events = [f"event:{name}" for name in filtered_events]
                sender_calls[sender].update(prefixed_events)
    
    return dict(sender_calls)


def jaccard_distance(set1: Set[str], set2: Set[str]) -> float:
    """Calculate Jaccard distance between two sets."""
    if not set1 and not set2:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 1.0
    
    jaccard_similarity = intersection / union
    return 1.0 - jaccard_similarity


def create_distance_matrix(sender_calls: Dict[str, Set[str]]) -> Tuple[np.ndarray, List[str]]:
    """Create a distance matrix for DBSCAN clustering."""
    senders = list(sender_calls.keys())
    n_senders = len(senders)
    
    distance_matrix = np.zeros((n_senders, n_senders))
    
    for i in range(n_senders):
        for j in range(i + 1, n_senders):
            distance = jaccard_distance(sender_calls[senders[i]], sender_calls[senders[j]])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
    
    return distance_matrix, senders


def analyze_eps_sensitivity(sender_calls: Dict[str, Set[str]], eps_range: List[float] = None) -> None:
    """Analyze how different eps values affect clustering results."""
    if eps_range is None:
        eps_range = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("\nEPS Sensitivity Analysis")
    print("=" * 30)
    print(f"{'eps':<6} {'clusters':<8} {'outliers':<8} {'largest':<8} {'avg_size':<8}")
    print("-" * 45)
    
    for eps in eps_range:
        clusters = cluster_senders(sender_calls, eps=eps, min_samples=2)
        
        num_clusters = len([cid for cid in clusters.keys() if cid != -1])
        outliers = len(clusters.get(-1, []))
        
        if num_clusters > 0:
            cluster_sizes = [len(senders) for cid, senders in clusters.items() if cid != -1]
            largest_cluster = max(cluster_sizes)
            avg_cluster_size = np.mean(cluster_sizes)
        else:
            largest_cluster = 0
            avg_cluster_size = 0
        
        print(f"{eps:<6.1f} {num_clusters:<8} {outliers:<8} {largest_cluster:<8} {avg_cluster_size:<8.1f}")


def plot_distance_distribution(sender_calls: Dict[str, Set[str]], sample_size: int = 1000) -> None:
    """Analyze the distribution of pairwise distances to help choose eps."""
    print("\nDistance Distribution Analysis")
    print("=" * 33)
    
    senders = list(sender_calls.keys())
    if len(senders) > sample_size:
        # Sample for performance with large datasets
        import random
        senders = random.sample(senders, sample_size)
    
    distances = []
    for i in range(len(senders)):
        for j in range(i + 1, min(i + 100, len(senders))):  # Limit for performance
            distance = jaccard_distance(sender_calls[senders[i]], sender_calls[senders[j]])
            distances.append(distance)
    
    distances = np.array(distances)
    
    print(f"Distance statistics (sample of {len(distances)} pairs):")
    print(f"  Min distance: {np.min(distances):.3f}")
    print(f"  Max distance: {np.max(distances):.3f}")
    print(f"  Mean distance: {np.mean(distances):.3f}")
    print(f"  Median distance: {np.median(distances):.3f}")
    print(f"  25th percentile: {np.percentile(distances, 25):.3f}")
    print(f"  75th percentile: {np.percentile(distances, 75):.3f}")
    print()
    print("Recommended eps values:")
    print(f"  Conservative (tight clusters): {np.percentile(distances, 15):.2f}")
    print(f"  Moderate (balanced): {np.percentile(distances, 30):.2f}")
    print(f"  Liberal (loose clusters): {np.percentile(distances, 50):.2f}")


def cluster_senders(sender_calls: Dict[str, Set[str]], eps: float = 0.5, min_samples: int = 2) -> Dict[int, List[str]]:
    """Cluster senders using DBSCAN based on their call patterns."""
    if len(sender_calls) < 2:
        return {0: list(sender_calls.keys())} if sender_calls else {}
    
    distance_matrix, senders = create_distance_matrix(sender_calls)
    
    # Use DBSCAN with precomputed distance matrix
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    cluster_labels = dbscan.fit_predict(distance_matrix)
    
    # Group senders by cluster
    clusters = defaultdict(list)
    for sender, label in zip(senders, cluster_labels):
        clusters[label].append(sender)
    
    return dict(clusters)


def write_outlier_sequences(outlier_senders: List[str], sender_calls: Dict[str, Set[str]], filename: str = "outlier_sequences.txt"):
    """Write outlier call sequences to a file, one line per outlier address."""
    try:
        with open(filename, 'w') as f:
            for sender in outlier_senders:
                calls = sender_calls.get(sender, set())
                # Sort calls for consistent output
                sorted_calls = sorted(calls)
                # Join calls with commas
                call_sequence = ", ".join(sorted_calls)
                f.write(f"{sender}: {call_sequence}\n")
        print(f"  Outlier sequences written to {filename}", file=sys.stderr)
    except Exception as e:
        print(f"  Warning: Could not write outlier sequences: {e}", file=sys.stderr)


def generate_cluster_name(cluster_id: int, senders: List[str], sender_calls: Dict[str, Set[str]], package_names: Dict[str, tuple]) -> str:
    """Generate a descriptive name for a cluster based on top move calls."""
    if cluster_id == -1:
        return "Outliers"
    
    # Count frequency of each call across all senders in the cluster
    call_frequency = defaultdict(int)
    for sender in senders:
        for call in sender_calls[sender]:
            call_frequency[call] += 1
    
    # Get top calls that appear in >25% of cluster members, limit to top 3
    frequent_calls = [(call, freq) for call, freq in call_frequency.items() 
                    if freq > len(senders) * 0.25]
    frequent_calls = sorted(frequent_calls, key=lambda x: x[1], reverse=True)[:3]
    
    if not frequent_calls:
        return f"Cluster_{cluster_id}"
    
    # Extract app names from the top calls
    app_names = []
    module_names = []
    
    for call, _ in frequent_calls:
        # Handle both regular calls and event types
        if call.startswith('event:'):
            # Remove "event:" prefix and extract package ID
            actual_call = call[6:]  # Remove "event:" prefix
            package_id = actual_call.split('::', 1)[0] if '::' in actual_call else ''
        else:
            # Regular move call
            package_id = call.split('::', 1)[0] if '::' in call else ''
        
        if package_id in package_names:
            app_name, _ = package_names[package_id]
            if app_name not in app_names:
                app_names.append(app_name)
        else:
            # Extract module name (second part after package ID)
            if call.startswith('event:'):
                actual_call = call[6:]  # Remove "event:" prefix
                parts = actual_call.split("::")
            else:
                parts = call.split("::")
            
            if len(parts) >= 2:
                module_name = parts[1]
                if module_name not in module_names:
                    module_names.append(module_name)
    
    # Create cluster name - prefer app names over module names
    if app_names:
        name_parts = app_names
    elif module_names:
        name_parts = module_names
    else:
        # Fallback to first part of the most frequent call
        first_call = frequent_calls[0][0]
        first_part = first_call.split("::")[0] if "::" in first_call else first_call
        name_parts = [first_part]
    
    # Join with '+' and limit length
    cluster_name = "+".join(name_parts)
    if len(cluster_name) > 30:  # Limit name length
        cluster_name = cluster_name[:27] + "..."
    
    return cluster_name


def print_cluster_analysis(clusters: Dict[int, List[str]], sender_calls: Dict[str, Set[str]], top_n: int = 10):
    """Print detailed cluster analysis."""
    print("Sui Sender Call Pattern Clustering Analysis")
    print("=" * 45)
    print()
    
    # Load package names
    package_names = load_package_names()
    
    # Calculate total number of senders for percentage calculation
    total_senders = len(sender_calls)
    
    # Sort clusters by size (largest first)
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    for cluster_id, senders in sorted_clusters:
        cluster_size = len(senders)
        percentage = (cluster_size / total_senders) * 100
        
        # Generate cluster name
        cluster_name = generate_cluster_name(cluster_id, senders, sender_calls, package_names)
        
        if cluster_id == -1:
            header = f"{cluster_name}: {cluster_size} senders ({percentage:.1f}% of all senders)"
            # Write outlier sequences to file
            write_outlier_sequences(senders, sender_calls)
        else:
            header = f"{cluster_name} (Cluster {cluster_id}): {cluster_size} senders ({percentage:.1f}% of all senders)"
        
        print(f"{Colors.GREEN}{Colors.BOLD}{header}{Colors.END}")
        print("-" * 40)
        
        if senders:
            # Count frequency of each call across all senders in the cluster
            call_frequency = defaultdict(int)
            for sender in senders:
                for call in sender_calls[sender]:
                    call_frequency[call] += 1
            
            # Find all unique calls in the cluster
            all_calls = set()
            for sender in senders:
                all_calls |= sender_calls[sender]
            
            # Get top-5 most frequent calls that appear in >25% of cluster members
            frequent_calls = [(call, freq) for call, freq in call_frequency.items() 
                            if freq > len(senders) * 0.25]
            frequent_calls = sorted(frequent_calls, key=lambda x: x[1], reverse=True)[:5]
            
            if frequent_calls:
                print(f"  Top function calls and events (>25% of members):")
                for call, frequency in frequent_calls:
                    # Handle both regular calls and event types
                    if call.startswith('event:'):
                        # Remove "event:" prefix and extract package ID
                        actual_call = call[6:]  # Remove "event:" prefix
                        package_id = actual_call.split('::', 1)[0] if '::' in actual_call else ''
                        # Shorten the call name by removing package prefix for display
                        short_call = "::".join(actual_call.split("::")[1:]) if "::" in actual_call else actual_call
                        call_type = "[EVENT]"
                    else:
                        # Regular move call
                        package_id = call.split('::', 1)[0] if '::' in call else ''
                        # Shorten the call name by removing package prefix for display
                        short_call = "::".join(call.split("::")[1:]) if "::" in call else call
                        call_type = "[CALL]"
                    
                    percentage = (frequency / len(senders)) * 100
                    
                    # Add package info if available
                    call_display = f"{call_type} {short_call}"
                    if package_id in package_names:
                        app_name, vertical = package_names[package_id]
                        call_display = f"{call_type} {short_call} [{Colors.RED}{app_name}{Colors.END} - {vertical}]"
                    
                    print(f"    {frequency:>3}/{len(senders)} ({percentage:>5.1f}%) - {call_display}")
            else:
                print(f"  No calls appear in >25% of cluster members")
            
            print(f"  Total unique calls and events in cluster: {len(all_calls)}")
        
        print()


def generate_html_output(clusters: Dict[int, List[str]], sender_calls: Dict[str, Set[str]], filename: str = "cluster_analysis.html"):
    """Generate a pretty HTML file with clustering analysis."""
    
    # Load package names
    package_names = load_package_names()
    
    # Calculate total number of senders for percentage calculation
    total_senders = len(sender_calls)
    
    # Sort clusters by size (largest first)
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sui Clustering Analysis</title>
    <style>
        body {{
            font-family: 'Segoe UI', sans-serif;
            line-height: 1.3;
            color: #333;
            max-width: none;
            width: 100%;
            margin: 0;
            padding: 10px;
            background: #f5f5f5;
            font-size: 13px;
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin: 10px 0 20px 0;
            font-size: 1.8em;
        }}
        .cluster {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 8px 0;
            padding: 12px;
        }}
        .cluster-header {{
            color: #28a745;
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 8px;
            padding: 6px 10px;
            background: #28a745;
            color: white;
            border-radius: 3px;
        }}
        .outlier-header {{
            background: #dc3545;
        }}
        .call-list {{
            margin: 8px 0;
        }}
        .call-item {{
            margin: 3px 0;
            padding: 4px 8px;
            background: #f8f9fa;
            border-radius: 3px;
            border-left: 3px solid #007bff;
            font-size: 12px;
        }}
        .app-name {{
            color: #dc3545;
            font-weight: bold;
        }}
        .percentage {{
            color: #666;
            font-weight: normal;
        }}
        .stats {{
            background: #e9ecef;
            padding: 6px 10px;
            border-radius: 3px;
            margin: 5px 0;
            font-size: 12px;
        }}
        .summary {{
            background: #fff;
            padding: 15px;
            border-radius: 4px;
            margin-top: 20px;
            border: 1px solid #ddd;
        }}
        .summary h2 {{
            color: #495057;
            margin: 0 0 10px 0;
            font-size: 1.3em;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 8px;
            margin-top: 10px;
        }}
        .stat-item {{
            background: #f8f9fa;
            padding: 8px;
            border-radius: 3px;
            text-align: center;
            border: 1px solid #e9ecef;
        }}
        .stat-value {{
            font-size: 1.4em;
            font-weight: bold;
            color: #007bff;
            line-height: 1;
        }}
        .stat-label {{
            color: #666;
            font-size: 11px;
            margin-top: 2px;
        }}
        .grid-layout {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 10px;
            padding: 0 5px;
        }}
        
        /* Responsive grid for different screen sizes */
        @media (min-width: 768px) {{
            .grid-layout {{
                grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            }}
        }}
        
        @media (min-width: 1200px) {{
            .grid-layout {{
                grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            }}
        }}
        
        @media (min-width: 1600px) {{
            .grid-layout {{
                grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            }}
        }}
        
        @media (min-width: 2000px) {{
            .grid-layout {{
                grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            }}
        }}
    </style>
</head>
<body>
    <h1>Sui Sender Clustering Analysis</h1>
    <div class="grid-layout">
"""

    for cluster_id, senders in sorted_clusters:
        cluster_size = len(senders)
        percentage = (cluster_size / total_senders) * 100
        
        # Generate cluster name
        cluster_name = generate_cluster_name(cluster_id, senders, sender_calls, package_names)
        
        if cluster_id == -1:
            header_class = "cluster-header outlier-header"
            header_text = f"{cluster_name} ({cluster_size}, {percentage:.1f}%)"
            # Write outlier sequences to file for HTML mode too
            write_outlier_sequences(senders, sender_calls)
        else:
            header_class = "cluster-header"
            header_text = f"{cluster_name} ({cluster_size}, {percentage:.1f}%)"
        
        html_content += f"""
        <div class="cluster">
            <div class="{header_class}">
                {header_text}
            </div>
"""
        
        if senders:
            # Count frequency of each call across all senders in the cluster
            call_frequency = defaultdict(int)
            for sender in senders:
                for call in sender_calls[sender]:
                    call_frequency[call] += 1
            
            # Find all unique calls in the cluster
            all_calls = set()
            for sender in senders:
                all_calls |= sender_calls[sender]
            
            # Get top-5 most frequent calls that appear in >25% of cluster members
            frequent_calls = [(call, freq) for call, freq in call_frequency.items() 
                            if freq > len(senders) * 0.25]
            frequent_calls = sorted(frequent_calls, key=lambda x: x[1], reverse=True)[:5]
            
            if frequent_calls:
                html_content += """
                <div class="call-list">
                    <strong>Top calls and events (>25%):</strong>
"""
                for call, frequency in frequent_calls:
                    # Handle both regular calls and event types
                    if call.startswith('event:'):
                        # Remove "event:" prefix and extract package ID
                        actual_call = call[6:]  # Remove "event:" prefix
                        package_id = actual_call.split('::', 1)[0] if '::' in actual_call else ''
                        # Shorten the call name by removing package prefix for display
                        short_call = "::".join(actual_call.split("::")[1:]) if "::" in actual_call else actual_call
                        call_type = "<strong>[EVENT]</strong>"
                    else:
                        # Regular move call
                        package_id = call.split('::', 1)[0] if '::' in call else ''
                        # Shorten the call name by removing package prefix for display
                        short_call = "::".join(call.split("::")[1:]) if "::" in call else call
                        call_type = "<strong>[CALL]</strong>"
                    
                    call_percentage = (frequency / len(senders)) * 100
                    
                    # Add package info if available
                    call_display = f"{call_type} {short_call}"
                    if package_id in package_names:
                        app_name, vertical = package_names[package_id]
                        call_display = f'{call_type} {short_call} [<span class="app-name">{app_name}</span>]'
                    
                    html_content += f"""
                    <div class="call-item">
                        {frequency}/{len(senders)} ({call_percentage:.0f}%) - {call_display}
                    </div>
"""
                html_content += "                </div>\n"
            else:
                html_content += """
                <div class="stats">
                    No calls or events >25% frequency
                </div>
"""
            
            html_content += f"""
            <div class="stats">
                {len(all_calls)} unique calls and events
            </div>
"""
        
        html_content += "        </div>\n"

    # Add summary statistics
    total_unique_calls = len(set().union(*sender_calls.values())) if sender_calls else 0
    call_counts = [len(calls) for calls in sender_calls.values()]
    avg_calls_per_sender = sum(call_counts) / len(call_counts) if call_counts else 0
    max_calls = max(call_counts) if call_counts else 0
    min_calls = min(call_counts) if call_counts else 0
    
    num_clusters = len([cid for cid in clusters.keys() if cid != -1])
    outliers = len(clusters.get(-1, []))
    
    if num_clusters > 0:
        cluster_sizes = [len(senders) for cid, senders in clusters.items() if cid != -1]
        avg_cluster_size = sum(cluster_sizes) / len(cluster_sizes)
        max_cluster_size = max(cluster_sizes)
    else:
        avg_cluster_size = 0
        max_cluster_size = 0

    html_content += f"""
        <div class="summary">
            <h2>Summary</h2>
            <div class="stat-grid">
                <div class="stat-item">
                    <div class="stat-value">{total_senders}</div>
                    <div class="stat-label">Senders</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{total_unique_calls}</div>
                    <div class="stat-label">Call Types</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{avg_calls_per_sender:.1f}</div>
                    <div class="stat-label">Avg Calls</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{num_clusters}</div>
                    <div class="stat-label">Clusters</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{outliers}</div>
                    <div class="stat-label">Outliers</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{max_cluster_size}</div>
                    <div class="stat-label">Max Size</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML report generated: {filename}", file=sys.stderr)
    except Exception as e:
        print(f"Error generating HTML file: {e}", file=sys.stderr)


def print_summary_statistics(sender_calls: Dict[str, Set[str]], clusters: Dict[int, List[str]]):
    """Print summary statistics."""
    print("Summary Statistics")
    print("=" * 18)
    
    total_senders = len(sender_calls)
    total_unique_calls = len(set().union(*sender_calls.values())) if sender_calls else 0
    
    # Calculate call distribution
    call_counts = [len(calls) for calls in sender_calls.values()]
    avg_calls_per_sender = np.mean(call_counts) if call_counts else 0
    max_calls = max(call_counts) if call_counts else 0
    min_calls = min(call_counts) if call_counts else 0
    
    print(f"Total senders analyzed: {total_senders}")
    print(f"Total unique call patterns: {total_unique_calls}")
    print(f"Average calls per sender: {avg_calls_per_sender:.2f}")
    print(f"Max calls by any sender: {max_calls}")
    print(f"Min calls by any sender: {min_calls}")
    print()
    
    # Cluster statistics
    num_clusters = len([cid for cid in clusters.keys() if cid != -1])
    outliers = len(clusters.get(-1, []))
    
    print(f"Number of clusters formed: {num_clusters}")
    print(f"Number of outliers: {outliers}")
    
    if num_clusters > 0:
        cluster_sizes = [len(senders) for cid, senders in clusters.items() if cid != -1]
        avg_cluster_size = np.mean(cluster_sizes)
        max_cluster_size = max(cluster_sizes)
        print(f"Average cluster size: {avg_cluster_size:.2f}")
        print(f"Largest cluster size: {max_cluster_size}")


def main():
    """Main function to process JSON from stdin and perform clustering analysis."""
    try:
        # Check for command line flags
        html_mode = '--html' in sys.argv
        analyze_eps = '--analyze-eps' in sys.argv
        
        # Read JSON from stdin
        data = json.load(sys.stdin)
        
        # Extract sender call profiles
        print("Extracting sender call profiles...", file=sys.stderr)
        sender_calls = extract_sender_call_profiles(data)
        
        if not sender_calls:
            print("No sender call data found in the input.", file=sys.stderr)
            sys.exit(1)
        
        print(f"Found {len(sender_calls)} senders with move calls.", file=sys.stderr)
        
        # Filter out senders with very few calls to reduce noise
        min_calls_threshold = 1
        filtered_sender_calls = {
            sender: calls for sender, calls in sender_calls.items() 
            if len(calls) >= min_calls_threshold
        }
        
        print(f"After filtering (min {min_calls_threshold} calls): {len(filtered_sender_calls)} senders.", file=sys.stderr)
        
        if len(filtered_sender_calls) < 2:
            print("Not enough senders for clustering analysis.", file=sys.stderr)
            sys.exit(1)
        
        # Calculate min_samples as 1% of unique senders (minimum 2)
        min_samples = min(max(2, int(len(filtered_sender_calls) * 0.0025)), 20)
        
        # Perform clustering
        print(f"Performing DBSCAN clustering (min_samples={min_samples}, 0.25% of {len(filtered_sender_calls)} senders)...", file=sys.stderr)
        clusters = cluster_senders(filtered_sender_calls, eps=0.7, min_samples=min_samples)
        
        # Print results
        if html_mode:
            # Generate HTML output
            generate_html_output(clusters, filtered_sender_calls)
        else:
            # Print to terminal
            print_cluster_analysis(clusters, filtered_sender_calls, top_n=10)
            print_summary_statistics(filtered_sender_calls, clusters)
        
        # Optionally run eps analysis
        if analyze_eps:
            plot_distance_distribution(filtered_sender_calls)
            analyze_eps_sensitivity(filtered_sender_calls)
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing data: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
