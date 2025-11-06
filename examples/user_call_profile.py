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

import csv
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.cluster import DBSCAN


# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    GREY = '\033[90m'
    BOLD = '\033[1m'
    STRIKE = '\033[9m'
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


def extract_programmable_transaction(transaction: dict) -> dict:
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


def analyze_subcluster(cluster_members: List[str], sender_calls: Dict[str, Set[str]], 
                      cluster_name: str, package_names: Dict[str, Tuple[str, str]]) -> Dict[int, List[str]]:
    """Perform detailed sub-clustering analysis on a specific cluster with low cohesion."""
    print(f"\n  {Colors.BOLD}Sub-cluster Analysis for {cluster_name}:{Colors.END}")
    print("  " + "-" * 50)
    
    # Extract only the calls for this cluster
    subcluster_calls = {sender: sender_calls[sender] for sender in cluster_members}
    
    # Use tighter clustering parameters for sub-clustering
    eps = 0.4  # Tighter than default 0.7
    min_samples = max(2, int(len(cluster_members) * 0.01))  # 1% of cluster size, min 2
    
    subclusters = cluster_senders(subcluster_calls, eps=eps, min_samples=min_samples)
    
    # Filter out tiny subclusters (less than 1% of original cluster)
    min_subcluster_size = max(5, int(len(cluster_members) * 0.01))
    significant_subclusters = {k: v for k, v in subclusters.items() 
                             if k == -1 or len(v) >= min_subcluster_size}
    
    num_subclusters = len([k for k in significant_subclusters.keys() if k != -1])
    
    if num_subclusters <= 1:
        print(f"  No meaningful sub-clusters found (tried eps={eps}, min_samples={min_samples})")
        return {}
    
    print(f"  Found {num_subclusters} significant sub-clusters (eps={eps}, min_samples={min_samples})")
    
    # Sort subclusters by size
    sorted_subclusters = sorted(significant_subclusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    for sub_id, sub_senders in sorted_subclusters:
        if sub_id == -1:
            continue  # Skip outliers in subcluster analysis
            
        sub_size = len(sub_senders)
        sub_percentage = (sub_size / len(cluster_members)) * 100
        
        # Generate name for this subcluster
        sub_name = generate_cluster_name(sub_id, sub_senders, sender_calls, package_names)
        
        print(f"\n    {Colors.GREEN}{sub_name} (Sub {sub_id}): {sub_size} senders ({sub_percentage:.1f}% of {cluster_name}){Colors.END}")
        
        # Analyze this sub-cluster's specific patterns
        call_frequency = defaultdict(int)
        for sender in sub_senders:
            for call in sender_calls[sender]:
                call_frequency[call] += 1
        
        # Get calls that appear in >50% of this sub-cluster (higher threshold for subclusters)
        frequent_calls = [(call, freq) for call, freq in call_frequency.items() 
                         if freq > len(sub_senders) * 0.5]
        frequent_calls = sorted(frequent_calls, key=lambda x: x[1], reverse=True)[:5]
        
        if frequent_calls:
            print(f"      Top calls and events (>50% of sub-cluster):")
            for call, frequency in frequent_calls:
                # Handle both regular calls and event types
                if call.startswith('event:'):
                    actual_call = call[6:]
                    package_id = actual_call.split('::', 1)[0] if '::' in actual_call else ''
                    short_call = "::".join(actual_call.split("::")[1:]) if "::" in actual_call else actual_call
                    call_type = "[EVENT]"
                else:
                    package_id = call.split('::', 1)[0] if '::' in call else ''
                    short_call = "::".join(call.split("::")[1:]) if "::" in call else call
                    call_type = "[CALL]"
                
                percentage = (frequency / len(sub_senders)) * 100
                
                # Add package info if available
                call_display = f"{call_type} {short_call}"
                if package_id in package_names:
                    app_name, vertical = package_names[package_id]
                    call_display = f"{call_type} {short_call} [{Colors.RED}{app_name}{Colors.END}]"
                
                print(f"        {frequency:>3}/{len(sub_senders)} ({percentage:>5.1f}%) - {call_display}")
        else:
            print(f"      No calls appear in >50% of sub-cluster members")
    
    # Analyze outliers from sub-clustering (senders that didn't fit into any sub-cluster)
    outlier_senders = significant_subclusters.get(-1, [])
    if outlier_senders:
        outlier_count = len(outlier_senders)
        outlier_percentage = (outlier_count / len(cluster_members)) * 100
        
        print(f"\n    {Colors.BOLD}Sub-cluster Outliers: {outlier_count} senders ({outlier_percentage:.1f}% of {cluster_name}){Colors.END}")
        
        # Analyze outlier call patterns
        outlier_call_frequency = defaultdict(int)
        for sender in outlier_senders:
            for call in sender_calls[sender]:
                outlier_call_frequency[call] += 1
        
        # Get top calls that appear in >25% of outliers (lower threshold since these are diverse)
        outlier_frequent_calls = [(call, freq) for call, freq in outlier_call_frequency.items() 
                                if freq > len(outlier_senders) * 0.25]
        outlier_frequent_calls = sorted(outlier_frequent_calls, key=lambda x: x[1], reverse=True)[:5]
        
        if outlier_frequent_calls:
            print(f"      Top calls and events in outliers (>25%):")
            for call, frequency in outlier_frequent_calls:
                # Handle both regular calls and event types
                if call.startswith('event:'):
                    actual_call = call[6:]
                    package_id = actual_call.split('::', 1)[0] if '::' in actual_call else ''
                    short_call = "::".join(actual_call.split("::")[1:]) if "::" in actual_call else actual_call
                    call_type = "[EVENT]"
                else:
                    package_id = call.split('::', 1)[0] if '::' in call else ''
                    short_call = "::".join(call.split("::")[1:]) if "::" in call else call
                    call_type = "[CALL]"
                
                percentage = (frequency / len(outlier_senders)) * 100
                
                # Add package info if available
                call_display = f"{call_type} {short_call}"
                if package_id in package_names:
                    app_name, vertical = package_names[package_id]
                    call_display = f"{call_type} {short_call} [{Colors.RED}{app_name}{Colors.END}]"
                
                print(f"        {frequency:>3}/{len(outlier_senders)} ({percentage:>5.1f}%) - {call_display}")
        else:
            print(f"      No common patterns in outliers (too diverse)")
    
    return significant_subclusters


def write_outlier_sequences(outlier_senders: List[str], sender_calls: Dict[str, Set[str]], filename: str = "outlier_sequences.txt") -> None:
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


def analyze_inter_cluster_distances(clusters: Dict[int, List[str]], 
                                   sender_calls: Dict[str, Set[str]]) -> Dict[int, Dict[int, float]]:
    """Analyze average distances between different clusters."""
    package_names = load_package_names()
    
    # Generate names for all clusters for display
    cluster_names = {}
    for cluster_id, senders in clusters.items():
        cluster_names[cluster_id] = generate_cluster_name(cluster_id, senders, sender_calls, package_names)
    
    # Calculate average distance between each pair of clusters
    cluster_ids = list(clusters.keys())
    cluster_distances = {}
    
    print(f"\n{Colors.BOLD}Inter-Cluster Distance Analysis{Colors.END}")
    print("=" * 35)
    
    for cluster_id in cluster_ids:
        cluster_distances[cluster_id] = {}
        
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            cluster1_id, cluster2_id = cluster_ids[i], cluster_ids[j]
            cluster1_senders = clusters[cluster1_id]
            cluster2_senders = clusters[cluster2_id]
            
            # Sample for performance if clusters are large
            sample_size = min(100, len(cluster1_senders), len(cluster2_senders))
            sample1 = random.sample(cluster1_senders, min(sample_size, len(cluster1_senders)))
            sample2 = random.sample(cluster2_senders, min(sample_size, len(cluster2_senders)))
            
            # Calculate all pairwise distances
            distances = []
            for sender1 in sample1:
                for sender2 in sample2:
                    distance = jaccard_distance(sender_calls[sender1], sender_calls[sender2])
                    distances.append(distance)
            
            if distances:
                avg_distance = np.mean(distances)
                
                # Store bidirectional distances
                cluster_distances[cluster1_id][cluster2_id] = avg_distance
                cluster_distances[cluster2_id][cluster1_id] = avg_distance
    
    return cluster_distances


def print_cluster_distances(cluster_id: int, cluster_name: str, cluster_distances: Dict[int, Dict[int, float]], 
                          cluster_names: Dict[int, str]) -> None:
    """Print distances from one cluster to all other clusters (only showing distances < 1.0)."""
    if cluster_id not in cluster_distances:
        return
        
    distances = cluster_distances[cluster_id]
    if not distances:
        return
    
    # Sort by distance (closest first) and filter for distances < 1.0
    filtered_distances = [(other_id, dist) for other_id, dist in distances.items() if dist < 1.0]
    
    if not filtered_distances:
        return  # No similar clusters to show
    
    sorted_distances = sorted(filtered_distances, key=lambda x: x[1])
    
    print(f"  Similar clusters (distance < 1.0):")
    for other_cluster_id, distance in sorted_distances:
        other_name = cluster_names.get(other_cluster_id, f"Cluster {other_cluster_id}")
        print(f"    → {other_name}: {distance:.3f}")
    print()


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


def print_cluster_analysis(clusters: Dict[int, List[str]], sender_calls: Dict[str, Set[str]], top_n: int = 10) -> None:
    """Print detailed cluster analysis."""
    print("Sui Sender Call Pattern Clustering Analysis")
    print("=" * 45)
    print()
    
    # Load package names
    package_names = load_package_names()
    
    # Calculate total number of senders for percentage calculation
    total_senders = len(sender_calls)
    
    # Calculate inter-cluster distances first (if we have multiple clusters)
    cluster_distances = {}
    cluster_names = {}
    if len(clusters) > 1:
        cluster_distances = analyze_inter_cluster_distances(clusters, sender_calls)
        # Generate cluster names for distance display
        for cluster_id, senders in clusters.items():
            cluster_names[cluster_id] = generate_cluster_name(cluster_id, senders, sender_calls, package_names)
    
    # Sort clusters by size (largest first)
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    # Precompute interesting clusters for CLI display using the same rule as HTML
    cluster_info = {}
    for cluster_id, senders in sorted_clusters:
        size = len(senders)
        call_frequency = defaultdict(int)
        for sender in senders:
            for call in sender_calls.get(sender, []):
                call_frequency[call] += 1
        # Apply same display filter as the UI: only consider calls that appear in >25% of members
        filtered_calls = [(call, freq) for call, freq in call_frequency.items() if freq > size * 0.25]
        top5 = sorted(filtered_calls, key=lambda x: x[1], reverse=True)[:5]
        if size == 0:
            interesting = False
        else:
            # If there are no calls passing the display filter, treat cluster as interesting
            if not top5:
                interesting = True
            else:
                # Cluster is interesting if any of the displayed top-5 calls is present in <95% of members
                interesting = any((freq / size) < 0.95 for _, freq in top5)
        cluster_info[cluster_id] = {"size": size, "interesting": interesting}

    adjusted_total_senders = sum(info["size"] for info in cluster_info.values() if info["interesting"]) or total_senders

    for cluster_id, senders in sorted_clusters:
        info = cluster_info.get(cluster_id, {"size": len(senders), "interesting": True})
        cluster_size = info["size"]

        # Compute percentage: interesting clusters use the adjusted denominator
        if info["interesting"]:
            denom = adjusted_total_senders if adjusted_total_senders > 0 else total_senders
            percentage = (cluster_size / denom) * 100
        else:
            percentage = (cluster_size / total_senders) * 100 if total_senders > 0 else 0
        
        # Generate cluster name
        cluster_name = generate_cluster_name(cluster_id, senders, sender_calls, package_names)
        # Build suffix: include percentage only for interesting clusters
        if info["interesting"]:
            if cluster_id == -1:
                suffix = f": {cluster_size} senders ({percentage:.1f}% of interesting senders)"
            else:
                suffix = f" (Cluster {cluster_id}): {cluster_size} senders ({percentage:.1f}% of interesting senders)"
        else:
            if cluster_id == -1:
                suffix = f": {cluster_size} senders"
            else:
                suffix = f" (Cluster {cluster_id}): {cluster_size} senders"

        # CLI: use green bold for interesting clusters, grey + strike-through name for not-interesting
        if info["interesting"]:
            header = f"{cluster_name}{suffix}"
            print(f"{Colors.GREEN}{Colors.BOLD}{header}{Colors.END}")
        else:
            # Strike-through the name but keep the rest grey
            struck = f"{Colors.GREY}{Colors.STRIKE}{cluster_name}{Colors.END}{Colors.GREY}"
            header = f"{struck}{suffix}{Colors.END}"
            print(header)
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
                
                # Check cohesion - if the top call is <75%, do subcluster analysis
                if frequent_calls and len(senders) >= 20:  # Only for clusters with 20+ members
                    top_call_percentage = (frequent_calls[0][1] / len(senders)) * 100
                    if top_call_percentage < 75.0:
                        # This cluster has low cohesion, perform sub-cluster analysis
                        analyze_subcluster(senders, sender_calls, cluster_name, package_names)
            else:
                print(f"  No calls appear in >25% of cluster members")
            
            print(f"  Total unique calls and events in cluster: {len(all_calls)}")
            
            # Show distances to similar clusters (< 1.0) right under each cluster
            if cluster_distances and cluster_names:
                print_cluster_distances(cluster_id, cluster_name, cluster_distances, cluster_names)
        
        print()


def generate_html_subcluster_analysis(cluster_members: List[str], sender_calls: Dict[str, Set[str]], 
                                     cluster_name: str, package_names: Dict[str, Tuple[str, str]]) -> str:
    """Generate HTML for subcluster analysis."""
    # Extract only the calls for this cluster
    subcluster_calls = {sender: sender_calls[sender] for sender in cluster_members}
    
    # Use tighter clustering parameters for sub-clustering
    eps = 0.4
    min_samples = max(2, int(len(cluster_members) * 0.01))
    
    subclusters = cluster_senders(subcluster_calls, eps=eps, min_samples=min_samples)
    
    # Filter out tiny subclusters
    min_subcluster_size = max(5, int(len(cluster_members) * 0.01))
    significant_subclusters = {k: v for k, v in subclusters.items() 
                             if k == -1 or len(v) >= min_subcluster_size}
    
    num_subclusters = len([k for k in significant_subclusters.keys() if k != -1])
    
    if num_subclusters <= 1:
        return ""
    
    html = f"""
                <div class="subcluster-analysis">
                    <h4>Sub-cluster Analysis ({num_subclusters} sub-clusters found)</h4>
"""
    
    # Sort subclusters by size
    sorted_subclusters = sorted(significant_subclusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    for sub_id, sub_senders in sorted_subclusters:
        if sub_id == -1:
            continue
            
        sub_size = len(sub_senders)
        sub_percentage = (sub_size / len(cluster_members)) * 100
        
        # Generate name for this subcluster
        sub_name = generate_cluster_name(sub_id, sub_senders, sender_calls, package_names)
        
        html += f"""
                    <div class="subcluster">
                        <div class="subcluster-header">
                            {sub_name} (Sub {sub_id}): {sub_size} senders ({sub_percentage:.1f}% of {cluster_name})
                        </div>
"""
        
        # Analyze this sub-cluster's patterns
        call_frequency = defaultdict(int)
        for sender in sub_senders:
            for call in sender_calls[sender]:
                call_frequency[call] += 1
        
        # Get calls that appear in >50% of this sub-cluster
        frequent_calls = [(call, freq) for call, freq in call_frequency.items() 
                         if freq > len(sub_senders) * 0.5]
        frequent_calls = sorted(frequent_calls, key=lambda x: x[1], reverse=True)[:5]
        
        if frequent_calls:
            html += """
                        <div class="subcall-list">
                            <strong>Top calls and events (>50%):</strong>
"""
            for call, frequency in frequent_calls:
                # Handle both regular calls and event types
                if call.startswith('event:'):
                    actual_call = call[6:]
                    package_id = actual_call.split('::', 1)[0] if '::' in actual_call else ''
                    short_call = "::".join(actual_call.split("::")[1:]) if "::" in actual_call else actual_call
                    call_type = "<strong>[EVENT]</strong>"
                else:
                    package_id = call.split('::', 1)[0] if '::' in call else ''
                    short_call = "::".join(call.split("::")[1:]) if "::" in call else call
                    call_type = "<strong>[CALL]</strong>"
                
                call_percentage = (frequency / len(sub_senders)) * 100
                
                # Add package info if available
                call_display = f"{call_type} {short_call}"
                if package_id in package_names:
                    app_name, vertical = package_names[package_id]
                    call_display = f'{call_type} {short_call} [<span class="app-name">{app_name}</span>]'
                
                html += f"""
                            <div class="subcall-item">
                                {frequency}/{len(sub_senders)} ({call_percentage:.0f}%) - {call_display}
                            </div>
"""
            html += "                        </div>\n"
        else:
            html += """
                        <div class="stats">
                            No calls >50% frequency
                        </div>
"""
        
        html += "                    </div>\n"
    
    # Add outlier analysis for sub-clustering
    outlier_senders = significant_subclusters.get(-1, [])
    if outlier_senders:
        outlier_count = len(outlier_senders)
        outlier_percentage = (outlier_count / len(cluster_members)) * 100
        
        html += f"""
                    <div class="subcluster subcluster-outliers">
                        <div class="subcluster-header">
                            Sub-cluster Outliers: {outlier_count} senders ({outlier_percentage:.1f}% of {cluster_name})
                        </div>
"""
        
        # Analyze outlier call patterns
        outlier_call_frequency = defaultdict(int)
        for sender in outlier_senders:
            for call in sender_calls[sender]:
                outlier_call_frequency[call] += 1
        
        # Get top calls that appear in >25% of outliers
        outlier_frequent_calls = [(call, freq) for call, freq in outlier_call_frequency.items() 
                                if freq > len(outlier_senders) * 0.25]
        outlier_frequent_calls = sorted(outlier_frequent_calls, key=lambda x: x[1], reverse=True)[:5]
        
        if outlier_frequent_calls:
            html += """
                        <div class="subcall-list">
                            <strong>Top calls and events in outliers (>25%):</strong>
"""
            for call, frequency in outlier_frequent_calls:
                # Handle both regular calls and event types
                if call.startswith('event:'):
                    actual_call = call[6:]
                    package_id = actual_call.split('::', 1)[0] if '::' in actual_call else ''
                    short_call = "::".join(actual_call.split("::")[1:]) if "::" in actual_call else actual_call
                    call_type = "<strong>[EVENT]</strong>"
                else:
                    package_id = call.split('::', 1)[0] if '::' in call else ''
                    short_call = "::".join(call.split("::")[1:]) if "::" in call else call
                    call_type = "<strong>[CALL]</strong>"
                
                call_percentage = (frequency / len(outlier_senders)) * 100
                
                # Add package info if available
                call_display = f"{call_type} {short_call}"
                if package_id in package_names:
                    app_name, vertical = package_names[package_id]
                    call_display = f'{call_type} {short_call} [<span class="app-name">{app_name}</span>]'
                
                html += f"""
                            <div class="subcall-item">
                                {frequency}/{len(outlier_senders)} ({call_percentage:.0f}%) - {call_display}
                            </div>
"""
            html += "                        </div>\n"
        else:
            html += """
                        <div class="stats">
                            No common patterns in outliers (too diverse)
                        </div>
"""
        
        html += "                    </div>\n"
    
    html += "                </div>\n"
    return html


def generate_html_output(clusters: Dict[int, List[str]], sender_calls: Dict[str, Set[str]], filename: str = "cluster_analysis.html", data=None) -> None:
    """Generate a pretty HTML file with clustering analysis."""
    
    # Load package names
    package_names = load_package_names()
    
    # Calculate total number of senders for percentage calculation
    total_senders = len(sender_calls)
    
    # Calculate inter-cluster distances first (if we have multiple clusters)
    cluster_distances = {}
    cluster_names = {}
    if len(clusters) > 1:
        cluster_distances = analyze_inter_cluster_distances(clusters, sender_calls)
        # Generate cluster names for distance display
        for cluster_id, senders in clusters.items():
            cluster_names[cluster_id] = generate_cluster_name(cluster_id, senders, sender_calls, package_names)
    
    # Sort clusters by size (largest first)
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Precompute which clusters are "interesting" (used to adjust percentage denominators)
    # A cluster is interesting if among its top-5 calls (by frequency) at least one call
    # is present in <95% of its members (i.e., there is some diversity). If the cluster
    # has no calls, treat it as interesting by default.
    cluster_info = {}
    for cluster_id, senders in sorted_clusters:
        size = len(senders)
        call_frequency = defaultdict(int)
        for sender in senders:
            for call in sender_calls.get(sender, []):
                call_frequency[call] += 1
        # Apply same display filter as the UI: only consider calls that appear in >25% of members
        filtered_calls = [(call, freq) for call, freq in call_frequency.items() if freq > size * 0.25]
        top5 = sorted(filtered_calls, key=lambda x: x[1], reverse=True)[:5]
        if size == 0:
            interesting = False
        else:
            # If there are no calls passing the display filter, treat cluster as interesting
            if not top5:
                interesting = True
            else:
                # Cluster is interesting if any of the displayed top-5 calls is present in <95% of members
                interesting = any((freq / size) < 0.95 for _, freq in top5)
        cluster_info[cluster_id] = {"size": size, "interesting": interesting, "top5": top5}

    # Adjusted total senders excludes non-interesting clusters so that percentages
    # for interesting clusters are computed only over the interesting subset.
    adjusted_total_senders = sum(info["size"] for info in cluster_info.values() if info["interesting"])
    if adjusted_total_senders == 0:
        adjusted_total_senders = total_senders

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
        /* Not-interesting clusters: grey header and strike-through the name */
        .cluster-header.not-interesting {{
            background: #6c757d;
            color: #ffffff;
        }}
        .cluster-header .cluster-name {{
            text-decoration: none;
        }}
        .cluster-header.not-interesting .cluster-name {{
            text-decoration: line-through;
            color: #f8f9fa;
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
        .subcluster-analysis {{
            margin-top: 15px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
            border-left: 4px solid #007bff;
        }}
        .subcluster-analysis h4 {{
            margin: 0 0 10px 0;
            color: #495057;
            font-size: 1.1em;
        }}
        .subcluster {{
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 3px;
            margin: 8px 0;
            padding: 8px;
        }}
        .subcluster-header {{
            color: #17a2b8;
            font-weight: bold;
            margin-bottom: 6px;
            font-size: 0.95em;
        }}
        .subcall-list {{
            margin: 6px 0;
        }}
        .subcall-item {{
            margin: 2px 0;
            padding: 3px 6px;
            background: #f1f3f4;
            border-radius: 2px;
            border-left: 2px solid #17a2b8;
            font-size: 11px;
        }}
        .subcluster-outliers {{
            border-left-color: #dc3545;
            background: #fdf2f2;
        }}
        .subcluster-outliers .subcluster-header {{
            color: #dc3545;
        }}
        .subcluster-outliers .subcall-item {{
            border-left-color: #dc3545;
            background: #f8d7da;
        }}
        .cluster-distances {{
            margin: 10px 0;
            padding: 8px;
            background: #f1f3f4;
            border-radius: 3px;
            border-left: 3px solid #6c757d;
        }}
        .cluster-distances h4 {{
            margin: 0 0 6px 0;
            color: #495057;
            font-size: 0.9em;
            font-weight: bold;
        }}
        .distance-list {{
            margin: 3px 0;
        }}
        .distance-item {{
            margin: 2px 0;
            padding: 3px 6px;
            background: white;
            border-radius: 2px;
            border-left: 3px solid #6c757d;
            font-size: 11px;
        }}
        /* Print-specific styles to make the report printer-friendly */
        @media print {{
            body {{
                background: white !important;
                color: #000 !important;
                padding: 0.5in !important;
                font-size: 12pt !important;
            }}

            /* Keep clusters intact on a single page when possible */
            .cluster {{
                background: white !important;
                border: 1px solid #000 !important;
                box-shadow: none !important;
                page-break-inside: avoid !important;
                break-inside: avoid !important;
                -webkit-column-break-inside: avoid !important;
            }}

            /* Simplify headers (background colors often don't print) */
            .cluster-header {{
                background: none !important;
                color: #000 !important;
                font-size: 1em !important;
                padding: 4px 0 !important;
                border-bottom: 1px solid #000 !important;
            }}

            /* Make call / stat items printer-friendly */
            .call-item, .stat-item, .subcall-item, .distance-item {{
                background: transparent !important;
                border-left-color: #000 !important;
                color: #000 !important;
                font-size: 10pt !important;
                padding: 2px 4px !important;
            }}

            .cluster-distances {{
                font-size: 10pt !important;
                padding: 4px !important;
            }}

            /* Summary should not be split across pages */
            .summary, .stat-grid {{
                page-break-inside: avoid !important;
                break-inside: avoid !important;
            }}

            /* Try to preserve colors when printing where supported */
            * {{
                -webkit-print-color-adjust: exact !important;
                print-color-adjust: exact !important;
            }}
        }}
    </style>
</head>
<body>
    <h1>Sui Sender Clustering Analysis</h1>
    <div class="grid-layout">
"""

    for cluster_id, senders in sorted_clusters:
        info = cluster_info.get(cluster_id, {"size": len(senders), "interesting": True})
        cluster_size = info["size"]

        # Compute percentage: interesting clusters use the adjusted denominator
        if info["interesting"]:
            denom = adjusted_total_senders if adjusted_total_senders > 0 else total_senders
            percentage = (cluster_size / denom) * 100
            header_class = "cluster-header"
        else:
            percentage = (cluster_size / total_senders) * 100 if total_senders > 0 else 0
            header_class = "cluster-header not-interesting"

        # Generate cluster name and wrap the name so we can strike-through only the name
        cluster_name = generate_cluster_name(cluster_id, senders, sender_calls, package_names)
        if info.get("interesting", True):
            header_text = f"<span class='cluster-name'>{cluster_name}</span> ({cluster_size}, {percentage:.1f}%)"
        else:
            # For not-interesting clusters we only show the count (no fraction)
            header_text = f"<span class='cluster-name'>{cluster_name}</span> ({cluster_size})"
        
        if cluster_id == -1:
            header_class += " outlier-header"
            # Write outlier sequences to file for HTML mode too
            write_outlier_sequences(senders, sender_calls)
        
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
                
                # Check cohesion - if the top call is <75%, do subcluster analysis
                if len(senders) >= 20:  # Only for clusters with 20+ members
                    top_call_percentage = (frequent_calls[0][1] / len(senders)) * 100
                    if top_call_percentage < 75.0:
                        # This cluster has low cohesion, add sub-cluster analysis
                        subcluster_html = generate_html_subcluster_analysis(senders, sender_calls, cluster_name, package_names)
                        html_content += subcluster_html
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
            
            # Add distance analysis for this cluster (only if distances < 1.0 exist)
            if cluster_distances and cluster_names and cluster_id in cluster_distances:
                distances = cluster_distances[cluster_id]
                filtered_distances = [(other_id, dist) for other_id, dist in distances.items() if dist < 1.0]
                
                if filtered_distances:
                    sorted_distances = sorted(filtered_distances, key=lambda x: x[1])
                    
                    html_content += """
            <div class="cluster-distances">
                <h4>Similar clusters (distance < 1.0):</h4>
                <div class="distance-list">
"""
                    
                    for other_cluster_id, distance in sorted_distances:
                        other_name = cluster_names.get(other_cluster_id, f"Cluster {other_cluster_id}")
                        # Color code by distance: closer to 0 = more similar (green), closer to 1 = more different (yellow)
                        color_intensity = int(distance * 255)
                        color = f"rgb({color_intensity}, {255-color_intensity}, 0)"
                        
                        html_content += f"""
                    <div class="distance-item" style="border-left-color: {color};">
                        → {other_name}: <strong>{distance:.3f}</strong>
                    </div>
"""
                    
                    html_content += """
                </div>
            </div>
"""
        
        html_content += "        </div>\n"

    # Add summary statistics
    # Count checkpoints and transactions if data is available
    num_checkpoints = 0
    total_transactions = 0
    if data:
        checkpoints = data.get('checkpoints', [])
        num_checkpoints = len(checkpoints)
        total_transactions = sum(len(checkpoint.get('transactions', [])) for checkpoint in checkpoints)
    
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
                    <div class="stat-value">{num_checkpoints}</div>
                    <div class="stat-label">Checkpoints</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{total_transactions}</div>
                    <div class="stat-label">Transactions</div>
                </div>
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


def print_summary_statistics(sender_calls: Dict[str, Set[str]], clusters: Dict[int, List[str]], data=None) -> None:
    """Print summary statistics."""
    print("Summary Statistics")
    print("=" * 18)
    
    # Count checkpoints and transactions if data is available
    if data:
        checkpoints = data.get('checkpoints', [])
        num_checkpoints = len(checkpoints)
        total_transactions = sum(len(checkpoint.get('transactions', [])) for checkpoint in checkpoints)
        print(f"Checkpoints analyzed: {num_checkpoints}")
        print(f"Transactions analyzed: {total_transactions}")
        print()
    
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


def main() -> None:
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
            generate_html_output(clusters, filtered_sender_calls, data=data)
        else:
            # Print to terminal
            print_cluster_analysis(clusters, filtered_sender_calls, top_n=10)
            print_summary_statistics(filtered_sender_calls, clusters, data)
        
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
