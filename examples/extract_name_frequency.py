#!/usr/bin/env python3
"""
Extract name frequency from Sui blockchain checkpoint JSON data.

This script analyzes checkpoint data and extracts frequency statistics for:
- Package::module::function names from MoveCall commands
- Package::module::type names from input objects  
- Package::module::type names from events

Results are printed sorted by frequency (most to least frequent) grouped by package.
"""

import json
import sys
import csv
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional


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


def format_package_header(package_id: str, total_activity: int, package_names: Dict[str, Tuple[str, str]]) -> str:
    """Format the package header with name and vertical if available."""
    if package_id in package_names:
        app_name, vertical = package_names[package_id]
        return f"Package: {package_id} - {app_name} ({vertical}) (Total: {total_activity} occurrences)"
    else:
        return f"Package: {package_id} (Total: {total_activity} occurrences)"


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
    return names


def extract_object_type_names(objects) -> List[str]:
    """Extract package::module::type names from object data."""
    names = []
    for obj in objects:
        if 'data' in obj and 'Move' in obj['data']:
            move_data = obj['data']['Move']
            if 'type_' in move_data:
                type_info = move_data['type_']
                
                # Handle string types like "GasCoin"
                if isinstance(type_info, str):
                    # Built-in types like GasCoin - we'll use a standard prefix
                    names.append(f"0x0000000000000000000000000000000000000000000000000000000000000002::coin::{type_info}")
                
                # Handle structured types
                elif isinstance(type_info, dict):
                    if 'Other' in type_info:
                        struct_info = type_info['Other']
                        package = struct_info.get('address', '')
                        module = struct_info.get('module', '')
                        name = struct_info.get('name', '')
                        if package and module and name:
                            # Normalize package ID to include 0x prefix
                            if not package.startswith('0x'):
                                package = f'0x{package}'
                            names.append(f"{package}::{module}::{name}")
                    elif 'struct' in type_info:
                        struct_info = type_info['struct']
                        package = struct_info.get('address', '')
                        module = struct_info.get('module', '')
                        name = struct_info.get('name', '')
                        if package and module and name:
                            # Normalize package ID to include 0x prefix
                            if not package.startswith('0x'):
                                package = f'0x{package}'
                            names.append(f"{package}::{module}::{name}")
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


def group_by_package(names_counter: Counter) -> Dict[str, Dict[str, int]]:
    """Group names by package and return as package -> {name: count} dict."""
    package_groups = defaultdict(dict)
    
    for name, count in names_counter.items():
        # Extract package from name (everything before the first ::)
        if '::' in name:
            package = name.split('::', 1)[0]
            package_groups[package][name] = count
    
    return dict(package_groups)


def print_combined_frequency_report(move_call_counter: Counter, input_object_counter: Counter, event_counter: Counter,
                                   move_call_senders: Dict, input_object_senders: Dict, event_senders: Dict):
    """Print a comprehensive frequency report organized by package."""
    
    # Load package names
    package_names = load_package_names()
    
    # Group data by package
    move_call_by_package = group_by_package(move_call_counter)
    input_object_by_package = group_by_package(input_object_counter)
    event_by_package = group_by_package(event_counter)
    
    # Get all packages and their total activity
    all_packages = set()
    all_packages.update(move_call_by_package.keys())
    all_packages.update(input_object_by_package.keys())
    all_packages.update(event_by_package.keys())
    
    # Calculate total activity per package for sorting
    package_totals = {}
    for package in all_packages:
        total = 0
        if package in move_call_by_package:
            total += sum(move_call_by_package[package].values())
        if package in input_object_by_package:
            total += sum(input_object_by_package[package].values())
        if package in event_by_package:
            total += sum(event_by_package[package].values())
        package_totals[package] = total
    
    # Print combined report organized by package
    print("Sui Blockchain Activity Report")
    print("=" * 30)
    print()
    
    for package in sorted(package_totals.keys(), key=lambda x: package_totals[x], reverse=True):
        total_activity = package_totals[package]
        header = format_package_header(package, total_activity, package_names)
        print(header)
        print("-" * max(86, len(header)))
        
        # Print MoveCall functions for this package
        if package in move_call_by_package:
            functions = move_call_by_package[package]
            total_calls = sum(functions.values())
            print(f"  MoveCall Functions ({total_calls} calls):")
            for func_name, count in sorted(functions.items(), key=lambda x: x[1], reverse=True):
                # Remove package prefix to shorten the name
                short_name = func_name.replace(f"{package}::", "")
                unique_senders = len(move_call_senders.get(func_name, set()))
                print(f"    {count:>4}  {short_name} ({unique_senders} unique senders)")
            print()
        
        # Print Input Object types for this package
        if package in input_object_by_package:
            types = input_object_by_package[package]
            total_objects = sum(types.values())
            print(f"  Input Object Types ({total_objects} objects):")
            for type_name, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
                # Remove package prefix to shorten the name
                short_name = type_name.replace(f"{package}::", "")
                unique_senders = len(input_object_senders.get(type_name, set()))
                print(f"    {count:>4}  {short_name} ({unique_senders} unique senders)")
            print()
        
        # Print Event types for this package
        if package in event_by_package:
            events = event_by_package[package]
            total_events_pkg = sum(events.values())
            print(f"  Event Types ({total_events_pkg} events):")
            for event_name, count in sorted(events.items(), key=lambda x: x[1], reverse=True):
                # Remove package prefix to shorten the name
                short_name = event_name.replace(f"{package}::", "")
                unique_senders = len(event_senders.get(event_name, set()))
                print(f"    {count:>4}  {short_name} ({unique_senders} unique senders)")
            print()
        
        print()  # Extra line between packages


def analyze_checkpoint_data(data):
    """Analyze checkpoint data and extract name frequencies with unique senders."""
    move_call_names = Counter()
    input_object_names = Counter()
    event_names = Counter()
    
    # Track unique senders for each name
    move_call_senders = defaultdict(set)
    input_object_senders = defaultdict(set)
    event_senders = defaultdict(set)
    
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
            
            # Extract MoveCall names from ProgrammableTransaction
            programmable_tx = extract_programmable_transaction(transaction)
            if programmable_tx:
                call_names = extract_move_call_names(programmable_tx)
                move_call_names.update(call_names)
                for name in call_names:
                    move_call_senders[name].add(sender)
            
            # Extract type names from input objects
            input_objects = tx_data.get('input_objects', [])
            input_type_names = extract_object_type_names(input_objects)
            input_object_names.update(input_type_names)
            for name in input_type_names:
                input_object_senders[name].add(sender)
            
            # Extract type names from events
            events = tx_data.get('events')
            if events:
                event_type_names = extract_event_type_names(events)
                event_names.update(event_type_names)
                for name in event_type_names:
                    event_senders[name].add(sender)
    
    return (move_call_names, input_object_names, event_names, 
            move_call_senders, input_object_senders, event_senders)


def main():
    """Main function to process JSON from stdin."""
    try:
        # Read JSON from stdin
        data = json.load(sys.stdin)
        
        # Analyze the data
        result = analyze_checkpoint_data(data)
        move_call_names, input_object_names, event_names = result[:3]
        move_call_senders, input_object_senders, event_senders = result[3:]
        
        # Print combined frequency report
        print_combined_frequency_report(move_call_names, input_object_names, event_names,
                                      move_call_senders, input_object_senders, event_senders)
        
        # Print summary statistics
        print("Summary Statistics")
        print("=" * 18)
        print(f"Total MoveCall commands: {sum(move_call_names.values())}")
        print(f"Unique MoveCall patterns: {len(move_call_names)}")
        print(f"Total input objects: {sum(input_object_names.values())}")
        print(f"Unique input object types: {len(input_object_names)}")
        print(f"Total events: {sum(event_names.values())}")
        print(f"Unique event types: {len(event_names)}")
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing data: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()