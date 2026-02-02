#!/usr/bin/env python3
"""
PTB Packages Histogram

Analyzes Sui blockchain checkpoint data and creates a histogram showing
the distribution of packages used in Programmable Transaction Block (PTB)
MoveCall commands.

Usage:
    cargo run -- --sample-count 1000 | python examples/ptb_packages_histogram.py

    # With specific checkpoint range:
    cargo run -- --sample-count 1000 --window 500000 | python examples/ptb_packages_histogram.py

    # Save histogram to file:
    cargo run -- --sample-count 1000 | python examples/ptb_packages_histogram.py --output histogram.png
"""

import json
import sys
import csv
import os
import argparse
from collections import Counter
from typing import Dict, List, Tuple, Optional


def load_package_names(csv_path: str = 'examples/sui_package_name.csv') -> Dict[str, Tuple[str, str]]:
    """Load package ID to (name, vertical) mapping from CSV file."""
    package_names = {}

    if not os.path.exists(csv_path):
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


def extract_packages_from_ptb(programmable_tx) -> List[str]:
    """Extract unique package IDs from MoveCall commands in a PTB."""
    packages = []
    if 'commands' in programmable_tx:
        for command in programmable_tx['commands']:
            if 'MoveCall' in command:
                move_call = command['MoveCall']
                package = move_call.get('package', '')
                if package:
                    # Normalize package ID to include 0x prefix
                    if not package.startswith('0x'):
                        package = f'0x{package}'
                    packages.append(package)
    return packages


def is_system_package(package_id: str) -> bool:
    """Check if a package is a system package (has many leading zeros)."""
    # Remove 0x prefix for checking
    hex_part = package_id[2:] if package_id.startswith('0x') else package_id
    # System packages have 10+ leading zeros
    return hex_part.startswith('0' * 10)


def analyze_checkpoint_data(data, include_system: bool = False) -> Tuple[Counter, int, int]:
    """
    Analyze checkpoint data and extract package usage statistics.

    Returns:
        Tuple of (package_counter, total_transactions, total_ptb_transactions)
    """
    package_counter = Counter()
    total_transactions = 0
    total_ptb_transactions = 0

    checkpoints = data.get('checkpoints', [])

    for checkpoint in checkpoints:
        transactions = checkpoint.get('transactions', [])

        for tx_data in transactions:
            total_transactions += 1
            transaction = tx_data.get('transaction', {})

            # Extract packages from ProgrammableTransaction
            programmable_tx = extract_programmable_transaction(transaction)
            if programmable_tx:
                total_ptb_transactions += 1
                packages = extract_packages_from_ptb(programmable_tx)

                for package in packages:
                    # Filter out system packages unless requested
                    if include_system or not is_system_package(package):
                        package_counter[package] += 1

    return package_counter, total_transactions, total_ptb_transactions


def print_text_histogram(package_counter: Counter, package_names: Dict[str, Tuple[str, str]],
                        top_n: int = 30, bar_width: int = 50):
    """Print a text-based histogram of package usage."""

    if not package_counter:
        print("No package data found.")
        return

    # Get top N packages
    top_packages = package_counter.most_common(top_n)

    # Find max count for scaling
    max_count = top_packages[0][1] if top_packages else 1

    print("\nPTB Package Usage Histogram")
    print("=" * 80)
    print(f"Top {min(top_n, len(top_packages))} packages by MoveCall count\n")

    for package, count in top_packages:
        # Get package name if available
        if package in package_names:
            app_name, vertical = package_names[package]
            label = f"{app_name} ({vertical})"
        else:
            # Shorten package ID for display
            label = f"{package[:10]}...{package[-6:]}"

        # Calculate bar length
        bar_length = int((count / max_count) * bar_width)
        bar = '#' * bar_length

        # Print the histogram row
        print(f"{label:35} | {bar:<{bar_width}} | {count:>6}")

    print()


def plot_matplotlib_histogram(package_counter: Counter, package_names: Dict[str, Tuple[str, str]],
                             output_file: Optional[str] = None, top_n: int = 25):
    """Create a matplotlib bar chart of package usage."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib", file=sys.stderr)
        return False

    if not package_counter:
        print("No package data to plot.")
        return False

    # Get top N packages
    top_packages = package_counter.most_common(top_n)

    # Prepare data for plotting
    labels = []
    counts = []
    colors = []

    # Color mapping by vertical
    vertical_colors = {
        'DeFi': '#2ecc71',
        'Gaming': '#e74c3c',
        'NFT': '#9b59b6',
        'Social': '#3498db',
        'Infrastructure': '#f39c12',
        'Bridge': '#1abc9c',
        'Oracle': '#e67e22',
        '': '#95a5a6'  # Unknown
    }

    for package, count in top_packages:
        if package in package_names:
            app_name, vertical = package_names[package]
            labels.append(app_name)
            colors.append(vertical_colors.get(vertical, '#95a5a6'))
        else:
            labels.append(f"{package[:8]}...")
            colors.append('#95a5a6')
        counts.append(count)

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))

    y_pos = range(len(labels))
    bars = ax.barh(y_pos, counts, color=colors)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # Top package at top
    ax.set_xlabel('Number of MoveCall Commands')
    ax.set_title('PTB Package Usage Distribution\n(Sampled from Sui Blockchain Checkpoints)')

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        width = bar.get_width()
        ax.text(width + max(counts) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{count:,}', ha='left', va='center', fontsize=9)

    # Add legend for verticals
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=vertical if vertical else 'Unknown')
                      for vertical, color in vertical_colors.items() if vertical]
    ax.legend(handles=legend_elements, loc='lower right', title='Vertical')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Histogram saved to: {output_file}")
    else:
        # Save to default location
        default_output = 'ptb_packages_histogram.png'
        plt.savefig(default_output, dpi=150, bbox_inches='tight')
        print(f"Histogram saved to: {default_output}")

    plt.close()
    return True


def print_summary_statistics(package_counter: Counter, total_transactions: int,
                            total_ptb_transactions: int, package_names: Dict[str, Tuple[str, str]]):
    """Print summary statistics about the analysis."""

    print("\nSummary Statistics")
    print("=" * 40)
    print(f"Total transactions analyzed:     {total_transactions:,}")
    print(f"Transactions with PTB:           {total_ptb_transactions:,} ({100*total_ptb_transactions/max(1,total_transactions):.1f}%)")
    print(f"Total MoveCall commands:         {sum(package_counter.values()):,}")
    print(f"Unique packages called:          {len(package_counter):,}")

    # Count known vs unknown packages
    known_count = sum(1 for p in package_counter if p in package_names)
    print(f"Known packages (in CSV):         {known_count:,}")
    print(f"Unknown packages:                {len(package_counter) - known_count:,}")

    # Vertical breakdown
    vertical_counts = Counter()
    for package, count in package_counter.items():
        if package in package_names:
            _, vertical = package_names[package]
            vertical_counts[vertical] += count
        else:
            vertical_counts['Unknown'] += count

    print("\nMoveCall Distribution by Vertical:")
    print("-" * 40)
    total_calls = sum(vertical_counts.values())
    for vertical, count in vertical_counts.most_common():
        pct = 100 * count / max(1, total_calls)
        print(f"  {vertical or 'Unknown':20} {count:>8,} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Create a histogram of packages used in PTB MoveCall commands',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage - sample 1000 checkpoints and show histogram
    cargo run -- --sample-count 1000 | python examples/ptb_packages_histogram.py

    # Save histogram to a specific file
    cargo run -- --sample-count 1000 | python examples/ptb_packages_histogram.py -o my_histogram.png

    # Include system packages in analysis
    cargo run -- --sample-count 1000 | python examples/ptb_packages_histogram.py --include-system

    # Show more packages in histogram
    cargo run -- --sample-count 1000 | python examples/ptb_packages_histogram.py --top 50
"""
    )
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output file for matplotlib histogram (default: ptb_packages_histogram.png)')
    parser.add_argument('--top', type=int, default=25,
                       help='Number of top packages to show (default: 25)')
    parser.add_argument('--include-system', action='store_true',
                       help='Include system packages (0x2, 0x3, etc.) in analysis')
    parser.add_argument('--text-only', action='store_true',
                       help='Only show text histogram, skip matplotlib chart')
    parser.add_argument('--csv-path', type=str, default='examples/sui_package_name.csv',
                       help='Path to package names CSV file')

    args = parser.parse_args()

    try:
        # Read JSON from stdin
        print("Reading checkpoint data from stdin...", file=sys.stderr)
        data = json.load(sys.stdin)

        num_checkpoints = len(data.get('checkpoints', []))
        print(f"Loaded {num_checkpoints} checkpoints", file=sys.stderr)

        # Load package names
        package_names = load_package_names(args.csv_path)
        print(f"Loaded {len(package_names)} package name mappings", file=sys.stderr)

        # Analyze the data
        print("Analyzing PTB package usage...", file=sys.stderr)
        package_counter, total_tx, total_ptb_tx = analyze_checkpoint_data(
            data, include_system=args.include_system
        )

        # Print text histogram
        print_text_histogram(package_counter, package_names, top_n=args.top)

        # Create matplotlib histogram unless text-only
        if not args.text_only:
            plot_matplotlib_histogram(package_counter, package_names,
                                     output_file=args.output, top_n=args.top)

        # Print summary statistics
        print_summary_statistics(package_counter, total_tx, total_ptb_tx, package_names)

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
