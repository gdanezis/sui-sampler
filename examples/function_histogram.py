#!/usr/bin/env python3
"""
Display ASCII histograms of MoveCall function calls grouped by execution status.

Reads sui-sampler JSON from stdin, extracts MoveCall function names, and prints
two bar-chart histograms: one for successful transactions, one for aborted.

Usage:
    ./target/release/sui-sampler --sample-count 200 | python3 examples/function_histogram.py
"""

import json
import os
import sys
from collections import Counter


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


def extract_move_call_labels(programmable_tx):
    """Extract module::function labels from MoveCall commands."""
    labels = []
    if 'commands' in programmable_tx:
        for command in programmable_tx['commands']:
            if 'MoveCall' in command:
                move_call = command['MoveCall']
                module = move_call.get('module', '')
                function = move_call.get('function', '')
                if module and function:
                    labels.append(f"{module}::{function}")
    return labels


def get_execution_status(tx_data):
    """Return 'success' or 'aborted' based on tx effects status."""
    effects = tx_data.get('effects', {})
    v2 = effects.get('V2', {})
    status = v2.get('status')
    if status == 'Success':
        return 'success'
    # {"Failure": {...}} or any non-Success value
    if status is not None:
        return 'aborted'
    return None


def render_histogram(counter, title, max_bars=40):
    """Render an ASCII horizontal bar chart to stdout."""
    if not counter:
        print(f"\n{title}")
        print("  (no data)")
        return

    top = counter.most_common(max_bars)
    max_count = top[0][1]
    total = sum(counter.values())

    # Determine available width for bars
    try:
        term_width = os.get_terminal_size().columns
    except OSError:
        term_width = 100

    # Find the longest label
    max_label_len = max(len(label) for label, _ in top)
    # count field: "  12345"
    count_width = len(str(max_count)) + 2
    # layout: label + " " + bar + " " + count
    bar_area = term_width - max_label_len - count_width - 3
    if bar_area < 10:
        bar_area = 10

    print(f"\n{title} ({total} total calls, {len(counter)} unique functions)")
    print("─" * min(term_width, len(title) + 40))

    for label, count in top:
        bar_len = int(count / max_count * bar_area) if max_count > 0 else 0
        bar = "█" * bar_len
        print(f"{label:>{max_label_len}} {bar} {count}")

    if len(counter) > max_bars:
        remaining = len(counter) - max_bars
        print(f"{'':>{max_label_len}} ... and {remaining} more")


def main():
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        sys.exit(1)

    success_counts = Counter()
    aborted_counts = Counter()

    checkpoints = data.get('checkpoints', [])
    tx_total = 0

    for checkpoint in checkpoints:
        for tx_data in checkpoint.get('transactions', []):
            tx_total += 1
            status = get_execution_status(tx_data)
            if status is None:
                continue

            transaction = tx_data.get('transaction', {})
            programmable_tx = extract_programmable_transaction(transaction)
            if not programmable_tx:
                continue

            labels = extract_move_call_labels(programmable_tx)
            if status == 'success':
                success_counts.update(labels)
            else:
                aborted_counts.update(labels)

    print(f"Processed {tx_total} transactions from {len(checkpoints)} checkpoints")

    render_histogram(success_counts, "Successful MoveCall Functions")
    render_histogram(aborted_counts, "Aborted MoveCall Functions")


if __name__ == '__main__':
    main()
