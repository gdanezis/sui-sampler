"""
Analyze Sui blockchain checkpoints JSON data.

This script processes checkpoint data from stdin and analyzes:
- Shared object usage patterns
- Gas consumption per sender and package
- Free tier transaction classification
- Move call statistics

For each checkpoint, it extracts ProgrammableTransaction data and provides
comprehensive statistics on transaction patterns and costs.
"""

import json
import sys
from collections import defaultdict


class TransactionData:
    """Holds data for a single transaction.
    
    Attributes:
        shared_objects: Set of shared object IDs that this transaction accesses
        sender: Address of the transaction sender
        gas_used: Total gas consumed by this transaction
        packages_used: Set of package IDs used in MoveCall commands
        json: Raw JSON data of the transaction for reference
    """
    
    def __init__(self):
        self.shared_objects = set()
        self.sender = None
        self.gas_used = 0
        self.packages_used = set()
        self.json = None


def _extract_programmable_transaction(transaction):
    """Extract ProgrammableTransaction from transaction data.
    
    Args:
        transaction: Transaction JSON data
        
    Returns:
        ProgrammableTransaction data if found, None otherwise
    """
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


class CheckpointStats:
    """Maintains statistics about transactions processed in a checkpoint.
    
    Tracks shared object access patterns, gas usage by sender and package,
    and provides free tier analysis capabilities.
    """

    def __init__(self):
        # Maps shared object ID to number of transactions that accessed it
        self.shared_object_access_per_transaction = defaultdict(int)

        # Gas usage and transaction counts per sender
        self.gas_used_per_sender = defaultdict(int)
        self.transaction_count_per_sender = defaultdict(int)

        # Gas usage and transaction counts per package
        self.gas_used_per_package = defaultdict(int)
        self.transaction_count_per_package = defaultdict(int)

        # Total transaction count and list of processed transactions
        self.transaction_count = 0
        self.transactions = []  # List of TransactionData objects


    def process_transaction(self, transaction_data):
        """Process a single transaction and update statistics.
        
        Args:
            transaction_data: Raw transaction JSON data
        """
        # Extract transaction components
        transaction = transaction_data.get('transaction', {})
        effects = transaction_data.get('effects', {})
        
        # Get sender from transaction data
        sender = self._extract_sender(transaction)
        if not sender:
            return
        
        # Check if this is a ProgrammableTransaction
        programmable_tx = _extract_programmable_transaction(transaction)
        if not programmable_tx:
            return  # Skip non-ProgrammableTransaction types
            
        # Calculate gas used for this transaction
        gas_used = self._calculate_gas_used(effects)
        
        # Create TransactionData object
        tx_data = TransactionData()
        tx_data.sender = sender
        tx_data.gas_used = gas_used
        tx_data.json = transaction_data

        # Extract shared objects from ProgrammableTransaction inputs (only mutable SharedObjects)
        if 'inputs' in programmable_tx:
            for input_obj in programmable_tx['inputs']:
                if 'Object' in input_obj and 'SharedObject' in input_obj['Object']:
                    shared_obj = input_obj['Object']['SharedObject']
                    if shared_obj.get('mutable', False):  # Only mutable shared objects
                        tx_data.shared_objects.add(shared_obj['id'])
        
        # Extract packages from MoveCall commands
        if 'commands' in programmable_tx:
            for command in programmable_tx['commands']:
                if 'MoveCall' in command:
                    move_call = command['MoveCall']
                    if 'package' in move_call:
                        package_id = self._normalize_package_id(move_call['package'])
                        tx_data.packages_used.add(package_id)
        
        # Store the TransactionData and update counters
        self.transactions.append(tx_data)
        self.transaction_count += 1
        
        # Update gas usage and transaction counts
        self.gas_used_per_sender[sender] += gas_used
        self.transaction_count_per_sender[sender] += 1
        
        # Update shared object access count
        for shared_obj_id in tx_data.shared_objects:
            self.shared_object_access_per_transaction[shared_obj_id] += 1
        
        # Update package usage statistics
        for package_id in tx_data.packages_used:
            self.gas_used_per_package[package_id] += gas_used
            self.transaction_count_per_package[package_id] += 1

    def _extract_sender(self, transaction):
        """Extract sender address from transaction data.
        
        Args:
            transaction: Transaction JSON data
            
        Returns:
            Sender address string if found, None otherwise
        """
        if 'data' in transaction and isinstance(transaction['data'], list):
            for data_item in transaction['data']:
                if 'intent_message' in data_item:
                    intent_message = data_item['intent_message']
                    if 'value' in intent_message:
                        value = intent_message['value']
                        if 'V1' in value:
                            return value['V1'].get('sender')
        return None

    def _calculate_gas_used(self, effects):
        """Calculate total gas used from effects.
        
        Args:
            effects: Transaction effects JSON data
            
        Returns:
            Total gas used (computation + storage costs)
        """
        if 'V2' in effects:
            gas_used = effects['V2'].get('gas_used', {})
            computation = int(gas_used.get('computationCost', '0'))
            storage = int(gas_used.get('storageCost', '0'))
            # NOTE: Intentionally ignore rebate and non_refundable for this calculation
            return computation + storage
        return 0

    def _normalize_package_id(self, package_id):
        """Normalize package ID format to ensure consistency.
        
        Args:
            package_id: Package ID string
            
        Returns:
            Normalized package ID with 0x prefix
        """
        if package_id and not package_id.startswith('0x'):
            return f'0x{package_id}'
        return package_id

    def print_stats(self, checkpoint_sequence_number):
        """Print statistics for this checkpoint.
        
        Args:
            checkpoint_sequence_number: Sequence number of the checkpoint
        """
        print(f"Checkpoint {checkpoint_sequence_number}:")
        print(f"  Transaction Count: {self.transaction_count}")
        print(f"  Shared Object Access Count: {len(self.shared_object_access_per_transaction)}")
        for obj_id, count in sorted(self.shared_object_access_per_transaction.items()):
            print(f"    {obj_id}: {count} transactions")
        
        print(f"  Gas Usage by Sender: {len(self.gas_used_per_sender)} senders")
        for sender, gas in sorted(self.gas_used_per_sender.items()):
            tx_count = self.transaction_count_per_sender.get(sender, 0)
            print(f"    {sender}: {gas} gas units ({tx_count} transactions)")
        
        print(f"  Gas Usage by Package: {len(self.gas_used_per_package)} packages")
        for package, gas in sorted(self.gas_used_per_package.items()):
            tx_count = self.transaction_count_per_package.get(package, 0)
            print(f"    {package}: {gas} gas units ({tx_count} transactions)")
        print()


    def free_tier_estimation(self, verbose=True):
        """Estimate transactions that qualify for free tier pricing.

        A transaction qualifies for free tier if:
        1. It uses no shared objects, OR
        2. All shared objects it uses are accessed only once in the checkpoint, AND
        3. The sender has only one transaction in the checkpoint (if using shared objects)

        Args:
            verbose: Whether to print detailed transaction information
            
        Returns:
            Tuple of (free_tier_count, free_tier_gas, non_free_tier_count, 
                     non_free_tier_gas, counterfactual_non_free_tier_gas,
                     free_tier_transactions, non_free_tier_transactions)
        """
        if verbose:
            print("Free Tier Estimation:")
        
        # Initialize counters
        free_tier_gas = 0
        non_free_tier_gas = 0
        free_tier_count = 0
        non_free_tier_count = 0
        counterfactual_non_free_tier_gas = 0

        free_tier_transactions = []
        non_free_tier_transactions = []
        
        for i, tx_data in enumerate(self.transactions):
            # Determine if transaction qualifies for free tier
            qualifies_for_free_tier = self._qualifies_for_free_tier(tx_data)
            
            # Print transaction details if verbose
            if verbose:
                shared_objects_str = ", ".join(tx_data.shared_objects) if tx_data.shared_objects else "none"
                tier_status = "FREE TIER" if qualifies_for_free_tier else "NON-FREE TIER"
                print(f"  Transaction {i+1}: {tx_data.gas_used} gas units, "
                      f"shared objects: [{shared_objects_str}] -> {tier_status}")
            
            # Calculate counterfactual gas (with penalty for multiple shared object usage)
            multiple_use_shared_objects = [
                obj_id for obj_id in tx_data.shared_objects 
                if self.shared_object_access_per_transaction.get(obj_id, 0) > 1
            ]
            penalty_factor = len(multiple_use_shared_objects) ** 2

            # Update counters
            if qualifies_for_free_tier:
                free_tier_gas += tx_data.gas_used
                free_tier_count += 1
                free_tier_transactions.append(tx_data)
            else:
                non_free_tier_gas += tx_data.gas_used
                counterfactual_non_free_tier_gas += tx_data.gas_used * penalty_factor
                non_free_tier_count += 1
                non_free_tier_transactions.append(tx_data)
        
        # Print summary if verbose
        if verbose:
            self._print_free_tier_summary(free_tier_count, free_tier_gas, 
                                        non_free_tier_count, non_free_tier_gas)

        return (free_tier_count, free_tier_gas, non_free_tier_count, non_free_tier_gas, 
                counterfactual_non_free_tier_gas, free_tier_transactions, non_free_tier_transactions)

    def _qualifies_for_free_tier(self, tx_data):
        """Check if a transaction qualifies for free tier pricing.
        
        Args:
            tx_data: TransactionData object
            
        Returns:
            True if transaction qualifies for free tier, False otherwise
        """
        # If transaction uses no shared objects, it qualifies for free tier
        if not tx_data.shared_objects:
            return True
        
        # Check if all shared objects are accessed only once
        for shared_obj_id in tx_data.shared_objects:
            access_count = self.shared_object_access_per_transaction.get(shared_obj_id, 0)
            if access_count > 1:
                return False
        
        # If using shared objects, sender must have only one transaction
        sender_tx_count = self.transaction_count_per_sender.get(tx_data.sender, 0)
        return sender_tx_count == 1

    def _print_free_tier_summary(self, free_tier_count, free_tier_gas, 
                                non_free_tier_count, non_free_tier_gas):
        """Print summary of free tier estimation.
        
        Args:
            free_tier_count: Number of free tier transactions
            free_tier_gas: Total gas used by free tier transactions
            non_free_tier_count: Number of non-free tier transactions
            non_free_tier_gas: Total gas used by non-free tier transactions
        """
        total_gas = free_tier_gas + non_free_tier_gas
        total_count = free_tier_count + non_free_tier_count
        
        print(f"  Summary:")
        print(f"    Free Tier: {free_tier_count} transactions, {free_tier_gas} gas units")
        print(f"    Non-Free Tier: {non_free_tier_count} transactions, {non_free_tier_gas} gas units")
        print(f"    Total: {total_count} transactions, {total_gas} gas units")
        
        if total_gas > 0:
            free_tier_percentage = (free_tier_gas / total_gas) * 100
            print(f"    Free Tier represents {free_tier_percentage:.1f}% of total gas usage")
        print()


def move_call_details(transaction_list, cutoff_percentage=0.01):
    """Analyze MoveCall commands from a list of transactions.
    
    Args:
        transaction_list: List of TransactionData objects
        cutoff_percentage: Minimum percentage threshold for inclusion in results
        
    Returns:
        Dictionary mapping (package_id, module, function) tuples to percentage of total calls
    """
    call_details = defaultdict(int)
    
    for tx in transaction_list:
        if not tx.json:
            continue
            
        transaction = tx.json.get('transaction', {})
        programmable_tx = _extract_programmable_transaction(transaction)
        if not programmable_tx or 'commands' not in programmable_tx:
            continue

        for command in programmable_tx['commands']:
            if 'MoveCall' in command:
                move_call = command['MoveCall']
                package_id = move_call['package']
                module = move_call['module']
                function = move_call['function']
                full_call_name = (package_id, module, function)
                call_details[full_call_name] += 1
    
    # Filter by cutoff percentage and convert to percentages
    total_calls = sum(call_details.values())
    if total_calls == 0:
        return {}
    
    filtered_calls = {
        k: v / total_calls 
        for k, v in call_details.items() 
        if (v / total_calls) > cutoff_percentage
    }
    
    return filtered_calls

def main():
    """Main function to process JSON from stdin."""
    try:
        # Read and parse JSON from stdin
        data = json.load(sys.stdin)
        
        # Process each checkpoint
        checkpoints = data.get('checkpoints', [])
        free_tier_stats = []
        
        for checkpoint in checkpoints:
            # Create stats tracker for this checkpoint
            stats = CheckpointStats()
            
            # Process each transaction in the checkpoint
            transactions = checkpoint.get('transactions', [])
            for transaction in transactions:
                stats.process_transaction(transaction)
            
            # Collect free tier statistics
            free_tier_stats.append(stats.free_tier_estimation(verbose=False))
    
        # Aggregate statistics across all checkpoints
        total_free_tier_count = sum(x[0] for x in free_tier_stats)
        total_free_tier_gas = sum(x[1] for x in free_tier_stats)
        total_non_free_tier_count = sum(x[2] for x in free_tier_stats)
        total_non_free_tier_gas = sum(x[3] for x in free_tier_stats)
        total_counterfactual_gas = sum(x[4] for x in free_tier_stats)

        # Collect all transactions by tier
        all_free_tier_transactions = []
        all_non_free_tier_transactions = []
        for x in free_tier_stats:
            all_free_tier_transactions.extend(x[5])
            all_non_free_tier_transactions.extend(x[6])
        
        # Analyze MoveCall patterns
        free_call_details = move_call_details(all_free_tier_transactions, cutoff_percentage=0.0035)
        non_free_call_details = move_call_details(all_non_free_tier_transactions, cutoff_percentage=0.0035)

        # Print overall summary
        _print_overall_summary(total_free_tier_count, total_free_tier_gas,
                              total_non_free_tier_count, total_non_free_tier_gas,
                              total_counterfactual_gas)

        # Print MoveCall analysis
        _print_move_call_analysis(free_call_details, non_free_call_details)

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing data: {e}", file=sys.stderr)
        sys.exit(1)


def _print_overall_summary(free_count, free_gas, non_free_count, non_free_gas, counterfactual_gas):
    """Print overall statistics summary.
    
    Args:
        free_count: Total free tier transaction count
        free_gas: Total free tier gas usage
        non_free_count: Total non-free tier transaction count  
        non_free_gas: Total non-free tier gas usage
        counterfactual_gas: Counterfactual gas with penalties
    """
    total_gas = free_gas + non_free_gas
    total_count = free_count + non_free_count
    
    print("Overall Free Tier Estimation Across All Checkpoints:")
    print(f"    Free Tier: {free_count} transactions, {free_gas} gas units")
    print(f"    Non-Free Tier: {non_free_count} transactions, {non_free_gas} gas units")
    print(f"    Total: {total_count} transactions, {total_gas} gas units")
    
    if total_gas > 0:
        penalty_multiplier = counterfactual_gas / total_gas
        free_tier_percentage = (free_gas / total_gas) * 100
        print(f"    Counterfactual Non-Free Tier Gas (with multiple shared object usage penalty): "
              f"{counterfactual_gas} gas units - ie {penalty_multiplier:.1f}x")
        print(f"    Free Tier represents {free_tier_percentage:.1f}% of total gas usage")
    print()


def _print_move_call_analysis(free_call_details, non_free_call_details):
    """Print MoveCall pattern analysis.
    
    Args:
        free_call_details: Dictionary of free tier call patterns
        non_free_call_details: Dictionary of non-free tier call patterns
    """
    print("Move Call Details for Free Tier Transactions (cutoff >0.1%):")
    for (package_id, module, function), percentage in sorted(free_call_details.items(), 
                                                            key=lambda x: x[1], reverse=True):
        print(f"    {percentage:.2%} {package_id}::{module}::{function}")
    print()

    print("Move Call Details for Non-Free Tier Transactions (cutoff >0.1%):")
    for (package_id, module, function), percentage in sorted(non_free_call_details.items(), 
                                                            key=lambda x: x[1], reverse=True):
        print(f"    {percentage:.2%} {package_id}::{module}::{function}")
    print()


if __name__ == '__main__':
    main()

