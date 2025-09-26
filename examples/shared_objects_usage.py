'''
Analyze checkpoints JSON.

- For each checkpoint in the standard input, parse the JSON data and populate the CheckpointStats.
- After processing all checkpoints, print the CheckpointStats for each checkpoint.

'''

import json
import sys


class TransactionData:
    """Holds data for a single transaction."""
    
    def __init__(self):
        self.shared_objects = set()
        self.sender = None
        self.gas_used = 0
        self.packages_used = set()
        self.json = None  # Store the raw JSON of the transaction for reference

def _extract_programmable_transaction(transaction):
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

'''
Maintain statistics about each checkpoint processed.

'''
class CheckpointStats:

    def __init__(self):
        # Maps the ID of a shared object to the number of transactions in the checkpoint that accessed it.
        self.shared_object_access_per_transaction = {}

        # Gas used per sender address -- each transaction contributes only once (since it only has a single sender).
        self.gas_used_per_sender = {}
        self.transaction_count_per_sender = {}

        # Gas used per function package - if a transaction uses a function from a package, 
        # accumulate its gas here keyed per package. Note each transaction should only contribute once per package.
        self.gas_used_per_package = {}
        self.transaction_count_per_package = {}

        # Number of transactions in the checkpoint
        self.transaction_count = 0

        # The list of transactions processed
        self.transactions = [] # List of TransactionData objects


    def process_transaction(self, transaction_data):
        """Process a single transaction and update statistics."""

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
        tx_data.json = transaction_data  # Store the raw JSON of the transaction

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
        
        # Store the TransactionData
        self.transactions.append(tx_data)
        
        # Increment transaction count
        self.transaction_count += 1
        
        # Update gas usage per sender
        if sender not in self.gas_used_per_sender:
            self.gas_used_per_sender[sender] = 0
        self.gas_used_per_sender[sender] += gas_used
        
        # Update transaction count per sender
        if sender not in self.transaction_count_per_sender:
            self.transaction_count_per_sender[sender] = 0
        self.transaction_count_per_sender[sender] += 1
        
        # Update shared object access count
        for shared_obj_id in tx_data.shared_objects:
            if shared_obj_id not in self.shared_object_access_per_transaction:
                self.shared_object_access_per_transaction[shared_obj_id] = 0
            self.shared_object_access_per_transaction[shared_obj_id] += 1
        
        # Update gas usage per package
        for package_id in tx_data.packages_used:
            if package_id not in self.gas_used_per_package:
                self.gas_used_per_package[package_id] = 0
            self.gas_used_per_package[package_id] += gas_used
        
        # Update transaction count per package
        for package_id in tx_data.packages_used:
            if package_id not in self.transaction_count_per_package:
                self.transaction_count_per_package[package_id] = 0
            self.transaction_count_per_package[package_id] += 1

    def _extract_sender(self, transaction):
        """Extract sender address from transaction data."""
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
        """Calculate total gas used from effects."""
        if 'V2' in effects:
            gas_used = effects['V2'].get('gas_used', {})
            computation = int(gas_used.get('computationCost', '0'))
            storage = int(gas_used.get('storageCost', '0'))
            rebate = int(gas_used.get('storageRebate', '0'))
            non_refundable = int(gas_used.get('nonRefundableStorageFee', '0'))
            return computation + storage # NOTE: on purpose ignore rebate and non_refundable
        return 0

    def _normalize_package_id(self, package_id):
        """Normalize package ID format to ensure consistency."""
        if package_id and not package_id.startswith('0x'):
            return f'0x{package_id}'
        return package_id

    def print_stats(self, checkpoint_sequence_number):
        """Print statistics for this checkpoint."""
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
        
        # Add free tier estimation
        print()


    def free_tier_estimation(self, verbose=True):
        """Go through all the transactions, and estimate how many would go in a free tier.

        A transaction goes into a free tier if it only uses shared objects that are accessed once in the checkpoint.
        And only one transaction per sender can qualify for free tier if it uses shared objects.
        For each transaction print the gas used, and whether it qualifies for free tier.
        Then sum up all the gas used by free tier transactions, and all the gas used by non-free tier transactions and print them.

        """
        if verbose:
            print("Free Tier Estimation:")
        
        free_tier_gas = 0
        non_free_tier_gas = 0
        free_tier_count = 0
        non_free_tier_count = 0

        free_tier_transactions = []
        non_free_tier_transactions = []
        
        for i, tx_data in enumerate(self.transactions):
            # Check if transaction qualifies for free tier
            qualifies_for_free_tier = True
            
            # A transaction qualifies for free tier if ALL its shared objects are accessed only once
            for shared_obj_id in tx_data.shared_objects:
                access_count = self.shared_object_access_per_transaction.get(shared_obj_id, 0)
                if access_count > 1:
                    qualifies_for_free_tier = False
                    break
            
            # If the transaction uses any shared object, and the sender has sent more than one transaction, it does not qualify for free tier
            if len(tx_data.shared_objects) > 0:
                sender_tx_count = self.transaction_count_per_sender.get(tx_data.sender, 0)
                if sender_tx_count > 1:
                    qualifies_for_free_tier = False

            # If transaction uses no shared objects, it also qualifies for free tier
            if len(tx_data.shared_objects) == 0:
                qualifies_for_free_tier = True
            
            # Print transaction details
            shared_objects_list = list(tx_data.shared_objects)
            shared_objects_str = ", ".join(shared_objects_list) if shared_objects_list else "none"
            tier_status = "FREE TIER" if qualifies_for_free_tier else "NON-FREE TIER"
            if verbose:
                print(f"  Transaction {i+1}: {tx_data.gas_used} gas units, shared objects: [{shared_objects_str}] -> {tier_status}")
            
            # Update counters
            if qualifies_for_free_tier:
                free_tier_gas += tx_data.gas_used
                free_tier_count += 1
                free_tier_transactions.append(tx_data)
            else:
                non_free_tier_gas += tx_data.gas_used
                non_free_tier_count += 1
                non_free_tier_transactions.append(tx_data)
        
        # Print summary
        total_gas = free_tier_gas + non_free_tier_gas
        if verbose:
            print(f"  Summary:")
            print(f"    Free Tier: {free_tier_count} transactions, {free_tier_gas} gas units")
            print(f"    Non-Free Tier: {non_free_tier_count} transactions, {non_free_tier_gas} gas units")
            print(f"    Total: {free_tier_count + non_free_tier_count} transactions, {total_gas} gas units")
            
            if total_gas > 0:
                free_tier_percentage = (free_tier_gas / total_gas) * 100
                print(f"    Free Tier represents {free_tier_percentage:.1f}% of total gas usage")
            print()

        return (free_tier_count, free_tier_gas, non_free_tier_count, non_free_tier_gas, free_tier_transactions, non_free_tier_transactions)


def move_call_details(transaction_list, cutoff_percentage=0.01):
    """Take a list of TransactionData objects and return a mapping from package ID to list of (function, count) tuples."""
    call_details = {}
    for tx in transaction_list:
        # Go through the JSON, and extract MoveCall commands
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

                if full_call_name not in call_details:
                    call_details[full_call_name] = 0
                call_details[full_call_name] += 1
    
    # Filter the call_details to only include those with count > cutoff_percentage
    total_calls = sum(call_details.values())
    call_details = {k: v for k, v in call_details.items() if (v / total_calls) > cutoff_percentage}

    # Substitute values with fractions
    call_details = {k: v / total_calls for k, v in call_details.items()}

    return call_details

def main():
    """Main function to process JSON from stdin."""
    try:
        # Read JSON from stdin
        data = json.load(sys.stdin)
        
        # Process each checkpoint
        checkpoints = data.get('checkpoints', [])
        free_tier_stats = [] 
        
        for checkpoint in checkpoints:
            # Create stats tracker for this checkpoint
            stats = CheckpointStats()
            
            # Get checkpoint sequence number
            sequence_number = checkpoint.get('checkpoint_summary', {}).get('data', {}).get('sequence_number', 'unknown')
            
            # Process each transaction in the checkpoint
            transactions = checkpoint.get('transactions', [])
            for transaction in transactions:
                stats.process_transaction(transaction)
            
            # Print statistics for this checkpoint
            free_tier_stats += [stats.free_tier_estimation(verbose=False)]
    
        # Compute the overall free tier statistics across all checkpoints
        total_free_tier_count = sum(x[0] for x in free_tier_stats)
        total_free_tier_gas = sum(x[1] for x in free_tier_stats)
        total_non_free_tier_count = sum(x[2] for x in free_tier_stats)
        total_non_free_tier_gas = sum(x[3] for x in free_tier_stats)

        # Make lists of all free tier and non-free tier transactions
        all_free_tier_transactions = []
        all_non_free_tier_transactions = []
        for x in free_tier_stats:
            all_free_tier_transactions += x[4]
            all_non_free_tier_transactions += x[5]
        
        free_call_details = move_call_details(all_free_tier_transactions, cutoff_percentage=0.001)
        non_free_call_details = move_call_details(all_non_free_tier_transactions, cutoff_percentage=0.001)

        print("Overall Free Tier Estimation Across All Checkpoints:")
        print(f"    Free Tier: {total_free_tier_count} transactions, {total_free_tier_gas} gas units")
        print(f"    Non-Free Tier: {total_non_free_tier_count} transactions, {total_non_free_tier_gas} gas units")
        total_gas = total_free_tier_gas + total_non_free_tier_gas
        print(f"    Total: {total_free_tier_count + total_non_free_tier_count} transactions, {total_gas} gas units")
        if total_gas > 0:
            free_tier_percentage = (total_free_tier_gas / total_gas) * 100
            print(f"    Free Tier represents {free_tier_percentage:.1f}% of total gas usage")
        print()

        print("Move Call Details for Free Tier Transactions (cutoff >0.1%):")
        for (package_id, module, function), percentage in sorted(free_call_details.items(), key=lambda x: x[1], reverse=True):
            print(f"    {percentage:.2%} {package_id}::{module}::{function}")
        print()

        print("Move Call Details for Non-Free Tier Transactions (cutoff >0.1%):")
        for (package_id, module, function), percentage in sorted(non_free_call_details.items(), key=lambda x: x[1], reverse=True):
            print(f"    {percentage:.2%} {package_id}::{module}::{function}")
        print()

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing data: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

