# sui-sampler

A command-line tool for randomly sampling SUI blockchain checkpoints and outputting their parsed data as JSON. Perfect for blockchain analytics, research, and feeding data to AI/ML models for chain analysis.

## Features

- 🎯 **Window-based sampling**: Specify how far back from the latest checkpoint to sample
- 🔄 **Random sampling**: Get representative data across checkpoint ranges
- 🚀 **Concurrent downloads**: Parallel checkpoint fetching for speed
- 🎲 **Reproducible results**: Use seeds for consistent sampling
- 📊 **Flexible output**: JSON or pretty-printed formats
- 🌐 **Latest checkpoint detection**: Automatically fetches the most recent checkpoint
- 🔧 **Configurable**: Customize sample counts, concurrency, and data sources

## Installation

```bash
# Clone the repository
git clone https://github.com/gdanezis/sui-sampler.git
cd sui-sampler

# Build the project
cargo build --release

# The binary will be available at target/release/sui-sampler
```

## Usage

### Basic Usage Examples

#### 1. Simple sampling (most common use case)
Sample 100 random checkpoints from the last 345,000 checkpoints (last 24h):
```bash
./sui-sampler
```

#### 2. Custom window size
Sample 100 from the last 1 million checkpoints:
```bash
./sui-sampler --window 1000000
```

#### 3. Specific checkpoint range
Sample 50 checkpoints from a specific ending point:
```bash
./sui-sampler --end-checkpoint 60000000 --sample-count 50
```

### Intermediate Examples

#### 4. Recent activity analysis
Focus on very recent checkpoints with pretty output:
```bash
./sui-sampler --window 1000 --sample-count 20 --output-format pretty
```

#### 5. High-throughput sampling
Increase concurrency for faster downloads:
```bash
./sui-sampler --sample-count 500 --concurrent 25
```

### Advanced Examples

#### 6. Historical analysis with custom endpoint
Analyze older data with specific configuration:
```bash
./sui-sampler \
  --window 5000000 \
  --end-checkpoint 50000000 \
  --sample-count 1000 \
  --seed 42 \
  --concurrent 20 \
  --checkpoints-url https://checkpoints.mainnet.sui.io
```

#### 8. Scripted data collection
Collect multiple samples for statistical analysis:
```bash
# Collect 10 different samples with different seeds
for i in {1..10}; do
  ./sui-sampler --seed $i --sample-count 100 > "sample_$i.json"
done
```

#### 9. Pipeline with jq for specific data extraction
Extract checkpoint sequence numbers:
```bash
./sui-sampler --sample-count 2 | jq '.checkpoints[].checkpoint_summary.data.sequence_number'
```

Filter checkpoints with programmable transactions:
```bash
./sui-sampler --sample-count 5 | jq '.checkpoints[] | select(.transactions[].transaction.data[].intent_message.value.V1.kind.ProgrammableTransaction) | .checkpoint_summary.data.sequence_number'
```

Extract transaction data from all checkpoints:
```bash
./sui-sampler --sample-count 10 | jq '.checkpoints[].transactions[]'
```

## Examples

The `examples/` directory contains Python scripts for analyzing sui-sampler output:

### extract_name_frequency.py

Analyzes checkpoint JSON data to extract and report frequency statistics for:
- **MoveCall Functions**: All Move function calls from transactions
- **Input Object Types**: Types of objects used as transaction inputs  
- **Event Types**: Events emitted by transactions
- **Unique Sender Tracking**: Number of unique wallet addresses using each function/type
- **Package Information**: Human-readable names and verticals for packages

#### Usage
```bash
# Analyze current network activity
./target/release/sui-sampler --sample-count 100 | python3 examples/extract_name_frequency.py

# Analyze saved checkpoint data
python3 examples/extract_name_frequency.py < checkpoint_data.json
```

#### Example Output
```
Package: 0x2c8d603bc51326b8c13cef9dd07031a408a48dddb541963357661df5d3204809 - DeepBook (DeFi) (Total: 317 occurrences)
----------------------------------------------------------------------------------------------------------------------
  MoveCall Functions (36 calls):
       8  balance_manager::generate_proof_as_owner (1 unique senders)
       8  pool::borrow_flashloan_base (5 unique senders)
       5  pool::place_limit_order (2 unique senders)

  Input Object Types (114 objects):
      70  pool::Pool (24 unique senders)
      43  balance_manager::BalanceManager (6 unique senders)

  Event Types (167 events):
      89  order_info::OrderInfo (2 unique senders)
      45  order_info::OrderPlaced (2 unique senders)
```

This provides insights into:
- Which protocols are most active (by transaction count)
- User adoption patterns (unique senders per function)
- Ecosystem diversity (DeFi, Gaming, Social, etc.)
- Popular vs niche functionality

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--window` | 345000 | Number of checkpoints back from end to start sampling |
| `--end-checkpoint` | Latest | Ending checkpoint sequence number (inclusive) |
| `--sample-count` | 100 | Number of checkpoints to randomly sample |
| `--checkpoints-url` | SUI mainnet | URL for checkpoint data store |
| `--output-format` | json | Output format: "json" or "pretty" |
| `--seed` | Random | Random seed for reproducible sampling |
| `--concurrent` | 10 | Number of concurrent downloads |

## Use Cases

### 🔬 Research & Analytics
- Analyze transaction patterns across different time periods
- Study validator behavior and checkpoint timing
- Research gas usage trends and fee patterns

### 🤖 AI/ML Training Data
- Generate datasets for blockchain analysis models
- Create training data for transaction classification
- Feed structured data to LLMs for chain insights

### 📊 Network Monitoring
- Sample recent activity for health checks
- Monitor transaction volume and types
- Track network performance metrics

### 🛠️ Development & Testing
- Generate test datasets for dApp development
- Validate parsing logic against real checkpoint data
- Benchmark transaction processing performance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the [Apache License 2.0](LICENSE) - see the LICENSE file for details.

## Support

For issues and questions:
- Create an issue on GitHub
