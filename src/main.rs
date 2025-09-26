use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use clap::Parser;
use futures::stream::{FuturesUnordered, StreamExt, TryStreamExt};
use log::{error, info, warn};
use object_store::{http::{HttpBuilder, HttpStore}, path::Path, ObjectStore};
use rand::{seq::SliceRandom, SeedableRng};
use serde::{Deserialize, Serialize, Serializer};
use sui_sdk::SuiClientBuilder;
use sui_types::full_checkpoint_content::CheckpointData;

#[derive(Parser, Debug)]
#[command(name = "sui-sampler")]
#[command(about = "A CLI tool to randomly sample SUI checkpoints and output their parsed data as JSON")]
#[command(version = "0.1.0")]
struct Args {
    /// Window size - number of checkpoints back from end checkpoint to start sampling from
    #[arg(long, default_value_t = 345000)]
    window: u64,

    /// Ending checkpoint sequence number (inclusive)
    #[arg(long)]
    end_checkpoint: Option<u64>,

    /// Number of checkpoints to randomly sample from the range
    #[arg(long, default_value_t = 100)]
    sample_count: u64,

    /// URL for checkpoint data store
    #[arg(long, default_value = "https://checkpoints.mainnet.sui.io")]
    checkpoints_url: String,

    /// Output format - "json" or "pretty"
    #[arg(long, default_value = "json")]
    output_format: OutputFormat,

    /// Random seed for reproducible sampling
    #[arg(long)]
    seed: Option<u64>,

    /// Number of concurrent downloads
    #[arg(long, default_value_t = 10)]
    concurrent: u64,
}

#[derive(Clone, Debug, clap::ValueEnum)]
enum OutputFormat {
    Json,
    Pretty,
}

#[derive(Serialize, Deserialize, Debug)]
struct OutputMetadata {
    start_checkpoint: u64,
    end_checkpoint: u64,
    sample_count: u64,
    sampled_checkpoints: Vec<u64>,
    seed: Option<u64>,
    timestamp: DateTime<Utc>,
}

// Custom serialization wrapper for CheckpointData that formats byte arrays as hex
#[derive(Serialize, Debug)]
struct CheckpointDataHex(#[serde(serialize_with = "serialize_checkpoint_with_hex")] CheckpointData);

impl From<CheckpointData> for CheckpointDataHex {
    fn from(checkpoint: CheckpointData) -> Self {
        CheckpointDataHex(checkpoint)
    }
}

fn serialize_checkpoint_with_hex<S>(checkpoint: &CheckpointData, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    // Convert to serde_json::Value first, then post-process to convert byte arrays to hex
    let mut value = serde_json::to_value(checkpoint).map_err(serde::ser::Error::custom)?;
    convert_byte_arrays_to_hex(&mut value);
    value.serialize(serializer)
}

fn serialize_checkpoints_with_hex<S>(checkpoints: &Vec<CheckpointData>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let hex_checkpoints: Vec<CheckpointDataHex> = checkpoints.iter().map(|c| CheckpointDataHex(c.clone())).collect();
    hex_checkpoints.serialize(serializer)
}

fn convert_byte_arrays_to_hex(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Array(arr) => {
            // Check if this looks like a byte array (all numbers 0-255)
            if arr.len() > 8 && arr.iter().all(|v| {
                if let serde_json::Value::Number(n) = v {
                    if let Some(i) = n.as_u64() {
                        i <= 255
                    } else {
                        false
                    }
                } else {
                    false
                }
            }) {
                // Convert to hex string
                let bytes: Vec<u8> = arr.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as u8))
                    .collect();
                *value = serde_json::Value::String(format!("0x{}", hex::encode(bytes)));
            } else {
                // Recursively process array elements
                for item in arr {
                    convert_byte_arrays_to_hex(item);
                }
            }
        }
        serde_json::Value::Object(obj) => {
            for (_, v) in obj.iter_mut() {
                convert_byte_arrays_to_hex(v);
            }
        }
        _ => {}
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct SamplerOutput {
    metadata: OutputMetadata,
    #[serde(serialize_with = "serialize_checkpoints_with_hex")]
    checkpoints: Vec<CheckpointData>,
}

async fn get_latest_checkpoint() -> Result<u64> {
    let full_node_url = "https://fullnode.mainnet.sui.io:443";
    let sui_client = SuiClientBuilder::default().build(full_node_url).await?;
    let latest_checkpoint = sui_client
        .read_api()
        .get_latest_checkpoint_sequence_number()
        .await?;
    Ok(latest_checkpoint)
}

async fn validate_args(args: &Args) -> Result<(u64, u64)> {
    // Get default end checkpoint (latest)
    let end_checkpoint = if let Some(end) = args.end_checkpoint {
        end
    } else {
        // Fetch the actual latest checkpoint
        get_latest_checkpoint().await.unwrap_or_else(|e| {
            warn!("Failed to fetch latest checkpoint: {}, using default", e);
            60000000 // Fallback default
        })
    };

    // Calculate start checkpoint using window
    let start_checkpoint = end_checkpoint.saturating_sub(args.window);

    // Validation rules
    if args.window > end_checkpoint {
        eprintln!("Warning: Window size ({}) is larger than end checkpoint ({}), start checkpoint will be 0", args.window, end_checkpoint);
    }

    if args.window == 0 {
        return Err(anyhow!("window must be > 0"));
    }

    let range_size = end_checkpoint - start_checkpoint + 1;
    if args.sample_count > range_size {
        return Err(anyhow!(
            "sample-count ({}) must be <= range size ({})",
            args.sample_count,
            range_size
        ));
    }

    if args.sample_count == 0 {
        return Err(anyhow!("sample-count must be > 0"));
    }

    if args.concurrent == 0 || args.concurrent > 100 {
        return Err(anyhow!("concurrent must be > 0 and <= 100"));
    }

    Ok((start_checkpoint, end_checkpoint))
}

fn generate_random_sample(
    start: u64,
    end: u64,
    count: u64,
    seed: Option<u64>,
) -> Vec<u64> {
    let mut rng = if let Some(seed) = seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_entropy()
    };

    let range: Vec<u64> = (start..=end).collect();
    range
        .choose_multiple(&mut rng, count as usize)
        .cloned()
        .collect()
}

async fn download_checkpoint(
    store: &HttpStore,
    checkpoint_number: u64,
) -> Result<CheckpointData> {
    let path = Path::from(format!("{}.chk", checkpoint_number));
    
    // Retry logic - up to 3 attempts
    let mut attempts = 0;
    loop {
        attempts += 1;
        match store.get(&path).await {
            Ok(response) => {
                match response.into_stream().try_fold(Vec::new(), |mut acc, chunk| async move {
                    acc.extend_from_slice(&chunk);
                    Ok(acc)
                }).await {
                    Ok(bytes) => {
                        match bcs::from_bytes::<(u8, CheckpointData)>(&bytes) {
                            Ok((_, checkpoint)) => {
                                info!("Successfully downloaded checkpoint {}", checkpoint_number);
                                return Ok(checkpoint);
                            }
                            Err(e) => {
                                error!("Failed to parse checkpoint {}: {}", checkpoint_number, e);
                                return Err(anyhow!("Parse error for checkpoint {}: {}", checkpoint_number, e));
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Failed to read response body for checkpoint {} (attempt {}): {}", 
                              checkpoint_number, attempts, e);
                        if attempts >= 3 {
                            return Err(anyhow!("Failed to read response body after {} attempts: {}", attempts, e));
                        }
                        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                    }
                }
            }
            Err(e) => {
                warn!("Failed to download checkpoint {} (attempt {}): {}", 
                      checkpoint_number, attempts, e);
                if attempts >= 3 {
                    return Err(anyhow!("Failed to download after {} attempts: {}", attempts, e));
                }
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            }
        }
    }
}

async fn download_checkpoints_concurrently(
    store: &HttpStore,
    checkpoint_numbers: Vec<u64>,
    concurrency: usize,
) -> Vec<CheckpointData> {
    let mut futures = FuturesUnordered::new();
    let mut results = Vec::new();
    let mut successful_downloads = 0;

    // Start initial batch of downloads
    let mut iter = checkpoint_numbers.into_iter();
    for _ in 0..concurrency.min(iter.len()) {
        if let Some(checkpoint_num) = iter.next() {
            let future = download_checkpoint(store, checkpoint_num);
            futures.push(future);
        }
    }

    // Process results and start new downloads
    while let Some(result) = futures.next().await {
        match result {
            Ok(checkpoint) => {
                successful_downloads += 1;
                results.push(checkpoint);
            }
            Err(e) => {
                eprintln!("Warning: {}", e);
            }
        }

        // Start next download if available
        if let Some(checkpoint_num) = iter.next() {
            let future = download_checkpoint(store, checkpoint_num);
            futures.push(future);
        }
    }

    info!("Successfully downloaded {} checkpoints", successful_downloads);
    results
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    // Validate arguments
    let (start_checkpoint, end_checkpoint) = match validate_args(&args).await {
        Ok(range) => range,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    info!("Sampling {} checkpoints from range [{}, {}]", 
          args.sample_count, start_checkpoint, end_checkpoint);

    // Generate random sample
    let mut sampled_checkpoints = generate_random_sample(
        start_checkpoint,
        end_checkpoint,
        args.sample_count,
        args.seed,
    );
    sampled_checkpoints.sort(); // Sort for consistent output

    info!("Selected checkpoints: {:?}", sampled_checkpoints);

    // Set up HTTP store for downloads
    let store = HttpBuilder::new()
        .with_url(&args.checkpoints_url)
        .build()
        .map_err(|e| anyhow!("Failed to build HTTP store: {}", e))?;

    // Download checkpoints concurrently
    let checkpoints = download_checkpoints_concurrently(
        &store,
        sampled_checkpoints.clone(),
        args.concurrent as usize,
    ).await;

    // Check if we got any successful downloads
    if checkpoints.is_empty() {
        eprintln!("Error: No checkpoints were successfully downloaded");
        std::process::exit(2);
    }

    // Create output structure
    let output = SamplerOutput {
        metadata: OutputMetadata {
            start_checkpoint,
            end_checkpoint,
            sample_count: args.sample_count,
            sampled_checkpoints,
            seed: args.seed,
            timestamp: Utc::now(),
        },
        checkpoints,
    };

    // Output JSON
    let json_output = match args.output_format {
        OutputFormat::Json => serde_json::to_string(&output)?,
        OutputFormat::Pretty => serde_json::to_string_pretty(&output)?,
    };

    println!("{}", json_output);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_validate_args_valid() {
        let args = Args {
            window: 1000,
            end_checkpoint: Some(2000),
            sample_count: 100,
            checkpoints_url: "test".to_string(),
            output_format: OutputFormat::Json,
            seed: None,
            concurrent: 10,
        };
        
        let result = validate_args(&args).await;
        assert!(result.is_ok());
        let (start, end) = result.unwrap();
        assert_eq!(start, 1000);
        assert_eq!(end, 2000);
    }

    #[tokio::test]
    async fn test_validate_args_zero_window() {
        let args = Args {
            window: 0,
            end_checkpoint: Some(1000),
            sample_count: 100,
            checkpoints_url: "test".to_string(),
            output_format: OutputFormat::Json,
            seed: None,
            concurrent: 10,
        };
        
        let result = validate_args(&args).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_random_sample() {
        let sample = generate_random_sample(1, 10, 5, Some(42));
        assert_eq!(sample.len(), 5);
        
        // With same seed, should get same result
        let sample2 = generate_random_sample(1, 10, 5, Some(42));
        assert_eq!(sample, sample2);
    }

    #[test]
    fn test_generate_random_sample_no_duplicates() {
        let sample = generate_random_sample(1, 10, 5, None);
        let mut sorted_sample = sample.clone();
        sorted_sample.sort();
        sorted_sample.dedup();
        assert_eq!(sample.len(), sorted_sample.len()); // No duplicates
    }
}