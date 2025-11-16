# GCP Remote GPU Setup

Keep editing locally while pushing training to a single GCP GPU VM.

## Launch the VM

```bash
cd ml_pipeline
chmod +x gcp_setup.sh  # Make script executable
./gcp_setup.sh  # provisions n1-standard-16 + T4 in us-central1-a
```

The script configures the firewall, drivers, and startup scripts.  
Customize on demand:

```bash
MACHINE_TYPE=n1-standard-8 ./gcp_setup.sh   # slower/cheaper
GPU_TYPE=nvidia-tesla-v100 ./gcp_setup.sh   # faster/more expensive
ZONE=us-west1-b ./gcp_setup.sh              # different region
```

## Connect & Run

1. **Install VS Code Remote SSH:** Get the "Remote - SSH" extension in VS Code
2. **Configure SSH:** Run `gcloud compute config-ssh` to add VM to `~/.ssh/config`
3. **Connect:** In VS Code, press `F1` → "Remote-SSH: Connect to Host" → select `gpu-training-vm`
4. **Open Notebook:** Navigate to `~/bench_research_ml_project/ml_pipeline/bench-imai-artifact.ipynb`
5. **Set Credentials:** In cell 2, configure your Kaggle API credentials:
   ```python
   os.environ["KAGGLE_USERNAME"] = "your_username"
   os.environ["KAGGLE_KEY"] = "your_api_key"
   ```
6. **Run:** Execute cells sequentially (the notebook handles everything else)

## Cost Snapshot

| Setup | vCPUs | ETA | Cost/hr | Approx. total* |
|-------|-------|-----|---------|----------------|
| **Default** (n1-standard-16 + T4) | 16 | ~2.5h @ batch 64 | ~$1.11 | **$2.8** |
| Lighter (n1-standard-8 + T4) | 8 | ~4h @ batch 32 | ~$0.73 | **$2.9** |

\*Disk costs (~$8/mo for 200GB) apply while the VM exists. Bigger CPUs shorten wall-clock time and usually lower the total spent.

## Control Costs

Stop when idle (preserves disk):

```bash
gcloud compute instances stop gpu-training-vm --zone=us-central1-a
gcloud compute instances start gpu-training-vm --zone=us-central1-a
```

Delete when finished:

```bash
gcloud compute instances delete gpu-training-vm --zone=us-central1-a
```

That’s it—`gcp_setup.sh` holds the rest of the knobs should you need them.
