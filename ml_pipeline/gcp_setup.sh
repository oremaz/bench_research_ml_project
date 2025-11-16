#!/bin/bash
# GCP Remote GPU Setup Script
# This script helps you set up a GCP VM with GPU for remote training

set -e

# Configuration
VM_NAME="${VM_NAME:-gpu-training-vm}"
ZONE="${ZONE:-us-central1-a}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-16}"  # 16 vCPUs for optimal data loading with batch_size=64
GPU_TYPE="${GPU_TYPE:-nvidia-tesla-t4}"
GPU_COUNT="${GPU_COUNT:-1}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-200GB}"

echo "🚀 GCP Remote GPU Setup"
echo "======================="
echo "VM Name: $VM_NAME"
echo "Zone: $ZONE"
echo "Machine Type: $MACHINE_TYPE"
echo "GPU: $GPU_TYPE x $GPU_COUNT"
echo "Boot Disk: $BOOT_DISK_SIZE"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install it first:"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if VM already exists
if gcloud compute instances describe "$VM_NAME" --zone="$ZONE" &> /dev/null; then
    echo "⚠️  VM '$VM_NAME' already exists in zone '$ZONE'"
    read -p "Do you want to delete and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Deleting existing VM..."
        gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet
    else
        echo "ℹ️  Using existing VM. Skipping creation."
        exit 0
    fi
fi

# Create the VM
echo "📦 Creating GCP VM with GPU..."
gcloud compute instances create "$VM_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --metadata="install-nvidia-driver=True"

echo "⏳ Waiting for VM to be ready..."
sleep 30

# SSH into the VM and set up the environment
echo "🔧 Setting up the environment on the VM..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
    set -e
    echo '📦 Installing dependencies...'
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install timm kaggle rich jupyter notebook
    
    echo '📂 Cloning repository...'
    if [ ! -d 'bench_research_ml_project' ]; then
        git clone https://github.com/oremaz/bench_research_ml_project
    else
        echo 'Repository already exists, pulling latest changes...'
        cd bench_research_ml_project && git pull && cd ..
    fi
    
    cd bench_research_ml_project
    pip install -r requirements.txt
    pip install numpy==1.24.3 scipy==1.10.1 scikit-learn==1.3.0
    
    echo '✅ Setup complete!'
    echo ''
    echo '🎮 GPU Information:'
    python3 -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}\")'
"

echo ""
echo "✅ GCP VM setup complete!"
echo ""
echo "📝 Next steps:"
echo ""
echo "1. Set up environment variables locally:"
echo "   export KAGGLE_USERNAME='your_username'"
echo "   export KAGGLE_KEY='your_api_key'"
echo "   export GCP_REMOTE_GPU='1'"
echo ""
echo "2. Connect to the VM using VS Code Remote SSH:"
echo "   • Install 'Remote - SSH' extension in VS Code"
echo "   • Run: gcloud compute config-ssh"
echo "   • In VS Code: F1 → 'Remote-SSH: Connect to Host' → select '$VM_NAME.$ZONE'"
echo "   • Open the notebook: bench_research_ml_project/ml_pipeline/bench-imai-artifact.ipynb"
echo ""
echo "3. Or use SSH tunnel for Jupyter:"
echo "   gcloud compute ssh $VM_NAME --zone=$ZONE -- -L 8888:localhost:8888"
echo "   # Then on the VM, run: jupyter notebook --no-browser --port=8888"
echo "   # Access at: http://localhost:8888"
echo ""
echo "4. When done, stop the VM to save costs:"
echo "   gcloud compute instances stop $VM_NAME --zone=$ZONE"
echo ""
echo "5. To delete the VM completely:"
echo "   gcloud compute instances delete $VM_NAME --zone=$ZONE"
echo ""
