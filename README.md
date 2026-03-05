# MNIST-NN

Simple MNIST neural network project with:
- Training script for an MLP classifier.
- Forward-pass visualizer showing node and edge activity layer by layer.
- Input image export for the same sample used in visualization.

## Project Structure

- `train.py`: train and save model checkpoint.
- `visualize_forward_pass.py`: create input image + animated NN run.
- `mnist_nn/model.py`: MLP model and activation trace helper.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

```bash
python train.py --epochs 8 --batch-size 128 --num-workers 0 --checkpoint checkpoints/mnist_mlp.pt
```

Quick smoke run:

```bash
python train.py --epochs 1 --num-workers 0 --max-train-samples 5000 --max-test-samples 1000
```

## Visualize Forward Pass (Nodes + Lines)

```bash
python visualize_forward_pass.py \
  --checkpoint checkpoints/mnist_mlp.pt \
  --split test \
  --index 3 \
  --output-dir outputs
```

Generated files:
- `outputs/mnist_input.png`: the exact MNIST input image used.
- `outputs/mnist_forward_pass.gif`: layer-by-layer animation from input to output.
- `outputs/mnist_forward_final.png`: final stage static visualization.

## Notes

- Red edges represent positive contribution.
- Blue edges represent negative contribution.
- Node color intensity reflects activation/probability strength.
- Input layer is visualized with selected high-intensity pixels to keep the graph readable.
