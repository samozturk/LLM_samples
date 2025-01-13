## LoRA
The LoRALinear class:

#### Implements the core LoRA mechanism
- Uses low-rank matrices A and B to approximate weight updates
- Properly scales the LoRA contribution using alpha/rank
- Initializes B to zero so LoRA starts as identity mapping


#### The LoRABERTClassifier class:

- Applies LoRA specifically to attention query and value projections
- Keeps key projections and other layers frozen
- Uses a simple classification head for the final output


#### Training optimizations:

- Higher learning rate (1e-3) since we're training fewer parameters
- Cosine learning rate scheduler with warm restarts
- Gradient clipping for stability
- Monitoring of LoRA update magnitudes


#### Memory efficiency:

- Saves only LoRA weights instead of full model
- Custom save/load functions for LoRA parameters
- Significantly smaller storage requirements



##### Key advantages of this LoRA implementation:

Memory Efficiency: Instead of storing full fine-tuned models, we only store small rank matrices
Training Efficiency: Fewer parameters to train means faster training and less memory usage
Modularity: LoRA weights can be easily swapped or combined
