# Comprehensive Guide to LoRA (Low-Rank Adaptation)

## 1. Introduction to LoRA

LoRA (Low-Rank Adaptation) is a technique that accelerates the fine-tuning of large models while consuming less memory. It represents weight updates using two smaller matrices through low-rank decomposition, allowing for efficient adaptation to new data while keeping the overall number of changes low.

## 2. Key Advantages

- Drastically reduces the number of trainable parameters
- Allows multiple lightweight and portable models for various tasks
- Compatible with other parameter-efficient methods
- Comparable performance to fully fine-tuned models
- No added inference latency (adapter weights can be merged)

## 3. How LoRA Works

- The original weight matrix remains frozen
- Two smaller update matrices are introduced
- These matrices are trained to adapt to new data
- The original and adapted weights are combined for final results

## 4. Implementation in Transformer Models

- Typically applied to attention blocks only
- The number of trainable parameters depends on:
  - Rank r of the update matrices
  - Shape of the original weight matrix

## 5. Key Parameters and Configuration

Use LoraConfig to define LoRA-specific parameters:

- **r**: Rank of update matrices (lower rank = fewer trainable parameters)
- **target_modules**: Modules to apply LoRA updates (e.g., attention blocks)
- **lora_alpha**: LoRA scaling factor
- **bias**: Controls training of bias parameters ('none', 'all', or 'lora_only')
- **use_rslora**: Enables Rank-Stabilized LoRA (uses lora_alpha/math.sqrt(r) scaling)
- **modules_to_save**: Additional modules to be trained and saved
- **layers_to_transform**: Specific layers for LoRA transformation
- **layers_pattern**: Pattern to match layer names in target_modules
- **rank_pattern**: Mapping of layer names to ranks (overrides default r)
- **alpha_pattern**: Mapping of layer names to alphas (overrides default lora_alpha)

## 6. Initialization Options

Control LoRA weight initialization using init_lora_weights:

- **Default**: Kaiming-uniform for weight A, zeros for weight B
- **"gaussian"**: Gaussian distribution for weight A, zeros for weight B
- **"loftq"**: LoftQ initialization for quantized models
- **False**: No identity transform (for debugging only)

## 7. Best Practices and Recommendations

### Dataset Size Considerations

- Larger datasets: Lower learning rate to avoid local minima
- Smaller datasets: More epochs, larger batch size to limit overfitting
- Minimum recommended sizes:
  - DPO/ORPO: At least 1000 samples
  - SFT (Supervised Fine-Tuning): Over 5000 samples

### Hyperparameters

- Sample packing: Increase learning rate to maintain effective learning rate per sample
- For small datasets (< 5000 for SFT):
  - Use large batch size, high epoch count, and medium learning rate
  - Example: 500 samples, batch size 64, 30 epochs, lr 0.00005 (adjust lr based on model)

### Other Tips

- Increase rank as much as possible (aim for 64 or 128)
- Use consistent prompt formats for training and inference
- Consider using GaLore optimizer and enabling rslora for high-rank effectiveness
- Stay updated with new techniques: Unsloth, ORPO, GaLore, rslora, loftq+

## 8. Utility Functions

- **merge_adapter()**: Merge LoRA layers into the base model
- **unmerge_adapter()**: Unmerge LoRA layers from the base model
- **unload()**: Revert to the original base model state
- **delete_adapter()**: Remove an existing adapter
- **add_weighted_adapter()**: Combine multiple LoRAs into a new adapter

## 9. Merging LoRA Weights

Use merge_and_unload() to combine adapter weights with the base model, creating a standalone model without latency issues during inference.

> Remember that while hyperparameters are important, the quality and appropriateness of your dataset often have a more significant impact on the fine-tuning results.
