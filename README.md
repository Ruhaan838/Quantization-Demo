# Quantization Techniques for Machine Learning Models

This repository demonstrates two common approaches to quantization in machine learning: **Post-Training Quantization (PTQ)** and **Quantization-Aware Training (QAT)**. It contains Python scripts and visual aids to help you understand and implement these techniques.

## Directory Structure

```
.
├── images
│   ├── post_training_quantization.png       # Visualization of Post-Training Quantization process
│   └── quantization_aware_training.png      # Visualization of Quantization-Aware Training process
├── .gitignore                               # Files and directories to ignore in version control
├── LICENSE                                  # License for the project
├── post_training_quantization.py           # Script demonstrating Post-Training Quantization
└── quantization_aware_training.py          # Script demonstrating Quantization-Aware Training
```

## Contents

### 1. Python Scripts
- **`post_training_quantization.py`**: 
  - Implements Post-Training Quantization, a method to reduce model size and improve inference speed by converting a pre-trained model's weights to lower precision without retraining.
  
- **`quantization_aware_training.py`**: 
  - Demonstrates Quantization-Aware Training, where the model is trained with quantization in mind, achieving better accuracy for lower-precision models.

### 2. Images
- **`post_training_quantization.png`**: 
  - Diagram explaining the process of Post-Training Quantization.

- **`quantization_aware_training.png`**: 
  - Diagram illustrating the workflow of Quantization-Aware Training.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/quantization-techniques.git
   cd quantization-techniques
   ```

2. Run the scripts:
   - For Post-Training Quantization:
     ```bash
     python post_training_quantization.py
     ```
   - For Quantization-Aware Training:
     ```bash
     python quantization_aware_training.py
     ```

## Requirements

- Python 3.8+
- TensorFlow or PyTorch (depending on the implementation)

## License

This project is licensed under the [MIT License](LICENSE).

## References

- [TensorFlow Quantization Documentation](https://www.tensorflow.org/model_optimization/guide/quantization)
- [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html)

## Contributions

Contributions are welcome! Feel free to open an issue or submit a pull request.
