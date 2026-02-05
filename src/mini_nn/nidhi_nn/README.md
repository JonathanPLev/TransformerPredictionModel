# Nidhi mini-nn 
Predicts NBA players beating projected line (over/under) using a neural network

## Performance Results
### Training Configuration
- Epochs: 100
- Batch Size: 32
- Learning Rate: 0.001

### Results
- Final Validation Accuracy: 52.43%
- Training Samples: 11190 
- Validation Samples: 2798

### Classification Report
             precision    recall  f1-score   support

       Under       0.53      0.54      0.54      1426
        Over       0.52      0.51      0.51      1372

    accuracy                           0.52      2798
   macro avg       0.52      0.52      0.52      2798
weighted avg       0.52      0.52      0.52      2798

### Confusion Matric: 
[[768 658]
 [673 699]]

## Outputs
- Model: `models/mini_nn/nidhi_mini_nn.pt`
- Training Plot: `models/mini_nn/training_history.png`

## Details
- Framework: PyTorch
- Data Preprocessing: StandardScaler normalization
- Regularization: Dropout (30%) + Batch Normalization
- Data Split: 80% train / 20% validation