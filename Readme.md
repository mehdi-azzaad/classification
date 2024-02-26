# Classification Model Project

This project develops a classification model to categorize URLs based on their content, title, and description into IAB taxonomy categories using BERT embeddings for feature extraction and a simple neural network for classification.

## Project Structure

The project is structured as follows:

- `src/`: Contains the source code for the project.
  - `data_preprocessing.py`: Script for data loading, preprocessing, and feature extraction using BERT.
  - `model_training.py`: Script for defining, training, and evaluating a simple neural network model.

## Installation

To set up the project environment, you will need Python 3.6 or later. It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install pandas torch transformers sklearn
```

## Usage

### Data Preprocessing

Before training the model, preprocess your data using `data_preprocessing.py`. This script reads your CSV file, preprocesses the data, and extracts features using BERT.

```bash
python src/data_preprocessing.py --filepath path/to/your/csv/file.csv --label_column your_label_column_name
```


### Model Training

After preprocessing, train your model using model_training.py. Ensure to adjust the script to load your dataset and extracted features.

```bash
python src/model_training.py
```

### Classification Model Evaluation Report (To be Calculate)

#### Evaluation Metrics

The model's performance has been assessed using the following metrics:

- **Accuracy**: The proportion of correctly predicted instances out of all predictions.
- **Precision**: The proportion of correctly predicted positive observations to the total predicted positives. It is calculated for each label and as a weighted average.
- **Recall**: The proportion of correctly predicted positive observations to all observations in actual class. It is calculated for each label and as a weighted average.
- **F1 Score**: The weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. It is calculated for each label and as a weighted average.

#### Data Split

The dataset was split into the following subsets for training and evaluation:

- Training set: _% of the data
- Validation set: _% of the data (if applicable)
- Test set: _% of the data

#### Model Configuration

- **Model Architecture**: Simple Neural Network
  - Input Layer Size: _
  - Hidden Layers: _ (Size: _)
  - Output Layer Size: _ (corresponding to the number of IAB categories)
- **Optimization Algorithm**: _
- **Loss Function**: CrossEntropyLoss
- **Learning Rate**: _
- **Batch Size**: _
- **Number of Epochs**: _

#### Evaluation Results

- **Accuracy**: _%
- **Precision (Weighted Average)**: _%
- **Recall (Weighted Average)**: _%
- **F1 Score (Weighted Average)**: _%

For detailed class-wise metrics, refer to the following table:

| Label | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| Label 1 | _% | _% | _% |
| Label 2 | _% | _% | _% |
| ... | ... | ... | ... |
| Label N | _% | _% | _% |

#### Model Analysis

- **Strengths**: 
  - [Describe any strengths of the model, such as high accuracy in certain categories, robustness to variations in input, etc.]
- **Weaknesses**:
  - [Describe any weaknesses or areas for improvement, such as poor performance in specific categories, overfitting to the training data, etc.]
- **Opportunities for Improvement**:
  - [Suggest any potential improvements, such as additional feature engineering, more complex model architectures, data augmentation, etc.]

#### Conclusion

[Provide a summary of the evaluation results, highlighting the key findings and any insights gained from the model's performance. Discuss the implications of these results for the project's objectives and future work.]
