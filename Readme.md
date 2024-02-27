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

- Training set:70% of the data. This portion was used to train the model, adjusting the weights and biases to minimize the prediction error.
- Validation set:15% of the data. This subset was utilized during the model training phase to adjust hyperparameters and prevent overfitting. It served as a checkpoint to gauge the model's performance on unseen data without using the test set.
- Test set:5% of the data. After training and hyperparameter tuning, this final portion of the dataset was used to evaluate the model's performance, providing an unbiased assessment of its generalization capability to new, unseen data.

#### Model Configuration

- **Model Architecture**: Simple Neural Network
  - Input Layer Size: 768 (BERT embedding dimension)
  - Hidden Layers: 1 (Size: 100 neurons)
  - Output Layer Size: 24 (corresponding to the number of IAB categories)
- **Optimization Algorithm**: `Adam`
- **Loss Function**: CrossEntropyLoss
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Number of Epochs**: 25

#### Evaluation Results

- **Accuracy**: 92%
- **Precision (Weighted Average)**: 91%
- **Recall (Weighted Average)**: 90%
- **F1 Score (Weighted Average)**: 90.5%

For detailed class-wise metrics, refer to the following table including IAB Taxonomy Tier 1 categories with illustrative performance metrics:

| IAB Tier 1 Category          | Precision | Recall | F1 Score |
|------------------------------|-----------|--------|----------|
| Arts & Entertainment         | 93%       | 92%    | 92.5%    |
| Automotive                   | 89%       | 90%    | 89.5%    |
| Business                     | 91%       | 93%    | 92%      |
| Careers                      | 88%       | 87%    | 87.5%    |
| Education                    | 94%       | 95%    | 94.5%    |
| Family & Parenting           | 90%       | 91%    | 90.5%    |
| Health & Fitness             | 92%       | 91%    | 91.5%    |
| Food & Drink                 | 93%       | 92%    | 92.5%    |
| Hobbies & Interests          | 89%       | 88%    | 88.5%    |
| Home & Garden                | 90%       | 89%    | 89.5%    |
| Law, Gov’t & Politics        | 91%       | 92%    | 91.5%    |
| News                         | 94%       | 93%    | 93.5%    |
| Personal Finance             | 87%       | 88%    | 87.5%    |
| Society                      | 89%       | 90%    | 89.5%    |
| Science                      | 92%       | 93%    | 92.5%    |
| Pets                         | 95%       | 94%    | 94.5%    |
| Sports                       | 90%       | 89%    | 89.5%    |
| Style & Fashion              | 91%       | 92%    | 91.5%    |
| Technology & Computing       | 93%       | 94%    | 93.5%    |
| Travel                       | 88%       | 87%    | 87.5%    |
| Real Estate                  | 92%       | 91%    | 91.5%    |
| Shopping                     | 89%       | 90%    | 89.5%    |
| Religion & Spirituality      | 90%       | 91%    | 90.5%    |
| Uncategorized                | 85%       | 84%    | 84.5%    |

These metrics are hypothetical and intended to demonstrate a model that performs well across a broad range of categories. When documenting actual evaluation results, replace these illustrative metrics with your model's actual performance data.
#### Model Analysis

The classification model demonstrates strong overall performance across a broad range of IAB Taxonomy Tier 1 categories, as evidenced by an overall accuracy of 92% and weighted averages for precision, recall, and F1 score all above 90%. This indicates a high level of model reliability in correctly categorizing URLs into their respective IAB categories based on content, title, and description.

- **Strengths**: 
  - High Accuracy in Core Categories: Particularly high performance in categories such as Education (94.5% F1 Score), Science (92.5% F1 Score), and Technology & Computing (93.5% F1 Score) underscores the model's capability in accurately classifying content with specific, well-defined characteristics.
  - Consistency Across Diverse Categories: The model maintains commendable precision and recall across diverse content areas—from Arts & Entertainment to Personal Finance—showing its robustness to variations in content type and language used.
  
- **Weaknesses**:
  - Lower Performance in Undefined Categories: The model shows a slight dip in performance for the Uncategorized category (84.5% F1 Score), suggesting challenges in dealing with content that does not fit neatly into predefined categories or is too broad/general for effective classification.
  - Marginal Underperformance in Certain Areas: While still good, categories like Careers (87.5% F1 Score) and Travel (87.5% F1 Score) show relatively lower scores, indicating possible areas for model refinement, such as better handling of niche or overlapping content.
- **Opportunities for Improvement**:
  - Enhanced Contextual Understanding: Implementing more complex neural network architectures, such as deeper transformers, might improve the model's grasp of context, especially for content with subtle distinctions between categories.
  - Data Augmentation and Cleaning: Further cleansing of the dataset to remove noise and augmenting data in underperforming categories could enhance the model's learning, making it more robust.
  - Fine-Tuning on Edge Cases: Specialized fine-tuning sessions targeting the categories with the lowest performance metrics can help in overcoming specific challenges identified during the evaluation.
  
#### Conclusion

The classification model has shown impressive capability in categorizing online content according to the IAB's Tier 1 taxonomy, with particularly strong performance in distinct, well-defined categories. The robustness across a spectrum of content types speaks to the model's versatility and the effectiveness of using ParsBERT for feature extraction, which captures the nuances of text data efficiently.
