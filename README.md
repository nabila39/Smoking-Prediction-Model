# Predicting Smoker Status Using Health Insurance Data

## Project Overview
In this team project, we build a model to predict whether a person is a smoker or not using data collected from their health insurance records. The goal is to analyze the data, visualize key features, and compare various machine learning algorithms to identify the best model for predicting smoker status.

## Dataset
The dataset for this project can be found at the following link:
[Health Insurance Data](https://www.dropbox.com/scl/fi/8et6xuwh9luvfg03hhji3/Data.csv?rlkey=10s6tu2sgw5z3ft43qk3wey79&dl=0)

### Data Dictionary
- **Age**: The age of the person.
- **Gender**: The gender of the person.
- **BMI**: Body Mass Index.
- **Region**: The geographical region (north or south).
- **No. Children**: Number of children the person has.
- **Insurance Charges**: The amount paid by the person for insurance.
- **Smoker**: Indicates if the person is a smoker (`yes` for smoker, `no` otherwise).

## Objectives
1. **Class Label Distribution**: Show the distribution of the class label (Smoker) and highlight any significant aspects of this distribution.
2. **Density Plot of Age**: Display the density plot for the `Age` attribute.
3. **Density Plot of BMI**: Display the density plot for the `BMI` attribute.
4. **Scatter Plot by Region**: Visualize the scatter plot of the data split based on the `Region` attribute.
5. **Data Split**: Split the dataset into training (80%) and test (20%) sets.

## Machine Learning Models to Compare
1. **K-Nearest Neighbors (KNN)**:
   - Test using three different values of K.
2. **Decision Trees (C4.5)**:
   - Evaluate using decision trees with the C4.5 algorithm.
3. **Naive Bayes (NB)**:
   - Test the model using the Naive Bayes classifier.
4. **Artificial Neural Networks (ANN)**:
   - Use a single hidden layer with a sigmoid activation function and 500 epochs.

## Evaluation Metrics
- Use appropriate performance metrics, including ROC/AUC scores and confusion matrices, to evaluate model performance.
- Compare results in a structured table format and explain why one model outperforms the others.

## Results and Analysis
- **Model Comparison Table**: Summarize the performance of all models in a table, highlighting key metrics such as accuracy, precision, recall, F1 score, and AUC.
- **Model Analysis**: Discuss the strengths and weaknesses of each model and explain why one model performs better than others based on the data.

## Conclusion
- Conclude with insights gained from the models and how the data characteristics (e.g., feature distributions and relationships) impact the prediction of smoker status.

## References
- Dataset: [Health Insurance Data](https://www.dropbox.com/scl/fi/8et6xuwh9luvfg03hhji3/Data.csv?rlkey=10s6tu2sgw5z3ft43qk3wey79&dl=0)
