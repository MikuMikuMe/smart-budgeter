# smart-budgeter

Creating a comprehensive budgeting tool using Python involves several key components including data input, processing, machine learning for categorization, and analytics for financial decisions. Below is a simplified version of such a program with comments and error handling. To keep it manageable, I’ll use a basic machine learning model for transaction categorization, and we’ll leverage libraries such as `pandas`, `scikit-learn`, and `numpy`.

Make sure to have the required packages installed:

```bash
pip install pandas scikit-learn numpy
```

Here’s a basic implementation:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import logging

# Set up logging for debugging and error handling
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Sample dataset for demonstration
# In a real-world application, this data should be loaded from a secure source
data = {
    'Description': ['coffee shop', 'grocery store', 'gas station', 'electric bill'],
    'Amount': [5.75, 45.00, 30.00, 60.00],
    'Category': ['Dining', 'Groceries', 'Transport', 'Utilities']
}
df = pd.DataFrame(data)

try:
    # Preprocessing
    df['Amount'] = df['Amount'].apply(lambda x: np.log(x + 1))  # Normalizing the amount

    # For simplicity, let's use only Description and Amount
    # In real applications, more sophisticated feature engineering would be needed
    feature_cols = ['Amount']
    
    # Convert descriptions to numeric features using a simple technique; more advanced NLP could be considered
    df = pd.get_dummies(df, columns=['Description'])
    X = df.drop('Category', axis=1)
    y = df['Category']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Machine Learning Model: A simple RandomForestClassifier for this example
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    logging.info("Model trained successfully.")

    # Predictions
    y_pred = clf.predict(X_test)

    # Report
    logging.info("Model evaluation:")
    logging.info("\n" + classification_report(y_test, y_pred))

except Exception as e:
    logging.error(f"An error occurred: {e}")

# Function to predict category of a new expense
def predict_category(description, amount):
    try:
        # For the purposes of a simple prediction example, let's log transform the amount directly
        amount = np.log(amount + 1)
        # To simulate the dummy transformation as done before
        transaction = pd.DataFrame([[amount]], columns=feature_cols)
        for desc in df.filter(like='Description').columns:
            transaction[desc] = 1 if desc == f"Description_{description}" else 0

        # Handle new descriptions gracefully
        if transaction.shape[1] != X.shape[1]:
            missing_cols = [col for col in X.columns if col not in transaction.columns]
            for missing in missing_cols:
                transaction[missing] = 0

        result = clf.predict(transaction)
        return result[0]
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return None

# Example usage
if __name__ == "__main__":
    new_description = "coffee shop"
    new_amount = 10.25

    category = predict_category(new_description, new_amount)
    if category:
        logging.info(f'The predicted category for the transaction is: {category}')
    else:
        logging.info("Could not categorize the transaction.")
```

### Key Points:
- **Data Representation**: The dataset is simplified and uses dummy features for descriptions. In a real scenario, you would need a comprehensive dataset and possibly apply more sophisticated NLP techniques for text processing.
- **Machine Learning Model**: A simple RandomForest classifier is used for categorization. Depending on the data's complexity and size, you might consider using different models or even deep learning approaches.
- **Error Handling**: Basic error handling is implemented using try-except blocks and logging for better diagnostics.
- **Dummy Variables**: Used to handle categorical data for transaction descriptions. Note that in large datasets, dimensional reduction techniques might be necessary.
- **Feature Engineering**: The sample code does simple log transformation for normalization. The choice of features and their transformations can greatly affect the performance and should be made considering the specifics of the domain.

This script serves as a foundational point that can be expanded with more sophisticated techniques as needed.