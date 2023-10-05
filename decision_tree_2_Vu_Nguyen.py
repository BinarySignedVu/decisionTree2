#-------------------------------------------------------------------------
# AUTHOR: Vu Nguyen
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# importing required libraries
from sklearn import tree
import csv

DATA_FILES = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']
TRANSFORMATION_MAP = {'Young': 1, 'Myope': 1, 'Normal': 1, 'Yes': 1,'Prepresbyopic': 2, 'Hypermetrope': 2, 'Reduced': 2, 'No': 2,'Presbyopic': 3}

def read_csv_data(filename):
    """Load data from a given CSV file."""
    data = []
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for idx, row in enumerate(reader):
                if idx > 0:  # Exclude header
                    data.append(row)
    except FileNotFoundError:
        print(f"File {filename} not found.")
    return data

def transform_data(input_data):
    """Transform categorical data to numerical based on the provided map."""
    features = []
    labels = []
    for row in input_data:
        transformed_row = [TRANSFORMATION_MAP[item] for item in row[:-1]]
        features.append(transformed_row)
        labels.append(TRANSFORMATION_MAP[row[-1]])
    return features, labels

def compute_accuracy(model, test_features, test_labels):
    """Calculate accuracy of the given model on the test data."""
    correct_predictions = sum([1 for predicted, true in zip(model.predict(test_features), test_labels) if predicted == true])
    return correct_predictions / len(test_features)

for data_file in DATA_FILES:
    training_data = read_csv_data(data_file)
    X_train, Y_train = transform_data(training_data)

    test_data = read_csv_data('contact_lens_test.csv')
    X_test, Y_test = transform_data(test_data)

    total_accuracy = 0  # To calculate average accuracy
    for iteration in range(10):
        # Train the model
        model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
        model.fit(X_train, Y_train)

        # Compute accuracy and add it to the total
        total_accuracy += compute_accuracy(model, X_test, Y_test)

    average_accuracy = total_accuracy / 10  # Calculate average accuracy over 10 runs
    print(f"Average accuracy when training with {data_file}: {average_accuracy:.4f}")
