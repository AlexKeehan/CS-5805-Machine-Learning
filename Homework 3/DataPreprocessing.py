import matplotlib.pyplot as plt
import pandas as pd

# Create class to do all necessary pre-processing
class DataPreProcessing:
    # Constructor
    def __init__(self):
        self.self = self

    # Function to normalize dataset
    def normalized(self, dataset):
        # Normalize
        for column in dataset.columns:
            dataset[column] = (dataset[column] - dataset[column].min()) / (dataset[column].max() - dataset[column].min())

        return dataset

    # Function to standardize dataset
    def standardized(self, dataset):
        # Standardize
        for column in dataset.columns:
            dataset[column] = (dataset[column] - dataset[column].mean()) / dataset[column].std()

        return dataset

    # Function to perform IQR transformations to dataset
    def iqr(self, dataset):
        # Iterate over each column
        for column in dataset.columns:
            # Calculate Q1 (25th percentile) and Q3 (75th percentile)
            Q1 = dataset[column].quantile(0.25)
            Q3 = dataset[column].quantile(0.75)
            IQR = Q3 - Q1

            # Apply IQR transformation
            dataset[column] = (dataset[column] - dataset[column].median()) / IQR

        return dataset

    # Graph dataset with no changes
    def show_original(self, dataset):
        dataset.plot(figsize=(12,8), use_index=True)
        self.set_labels(dataset)
        plt.title('Original AAPL data set')
        plt.show()

    # Graph normalized dataset
    def show_normalized(self, dataset):
        norm_dataset = self.normalized(dataset)
        norm_dataset.plot(figsize=(12,8), use_index=True)
        self.set_labels(norm_dataset)
        plt.title("Normalized AAPL data set")
        plt.show()

    # Graph standardized dataset
    def show_standardized(self, dataset):
        stand_dataset = self.standardized(dataset)
        stand_dataset.plot(figsize=(12,8), use_index=True)
        self.set_labels(stand_dataset)
        plt.title("Standarized AAPL data set")
        plt.show()

    # Graph IQR transformed dataset
    def show_iqr(self, dataset):
        iqr_dataset = self.iqr(dataset)
        iqr_dataset.plot(figsize=(12,8), use_index=True)
        self.set_labels(iqr_dataset)
        plt.title("IQR transformation AAPL data set")
        plt.show()

    # Split off logic for labels & ticks to avoid repeated code
    def set_labels(self, dataset):
        # Labels
        plt.xlabel('Date')
        plt.legend()

        # Use date_range to separate the date ticks out & specify how many I want
        num_ticks = 6

        tick_dates = pd.date_range(start=dataset.index.min(),
                                   end=dataset.index.max(),
                                   periods=num_ticks)

        # Set x-ticks
        plt.xticks(tick_dates, rotation=0, ha='center')
        plt.grid()