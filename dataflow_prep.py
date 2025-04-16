import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import pickle
import os

# Custom pipeline options
class PreprocessingOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_argument('--input_file', required=True, help='Path to the input CSV file')
        parser.add_argument('--output_train', required=True, help='Output path for training data')
        parser.add_argument('--output_validation', required=True, help='Output path for validation data')
        parser.add_argument('--output_test', required=True, help='Output path for test data')
        parser.add_argument('--output_preprocessors', required=True, help='Output path for preprocessor objects')

# DoFn to read and parse the CSV data
class ReadCSVDoFn(beam.DoFn):
    def process(self, file_path):
        try:
            print(f"Reading data from {file_path}")
            df = pd.read_csv(file_path)
            print(f"Successfully read data with shape {df.shape}")
            return [df]
        except Exception as e:
            print(f"Error reading CSV: {str(e)}")
            raise

# DoFn to preprocess the data
class PreprocessDataDoFn(beam.DoFn):
    def process(self, df):
        try:
            print("Starting data preprocessing")
            
            # Keep a copy of the original data with 'Time' column removed
            df_original = df.drop(['Time'], axis=1)
            
            # Handle missing values (if any)
            df.fillna(df.mean(), inplace=True)
            
            # Create feature 'hour' from 'Time'
            df['hour'] = df['Time'].apply(lambda x: np.floor(x / 3600))
            
            # Drop the 'Time' column
            df = df.drop(['Time'], axis=1)
            
            # Separate the features from the target variable
            X = df.drop(['Class'], axis=1)
            y = df['Class']
            
            # Feature scaling using StandardScaler for amount column
            amount_scaler = StandardScaler()
            X['Amount'] = amount_scaler.fit_transform(X[['Amount']])
            
            # Robust scaling for other numerical features to handle outliers
            num_cols = X.select_dtypes(include=['float64', 'int64']).columns
            num_cols = num_cols.drop('Amount')  # Exclude Amount as it's already scaled
            
            robust_scaler = RobustScaler()
            X[num_cols] = robust_scaler.fit_transform(X[num_cols])
            
            # Combine features and target into a processed dataframe
            processed_df = pd.concat([X, y], axis=1)
            
            # Create preprocessors dictionary
            preprocessors = {
                'amount_scaler': amount_scaler,
                'robust_scaler': robust_scaler,
                'num_cols': num_cols.tolist(),
            }
            
            print(f"Preprocessing completed. Processed data shape: {processed_df.shape}")
            
            return [(processed_df, preprocessors)]
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise

# DoFn to split data
class SplitDataDoFn(beam.DoFn):
    def process(self, element):
        try:
            print("Starting data splitting")
            
            df, preprocessors = element
            
            # Separate features and target
            X = df.drop(['Class'], axis=1)
            y = df['Class']
            
            # First split: training+validation vs test (80% vs 20%)
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Second split: training vs validation (75% vs 25% of the training+validation set)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
            )
            
            # Create dataframes
            train_df = pd.concat([X_train, pd.DataFrame(y_train, columns=['Class'])], axis=1)
            val_df = pd.concat([X_val, pd.DataFrame(y_val, columns=['Class'])], axis=1)
            test_df = pd.concat([X_test, pd.DataFrame(y_test, columns=['Class'])], axis=1)
            
            print(f"Data split completed. Train: {train_df.shape}, Validation: {val_df.shape}, Test: {test_df.shape}")
            
            # Return all datasets with their types
            return [
                ('train', train_df),
                ('validation', val_df),
                ('test', test_df),
                ('preprocessors', preprocessors)
            ]
        except Exception as e:
            print(f"Error in data splitting: {str(e)}")
            raise

# Class to write dataframe to CSV
class WriteDataFrameToCSV(beam.DoFn):
    def process(self, element):
        try:
            data_type, data = element
            
            if data_type == 'preprocessors':
                return  # Preprocessors will be handled by a separate DoFn
            
            print(f"Writing {data_type} data to CSV")
            return [(data_type, data.to_csv(index=False))]
        except Exception as e:
            print(f"Error writing dataframe to CSV: {str(e)}")
            raise

# Class to write preprocessors to pickle
class WritePreprocessors(beam.DoFn):
    def process(self, element):
        try:
            data_type, preprocessors = element
            
            if data_type != 'preprocessors':
                return  # Only handle preprocessors
            
            print(f"Serializing preprocessors")
            return [(data_type, pickle.dumps(preprocessors))]
        except Exception as e:
            print(f"Error serializing preprocessors: {str(e)}")
            raise

def run(argv=None):
    parser = argparse.ArgumentParser()
    known_args, pipeline_args = parser.parse_known_args(argv)
    
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(StandardOptions).runner = 'DataflowRunner'
    preprocessing_options = pipeline_options.view_as(PreprocessingOptions)
    
    # Define file output paths
    train_output = preprocessing_options.output_train
    validation_output = preprocessing_options.output_validation
    test_output = preprocessing_options.output_test
    preprocessors_output = preprocessing_options.output_preprocessors
    
    print(f"Starting preprocessing pipeline")
    print(f"Input file: {preprocessing_options.input_file}")
    print(f"Output paths: Train: {train_output}, Validation: {validation_output}, Test: {test_output}")
    
    with beam.Pipeline(options=pipeline_options) as p:
        # Read the data
        data = (
            p 
            | "Create file path" >> beam.Create([preprocessing_options.input_file])
            | "Read CSV" >> beam.ParDo(ReadCSVDoFn())
        )
        
        # Preprocess the data
        preprocessed = (
            data
            | "Preprocess data" >> beam.ParDo(PreprocessDataDoFn())
        )
        
        # Split the data
        split_data = (
            preprocessed
            | "Split data" >> beam.ParDo(SplitDataDoFn())
        )
        
        # Separate the datasets
        datasets = {
            'train': split_data | "Filter train data" >> beam.Filter(lambda x: x[0] == 'train'),
            'validation': split_data | "Filter validation data" >> beam.Filter(lambda x: x[0] == 'validation'),
            'test': split_data | "Filter test data" >> beam.Filter(lambda x: x[0] == 'test'),
            'preprocessors': split_data | "Filter preprocessors" >> beam.Filter(lambda x: x[0] == 'preprocessors')
        }
        
        # Process and write datasets
        for data_type in ['train', 'validation', 'test']:
            output_path = {
                'train': train_output,
                'validation': validation_output,
                'test': test_output
            }[data_type]
            
            _ = (
                datasets[data_type]
                | f"Convert {data_type} to CSV" >> beam.ParDo(WriteDataFrameToCSV())
                | f"Write {data_type} to GCS" >> beam.io.WriteToText(
                    output_path, 
                    file_name_suffix='.csv',
                    shard_name_template='',
                )
            )
        
        # Process and write preprocessors
        _ = (
            datasets['preprocessors']
            | "Serialize preprocessors" >> beam.ParDo(WritePreprocessors())
            | "Write preprocessors to GCS" >> beam.io.WriteToText(
                preprocessors_output,
                file_name_suffix='.pkl',
                shard_name_template='',
            )
        )

if __name__ == '__main__':
    print("Starting Dataflow preprocessing pipeline")
    run()
    