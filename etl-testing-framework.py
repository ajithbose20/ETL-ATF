# ETL Testing Framework
# Author: Ajith Bose
# This framework provides automated validation for ETL processes

import pandas as pd
import sqlite3
import logging
import time
import os
from datetime import datetime

class ETLValidator:
    """
    A framework for validating ETL processes through automated testing.
    Provides functionality for source-to-target validation, data quality checks,
    and performance monitoring.
    """
    
    def __init__(self, source_conn=None, target_conn=None, log_path="etl_validation.log"):
        """
        Initialize the ETL validator with source and target connections.
        
        Args:
            source_conn: Connection string or object for source database
            target_conn: Connection string or object for target database
            log_path: Path to log file
        """
        self.source_conn = source_conn
        self.target_conn = target_conn
        
        # Set up logging
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("ETL Validator initialized")
        
        # Results tracking
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'total': 0
        }
    
    def connect_to_database(self, conn_string, db_type="sqlite"):
        """
        Establish connection to a database.
        
        Args:
            conn_string: Connection string for the database
            db_type: Type of database (sqlite, mysql, oracle, etc.)
            
        Returns:
            Database connection object
        """
        try:
            if db_type.lower() == "sqlite":
                conn = sqlite3.connect(conn_string)
                self.logger.info(f"Successfully connected to SQLite database: {conn_string}")
                return conn
            # Add support for other database types as needed
            else:
                self.logger.error(f"Unsupported database type: {db_type}")
                return None
        except Exception as e:
            self.logger.error(f"Error connecting to database: {str(e)}")
            return None
    
    def execute_query(self, conn, query):
        """
        Execute a SQL query and return the results.
        
        Args:
            conn: Database connection
            query: SQL query to execute
            
        Returns:
            Pandas DataFrame with query results
        """
        try:
            start_time = time.time()
            df = pd.read_sql_query(query, conn)
            execution_time = time.time() - start_time
            
            self.logger.info(f"Query executed in {execution_time:.2f} seconds, returned {len(df)} rows")
            return df
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            self.logger.error(f"Query: {query}")
            return None
    
    def validate_record_counts(self, source_query, target_query):
        """
        Compare record counts between source and target.
        
        Args:
            source_query: SQL query for source data
            target_query: SQL query for target data
            
        Returns:
            True if counts match, False otherwise
        """
        self.test_results['total'] += 1
        
        source_df = self.execute_query(self.source_conn, source_query)
        target_df = self.execute_query(self.target_conn, target_query)
        
        if source_df is None or target_df is None:
            self.logger.error("Validation failed due to query execution error")
            self.test_results['failed'] += 1
            return False
        
        source_count = len(source_df)
        target_count = len(target_df)
        
        if source_count == target_count:
            self.logger.info(f"Record count validation passed: {source_count} records in both source and target")
            self.test_results['passed'] += 1
            return True
        else:
            self.logger.error(f"Record count mismatch: Source={source_count}, Target={target_count}")
            self.test_results['failed'] += 1
            return False
    
    def validate_data_consistency(self, source_query, target_query, key_columns, tolerance=0):
        """
        Validate data consistency between source and target tables.
        
        Args:
            source_query: SQL query for source data
            target_query: SQL query for target data
            key_columns: List of key columns for matching records
            tolerance: Tolerance for numeric discrepancies (percentage)
            
        Returns:
            Tuple of (success boolean, DataFrame of discrepancies)
        """
        self.test_results['total'] += 1
        
        source_df = self.execute_query(self.source_conn, source_query)
        target_df = self.execute_query(self.target_conn, target_query)
        
        if source_df is None or target_df is None:
            self.logger.error("Validation failed due to query execution error")
            self.test_results['failed'] += 1
            return False, None
        
        # Ensure key columns exist in both dataframes
        for col in key_columns:
            if col not in source_df.columns or col not in target_df.columns:
                self.logger.error(f"Key column {col} not found in one or both datasets")
                self.test_results['failed'] += 1
                return False, None
        
        # Set key columns as index
        source_df = source_df.set_index(key_columns)
        target_df = target_df.set_index(key_columns)
        
        # Find common columns for comparison
        common_cols = [col for col in source_df.columns if col in target_df.columns]
        
        if not common_cols:
            self.logger.error("No common columns found for comparison")
            self.test_results['failed'] += 1
            return False, None
        
        # Find all discrepancies
        discrepancies = pd.DataFrame(columns=['key', 'column', 'source_value', 'target_value'])
        
        for idx in source_df.index:
            if idx in target_df.index:
                for col in common_cols:
                    source_val = source_df.at[idx, col]
                    target_val = target_df.at[idx, col]
                    
                    # Handle numeric comparisons with tolerance
                    if pd.api.types.is_numeric_dtype(type(source_val)) and pd.api.types.is_numeric_dtype(type(target_val)):
                        if source_val == 0 and target_val == 0:
                            continue
                        elif source_val == 0:
                            diff_pct = float('inf')
                        else:
                            diff_pct = abs((source_val - target_val) / source_val * 100)
                        
                        if diff_pct > tolerance:
                            discrepancies = pd.concat([discrepancies, pd.DataFrame({
                                'key': [str(idx)],
                                'column': [col],
                                'source_value': [source_val],
                                'target_value': [target_val],
                                'difference_pct': [diff_pct]
                            })])
                    # String and other type comparisons
                    elif source_val != target_val:
                        discrepancies = pd.concat([discrepancies, pd.DataFrame({
                            'key': [str(idx)],
                            'column': [col],
                            'source_value': [source_val],
                            'target_value': [target_val],
                            'difference_pct': [None]
                        })])
            else:
                self.logger.warning(f"Record with key {idx} found in source but not in target")
                self.test_results['warnings'] += 1
        
        # Check for records in target but not in source
        for idx in target_df.index:
            if idx not in source_df.index:
                self.logger.warning(f"Record with key {idx} found in target but not in source")
                self.test_results['warnings'] += 1
        
        if discrepancies.empty:
            self.logger.info("Data consistency validation passed: No discrepancies found")
            self.test_results['passed'] += 1
            return True, None
        else:
            self.logger.error(f"Data consistency validation failed: {len(discrepancies)} discrepancies found")
            self.test_results['failed'] += 1
            return False, discrepancies
    
    def validate_data_quality(self, data, rules):
        """
        Validate data quality based on specified rules.
        
        Args:
            data: DataFrame to validate
            rules: Dictionary of rules (column: rule function)
            
        Returns:
            DataFrame with validation issues
        """
        self.test_results['total'] += 1
        
        if data is None or data.empty:
            self.logger.error("Cannot validate data quality: No data provided")
            self.test_results['failed'] += 1
            return None
        
        issues = pd.DataFrame(columns=['row_index', 'column', 'value', 'issue'])
        
        for column, rule_func in rules.items():
            if column not in data.columns:
                self.logger.warning(f"Column {column} not found in dataset")
                continue
                
            for idx, value in enumerate(data[column]):
                issue = rule_func(value)
                if issue:
                    issues = pd.concat([issues, pd.DataFrame({
                        'row_index': [idx],
                        'column': [column],
                        'value': [value],
                        'issue': [issue]
                    })])
        
        if issues.empty:
            self.logger.info("Data quality validation passed: No issues found")
            self.test_results['passed'] += 1
        else:
            self.logger.error(f"Data quality validation failed: {len(issues)} issues found")
            self.test_results['failed'] += 1
            
        return issues
    
    def measure_performance(self, query, connection, iterations=3):
        """
        Measure performance of a query.
        
        Args:
            query: SQL query to execute
            connection: Database connection
            iterations: Number of iterations for measuring average performance
            
        Returns:
            Dictionary with performance metrics
        """
        execution_times = []
        
        for i in range(iterations):
            start_time = time.time()
            result = self.execute_query(connection, query)
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            
            # Small delay between iterations
            time.sleep(0.1)
        
        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        
        metrics = {
            'average_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'row_count': len(result) if result is not None else 0
        }
        
        self.logger.info(f"Performance metrics: Avg={avg_time:.2f}s, Min={min_time:.2f}s, Max={max_time:.2f}s, Rows={metrics['row_count']}")
        return metrics
    
    def generate_report(self, output_path=None):
        """
        Generate validation report.
        
        Args:
            output_path: Path to save the report, if None, return as string
            
        Returns:
            Report as string if output_path is None
        """
        report = []
        report.append("=" * 50)
        report.append("ETL VALIDATION REPORT")
        report.append("=" * 50)
        report.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("TEST SUMMARY")
        report.append("-" * 50)
        report.append(f"Total Tests: {self.test_results['total']}")
        report.append(f"Passed: {self.test_results['passed']}")
        report.append(f"Failed: {self.test_results['failed']}")
        report.append(f"Warnings: {self.test_results['warnings']}")
        
        success_rate = (self.test_results['passed'] / self.test_results['total'] * 100) if self.test_results['total'] > 0 else 0
        report.append(f"Success Rate: {success_rate:.2f}%")
        report.append("")
        
        report_text = "\n".join(report)
        
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    f.write(report_text)
                self.logger.info(f"Report generated and saved to {output_path}")
            except Exception as e:
                self.logger.error(f"Error saving report: {str(e)}")
        
        return report_text


# Example usage
if __name__ == "__main__":
    # Example data quality rules
    def check_not_null(value):
        return "Value is NULL" if pd.isna(value) else None
    
    def check_positive(value):
        try:
            return "Value should be positive" if float(value) <= 0 else None
        except:
            return "Value is not numeric"
    
    def check_email_format(value):
        if pd.isna(value):
            return None
        if '@' not in str(value) or '.' not in str(value):
            return "Invalid email format"
        return None
    
    # Create example source and target databases
    source_db = "source_db.sqlite"
    target_db = "target_db.sqlite"
    
    # Sample data
    source_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['John', 'Jane', 'Bob', 'Alice', 'Mike'],
        'email': ['john@example.com', 'jane@example.com', 'invalid_email', 'alice@example.com', 'mike@example.com'],
        'amount': [100, 200, -50, 400, 500]
    })
    
    target_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 6],  # ID 5 missing, ID 6 added
        'name': ['John', 'Jane', 'Bob', 'Alice', 'Sarah'],
        'email': ['john@example.com', 'jane@example.com', 'invalid_email', 'alice@example.com', 'sarah@example.com'],
        'amount': [100, 205, -50, 400, 600]  # Value for ID 2 changed
    })
    
    # Set up source database
    source_conn = sqlite3.connect(source_db)
    source_data.to_sql('customer', source_conn, if_exists='replace', index=False)
    
    # Set up target database
    target_conn = sqlite3.connect(target_db)
    target_data.to_sql('customer_dim', target_conn, if_exists='replace', index=False)
    
    # Initialize validator
    validator = ETLValidator(source_conn, target_conn)
    
    # Validate record counts
    print("Validating record counts...")
    validator.validate_record_counts(
        "SELECT * FROM customer",
        "SELECT * FROM customer_dim"
    )
    
    # Validate data consistency
    print("Validating data consistency...")
    success, discrepancies = validator.validate_data_consistency(
        "SELECT * FROM customer",
        "SELECT * FROM customer_dim",
        ['id'],
        tolerance=1  # 1% tolerance for numeric differences
    )
    
    if not success and discrepancies is not None:
        print(f"Found {len(discrepancies)} discrepancies:")
        print(discrepancies)
    
    # Validate data quality
    print("Validating data quality...")
    source_data = validator.execute_query(source_conn, "SELECT * FROM customer")
    quality_rules = {
        'email': check_email_format,
        'amount': check_positive
    }
    issues = validator.validate_data_quality(source_data, quality_rules)
    
    if issues is not None and not issues.empty:
        print(f"Found {len(issues)} data quality issues:")
        print(issues)
    
    # Measure performance
    print("Measuring query performance...")
    metrics = validator.measure_performance(
        "SELECT c.* FROM customer c JOIN (SELECT id FROM customer) t ON c.id = t.id",
        source_conn
    )
    print(f"Average execution time: {metrics['average_time']:.4f} seconds")
    
    # Generate report
    print("\nGenerating validation report...")
    report = validator.generate_report("etl_validation_report.txt")
    print(report)
    
    # Clean up
    source_conn.close()
    target_conn.close()
    
    if os.path.exists(source_db):
        os.remove(source_db)
    if os.path.exists(target_db):
        os.remove(target_db)
