"""
Data Processor

Handles tick data upload, validation, and preprocessing for backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
from dataclasses import dataclass
from datetime import datetime
import pytz
import re
from error_handling import (
    ErrorHandler, ValidationErrorCollector, DataProcessingError,
    ErrorDetails, ErrorCategory, ErrorSeverity
)


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


@dataclass
class DataConfig:
    """Configuration for data preprocessing"""
    timezone: str = "UTC"
    fill_gaps: bool = True
    remove_outliers: bool = True
    outlier_threshold: float = 3.0


class DataProcessor:
    """Handles tick data upload, validation, and preprocessing"""
    
    # Common datetime formats to try
    DATETIME_FORMATS = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
        '%d/%m/%Y %H:%M:%S',
        '%m/%d/%Y %H:%M:%S',
        '%Y-%m-%d',
        '%d/%m/%Y',
        '%m/%d/%Y',
        '%Y%m%d %H:%M:%S',
        '%Y%m%d',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S.%fZ'
    ]
    
    # Required columns for tick data
    REQUIRED_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close']
    OPTIONAL_COLUMNS = ['volume']
    
    def __init__(self):
        """Initialize data processor with error handling"""
        self.error_handler = ErrorHandler("data_processor")
    
    def validate_tick_data(self, data: pd.DataFrame) -> ValidationResult:
        """Validate uploaded tick data format and completeness with comprehensive error handling"""
        validator = ValidationErrorCollector()
        
        try:
            # Check if DataFrame is empty
            if data.empty:
                validator.add_error(
                    "Data is empty",
                    suggestions=[
                        "Check that your CSV file contains data",
                        "Verify the file was uploaded correctly",
                        "Ensure the file is not corrupted"
                    ]
                )
                return ValidationResult(
                    is_valid=False, 
                    errors=[error.message for error in validator.errors], 
                    warnings=[]
                )
            
            # Check DataFrame size limits
            if len(data) > 1000000:  # 1M rows
                validator.add_warning(
                    f"Large dataset detected ({len(data):,} rows) - processing may be slow",
                    suggestions=[
                        "Consider filtering to a smaller date range",
                        "Use higher timeframes if possible",
                        "Ensure sufficient system memory"
                    ]
                )
            elif len(data) < 100:
                validator.add_warning(
                    f"Small dataset detected ({len(data)} rows) - results may not be reliable",
                    suggestions=[
                        "Use more historical data for better results",
                        "Consider using a longer time period",
                        "Verify this is the complete dataset"
                    ]
                )
            
            # Check for required columns with smart matching
            missing_columns = []
            column_mapping = {}
            
            for required_col in self.REQUIRED_COLUMNS:
                found_col = self._find_column_match(data.columns, required_col)
                if found_col:
                    column_mapping[required_col] = found_col
                    if found_col.lower() != required_col.lower():
                        validator.add_warning(
                            f"Using column '{found_col}' for '{required_col}' (case/name mismatch)",
                            suggestions=[f"Consider renaming '{found_col}' to '{required_col}' for clarity"]
                        )
                else:
                    missing_columns.append(required_col)
            
            if missing_columns:
                validator.add_error(
                    f"Missing required columns: {missing_columns}",
                    suggestions=[
                        f"Ensure your CSV has columns: {', '.join(self.REQUIRED_COLUMNS)}",
                        "Check column names for typos",
                        "Verify CSV format and structure"
                    ]
                )
            
            # Check for optional columns
            for col in self.OPTIONAL_COLUMNS:
                if not self._find_column_match(data.columns, col):
                    validator.add_warning(
                        f"Optional column '{col}' not found - will use default values",
                        suggestions=[f"Add '{col}' column for more accurate backtesting"]
                    )
            
            # Validate timestamp column if present
            timestamp_col = column_mapping.get('timestamp')
            if timestamp_col:
                timestamp_errors = self._validate_timestamps(data[timestamp_col])
                for error in timestamp_errors:
                    validator.add_error(error, suggestions=[
                        "Check timestamp format in your data",
                        "Ensure timestamps are in a standard format",
                        "Consider using ISO format (YYYY-MM-DD HH:MM:SS)"
                    ])
            
            # Validate OHLC data if present
            ohlc_cols = {col: column_mapping.get(col) for col in ['open', 'high', 'low', 'close']}
            available_ohlc = {k: v for k, v in ohlc_cols.items() if v is not None}
            
            if len(available_ohlc) >= 4:
                ohlc_data = data[[col for col in available_ohlc.values()]]
                ohlc_data.columns = list(available_ohlc.keys())  # Standardize column names
                ohlc_errors = self._validate_ohlc_data(ohlc_data)
                for error in ohlc_errors:
                    validator.add_error(error, suggestions=[
                        "Check for data entry errors in price columns",
                        "Verify OHLC relationships (High >= Open,Close; Low <= Open,Close)",
                        "Look for negative or zero prices"
                    ])
            
            # Check for missing values with detailed analysis
            missing_analysis = self._analyze_missing_data(data, column_mapping)
            for analysis in missing_analysis:
                if analysis['severity'] == 'error':
                    validator.add_error(analysis['message'], suggestions=analysis['suggestions'])
                else:
                    validator.add_warning(analysis['message'], suggestions=analysis['suggestions'])
            
            # Check data types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                mapped_col = column_mapping.get(col) or self._find_column_match(data.columns, col)
                if mapped_col and mapped_col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[mapped_col]):
                        # Try to convert to numeric
                        try:
                            pd.to_numeric(data[mapped_col], errors='raise')
                            validator.add_warning(
                                f"Column '{mapped_col}' is not numeric but can be converted",
                                suggestions=[f"Convert '{mapped_col}' to numeric format in your data"]
                            )
                        except (ValueError, TypeError):
                            validator.add_error(
                                f"Column '{mapped_col}' contains non-numeric values",
                                suggestions=[
                                    f"Check '{mapped_col}' column for text or invalid values",
                                    "Remove or fix non-numeric entries",
                                    "Ensure decimal separator is correct (. not ,)"
                                ]
                            )
            
            # Check for data quality issues
            quality_issues = self._check_data_quality(data, column_mapping)
            for issue in quality_issues:
                if issue['severity'] == 'error':
                    validator.add_error(issue['message'], suggestions=issue['suggestions'])
                else:
                    validator.add_warning(issue['message'], suggestions=issue['suggestions'])
            
        except Exception as e:
            self.error_handler.handle_error(e, {'data_shape': data.shape})
            validator.add_error(
                f"Unexpected error during validation: {str(e)}",
                suggestions=[
                    "Check your data file format",
                    "Try with a smaller sample of data",
                    "Contact support if the issue persists"
                ]
            )
        
        # Return validation result
        is_valid = not validator.has_errors()
        return ValidationResult(
            is_valid=is_valid,
            errors=[error.message for error in validator.errors],
            warnings=[warning.message for warning in validator.warnings]
        )
    
    def _find_column_match(self, columns: List[str], target: str) -> Union[str, None]:
        """Find column that matches target (case insensitive, with common variations)"""
        # Exact match first
        if target in columns:
            return target
        
        # Case insensitive match
        for col in columns:
            if col.lower() == target.lower():
                return col
        
        # Common variations
        variations = {
            'timestamp': ['time', 'datetime', 'date', 'ts'],
            'open': ['o', 'opening', 'open_price'],
            'high': ['h', 'hi', 'high_price'],
            'low': ['l', 'lo', 'low_price'],
            'close': ['c', 'closing', 'close_price', 'last'],
            'volume': ['vol', 'v', 'size', 'quantity']
        }
        
        if target.lower() in variations:
            for variation in variations[target.lower()]:
                for col in columns:
                    if col.lower() == variation:
                        return col
        
        return None
    
    def _analyze_missing_data(self, data: pd.DataFrame, column_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
        """Analyze missing data patterns and provide detailed feedback"""
        analyses = []
        
        missing_data = data.isnull().sum()
        total_rows = len(data)
        
        for col, count in missing_data.items():
            if count > 0:
                percentage = (count / total_rows) * 100
                
                # Determine severity based on column importance and percentage
                is_required = col in [column_mapping.get(req_col) for req_col in self.REQUIRED_COLUMNS]
                
                if is_required and percentage > 10:
                    severity = 'error'
                    message = f"Critical column '{col}' has {count} missing values ({percentage:.1f}%)"
                    suggestions = [
                        "Fill missing values or remove incomplete rows",
                        "Check data source for completeness",
                        "Consider using a different dataset"
                    ]
                elif is_required and percentage > 0:
                    severity = 'warning'
                    message = f"Required column '{col}' has {count} missing values ({percentage:.1f}%)"
                    suggestions = [
                        "Fill missing values with appropriate method",
                        "Remove rows with missing critical data",
                        "Check for data collection issues"
                    ]
                else:
                    severity = 'warning'
                    message = f"Column '{col}' has {count} missing values ({percentage:.1f}%)"
                    suggestions = [
                        "Consider filling missing values",
                        "Missing values will be handled during preprocessing"
                    ]
                
                analyses.append({
                    'severity': severity,
                    'message': message,
                    'suggestions': suggestions
                })
        
        return analyses
    
    def _check_data_quality(self, data: pd.DataFrame, column_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
        """Check for data quality issues"""
        issues = []
        
        # Check for duplicate timestamps
        timestamp_col = column_mapping.get('timestamp')
        if timestamp_col and timestamp_col in data.columns:
            duplicates = data[timestamp_col].duplicated().sum()
            if duplicates > 0:
                issues.append({
                    'severity': 'warning',
                    'message': f"Found {duplicates} duplicate timestamps",
                    'suggestions': [
                        "Remove duplicate rows",
                        "Check data source for duplicate entries",
                        "Consider aggregating duplicate timestamps"
                    ]
                })
        
        # Check for unrealistic price movements
        close_col = column_mapping.get('close')
        if close_col and close_col in data.columns:
            try:
                price_changes = data[close_col].pct_change().abs()
                extreme_changes = (price_changes > 0.5).sum()  # >50% price change
                
                if extreme_changes > 0:
                    issues.append({
                        'severity': 'warning',
                        'message': f"Found {extreme_changes} extreme price movements (>50%)",
                        'suggestions': [
                            "Check for data errors or stock splits",
                            "Verify price data accuracy",
                            "Consider outlier removal"
                        ]
                    })
            except Exception:
                pass  # Skip if calculation fails
        
        # Check for constant values
        numeric_cols = ['open', 'high', 'low', 'close']
        for col in numeric_cols:
            mapped_col = column_mapping.get(col)
            if mapped_col and mapped_col in data.columns:
                try:
                    if data[mapped_col].nunique() == 1:
                        issues.append({
                            'severity': 'error',
                            'message': f"Column '{mapped_col}' has constant values",
                            'suggestions': [
                                "Check data source for errors",
                                "Verify data collection process",
                                "Use different time period or instrument"
                            ]
                        })
                except Exception:
                    pass
        
        return issues
    def _validate_timestamps(self, timestamps: pd.Series) -> List[str]:
        """Validate timestamp column with comprehensive error checking"""
        errors = []
        
        try:
            # Check if timestamps can be parsed
            if not pd.api.types.is_datetime64_any_dtype(timestamps):
                # Try to parse as datetime
                try:
                    parsed_timestamps = self._parse_timestamps(timestamps)
                    if parsed_timestamps is None:
                        errors.append("Unable to parse timestamp column - unrecognized format")
                    else:
                        # Check for reasonable date range
                        min_date = parsed_timestamps.min()
                        max_date = parsed_timestamps.max()
                        
                        if min_date.year < 1990:
                            errors.append(f"Timestamps contain very old dates (earliest: {min_date})")
                        if max_date.year > 2030:
                            errors.append(f"Timestamps contain future dates (latest: {max_date})")
                        
                        # Check for chronological order
                        if not parsed_timestamps.is_monotonic_increasing:
                            errors.append("Timestamps are not in chronological order")
                            
                except Exception as e:
                    errors.append(f"Timestamp parsing error: {str(e)}")
            else:
                # Already datetime, check for issues
                if timestamps.isnull().any():
                    errors.append("Timestamp column contains null values")
                
                # Check date range
                try:
                    min_date = timestamps.min()
                    max_date = timestamps.max()
                    
                    if pd.isna(min_date) or pd.isna(max_date):
                        errors.append("Unable to determine timestamp range")
                    else:
                        if min_date.year < 1990:
                            errors.append(f"Timestamps contain very old dates (earliest: {min_date})")
                        if max_date.year > 2030:
                            errors.append(f"Timestamps contain future dates (latest: {max_date})")
                except Exception as e:
                    errors.append(f"Error checking timestamp range: {str(e)}")
        
        except Exception as e:
            errors.append(f"Unexpected error validating timestamps: {str(e)}")
        
        return errors
    
    def _validate_ohlc_data(self, ohlc_data: pd.DataFrame) -> List[str]:
        """Validate OHLC data consistency with comprehensive checks"""
        errors = []
        
        try:
            # Check OHLC relationships: High >= Open, Close, Low <= Open, Close
            invalid_high = (ohlc_data['high'] < ohlc_data['open']) | (ohlc_data['high'] < ohlc_data['close'])
            invalid_low = (ohlc_data['low'] > ohlc_data['open']) | (ohlc_data['low'] > ohlc_data['close'])
            
            if invalid_high.any():
                count = invalid_high.sum()
                errors.append(f"Found {count} rows where High < Open or High < Close")
            
            if invalid_low.any():
                count = invalid_low.sum()
                errors.append(f"Found {count} rows where Low > Open or Low > Close")
            
            # Check for negative prices
            negative_prices = (ohlc_data < 0).any(axis=1)
            if negative_prices.any():
                count = negative_prices.sum()
                errors.append(f"Found {count} rows with negative prices")
            
            # Check for zero prices
            zero_prices = (ohlc_data == 0).any(axis=1)
            if zero_prices.any():
                count = zero_prices.sum()
                errors.append(f"Found {count} rows with zero prices")
            
            # Check for unrealistic price ranges
            for col in ['open', 'high', 'low', 'close']:
                if col in ohlc_data.columns:
                    col_data = ohlc_data[col]
                    if col_data.max() / col_data.min() > 1000:  # 1000x price range
                        errors.append(f"Column '{col}' has unrealistic price range (min: {col_data.min():.6f}, max: {col_data.max():.6f})")
            
            # Check for identical OHLC values (suspicious)
            identical_ohlc = (
                (ohlc_data['open'] == ohlc_data['high']) & 
                (ohlc_data['high'] == ohlc_data['low']) & 
                (ohlc_data['low'] == ohlc_data['close'])
            )
            if identical_ohlc.any():
                count = identical_ohlc.sum()
                if count > len(ohlc_data) * 0.1:  # More than 10% identical
                    errors.append(f"Found {count} rows with identical OHLC values (may indicate data issues)")
        
        except Exception as e:
            errors.append(f"Error validating OHLC data: {str(e)}")
        
        return errors
    
    def _parse_timestamps(self, timestamps: pd.Series) -> Union[pd.Series, None]:
        """Parse timestamps using multiple formats with comprehensive error handling"""
        if pd.api.types.is_datetime64_any_dtype(timestamps):
            return timestamps
        
        # Try different datetime formats
        for fmt in self.DATETIME_FORMATS:
            try:
                parsed = pd.to_datetime(timestamps, format=fmt, errors='raise')
                self.error_handler.logger.info(f"Successfully parsed timestamps using format: {fmt}")
                return parsed
            except (ValueError, TypeError):
                continue
        
        # Try pandas automatic parsing as last resort
        try:
            parsed = pd.to_datetime(timestamps, infer_datetime_format=True, errors='raise')
            self.error_handler.logger.info("Successfully parsed timestamps using automatic inference")
            return parsed
        except (ValueError, TypeError):
            pass
        
        # Try with different error handling
        try:
            parsed = pd.to_datetime(timestamps, errors='coerce')
            if parsed.isnull().all():
                return None
            
            null_count = parsed.isnull().sum()
            if null_count > 0:
                self.error_handler.logger.warning(f"Parsed timestamps with {null_count} null values")
            
            return parsed
        except Exception:
            return None
    
    def preprocess_data(self, data: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
        """Clean and prepare data for backtesting with comprehensive error handling"""
        try:
            processed_data = data.copy()
            
            # Parse timestamps if needed
            if 'timestamp' in processed_data.columns:
                try:
                    parsed_timestamps = self._parse_timestamps(processed_data['timestamp'])
                    if parsed_timestamps is None:
                        error_details = ErrorDetails(
                            category=ErrorCategory.DATA_ERROR,
                            severity=ErrorSeverity.ERROR,
                            message="Failed to parse timestamps during preprocessing",
                            user_message="Unable to process timestamp data. Please check your date format.",
                            suggestions=[
                                "Use standard date formats (YYYY-MM-DD HH:MM:SS)",
                                "Check for invalid or missing timestamps",
                                "Verify timezone information"
                            ]
                        )
                        raise DataProcessingError(error_details)
                    
                    processed_data['timestamp'] = parsed_timestamps
                    
                    # Always ensure timezone awareness for consistency
                    try:
                        tz = pytz.timezone(config.timezone)
                        if processed_data['timestamp'].dt.tz is None:
                            # Localize naive timestamps to the specified timezone
                            processed_data['timestamp'] = processed_data['timestamp'].dt.tz_localize(tz)
                        else:
                            # Convert to the specified timezone
                            processed_data['timestamp'] = processed_data['timestamp'].dt.tz_convert(tz)
                    except Exception as e:
                        self.error_handler.logger.warning(f"Timezone conversion failed: {str(e)}")
                        # Fallback: localize to UTC if timezone conversion fails
                        if processed_data['timestamp'].dt.tz is None:
                            processed_data['timestamp'] = processed_data['timestamp'].dt.tz_localize('UTC')
                    
                    # Sort by timestamp
                    processed_data = processed_data.sort_values('timestamp').reset_index(drop=True)
                    
                except Exception as e:
                    error_details = ErrorDetails(
                        category=ErrorCategory.DATA_ERROR,
                        severity=ErrorSeverity.ERROR,
                        message=f"Timestamp processing failed: {str(e)}",
                        user_message="Error processing timestamp data.",
                        suggestions=[
                            "Check timestamp format in your data",
                            "Ensure timestamps are valid",
                            "Try a different timezone setting"
                        ]
                    )
                    raise DataProcessingError(error_details)
            
            # Fill gaps if requested
            if config.fill_gaps and 'timestamp' in processed_data.columns:
                try:
                    processed_data = self._fill_data_gaps(processed_data)
                except Exception as e:
                    self.error_handler.logger.warning(f"Gap filling failed: {str(e)}")
                    # Continue without gap filling
            
            # Remove outliers if requested
            if config.remove_outliers:
                try:
                    processed_data = self._remove_outliers(processed_data, config.outlier_threshold)
                except Exception as e:
                    self.error_handler.logger.warning(f"Outlier removal failed: {str(e)}")
                    # Continue without outlier removal
            
            # Final validation
            if processed_data.empty:
                error_details = ErrorDetails(
                    category=ErrorCategory.DATA_ERROR,
                    severity=ErrorSeverity.ERROR,
                    message="Data preprocessing resulted in empty dataset",
                    user_message="All data was removed during preprocessing.",
                    suggestions=[
                        "Check your data quality",
                        "Adjust preprocessing parameters",
                        "Use different outlier thresholds"
                    ]
                )
                raise DataProcessingError(error_details)
            
            return processed_data
            
        except DataProcessingError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            error_details = ErrorDetails(
                category=ErrorCategory.DATA_ERROR,
                severity=ErrorSeverity.ERROR,
                message=f"Unexpected error during data preprocessing: {str(e)}",
                technical_details=str(e),
                user_message="An unexpected error occurred while processing your data.",
                suggestions=[
                    "Check your data format and content",
                    "Try with a smaller dataset",
                    "Contact support if the issue persists"
                ]
            )
            self.error_handler.handle_error(e, {'data_shape': data.shape, 'config': config})
            raise DataProcessingError(error_details)
    
    def _fill_data_gaps(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill gaps in time series data with error handling"""
        if len(data) < 2:
            return data
        
        try:
            # Forward fill missing values
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data[numeric_cols] = data[numeric_cols].fillna(method='ffill')
                
                # If still missing values at the beginning, use backward fill
                remaining_nulls = data[numeric_cols].isnull().sum().sum()
                if remaining_nulls > 0:
                    data[numeric_cols] = data[numeric_cols].fillna(method='bfill')
            
            return data
            
        except Exception as e:
            self.error_handler.logger.warning(f"Error filling data gaps: {str(e)}")
            return data
    
    def _remove_outliers(self, data: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """Remove outliers using z-score method with error handling"""
        numeric_cols = ['open', 'high', 'low', 'close']
        available_cols = [col for col in numeric_cols if col in data.columns]
        
        if not available_cols:
            return data
        
        try:
            # Calculate z-scores for price columns
            for col in available_cols:
                if data[col].std() == 0:  # Skip constant columns
                    continue
                    
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outlier_mask = z_scores > threshold
                
                # Replace outliers with median value
                if outlier_mask.any():
                    median_value = data[col].median()
                    outlier_count = outlier_mask.sum()
                    data.loc[outlier_mask, col] = median_value
                    self.error_handler.logger.info(f"Replaced {outlier_count} outliers in column '{col}' with median value")
            
            return data
            
        except Exception as e:
            self.error_handler.logger.warning(f"Error removing outliers: {str(e)}")
            return data