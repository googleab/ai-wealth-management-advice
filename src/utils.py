# src/utils.py
import logging

def setup_logging(log_file: str = 'data_lineage.log'):
    """Configure logging to output data lineage to a file."""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

def log_data_lineage(action: str, details: str):
    """Log key events in the data pipeline."""
    logging.info(f"{action}: {details}")

# Example usage when run directly
if __name__ == '__main__':
    setup_logging()
    log_data_lineage("DATA_LOAD", "Loaded client_data.csv with shape (5000, 6)")
    log_data_lineage("DATA_VALIDATION", "Data passed all quality checks.")
