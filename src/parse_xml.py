import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from src.utils import setup_logger
import os
from dotenv import load_dotenv

logger = setup_logger(__name__)
load_dotenv()


def extract_record_types(
    records: pd.DataFrame, type_of_record: str = None
) -> np.ndarray:
    """
    Extracts the names of records from Apple HealthKit Data based on the value in type_of_record

    Parameters:
    ----
    records: pandas.DataFrame. DataFrame of records parsed from Apple HealthKit XML Export. Attribute in XML is "Record"
    type_of_record: str. Must be either "Quantity" or "Category".

    Returns:
    ----
    np.ndarray: np.ndarray of record type.
    """

    if type_of_record not in ["Quantity", "Category"]:
        logger.error("Invalid type_of_record provided: %s", type_of_record)
        raise ValueError(
            'Unknown type of record. Please choose one of the following: "Quantity" or "Category"'
        )
    record_types = get_filtered_record_types(records, type_of_record)

    return record_types.str.replace(f"HK{type_of_record}TypeIdentifier", "").values


def clean_record_types(
    records: pd.DataFrame, type_of_record: str = None
) -> pd.DataFrame:
    """
    Cleans the names of records from Apple HealthKit Data based on the value in type_of_record

    Parameters:
    ----
    records (pd.DataFrame): DataFrame of records parsed from Apple HealthKit XML Export. Attribute in XML is "Record"
    type_of_record (str): Must be either "Quantity" or "Category".

    Returns:
    ----
    records (pandas.DataFrame): DataFrame of records parsed from Apple HealthKit XML Export with the 'type' column cleaned.
    """

    if type_of_record not in ["Quantity", "Category"]:
        logger.error("Invalid type_of_record provided: %s", type_of_record)
        raise ValueError(
            'Unknown type of record. Please choose one of the following: "Quantity" or "Category"'
        )

    record_types = get_filtered_record_types(records, type_of_record).str.replace(
        f"HK{type_of_record}TypeIdentifier", ""
    )
    records.loc[:, "type"] = record_types

    return records


def get_filtered_record_types(records: pd.DataFrame, type_of_record: str) -> pd.Series:
    """
    Filters the 'type' column of the records dataframe by the given 'type_of_record'.

    Parameters:
    ----
    records (pd.DataFrame): DataFrame of records parsed from Apple HealthKit XML Export. Attribute in XML is "Record"
    type_of_record (str): Record type that the 'type' column is filtered on. Must be either "Quantity" or "Category".

    Returns:
    ----
    filtered_record_types (pd.Series): 'type' column filtered by 'type_of_record'.
    """
    filtered_record_types = records.loc[
        records["type"].str.contains(f"HK{type_of_record}TypeIdentifier"), "type"
    ]
    return filtered_record_types


def parse_health_data(xml_path):
    """
    Parses an XML file containing health data and converts specified date columns to datetime objects.

    This function extracts records from the specified XML file path, reads each "Record" element, and constructs
    a pandas DataFrame. It also converts columns that are expected to contain date information into pandas datetime
    objects for better manipulation and analysis.

    Parameters:
    ----
    xml_path (str): The file path to the XML file containing the health data.

    Returns:
    ----
    records (pd.DataFrame): A pandas DataFrame containing the health records with date columns converted to datetime objects.

    Raises:
    ----
    Exception: Raises an exception if there is an error during the parsing of the XML file or during the data conversion. Errors will be logged with specific details.

    Example:
    --------
    >>> records = parse_health_data("path/to/health_data.xml")
    >>> print(records.head())
    """
    try:
        # Extracting Records from the XML export file
        tree = ET.parse(xml_path)
        root = tree.getroot()
        record_list = [x.attrib for x in root.iter("Record")]

        # Converting Date/Time columns to pandas datetime objects
        records = pd.DataFrame(record_list)
        date_cols = ["creationDate", "startDate", "endDate"]
        records[date_cols] = records[date_cols].apply(pd.to_datetime)
        return records
    except Exception as e:
        logger.error(f"Failed to parse XML data: {e}")
        raise e


# There are two types of records - QuantityTypes and CategoryTypes
# The next two lines extract the names of these two types of records
def process_health_records(xml_path, save_path):
    """
    Processes health records from an XML file, extracts and logs types of health records,
    cleans the record types by removing specific prefixes, and saves the cleaned data to a CSV file.

    The function leverages the `parse_health_data` to parse the XML, `extract_record_types` to extract the types of records,
    and `clean_record_types` to clean the record names from the prefixes used in Apple HealthKit data. After cleaning,
    it drops unnecessary columns and saves the resulting DataFrame to a CSV file.

    Parameters:
    ----
    xml_path (str): The file path to the XML file containing the health data.
    save_path (str): The file path where the cleaned data should be saved in CSV format.

    Raises:
    ----
    Exception: Raises an exception if there is an error during any step of the process, including parsing, extracting, cleaning, and saving data. Errors will be logged with specific details.

    Example:
    --------
    >>> process_health_records("path/to/health_data.xml", "path/to/save_data.csv")
    """
    try:
        records = parse_health_data(xml_path)
        quantity_types = extract_record_types(records, "Quantity")
        category_types = extract_record_types(records, "Category")

        logger.info("Quantity Types:\n" + "\n".join(quantity_types) + "\n")
        logger.info("Category Types:\n" + "\n".join(category_types) + "\n")

        # The next few lines remove the prefix "HKQuantityTypeIdentifier" and
        # "HKCategoryTypeIdentifier" from the names of the records.
        records = clean_record_types(records, type_of_record="Quantity")

        records.drop(columns="device", inplace=True)

        records.to_csv(save_path)

        logger.info("Data processing completed successfully")
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")
        raise e


if __name__ == "__main__":
    DATA = os.environ.get("DATA")
    export_path = os.path.join(DATA, "apple_health_export", "export.xml")
