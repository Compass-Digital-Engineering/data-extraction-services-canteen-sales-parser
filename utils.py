import json
import re
import pandas as pd # type: ignore
from unit_convert import UnitConvert # type: ignore
from collections import defaultdict
# from sentence_transformers import SentenceTransformer, util
from google.cloud import storage # type: ignore
from Workflow.google_storage_workflow import read_csv_from_gcs
from google.cloud import secretmanager # type: ignore
import os
from typing import Tuple, Any
import numpy as np
import logging
import sys
from logging.handlers import RotatingFileHandler


def extract_json_from_string(text: str):
    """Extracts first JSON object present from a string of text if present, else returns None"""
    # Look for the largest { ... } in text
    match = re.search(r"\{(.*)\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed_response = json.loads(match.group())
        return parsed_response
    except Exception as e:
        return None
    
    
def sanitize_filename(filename: str):
    """Removes any special characters that won't work in filenames and replaces them with underscores"""
    return "".join(
        c if c.isalnum() or c in {" ", ".", "-", "_"} else "_" for c in filename
    )

def convert_volume_to_oz(volume_quantity, volume_unit):
    """
    Convert volume to ounces directly based on the volume unit.

    Args:
    - volume_quantity (float): The volume value to convert.
    - volume_unit (str): The unit of the volume (e.g., "ml", "l", "fl oz").

    Returns:
    - float: The equivalent volume in ounces.
    """
    # Conversion factors for volume to ounces (direct conversion)
    volume_to_oz = {
        'ml': 0.033814,   
        'l': 33.814,     
        'gal': 128.0  
    }

    # Check if the provided volume unit exists in our dictionary
    if volume_unit.lower() in volume_to_oz:
        return volume_quantity * volume_to_oz[volume_unit.lower()]
    else:
        raise ValueError(f"Unsupported volume unit: {volume_unit}")

# Updated extract_quantity_and_unit function
def extract_quantity_and_unit(size_str):
    """
    Extract quantity and unit from a given string.
    
    Args:
    - size_str (str): The input string containing size information.
    
    Returns:
    - tuple: (quantity, unit) or (None, None) if no match is found.
    """
    # Updated regex pattern
    pattern = r'(\d*?\.?\d+)\s*(oz|lb|l|ml|mg|gal|g|fl.*oz|ounce|millilitre|gram|fluidOunceUS|pound)'
    
    # Perform the regex search on the input string
    match = re.search(pattern, size_str, re.IGNORECASE)
    
    if match:
        quantity = float(match.group(1))  # Extract the quantity
        unit = match.group(2).lower()  # Extract the unit and convert to lowercase
        unit = unit.replace('lb', 'lbs')
        unit = unit.replace('ounce', 'oz')
        unit = unit.replace('millilitre', 'ml')
        unit = unit.replace('gram', 'g')
        unit = unit.replace('fluidOunceUS', 'oz')
        unit = unit.replace('pound', 'lbs')
        if 'oz' in unit:
            unit = 'oz' 
        return quantity, unit
    return None, None

def is_serving_size_match(product_name, serving_size, margin_of_error=0.2):
    """
    Check if the serving size matches the product quantity and unit in the product name.
    
    Args:
    - product_name (str): The product name containing the size information.
    - serving_size (str): The serving size string to match against.
    - margin_of_error (float): The acceptable margin of error in the comparison (in oz).
    
    Returns:
    - bool: True if the serving size matches the product size, False otherwise.
    """
    # Check if serving size is NaN
    if pd.isna(serving_size):
        return False

    # Extract quantity and unit from serving size
    serving_quantity, serving_unit = extract_quantity_and_unit(serving_size)

    # If no unit is found in the product name, return True
    if not serving_unit:
        return False

    # Extract quantity and unit from product name
    product_quantity, product_unit = extract_quantity_and_unit(product_name)

    # If any of the quantities or units are missing, return False
    if not product_quantity or not serving_quantity or not product_unit or not serving_unit:
        return True

    # If the unit is a volume (e.g., ml, l, fl oz), convert it to ounces
    if product_unit in ['ml', 'l', 'gal']:
        try:
            product_quantity_in_oz = convert_volume_to_oz(product_quantity, product_unit)
        except ValueError:
            return False
    else:
        try:
            product_quantity_in_oz = UnitConvert(**{product_unit: product_quantity})['oz']
        except KeyError:
            return False

    # Convert serving size to ounces
    if serving_unit in ['ml', 'l', 'gal']:
        try:
            serving_quantity_in_oz = convert_volume_to_oz(serving_quantity, serving_unit)
        except ValueError:
            return False
    else:
        try:
            serving_quantity_in_oz = UnitConvert(**{serving_unit: serving_quantity})['oz']
        except KeyError:
            return False

    # Compare the quantities within the margin of error
    if abs(product_quantity_in_oz - serving_quantity_in_oz) <= margin_of_error:
        return True
    else:
        return False



def invert_dictionary(my_dict):

    # Inverted dictionary using defaultdict
    inverted_dict = defaultdict(list)
    for key, values in my_dict.items():
        for value in values:
            inverted_dict[value].append(key)

    # Post-process to make single-value lists into single values
    for key, value in inverted_dict.items():
        if len(value) == 1:  # If there's only one key for this value
            inverted_dict[key] = value[0]  # Replace list with the single value

    # Convert defaultdict to a regular dictionary and print
    return dict(inverted_dict)


def generate_sitemap(site_path:str) -> list:
    """
    Generates a sitemap by reading a CSV file from Google Cloud Storage (GCS).

    This function retrieves a CSV file from GCS, processes its content, and returns
    a list of site entries.

    Args:
        site_path (str): The GCS file path (e.g., "gs://bucket-name/path/to/file.csv").

    Returns:
        list: A list containing the site entries extracted from the CSV file.
    """
    # Initialize empty sitemap list
    sitemap = []

    # Call read csv from gcs 
    site_csv = read_csv_from_gcs(site_path)

    # Append to sitemap list
    for site in site_csv:    
        sitemap.append(site)

    return pd.DataFrame(sitemap)

def list_folders_in_bucket(bucket_name:str, folder_path:str) -> list:
    """
    Lists all folder names within a specified path in a Google Cloud Storage (GCS) bucket.

    This function retrieves the names of all "folders" (i.e., common prefixes) within 
    a given folder path inside a GCS bucket. It uses the `delimiter='/'` parameter 
    to treat objects as directory-like structures.

    Args:
        bucket_name (str): The name of the GCS bucket.
        folder_path (str): The path within the bucket where folders should be listed. 
                        This should end with a `/` to properly target a directory.

    Returns:
        list[str]: A list of folder names (subdirectories) within the specified path.
    """
    # Initialize the Google Cloud Storage client
    client = storage.Client()

    # Access the specified bucket
    bucket = client.bucket(bucket_name)

    # List blobs in the specified folder
    blobs = client.list_blobs(bucket_name, prefix=folder_path, delimiter='/')

    # Extract folder names
    folders = []
    for page in blobs.pages:
        folders.extend(page.prefixes)  # `prefixes` contains only folder names

    # Print or return folder names
    folder_names = [folder[len(folder_path):].rstrip('/') for folder in folders]
    
    return folder_names


def get_all_ids(
        bucket_name: str,
        folder_path: str,
    ) -> set:
    """
    Fetches all blobs from a specified GCS bucket folder and filters them based on metadata.

    Parameters:
        bucket_name (str): The name of the GCS bucket.
        folder_path (str): The folder path within the bucket (e.g., "data/subfolder/").

    Returns:
        Set: A set of all ids.
    """

    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # List all blobs in the given folder
    blobs = bucket.list_blobs(prefix=folder_path)

    # Filter blobs based on metadata
    results_set = set()
    for blob in blobs:
        if blob.metadata is not None:
            results_set.add(blob.metadata.get('id'))
    
    return results_set

def generate_intersecting_sitemap_df(bucket_name:str,base_path:str,sitemap:pd.DataFrame, identifier:str):
    """
    Generates a DataFrame containing only the SKUs that exist in both a given sitemap 
    and the scraped products from a Google Cloud Storage (GCS) bucket.

    This function retrieves product folder names from the specified GCS bucket path, 
    extracts their SKUs, and compares them against the SKUs present in the provided sitemap. 
    It then filters the sitemap to only include SKUs that are present in both sources.

    Args:
        bucket_name (str): The name of the GCS bucket.
        base_path (str): The base folder path in the bucket where product folders are stored.
        sitemap (pd.DataFrame): A DataFrame containing the sitemap entries.
        identifier (str): The identifier for the product folder (e.g., 'SKU').

    Returns:
        pd.DataFrame: A DataFrame containing only the entries from the sitemap where the SKU 
                      exists in the scraped product folders.
    """
    
    # Get all scraped product IDs from the specified GCS bucket path
    scraped_ids = get_all_ids(bucket_name, base_path)

    # Filter the sitemap to only include SKUs that exist in the scraped products
    intersecting_sitemap_df = sitemap[sitemap[identifier].isin(scraped_ids)].copy().reset_index(drop=True)
    
    return intersecting_sitemap_df

def get_secret(secret_name: str, project_id: str) -> dict:
    """
    Retrieve a secret from GCP Secret Manager and parse it as a dictionary.

    Args:
        secret_name (str): The name of the secret.
        project_id (str): The GCP project ID.

    Returns:
        dict: A dictionary containing the secret's key-value pairs.
    """
    try:
        # Create a Secret Manager client
        client = secretmanager.SecretManagerServiceClient()

        # Build the resource name of the secret
        secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"

        # Fetch the secret
        response = client.access_secret_version(request={"name": secret_path})
        secret_json = response.payload.data.decode("UTF-8")  # Decode secret value

        # Parse the JSON secret
        secret_dict = json.loads(secret_json)

        return secret_dict

    except Exception as e:
        raise RuntimeError(f"Failed to retrieve secret: {e}")

def store_secret(secret_name:str,project_id:str):
    """
    Store a secret from GCP Secret Manager into env variables

    Args:
        secret_name (str): The name of the secret.
        project_id (str): The GCP project ID.

    Returns:
        dict: A dictionary containing the secret's key-value pairs.
    """
    # Fetch the secret
    secrets = get_secret(secret_name, project_id)

    # Set each secret as an environment variable
    for key, value in secrets.items():
        os.environ[key] = value  # Store in environment

    pass


def get_eval_metrics(intermediate_filepath:str, final_path:str) -> Tuple[pd.DataFrame, dict]:
    """
    This function calculates the precision and recall of the final mapping results.

    Args:
        - intermediate_filepath (str): The path to the intermediate CSV file.
        - final_path (str): The path to the final CSV file.
    Returns:
        - final_df (pd.DataFrame): The final DataFrame containing the mapping results.
    """

    final_df = pd.read_csv(intermediate_filepath)
    final_df['MATCHED_SKU'] = safe_round_and_convert(final_df['MATCHED_SKU'])
    final_df['SKU_NUMBER'] = safe_round_and_convert(final_df['SKU_NUMBER'])
    final_df['IS_CORRECT'] = final_df['MATCHED_SKU']==final_df['SKU_NUMBER']
    final_df_precision = final_df['IS_CORRECT'].sum()/len(final_df[final_df['MATCH_EXISTS']==True])
    final_df_recall = final_df['IS_CORRECT'].sum()/len(final_df[final_df['MATCH_EXISTS'].isna()==False])
    total_skus_run = len(final_df[final_df["MATCH_EXISTS"].isna()==False])
    total_llm_identified_matches = len(final_df[final_df["MATCH_EXISTS"]==True])
    total_correct_matches = final_df["IS_CORRECT"].sum()

    print(f'Precision: {final_df_precision}')
    print(f'Recall: {final_df_recall}')
    print(f'Total SKUs run: {total_skus_run}')
    print(f'Total LLM identified matches: {total_llm_identified_matches}')
    print(f'Total correct matches: {total_correct_matches}')


    metrics = {
        "Precision": final_df_precision,
        "Recall": final_df_recall,
        "Total SKUs run": total_skus_run,
        "Total LLM identified matches": total_llm_identified_matches,
        "Total correct matches": total_correct_matches
    }

    final_df.to_csv(final_path,index=False)

    return final_df, metrics



def transform_to_inference_df(
    source_type: str,
    test_set_df: pd.DataFrame,
    subattribute_df: pd.DataFrame
    ) -> pd.DataFrame:
    """ 
        Transform the test set dataframe to the inference dataframe based on the source type.

    Args:
        - source_type (str): The type of source data ('vms' or 'jde').
        - test_set_df (pd.DataFrame): The test set dataframe to transform.
    Returns:
        - inference_df (pd.DataFrame): The transformed inference dataframe.
    """
    # Data source configuration
    if source_type == 'vms':
        inference_df = test_set_df
        inference_df['SOURCE_SKU'] = test_set_df['PROVIDER_SKU'] + '-' + test_set_df['PROVIDER_UPC']
        inference_df['SOURCE_NAME'] = test_set_df['PROVIDER_PRODUCT_NAME']
    elif source_type == 'jde':
        inference_df = test_set_df
        inference_df['SOURCE_SKU'] = test_set_df['SOURCE_SKU'] 
        inference_df['SOURCE_NAME'] = test_set_df['ITEM_DESCRIPTION']
    else:
        raise ValueError('Invalid source type')
    
    if 'SIZE' in subattribute_df.columns:
        subattribute_df['SOURCE_SIZE'] = subattribute_df['SIZE']
        inference_df = left_join_with_single_column(inference_df, subattribute_df, join_key="SOURCE_SKU", right_column="SOURCE_SIZE")
    if 'UNIT_OF_MEASUREMENT' in subattribute_df.columns:
        subattribute_df['SOURCE_UNIT_OF_MEASUREMENT'] = subattribute_df['UNIT_OF_MEASUREMENT']
        inference_df = left_join_with_single_column(inference_df, subattribute_df, join_key="SOURCE_SKU", right_column="SOURCE_UNIT_OF_MEASUREMENT")
    if 'MANUFACTURER_NAME' in subattribute_df.columns:
        subattribute_df['SOURCE_MANUFACTURER_NAME'] = subattribute_df['MANUFACTURER_NAME']
        inference_df = left_join_with_single_column(inference_df, subattribute_df, join_key="SOURCE_SKU", right_column="SOURCE_MANUFACTURER_NAME")
    if 'FLAVOR' in subattribute_df.columns:
        subattribute_df['SOURCE_FLAVOR'] = subattribute_df['FLAVOR']
        inference_df = left_join_with_single_column(inference_df, subattribute_df, join_key="SOURCE_SKU", right_column="SOURCE_FLAVOR")
    if 'BRAND' in subattribute_df.columns:
        subattribute_df['SOURCE_BRAND'] = subattribute_df['BRAND']
        inference_df = left_join_with_single_column(inference_df, subattribute_df, join_key="SOURCE_SKU", right_column="SOURCE_BRAND")
    if 'PACKAGING_CODE' in subattribute_df.columns:
        subattribute_df['SOURCE_PACKAGE_CODE'] = subattribute_df['PACKAGING_CODE']
        inference_df = left_join_with_single_column(inference_df, subattribute_df, join_key="SOURCE_SKU", right_column="SOURCE_PACKAGE_CODE")
    if 'PACKAGING_QTY' in subattribute_df.columns:
        subattribute_df['SOURCE_PACKAGE_QTY'] = subattribute_df['PACKAGING_QTY']
        inference_df = left_join_with_single_column(inference_df, subattribute_df, join_key="SOURCE_SKU", right_column="SOURCE_PACKAGE_QTY")
        # Manual overwrite with source information
        if source_type == 'jde':
            inference_df['SOURCE_PACKAGE_CODE'] = test_set_df['PACKAGE_CODE']
            inference_df['SOURCE_PACKAGE_QTY'] = test_set_df['PACKAGE_QTY']

    # Drop any duplicates to preserve unique granularity
    inference_df_post_duplicate_drop = (
        inference_df
        .drop_duplicates(subset="SOURCE_SKU", keep="first")
        .reset_index(drop=True)
    )
    
    # Log the lengths of the DataFrames
    logger = logging.getLogger("app_logger")
    logger.info(f"Post-duplicate drop length: {len(inference_df_post_duplicate_drop)}")
    logger.info(f"Pre-duplicate drop length: {len(inference_df)}")

    # Add source type column
    inference_df_post_duplicate_drop = inference_df_post_duplicate_drop.reset_index(drop=True)
    inference_df_post_duplicate_drop['SOURCE'] = source_type

    return inference_df_post_duplicate_drop

def transform_pma_to_sku_grain(pma_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a PMA DataFrame to SKU grain by selecting SKU-related columns, 
    removing duplicates, and renaming columns for consistency.

    Args:
        pma_df (pd.DataFrame): 
            A DataFrame containing at least 'SKU_NUMBER' and 'PRODUCT_LONG_NAME' columns.

    Returns:
        pd.DataFrame: 
            A transformed DataFrame with unique SKU entries and renamed columns:
            - 'SKU_NUMBER' (retained)
            - 'PRODUCT_LONG_NAME' (retained)
            - 'SOURCE_SKU' (duplicate of 'SKU_NUMBER')
            - 'SOURCE_NAME' (duplicate of 'PRODUCT_LONG_NAME')

    Example:
        >>> data = {
        ...     'SKU_NUMBER': [101, 102, 101, 103],
        ...     'PRODUCT_LONG_NAME': ['Product A', 'Product B', 'Product A', 'Product C']
        ... }
        >>> pma_df = pd.DataFrame(data)
        >>> transformed_df = transform_pma_to_sku_grain(pma_df)
        >>> print(transformed_df)
           SKU_NUMBER PRODUCT_LONG_NAME  SOURCE_SKU  SOURCE_NAME
        0        101        Product A        101  Product A
        1        102        Product B        102  Product B
        2        103        Product C        103  Product C
    """
    # Select relevant columns and remove duplicates
    pma_df = pma_df.drop_duplicates().reset_index(drop=True)

    # Create additional reference columns
    pma_df['SOURCE_SKU'] = pma_df['SKU_NUMBER']
    pma_df['SOURCE_NAME'] = pma_df['PRODUCT_LONG_NAME']

    return pma_df


def safe_round_and_convert(series: pd.Series) -> pd.Series:
    """
    Safely rounds numeric values in a Pandas Series to the nearest integer
    and converts them to strings, while keeping non-numeric values unchanged.

    Args:
        series (pd.Series): The input Pandas Series.

    Returns:
        pd.Series: A Series where numeric values are rounded and converted to strings,
                   while non-numeric values remain unchanged.
    """
    def process_value(x):
        if pd.api.types.is_numeric_dtype(type(x)):  # Check if x is a number
            if pd.notna(x):  # Ensure x is not NaN
                return str(int(round(x)))  # Round and convert to string
            return ""  # Handle NaNs safely
        return str(x)  # Keep strings unchanged

    return series.apply(process_value)


def left_join_with_single_column(
    left_df: pd.DataFrame, right_df: pd.DataFrame, join_key: str, right_column: str
) -> pd.DataFrame:
    """
    Performs a left join between two DataFrames while keeping all columns from the left DataFrame 
    and only a single specified column from the right DataFrame.

    Args:
        left_df (pd.DataFrame): The left DataFrame (all columns retained).
        right_df (pd.DataFrame): The right DataFrame (only one column retained).
        join_key (str): The column name to join on.
        right_column (str): The column from the right DataFrame to retain.

    Returns:
        pd.DataFrame: The left DataFrame with an additional column from the right DataFrame.
    
    Example:
        >>> df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['Alice', 'Bob', 'Charlie']})
        >>> df2 = pd.DataFrame({'ID': [1, 2, 3], 'Score': [85, 90, 78]})
        >>> merged_df = left_join_with_single_column(df1, df2, join_key="ID", right_column="Score")
        >>> print(merged_df)
           ID     Name  Score
        0   1   Alice     85
        1   2     Bob     90
        2   3  Charlie     78
    """
    return left_df.merge(
        right_df[[join_key, right_column]],  # Select only the join column + desired right column
        on=join_key,
        how="left"
    )

def convert_numpy(obj:Any) -> Any:
    """ 
    Convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    

def clean_text(text: str) -> str:
    """ 
    Function to clean text data.
    Args:
        text: str: text data to be cleaned
    Returns:
        text: str: cleaned
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()                     # lowercase and trim
    text = re.sub(r'\s+', ' ', text)                # normalize multiple spaces
    text = re.sub(r'[^\w\s./%-]', '', text)         # remove special characters (customizable)
    return text

def setup_logger(
    name: str = "app_logger",
    log_file: str = "logging/app.log",
    level: int = logging.INFO,
    max_bytes: int = 10_000_000,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up a logger that writes to both console and a rotating log file.

    Args:
        name (str): Name of the logger.
        log_file (str): File path to log to.
        level (int): Logging level (e.g., logging.INFO).
        max_bytes (int): Max size in bytes before rotating.
        backup_count (int): Number of backup files to keep.

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs("logging", exist_ok=True)  # Ensure the logging directory exists
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Avoid duplicate handlers on multiple calls
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger



def upload_log_to_gcs(bucket_name: str, destination_blob_path: str, log_file: str) -> None:
    """
    Uploads the log file to a GCP bucket.

    Args:
        bucket_name (str): Name of the GCS bucket.
        destination_blob_path (str): Path in the bucket where the log should go.
        log_file (str): Local path to the log file.
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_path)
        blob.upload_from_filename(log_file)
        print(f"✅ Uploaded log to gs://{bucket_name}/{destination_blob_path}")
    except Exception as e:
        print(f"❌ Failed to upload log: {e}")



def retrieve_secret(secret_name: str, project_id: str) -> dict:
    """
    Retrieve a secret from GCP Secret Manager and parse it as a dictionary, loading it in as environment variables.

    Args:
        secret_name (str): The name of the secret.
        project_id (str): The GCP project ID.

    Returns:
        dict: A dictionary containing the secret's key-value pairs.
    """
    # Get logger
    logger = logging.getLogger(__name__)

    # Create a Secret Manager client
    client = secretmanager.SecretManagerServiceClient()

    # Build the resource name of the secret
    secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"

    # Fetch the secret
    response = client.access_secret_version(request={"name": secret_path})
    secret_json = response.payload.data.decode("UTF-8")  # Decode secret value

    # disable HuggingFace tokenizers’ parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Parse the JSON secret
    secret_dict = json.loads(secret_json)

    # Set each secret as an environment variable
    for key, value in secret_dict.items():
        os.environ[key] = value  # Store in environment

    return secret_dict
