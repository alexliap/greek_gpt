import argparse
import logging
import os

from azure.storage.blob import BlobServiceClient

logging.basicConfig(level=logging.INFO)

account_name = ""
blob_key = ""

account_url = f"https://{account_name}.blob.core.windows.net"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload local folder to Azure")
    parser.add_argument("--local_folder", required=True)
    parser.add_argument("--container", required=True)
    args = parser.parse_args()

    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient(account_url, credential=blob_key)

    container_client = blob_service_client.get_container_client(
        container=args.container
    )

    for root, dirs, files in os.walk(args.local_folder):
        for file in files:
            file_path = os.path.join(root, file)
            logging.info(
                f"Uploading file: {file_path} to Azure Storage Container: {args.container} ..."
            )
            if ".DS_Store" not in file_path:
                with open(file=file_path, mode="rb") as data:
                    try:
                        container_client.upload_blob(
                            name=file_path, data=data, overwrite=True
                        )
                        logging.info(f"File: {file_path} was uploaded successfully.")
                    except Exception as e:
                        message = f"Upload for file {file_path} failed."
                        logging.error(f"{message}: {e}")
                        raise Exception(f"{message}: {e}")
