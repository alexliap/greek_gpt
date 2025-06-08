import argparse
import logging
import os

from azure.storage.blob import BlobServiceClient

logging.basicConfig(level=logging.INFO)

account_name = ""
blob_key = ""

account_url = f"https://{account_name}.blob.core.windows.net"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download folder from Azure")
    parser.add_argument("--container", required=True)
    parser.add_argument("--sub_dir", required=True)
    args = parser.parse_args()

    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient(account_url, credential=blob_key)

    container_client = blob_service_client.get_container_client(
        container=args.container
    )

    os.makedirs(args.sub_dir, exist_ok=True)

    for blob_name in container_client.list_blob_names():
        if args.sub_dir in blob_name:
            blob_client = container_client.get_blob_client(blob_name)
            logging.info(f"Downloading {blob_name} ...")
            with open(file=os.path.join("./", blob_name), mode="wb") as sample_blob:
                try:
                    download_stream = blob_client.download_blob()
                    sample_blob.write(download_stream.readall())
                    logging.info(f"Downloading succeded for file: {blob_name}.")
                except Exception as e:
                    message = f"Downloading failed for file {blob_name}."
                    logging.error(f"{message}: {e}")
                    raise Exception(f"{message}: {e}")
