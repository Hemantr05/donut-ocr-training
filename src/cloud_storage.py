from google.cloud import storage

storage_client = storage.Client.from_service_account_json('../service-account-file.json')

def create_bucket(bucket_name):
    bucket = storage_client.create_bucket(bucket_name)
    print(f"Bucket {bucket.name} created.")
    return bucket_name

def bucket_exists(bucket_name):
    flag = False
    if bucket_name in storage_client.list_buckets():
        flag = True
    return flag

def model_to_bucket(bucket_name, run_name, path_to_model):
    if bucket_exists(bucket_name):
        bucket = storage_client.get_bucket(bucket_name)
    else:
        bucket = create_bucket(bucket_name)

    blob = bucket.blob(run_name)
    blob.upload_from_filename(path_to_model)

    print(f"{run_name} stored at {blob.public_url}.")