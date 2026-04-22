import os
import zipfile
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Root directory where your DHS folders are stored
root_dir = os.getenv("DATABASE_PATH")

for country_folder in os.listdir(root_dir):
    country_path = os.path.join(root_dir, country_folder)

    if os.path.isdir(country_path):
        # Create new top-level folder for unzipped files
        unzipped_country_folder = os.path.join(root_dir, f"{country_folder}_unzipped")
        os.makedirs(unzipped_country_folder, exist_ok=True)

        print(f"\nProcessing: {country_folder} → {unzipped_country_folder}")

        for file in os.listdir(country_path):
            if file.endswith(".zip"):
                zip_path = os.path.join(country_path, file)

                # Create a folder for each zip file inside the unzipped country folder
                extract_folder = os.path.join(unzipped_country_folder, file.replace(".zip", ""))
                os.makedirs(extract_folder, exist_ok=True)

                print(f"  Unzipping: {file} → {extract_folder}")

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_folder)

print("\n✅ All files extracted into *_unzipped folders.")