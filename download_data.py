import gdown
import os

# URL of your Google Drive folder
folder_url = "https://drive.google.com/drive/folders/12W72ZOq6MM5PmrTBWWBvOFvGWylVCNFQ?usp=drive_link"

output_dir = "."

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Convert folder link into the "download all" format
# gdown needs the folder ID only
folder_id = folder_url.split("/")[-1].split("?")[0]

print("Downloading all files from Google Drive folder...")
gdown.download_folder(id=folder_id, output=output_dir, quiet=False, use_cookies=False)
print("✅ Download complete!")
