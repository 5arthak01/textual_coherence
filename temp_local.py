from prepare_data import *
import os
import requests
import tarfile

print("Downloading WikiCoherence Corpus...")
download_file_from_google_drive(
    "1Il9mZt111kRAkzy8IXirp_7NUZ2dYAs2", "WikiCoherence.tar.gz"
)
tar = tarfile.open("WikiCoherence.tar.gz", "r:gz")
tar.extractall()
tar.close()
os.remove("WikiCoherence.tar.gz")
