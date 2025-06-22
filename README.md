📥 Dataset
You can find the training and test datasets on https://www.kaggle.com/competitions/dog-breed-identification/data
Download the files (train, test, and labels.csv)

Place them in an accessible local directory

Replace the hardcoded paths in the script with your local paths as needed

Example:
python
Copy
Edit
# Original path (you should change this to match your setup)
directory=r'c:\Users\aryan\Downloads\dog\train'

# Update to your own local path like:
directory='/path/to/your/downloaded/train'
🔄 This flexibility ensures the script runs on any system — just customize the directory argument in flow_from_dataframe.
