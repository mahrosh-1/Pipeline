import requests

url = "http://localhost:8000/predict"

pdf_file_path = r"E:\VSCODE\visionapiflask\inv-ah4vW-1647414055.pdf"
files = {"files": ("filename.pdf", open(pdf_file_path, "rb"), "application/pdf")}

# response = requests.post(url, files=files)
response = requests.post(url, files=files)

if response.status_code == 200:
    # Response is successful, parse the JSON response
    json_response = response.json()
    print(json_response)
else:
    # Response failed, handle the error
    print(f"Error: {response.status_code} - {response.text}")