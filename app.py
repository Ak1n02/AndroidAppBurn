import os
import cv2
import timm
import torch

from flask import Flask, request, render_template, url_for

import torchvision.transforms as transforms

import download_model as download_model

app = Flask(__name__)

UPLOAD_FOLDER = 'WebApp/static/uploads'
RESULT_FOLDER = 'WebApp/static/results'
MODEL_PATH = "regnet94valacc.pth"
class_names = {0: "First Degree Burn", 1: "Second Degree Burn", 2: "Third Degree Burn"}

def setup_folders():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['RESULT_FOLDER'] = RESULT_FOLDER

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('regnety_080', pretrained=True)
    model.reset_classifier(num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cuda')))
    model.eval()
    return model

def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def handle_post_request(file, model, transform):
    from removal import process_and_save
    from preprocess_ak1n import process_image

    if file.filename == '':
        return "No file selected", 400

    upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(upload_path)

    removed_path = os.path.join(RESULT_FOLDER, 'removed_' + file.filename)
    process_and_save(upload_path, removed_path)

    preprocessed_image = process_image(removed_path)
    if preprocessed_image is None:
        return "Error during preprocessing", 500

    if preprocessed_image.shape[2] == 3:
        preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
    elif preprocessed_image.shape[2] == 4:
        preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGRA2RGB)

    input_tensor = transform(preprocessed_image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        burn_degree = class_names.get(predicted.item(), "Unknown")

    result_filename = 'final_' + file.filename
    result_image_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_image_path, cv2.cvtColor(preprocessed_image, cv2.COLOR_RGB2BGR))

    result_image_url = url_for('static', filename=f'results/{result_filename}')
    return render_template('result.html', burn_degree=burn_degree, result_image=result_image_url)

@app.route('/', methods=['GET', 'POST'])
def index():
    model = load_model()
    transform = get_transform()
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image file part", 400
        file = request.files['image']
        return handle_post_request(file, model, transform)
    return render_template('index.html')

def main():
    setup_folders()
    download_model.download()

    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

if __name__ == '__main__':
    main()