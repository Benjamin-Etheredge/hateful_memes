import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms
import os
import dlib
from icecream import ic
import pytorch_lightning as pl

class FairFaceModel():
    def __init__(self, device, max_length=5):
        self.max_length = max_length
        self.device = device

        cnn_face_detector_fp = os.path.join('.', 'master', 'dlib_models', 'mmod_human_face_detector.dat')
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detector_fp)
        sp_fp = os.path.join('.', 'master', 'dlib_models','shape_predictor_5_face_landmarks.dat')
        self.sp = dlib.shape_predictor(sp_fp)

        self.model_fair_4 = torchvision.models.resnet34(pretrained=True)
        self.model_fair_4.fc = nn.Linear(self.model_fair_4.fc.in_features, 18)
        model_fair_4_fp = os.path.join('.', 'master', 'fair_face_models', 'res34_fair_align_multi_4_20190809.pt')
        self.model_fair_4.load_state_dict(
            torch.load(model_fair_4_fp, map_location=self.device)
        )
        self.model_fair_4 = self.model_fair_4.to(self.device)
        self.model_fair_4.eval()

    def detect_faces(self, image, default_max_size=800, size=300, padding=0.25):
        old_height, old_width, _ = image.shape

        if old_width > old_height:
            new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
        else:
            new_width, new_height =  int(default_max_size * old_width / old_height), default_max_size
        image = dlib.resize_image(image, rows=new_height, cols=new_width)

        dets = self.cnn_face_detector(image, 1)
        num_faces = len(dets)

        images = []
        # Find the 5 face landmarks we need to do the alignment.
        if num_faces > 0:
            faces = dlib.full_object_detections()
            for detection in dets:
                rect = detection.rect
                faces.append(self.sp(image, rect))
            images = dlib.get_face_chips(image, faces, size=size, padding = padding)
        return images

    def predict_age_gender_race(self, image):
        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = trans(image)
        image = image.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
        image = image.to(self.device)

        pred_weights = self.model_fair_4(image)
        pred_weights = torch.squeeze(pred_weights)
        return pred_weights

    def forward(self, image):
        '''
        Returns a zero-padded tensor of prediction weights for each face in the image,
        up to the maximum number of faces specified by max_length.
        output: shape (max_length, 18)
        '''
        output = torch.zeros(self.max_length, 18) # 18 = length of pred_weights
        
        faces = self.detect_faces(image)
        num_faces = len(faces)
        
        if num_faces > self.max_length:
            faces = faces[:self.max_length] 
        
        for idx, face in enumerate(faces):
            output[idx] = self.predict_age_gender_race(face)

        return output

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_fp = os.path.join('.', 'master', 'test', 'race_Latino.jpg') 
    image = dlib.load_rgb_image(image_fp)

    model = FairFaceModel(device=device)

    outs = model.forward(image)
    ic(outs)

if __name__ == "__main__":
    pl.seed_everything(42)
    main()