import torch
from torchvision import transforms
from PIL import Image
import argparse
import segmentation_models_pytorch as smp
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
def load_model(checkpoint_path):
    model = smp.UnetPlusPlus(
        encoder_name="timm-efficientnet-b8",
        encoder_weights="advprop",
        in_channels=3,
        classes=3
    )
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    return model
def predict(model, image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    return output
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference script for image segmentation')
    parser.add_argument('--image_path', type=str, help='Path to input image', required=True)
    args = parser.parse_args()
    model = load_model('model.pth')
    output = predict(model, args.image_path)
    output_image = transforms.ToPILImage()(output.squeeze(0))
    output_image.save('output_segmented_image.png')
    print("Segmented image saved as output_segmented_image.png")


