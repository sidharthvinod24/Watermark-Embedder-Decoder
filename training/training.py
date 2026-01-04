import torch
import os
from torch import nn
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.UNET import UNet
from models.RESNET import ResNetExtractor
from data.ImageLoader import MyImageFolder
from PIL import Image
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import time


# General CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-4
ADAM_BETA = 0.5
NUM_EPOCHS = 50

DATA_DIR = f"{os.getcwd()}/data"
train_dir = os.path.join(DATA_DIR, 'coco_train')
val_dir = os.path.join(DATA_DIR, "coco_test")

checkpoints = os.makedirs("checkpoints", exist_ok=True)
CHECKPOINT_DIR = f"{os.getcwd()}/checkpoints"

SECRET_IMG_DIR = f"{os.getcwd()}/secret"
data_pics = os.makedirs("datapics", exist_ok=True)
DATAPICS_DIR = f"{os.getcwd()}/datapics"

# Save the graphs
os.makedirs("graphs", exist_ok=True)
GRAPH_DIR = f"{os.getcwd()}/graphs"

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(str(net))
    print(f"Total Number of parameters is {num_params}")


def plot_curve(train, val, title, ylabel, filename):
    epochs = range(1, len(train) + 1)
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.plot(epochs, train, label="Training")
    plt.plot(epochs, val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(f"{GRAPH_DIR}/{filename}")
    plt.close()


def save_pics(batch_size, cover, secret, unet_output, restnet_output_nw, restnet_output, epoch, i, dir=None):
    os.makedirs(f"{DATAPICS_DIR}/{dir}", exist_ok=True)
    save_path = f"{DATAPICS_DIR}/{dir}"

    def process(tensor):
        img = tensor.detach().cpu().clone()
        # Ensure it is 3 channels (Handle grayscale if necessary, though usually RGB)
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)
        return img

    # Stack images vertically in this specific order for easy comparison
    img_list = [
        process(cover),               # Row 1: Original Cover
        process(secret),              # Row 2: Original Secret
        process(unet_output),         # Row 3: Container (Unet Output)
        process(restnet_output_nw),   # Row 4: Secret extracted from non-watermarked
        process(restnet_output),      # Row 5: Secret extracted from watermarked
    ]
    showResult = torch.cat(img_list, dim=0)

    resultImgName = '%s/ResultPics_epoch%03d_batch%04d.png' % (
        save_path, epoch, i)

    # nrow=this_batch_size means each row in the grid will contain one "type" of image
    vutils.save_image(showResult, resultImgName,
                      nrow=batch_size, padding=1, normalize=False)


def train(train_loader, epoch, unetModel, resnetModel):
    unetModel.train()
    resnetModel.train()

    loader = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()])
    clean_img_template = loader(Image.open(f"{SECRET_IMG_DIR}/clean.png"))
    secret_img_template = loader(Image.open(f"{SECRET_IMG_DIR}/cat.png"))

    FidelityLoss, WatermarkLoss, NonWatermarkLoss, JointLoss = [], [], [], []

    start_time = time.time()
    for i, data in enumerate(train_loader):
        batch_size = data.shape[0]
        cover_img = data

        # Create fresh copies for each iteration (avoid in-place modifications)
        secret_img = secret_img_template.clone().repeat(batch_size, 1, 1, 1)
        secret_img = secret_img[0:batch_size, :, :, :]

        clean_img = clean_img_template.clone().repeat(batch_size, 1, 1, 1)
        clean_img = clean_img[0:batch_size, :, :, :]

        cover_img = data.to(DEVICE)
        secret_img = secret_img.to(DEVICE)
        clean_img = clean_img.to(DEVICE)

        # PHASE 1 Training  Unet + ResNet

        unetModel.zero_grad()
        resnetModel.zero_grad()

        # COMBINATION OF the Original Image + Watermark Image
        concat_img = torch.concat((cover_img, secret_img), dim=1)

        # Output of Unet
        unet_output_img = unetModel(cover_img, concat_img)

        # Output of RESNET for UNET OUTPUT
        resnet_output_img = resnetModel(unet_output_img)

        # Output of RESNET for Non Watermark -> Cover IMG
        resnet_output_img_nw = resnetModel(cover_img)

        # LOSSES
        # Fidelity Loss
        fidelity_loss = L1_Loss(unet_output_img, cover_img)
        watermark_loss = L2_Loss(resnet_output_img, secret_img)

        # Non-Watermark Loss - extract something FAR from secret for clean images
        margin = 0.01
        non_watermark_loss = torch.relu(margin - L2_Loss(resnet_output_img_nw, secret_img))

        joint_loss = 1 * fidelity_loss + 1 * watermark_loss + 1 * non_watermark_loss

        FidelityLoss.append(fidelity_loss.item())
        WatermarkLoss.append(watermark_loss.item())
        NonWatermarkLoss.append(non_watermark_loss.item())
        JointLoss.append(joint_loss.item())

        joint_loss.backward()
        optimizerUnet.step()
        optimizerResNet.step()

        if i % 10 == 0:
            print(f"""
        ##################################
        EPOCH: {epoch}
        ##################################
        BATCH: {i}/{len(train_loader)}
        Fidelity Loss: {fidelity_loss:.6f}
        Watermark Loss: {watermark_loss:.6f}
        Non Watermark Loss: {non_watermark_loss:.6f}
        Joint Loss: {joint_loss:.6f}
        ##################################
        """)
        if i % 50 == 0:
            save_pics(
                batch_size,
                cover_img,
                secret_img,
                unet_output_img,
                resnet_output_img_nw,
                resnet_output_img,
                epoch,
                i,
                "training"
            )
    
    # Return AVERAGED losses for the epoch
    return (
        np.mean(FidelityLoss),
        np.mean(WatermarkLoss),
        np.mean(NonWatermarkLoss),
        np.mean(JointLoss)
    )


def validate(val_loader, epoch, unetModel, resnetModel):
    print("Validation Begins")
    unetModel.eval()
    resnetModel.eval()

    FidelityLoss, WatermarkLoss, NonWatermarkLoss, JointLoss = [], [], [], []

    with torch.no_grad():
        loader = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()])
        clean_img_template = loader(Image.open(f"{SECRET_IMG_DIR}/clean.png"))
        secret_img_template = loader(Image.open(f"{SECRET_IMG_DIR}/cat.png"))

        for i, data in enumerate(val_loader):
            batch_size = data.shape[0]
            cover_img = data

            # Create fresh copies for each iteration (avoid in-place modifications)
            secret_img = secret_img_template.clone().repeat(batch_size, 1, 1, 1)
            secret_img = secret_img[0:batch_size, :, :, :]

            clean_img = clean_img_template.clone().repeat(batch_size, 1, 1, 1)
            clean_img = clean_img[0:batch_size, :, :, :]

            cover_img = data.to(DEVICE)
            secret_img = secret_img.to(DEVICE)
            clean_img = clean_img.to(DEVICE)

            # COMBINATION OF the Original Image + Watermark Image
            concat_img = torch.concat((cover_img, secret_img), dim=1)

            # Output of Unet
            unet_output_img = unetModel(cover_img, concat_img)

            # Output of RESNET for UNET OUTPUT
            resnet_output_img = resnetModel(unet_output_img)

            # Output of RESNET for Non Watermark -> Cover IMG
            resnet_output_img_nw = resnetModel(cover_img)

            # LOSSES
            # Fidelity Loss
            fidelity_loss = L1_Loss(unet_output_img, cover_img)

            # Watermark Loss - extract secret from watermarked images
            watermark_loss = L2_Loss(resnet_output_img, secret_img)

            # Non-Watermark Loss - extract something FAR from secret for clean images
            margin = 0.01
            non_watermark_loss = torch.relu(margin - L2_Loss(resnet_output_img_nw, secret_img))

            joint_loss = 1 * fidelity_loss + 1 * watermark_loss + 1 * non_watermark_loss

            FidelityLoss.append(fidelity_loss.item())
            WatermarkLoss.append(watermark_loss.item())
            NonWatermarkLoss.append(non_watermark_loss.item())
            JointLoss.append(joint_loss.item())

            if i % 50 == 0:
                save_pics(
                    batch_size,
                    cover_img,
                    secret_img,
                    unet_output_img,
                    resnet_output_img_nw,
                    resnet_output_img,
                    epoch,
                    i,
                    "validation"
                )
        
        # Calculate averages
        val_fidelity_avg = np.mean(FidelityLoss)
        val_watermark_avg = np.mean(WatermarkLoss)
        val_nonwatermark_avg = np.mean(NonWatermarkLoss)
        val_joint_avg = np.mean(JointLoss)
        
        print(f"""
        ##################################
        EPOCH: {epoch}
        ##################################
        VALIDATION COMPLETE
        Fidelity Loss: {val_fidelity_avg:.6f}
        Watermark Loss: {val_watermark_avg:.6f}
        Non Watermark Loss: {val_nonwatermark_avg:.6f}
        Joint Loss: {val_joint_avg:.6f}
        ##################################
        """)
        print("Validation END")
        
        # Return AVERAGED losses for the epoch
        return (
            val_fidelity_avg,
            val_watermark_avg,
            val_nonwatermark_avg,
            val_joint_avg
        )


def main():
    global L1_Loss, L2_Loss, optimizerResNet, optimizerUnet
    train_dataset = MyImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    )

    val_dataset = MyImageFolder(
        val_dir,
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    )

    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=16,
                            shuffle=False, num_workers=8)

    unetModel = UNet(in_channels=6, out_channels=3, alpha=0.15).to(DEVICE)
    resnetModel = ResNetExtractor(image_channels=3).to(DEVICE)

    for param in unetModel.parameters():
        param.requires_grad = True

    for param in resnetModel.parameters():
        param.requires_grad = True

    optimizerUnet = optim.Adam(
        unetModel.parameters(), lr=LR, betas=(ADAM_BETA, 0.999))
    optimizerResNet = optim.Adam(
        resnetModel.parameters(), lr=LR, betas=(ADAM_BETA, 0.999))

    schedularUnet = ReduceLROnPlateau(
        optimizerUnet, mode="min", factor=0.2, patience=5)
    schedularResNet = ReduceLROnPlateau(
        optimizerResNet, mode="min", factor=0.2, patience=5)

    # Two different losses needed
    # For FIDELITY: we should use L1 loss since image similarity is the most important measure
    L1_Loss = nn.L1Loss()
    L2_Loss = nn.MSELoss()

    # Lists to store per-epoch averages for plotting
    train_fidelity_history = []
    train_watermark_history = []
    train_nonwatermark_history = []
    train_joint_history = []
    
    val_fidelity_history = []
    val_watermark_history = []
    val_nonwatermark_history = []
    val_joint_history = []

    best_score = float("inf")
    for epoch in range(NUM_EPOCHS):
        # Get averaged losses for this epoch
        train_fidelity, train_watermark, train_nonwatermark, train_joint = train(
            train_loader, epoch, unetModel, resnetModel)
        val_fidelity, val_watermark, val_nonwatermark, val_joint = validate(
            val_loader, epoch, unetModel, resnetModel)

        # Append to history
        train_fidelity_history.append(train_fidelity)
        train_watermark_history.append(train_watermark)
        train_nonwatermark_history.append(train_nonwatermark)
        train_joint_history.append(train_joint)
        
        val_fidelity_history.append(val_fidelity)
        val_watermark_history.append(val_watermark)
        val_nonwatermark_history.append(val_nonwatermark)
        val_joint_history.append(val_joint)

        # Plot curves with accumulated history
        plot_curve(train_fidelity_history, val_fidelity_history, 
                   "Fidelity Loss", "Loss", "Fidelity_Loss.png")
        plot_curve(train_watermark_history, val_watermark_history, 
                   "Watermark Loss", "Loss", "Watermark_Loss.png")
        plot_curve(train_nonwatermark_history, val_nonwatermark_history, 
                   "Non Watermark Loss", "Loss", "NonWatermark_Loss.png")
        plot_curve(train_joint_history, val_joint_history, 
                   "Joint Loss", "Loss", "Joint_Loss.png")

        # Update schedulers
        schedularUnet.step(val_fidelity)
        schedularResNet.step(val_watermark)

        # Save best model
        if val_joint < best_score:
            best_score = val_joint
            torch.save(unetModel.state_dict(),
                       f"{CHECKPOINT_DIR}/UNET_epoch_{epoch},JointLoss={val_joint:.6f}.pth")
            torch.save(resnetModel.state_dict(),
                       f"{CHECKPOINT_DIR}/RESNET_epoch_{epoch},JointLoss={val_joint:.6f}.pth")
            print(f"✓ New best model saved at epoch {epoch} with joint loss {val_joint:.6f}")


if __name__ == "__main__":
    main()