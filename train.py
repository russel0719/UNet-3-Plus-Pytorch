from datetime import datetime
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import torch
from torch import nn, optim
import torch.backends.cudnn as cudnn

from data_generator import DataGenerator, get_data_loader
from data_preparation.verify_data import verify_data
from utils.general_utils import create_directory, join_paths, set_gpus, get_gpus_count
from models.model import prepare_model
from losses.loss import dice_coef
from losses.unet_loss import unet3p_hybrid_loss

def create_training_folders(cfg: DictConfig):
    """
    Create directories to store Model CheckPoint and TensorBoard logs.
    """
    create_directory(
        join_paths(
            cfg.WORK_DIR,
            cfg.CALLBACKS.MODEL_CHECKPOINT.PATH
        )
    )
    create_directory(
        join_paths(
            cfg.WORK_DIR,
            cfg.CALLBACKS.LOGGING.PATH
        )
    )


def train(cfg: DictConfig):
    """
    Training method
    """
    print("Verifying data ...")
    verify_data(cfg)

    if cfg.MODEL.TYPE == "unet3plus_deepsup_cgm":
        raise ValueError(
            "UNet3+ with Deep Supervision and Classification Guided Module"
            "\nModel exist but training script is not supported for this variant"
            "please choose other variants from config file"
        )

    if cfg.USE_MULTI_GPUS.VALUE:
        # change number of visible gpus for training
        set_gpus(cfg.USE_MULTI_GPUS.GPU_IDS)
        # change batch size according to available gpus
        cfg.HYPER_PARAMETERS.BATCH_SIZE = \
            cfg.HYPER_PARAMETERS.BATCH_SIZE * get_gpus_count()

    # create folders to store training checkpoints and logs
    create_training_folders(cfg)

    # data generators
    train_loader = get_data_loader(cfg, mode="TRAIN")
    val_loader = get_data_loader(cfg, mode="VAL")

    # set device with gpu id
    gpu_id = cfg.GPU_ID
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    # create model
    model = prepare_model(cfg, training=True).to(device)
    
    if cfg.USE_MULTI_GPUS.VALUE:
        model = nn.DataParallel(model)

    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=cfg.HYPER_PARAMETERS.LEARNING_RATE)
    criterion = unet3p_hybrid_loss

    # EarlyStopping, ModelCheckpoint and CSVLogger callbacks
    checkpoint_path = join_paths(
        cfg.WORK_DIR,
        cfg.CALLBACKS.MODEL_CHECKPOINT.PATH,
        f"{cfg.MODEL.WEIGHTS_FILE_NAME}.pt"
    )
    print("Weights path\n" + checkpoint_path)

    csv_log_path = join_paths(
        cfg.WORK_DIR,
        cfg.CALLBACKS.CSV_LOGGER.PATH,
        f"training_logs_{cfg.MODEL.TYPE}.csv"
    )
    print("Logs path\n" + csv_log_path)

    txt_log_path = join_paths(
        cfg.WORK_DIR,
        cfg.CALLBACKS.LOGGING.PATH,
        f"training_logs_{cfg.MODEL.TYPE}.txt"
    )
    print("Text Logs path\n" + txt_log_path)

    best_val_score = 0.0

    for epoch in range(cfg.HYPER_PARAMETERS.EPOCHS):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        for batch_images, batch_masks in tqdm(train_loader):
            batch_images, batch_masks = batch_images.to(device), batch_masks.to(device)
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(batch_masks, outputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_images.size(0)
            train_dice += dice_coef(outputs, batch_masks).item() * batch_images.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_dice /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for batch_images, batch_masks in val_loader:
                batch_images, batch_masks = batch_images.to(device), batch_masks.to(device)
                outputs = model(batch_images)
                loss = criterion(outputs, batch_masks)
                val_loss += loss.item() * batch_images.size(0)
                val_dice += dice_coef(outputs, batch_masks).item() * batch_images.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{cfg.HYPER_PARAMETERS.EPOCHS}, "
              f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        with open(txt_log_path, 'w') as log_file:
            log_file.write(f"Epoch {epoch+1}/{cfg.HYPER_PARAMETERS.EPOCHS}, "
                           f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}\n")

        # EarlyStopping and ModelCheckpoint logic
        if val_dice > best_val_score:
            best_val_score = val_dice
            torch.save(model.state_dict(), checkpoint_path)
            print("Saved best model")

        with open(csv_log_path, 'a') as f:
            f.write(f"{epoch+1},{train_loss},{train_dice},{val_loss},{val_dice}\n")

        if val_dice > best_val_score - cfg.CALLBACKS.EARLY_STOPPING.DELTA:
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg.CALLBACKS.EARLY_STOPPING.PATIENCE:
                print("Early stopping")
                break


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Read config file and pass to train method for training
    """
    train(cfg)


if __name__ == "__main__":
    main()
