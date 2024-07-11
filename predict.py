"""
Prediction script used to visualize model output
"""
import os
import hydra
from omegaconf import DictConfig
import torch

from data_generator import get_data_loader
from utils.general_utils import join_paths
from utils.images_utils import display
from utils.images_utils import postprocess_mask, denormalize_mask
from models.model import prepare_model


@torch.no_grad()
def predict(cfg: DictConfig):
    """
    Predict and visualize given data
    """

    # set batch size to one
    cfg.HYPER_PARAMETERS.BATCH_SIZE = 1

    # data generator
    val_generator = get_data_loader(cfg, mode="VAL")

    # set device with gpu id
    gpu_id = cfg.GPU_ID
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # create model
    model = prepare_model(cfg)

    # weights model path
    checkpoint_path = join_paths(
        cfg.WORK_DIR,
        cfg.CALLBACKS.MODEL_CHECKPOINT.PATH,
        f"{cfg.MODEL.WEIGHTS_FILE_NAME}.pt"
    )

    assert os.path.exists(checkpoint_path), \
        f"Model weight's file does not exist at \n{checkpoint_path}"

    # load model weights
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    model = model.to(device)
    # model.summary()

    # check mask are available or not
    mask_available = True
    if cfg.DATASET.VAL.MASK_PATH is None or \
            str(cfg.DATASET.VAL.MASK_PATH).lower() == "none":
        mask_available = False

    showed_images = 0
    for batch_data in val_generator:  # for each batch
        batch_images = batch_data[0].to(device)
        if mask_available:
            batch_mask = batch_data[1]

        # make prediction on batch
        batch_predictions = model(batch_images)
        
        for index in range(len(batch_images)):

            image = batch_images[index].cpu().numpy()  # for each image
            if cfg.SHOW_CENTER_CHANNEL_IMAGE:
                # for UNet3+ show only center channel as image
                image = image[1, :, :]

            # do postprocessing on predicted mask
            prediction = batch_predictions[index]
            prediction = postprocess_mask(prediction.cpu().numpy())
            # denormalize mask for better visualization
            prediction = denormalize_mask(prediction, cfg.OUTPUT.CLASSES + 1)

            if mask_available:
                mask = batch_mask[index]
                mask = postprocess_mask(mask.numpy())
                mask = denormalize_mask(mask, cfg.OUTPUT.CLASSES)
            
            # if np.unique(mask).shape[0] == 2:
            result_path = join_paths(cfg.WORK_DIR, cfg.RESULT_DIR, f'output_{showed_images}.png')
            if mask_available:
                display([image, mask, prediction], result_path, show_true_mask=True)
            else:
                display([image, prediction], result_path, show_true_mask=False)

            showed_images += 1
        # stop after displaying below number of images
        if showed_images >= 10: break


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Read config file and pass to prediction method
    """
    predict(cfg)


if __name__ == "__main__":
    main()
