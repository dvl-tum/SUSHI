from PIL import Image
from skimage.io import imread
from numpy import pad
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import numpy as np

class BoundingBoxDataset(Dataset):
    """
    Class used to process detections. Given a DataFrame (det_df) with detections of a MOT sequence, it returns
    the image patch corresponding to the detection's bounding box coordinates
    """
    def __init__(self, det_df, seq_info_dict, pad_=True, pad_mode='mean', output_size=(128, 64),
                 return_det_ids_and_frame=False, transforms=None):
        self.det_df = det_df
        self.seq_info_dict = seq_info_dict
        self.pad = pad_
        self.pad_mode = pad_mode
        if transforms is None:
            self.transforms = Compose((Resize(output_size), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])))
        else:
            self.transforms = transforms                                                                            

        # Initialize two variables containing the path and img of the frame that is being loaded to avoid loading multiple
        # times for boxes in the same image
        self.curr_img = None
        self.curr_img_path = None

        self.return_det_ids_and_frame = return_det_ids_and_frame

    def __len__(self):
        return self.det_df.shape[0]

    def __getitem__(self, ix):
        row = self.det_df.iloc[ix]

        # Load this bounding box' frame img, in case we haven't done it yet
        if row['frame_path'] != self.curr_img_path:
            self.curr_img = imread(row['frame_path'])
            self.curr_img_path = row['frame_path']

        frame_img = self.curr_img

        # Crop the bounding box, and pad it if necessary to
        bb_img = frame_img[int(max(0, row['bb_top'])): int(max(0, row['bb_bot'])),
                   int(max(0, row['bb_left'])): int(max(0, row['bb_right']))]
        if self.pad:
            x_height_pad = np.abs(row['bb_top'] - max(row['bb_top'], 0)).astype(int)
            y_height_pad = np.abs(row['bb_bot'] - min(row['bb_bot'], self.seq_info_dict['frame_height'])).astype(int)

            x_width_pad = np.abs(row['bb_left'] - max(row['bb_left'], 0)).astype(int)
            y_width_pad = np.abs(row['bb_right'] - min(row['bb_right'], self.seq_info_dict['frame_width'])).astype(int)

            bb_img = pad(bb_img, ((x_height_pad, y_height_pad), (x_width_pad, y_width_pad), (0, 0)), mode=self.pad_mode)

        bb_img = Image.fromarray(bb_img)

        if self.transforms is not None:
            bb_img = self.transforms(bb_img)

        if self.return_det_ids_and_frame:
            return row['frame'], row['detection_id'], bb_img
        else:
            return bb_img