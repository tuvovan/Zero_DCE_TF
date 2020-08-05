# MASK HDR
The Keras Implementation of the [Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf) - CVPR 2020
## Content
- [Zero-DCE](#mask-hdr)
- [Getting Started](#getting-tarted)
- [Running](#running)
- [References](#references)
- [Citations](#citation)

## Getting Started

- Clone the repository

### Prerequisites

- Tensorflow 2.2.0+
- Python 3.6+
- Keras 2.3.0
- PIL
- numpy

```python
pip install -r requirements.txt
```

## Running
### Training Pretraining
- Preprocess
    - Download the training data at [Google Drive](https://drive.google.com/file/d/1HiLtYiyT9R7dR9DRTLRlUUrAicC4zzWN/view).

    - Run this file to generate data. (Please remember to change path first)

    ```
    python prepare_data.py
    ```

- Train ZERO_DCE 
    - Please note that the network I'm using here is different from the network from the original repo, however, the loss function is same, as the idea of the paper [[2]](#references) is to train the network using Place2 [[3]](#references) dataset using the loss function from [[2]](#references) and then do fine-tunning on HDR dataset.

    ```
    python train.py
    ```
- Test ZERO_DCE
    ```
    python inpainting/test.py --snapshot <snapshot_path>
    ```

## Usage
### Training
```
python train.py [-h] [--lowlight_images_path LOWLIGHT_IMAGES_PATH] [--lr LR]
                [--weight_decay WEIGHT_DECAY]
                [--grad_clip_norm GRAD_CLIP_NORM] [--num_epochs NUM_EPOCHS]
                [--train_batch_size TRAIN_BATCH_SIZE]
                [--val_batch_size VAL_BATCH_SIZE] [--num_workers NUM_WORKERS]
                [--display_iter DISPLAY_ITER] [--snapshot_iter SNAPSHOT_ITER]
                [--snapshots_folder SNAPSHOTS_FOLDER]
                [--load_pretrain LOAD_PRETRAIN] [--pretrain_dir PRETRAIN_DIR]
```

```
optional arguments: -h, --help            show this help message and exit
                    --lowlight_images_path LOWLIGHT_IMAGES_PATH
                    --lr LR
                    --weight_decay WEIGHT_DECAY
                    --grad_clip_norm GRAD_CLIP_NORM
                    --num_epochs NUM_EPOCHS
                    --train_batch_size TRAIN_BATCH_SIZE
                    --val_batch_size VAL_BATCH_SIZE
                    --num_workers NUM_WORKERS
                    --display_iter DISPLAY_ITER
                    --snapshot_iter SNAPSHOT_ITER
                    --snapshots_folder SNAPSHOTS_FOLDER
                    --load_pretrain LOAD_PRETRAIN
                    --pretrain_dir PRETRAIN_DIR
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## References
[1] Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement - CVPR 2020 [link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)

[3] Low-light dataset - [link](https://drive.google.com/file/d/1HiLtYiyT9R7dR9DRTLRlUUrAicC4zzWN/view)

## Citation
```
    @misc{guo2020zeroreference,
    title={Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement},
    author={Chunle Guo and Chongyi Li and Jichang Guo and Chen Change Loy and Junhui Hou and Sam Kwong and Runmin Cong},
    year={2020},
    eprint={2001.06826},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
## Acknowledgments

- This repo is the re-production of the original pytorch [version](https://github.com/Li-Chongyi/Zero-DCE)
- Thanks you for helping understand more pains that tensorflow may cause.
- Final words:
    - Any ideas on updating or misunderstanding, please send me an email: <vovantu.hust@gmail.com>
    - If you find this repo helpful, kindly give me a star.

