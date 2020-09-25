# multi-scale-style-transfer

Multi-scale style transfer with a pyramid of fully convolutional GANs inspired from [SinGAN: Learning a Generative Model from a Single Natural Image (ICCV 2019)](https://arxiv.org/abs/1905.01164) 

## Repository Structure

- The `SinGAN_master` directory contains the original, unaltered version of SinGAN. 
- In `architectures` you will find different networks that can be used for the Pyramid Model.
- Images (content and style) should be stored in `Images`.
- `main.py` is the ... main file you will want to run using command line arguments similar to the ones used by SinGAN.
- `parse.py` contains the necessary command line arguments.
- `training.py` handles all of the training for a given Pyramid Model.
- `utilities.py` has the necessary functions that don't belong elsewhere.

## Usage

The basic structure of a command looks like this:

```python main.py --content_input <content input image> --style_input <style input image>```


### Visualization
For visualization, you will need to install visdom first: `pip install visdom`. Then run the visdom server in a separate terminal:

```python -m visdom.server``` 

and lastly run `main.py` as usual with the following flag:

```python main.py ... --vis True```
