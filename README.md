# Yu_HoloXR

Yu's implementation on 'Importing self-attention into CGH'

## Program structure

The code is organized as follows:

- `pairs_loader.py` Loads (phase, captured) tuples for forward model training
- `prop_attention.py` core: the implementation of self-attention wave propagation model.
- `prop_CNNpropCNN.py` the implementation of CNNpropCNN wave-propagation model.
- `prop_idea.py` the implementation of ideal(ASM or fresnel) wave-propagation model.
- Other support modules: `prop_submodules.py`: submodules for propagation, `prop_submodules.py`: functions for zernike basis, `unet.py`: U-net generation, `utils.py`: utils functions
