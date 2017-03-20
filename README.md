# DeepVel

DeepVel is an end-to-end deep neural network to estimate horizontal velocity fields from pairs of solar granulation images separated in time. The velocities returned by `DeepVel` are computed assuming that the images are separated by 30 seconds. It is described in the paper, that you can find in [https://arxiv.org/abs/1703.05128](https://arxiv.org/abs/1703.05128).

## Abstract
Many phenomena taking place in the solar photosphere are controlled by plasma motions. Although the line-of-sight component of the velocity can be estimated using the Doppler effect, we do not have direct spectroscopic access to the components that are perpendicular to the line-of-sight. These components are typically estimated using methods based on local correlation tracking. We have designed `DeepVel`, an end-to-end deep neural network that produces an estimation of the velocity at every single pixel and at every time step and at three different heights in the atmosphere from just two consecutive continuum images. We have confronted `DeepVel` with local correlation tracking, pointing out that they give very similar results in the time- and spatially-averaged cases. We use the network to study the evolution in height of the horizontal velocity field in fragmenting granules, supporting the buoyancy-braking mechanism for the formation of integranular lanes in these granules. We also show that `DeepVel` can capture very small vortices, so that we can potentially expand the scaling cascade of vortices to very small sizes and durations.

## Dependencies
`train_deepvel.py` and `deepvel.py` depend on the following set of non-standard packages:

 - Keras (tested with v1.2.2 and v2): used for the neural network definition and computations
 - Tensorflow (tested with v1.0.0): used to do the real GPU calculations
 
## Using `DeepVel` for prediction
We provide DeepVel trained as described in the paper. You can use it straightforward from the command line if you have Keras and Tensorflow installed. The script we provide reads the observations from a FITS file, but you can modify the script to provide the observations from any other file. It is used as:

    python deepvel.py -i samples/sample.fits -o output/output.fits -b 100    

We provide the sample file containing two frames from the IMaX observations for testing. The output file is a FITS file containing an array of size `(n_frames x nx_new x ny_new x 6)`. The first dimension is the number of frames of the outpu (which is always one less than the input). The second and third dimensions are the size of the output image (which takes into account the removed borders). Finally, the last dimension contains the vx and vy velocity maps for the three optical depths considered: 1, 0.1 and 0.01.    

We provide scripts for Keras 2 (`deepvel_k2.py`) and, as legacy, for Keras 1 (`deepvel.py`).

## Using `DeepVel` for training
If you want to train `DeepVel` with your own images, we provide the script `train_deepvel.py` to this aim.
    
    python train_deepvel.py -a start -e 30 -o networks/model1 -n 1e-3

The parameters for `train_deepvel.py` are:

    - The action (indicated with `-a`) can be `start` to start the training of the network or `continue`, to continue a previous training.
    - The output (indicated with `-o`) gives the path to the output files. Some extensions will be added to save weights, plots, etc.
    - The number of epochs for training is indicated with `-e`.
    - The noise added during the training is indicated with `-n`.

The training phase is hardwired with some paths for our training data. We do not provide them in this repository, so you will have to work with the script to adapt it to your training data.
