import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import utils
from unet import UnetGenerator, init_weights, norm_layer
import prop_ideal
from prop_submodules import Field2Input, Output2Field, Conv2dField,\
    ContentDependentField, LatentCodedMLP, SourceAmplitude
from prop_zernike import compute_zernike_basis, combine_zernike_basis


class PropModel(pl.LightningModule):
    """
    A parameterized model trained on captured images at multiple planes

    Class initialization parameters
    -------------------------------
    :param roi_res:
    :param plane_idxs:
    :param loss_func:
    :param lr:
    """

    def __init__(self, roi_res=(1080, 1920), plane_idxs=None, loss_func=F.l1_loss, lr=4e-4):
        super(PropModel, self).__init__()
        self.roi_res = roi_res
        self.plane_idxs = plane_idxs
        self.loss_func = loss_func
        self.lr = lr
        self.recon_amp = {}
        self.target_amp = {}

    def perform_step(self, batch, batch_idx, prefix):
        slm_phase, target_amps = batch

        # output field at target planes.
        recon_fields = self.forward(slm_phase)

        # calculate losses
        loss, loss_mse, loss_mse_all = self.mp_loss(recon_fields, target_amps,
                                                    self.plane_idxs, self.loss_func, prefix)

        with torch.no_grad():
            self.log(f'loss_{prefix}', loss, on_step=True, on_epoch=True)
            self.log(f'PSNR_{prefix}', 10*math.log10(1 /
                     loss_mse), on_step=True, on_epoch=True)
            self.log(f'PSNR_except_held_out_{prefix}',
                     10*math.log10(1/loss_mse_all), on_step=True, on_epoch=True)

        return loss

    def mp_loss(self, recon_fields, target_amps, plane_idxs, loss_func, prefix):
        """ Loss function on multiplane amplitudes"""

        # take the amplitudes.
        target_amps = utils.crop_image(
            target_amps, self.roi_res, stacked_complex=False)
        recon_amps = utils.crop_image(
            recon_fields, self.roi_res, stacked_complex=False).abs()

        with torch.no_grad():
            self.recon_amp[prefix] = recon_amps
            self.target_amp[prefix] = target_amps

        # calculate loss values.
        loss = 0.  # the loss you penalize.
        mse_loss = 0.  # PSNR on planes you penalize.
        mse_loss_all = 0.  # PSNR on all planes except the held-out plane.

        if plane_idxs is None:
            # penalize all planes
            loss += loss_func(recon_amps, target_amps)
        else:
            # penalize only selected planes
            for i in range(target_amps.shape[1]):
                if i in plane_idxs[prefix]:
                    # selected planes
                    loss_i = loss_func(
                        recon_amps[:, i, ...], target_amps[:, i, ...])
                    loss += loss_i / len(plane_idxs[prefix])
                else:
                    # these are not penalized, just for tensorboard
                    with torch.no_grad():
                        loss_i = loss_func(
                            recon_amps[:, i, ...], target_amps[:, i, ...])

                # report PSNR
                with torch.no_grad():
                    # this is for evaluation so just use the min-mse scaling.
                    s = (recon_amps[:, i, ...] * target_amps[:, i, ...]).mean() \
                        / (recon_amps[:, i, ...]**2).mean()
                    mse_loss_i = F.mse_loss(
                        s*recon_amps[:, i, ...], target_amps[:, i, ...])

                    if i in self.plane_idxs[prefix]:
                        mse_loss += mse_loss_i / len(plane_idxs[prefix])

                    if not i in self.plane_idxs['heldout']:
                        # exclude held-out plane
                        mse_loss_all += mse_loss_i / len(plane_idxs['train'])

                    self.log(f'loss_{prefix}/plane_{i}',
                             loss_i, on_step=True, on_epoch=True)
                    self.log(f'PSNR_{prefix}/plane_{i}',
                             10*math.log10(1./mse_loss_i), on_step=True, on_epoch=True)

        return loss, mse_loss, mse_loss_all

    def training_step(self, batch, batch_idx):
        return self.perform_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.perform_step(batch, batch_idx, 'validation')

    def test_step(self, batch, batch_idx):
        return self.perform_step(batch, batch_idx, 'test')

    def test_epoch_end(self, outputs) -> None:
        self.epoch_end_images('test')

    def training_epoch_end(self, outputs) -> None:
        self.epoch_end_images('train')

    def validation_epoch_end(self, outputs) -> None:
        self.epoch_end_images('validation')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class CNNpropCNN(PropModel):
    """
    A parameterized model with CNNs

    Class initialization parameters
    -------------------------------
    :param prop_dist: Propagation distance from SLM to Intermediate plane, float.
    :param wavelength: wavelength, float.
    :param feature_size: Pixel pitch of SLM, float.
    :param prop_type: Type of propagation, string, default 'ASM'.
    :param F_aperture: Level of filtering at Fourier plane, float, default 1.0 (and circular).
    :param prop_dists_from_wrp: An array of propagation distances from Intermediate plane to Target planes.
    :param linear_conv: If true, pad before taking Fourier transform to ensure the linear convolution.
    :param slm_res: Resolution of SLM.
    :param roi_res: Resolution of Region of Interest.
    :param num_downs_slm: Number of layers of U-net at SLM network.
    :param num_feats_slm_min: Number of features at the top layer of SLM network.
    :param num_feats_slm_max: Number of features at the very bottom layer of SLM network.
    :param num_downs_target: Number of layers of U-net at target network.
    :param num_feats_target_min: Number of features at the top layer of target network.
    :param num_feats_target_max: Number of features at the very bottom layer of target network.
    :param norm: normalization layers.
    :param slm_coord: input/output format of SLM network.
    :param target_coord: input/output format of target network.).
    :param loss_func: Loss function to train the model.
    :param lr: a learning rate.
    :param plane_idxs: a dictionary that has plane idxs for 'train', 'val', 'test', 'heldout'.
    """

    def __init__(self, prop_dist, wavelength, feature_size, prop_type='ASM', F_aperture=1.0,
                 prop_dists_from_wrp=None, linear_conv=True, slm_res=(1080, 1920), roi_res=(960, 1680),
                 num_downs_slm=0, num_feats_slm_min=0, num_feats_slm_max=0,
                 num_downs_target=0, num_feats_target_min=0, num_feats_target_max=0,
                 norm=nn.InstanceNorm2d, slm_coord='rect', target_coord='rect',
                 loss_func=F.l1_loss, lr=4e-4,
                 plane_idxs=None):
        super(CNNpropCNN, self).__init__(roi_res=roi_res,
                                         plane_idxs=plane_idxs, loss_func=loss_func, lr=lr)

        ##################
        # Model pipeline #
        ##################
        # SLM Network
        if num_downs_slm > 0:
            slm_cnns = []
            slm_cnn_res = tuple(res if res % (2 ** num_downs_slm) == 0 else
                                res + (2 ** num_downs_slm - res %
                                       (2 ** num_downs_slm))
                                for res in slm_res)
            print(slm_cnn_res, ' res')

            slm_input = Field2Input(slm_cnn_res, coord=slm_coord)
            slm_cnns += [slm_input]

            if num_downs_slm > 0:
                slm_cnn = UnetGenerator(input_nc=4 if 'both' in slm_coord else 2, output_nc=2,
                                        num_downs=num_downs_slm, nf0=num_feats_slm_min,
                                        max_channels=num_feats_slm_max, norm_layer=norm, outer_skip=True)
                init_weights(slm_cnn, init_type='normal')
                slm_cnns += [slm_cnn]

            slm_output = Output2Field(slm_res, slm_coord)
            slm_cnns += [slm_output]
            self.slm_cnn = nn.Sequential(*slm_cnns)
        else:
            self.slm_cnn = None

        # Propagation from the SLM plane to the WRP.
        if prop_dist != 0.:
            self.prop_slm_wrp = prop_ideal.Propagation(prop_dist, wavelength, feature_size,
                                                       prop_type=prop_type, linear_conv=linear_conv,
                                                       F_aperture=F_aperture)
        else:
            self.prop_slm_wrp = None

        # Propagation from the WRP to other planes.
        if prop_dists_from_wrp is not None:
            self.prop_wrp_target = prop_ideal.Propagation(prop_dists_from_wrp, wavelength, feature_size,
                                                          prop_type=prop_type, linear_conv=1.0,
                                                          F_aperture=F_aperture)
        else:
            self.prop_wrp_target = None

        # Target network (This is either included (prop later) or not (prop before, which is then basically NH3D).
        if num_downs_target > 0:
            target_cnn_res = tuple(res if res % (2 ** num_downs_target) == 0 else
                                   res + (2 ** num_downs_target - res % (2 ** num_downs_target)) for res in slm_res)
            target_input = Field2Input(
                target_cnn_res, coord=target_coord, shared_cnn=True)
            input_nc_target = 4 if 'both' in target_coord else 2 if target_coord != 'amp' else 1
            output_nc_target = 2 if target_coord != 'amp' and (
                '1ch_output' not in target_coord) else 1
            target_cnn = UnetGenerator(input_nc=input_nc_target, output_nc=output_nc_target,
                                       num_downs=num_downs_target, nf0=num_feats_target_min,
                                       max_channels=num_feats_target_max, norm_layer=norm, outer_skip=True)
            init_weights(target_cnn, init_type='normal')

            # shared target cnn requires permutation in channels here.
            num_ch_output = 1 if not prop_dists_from_wrp else len(
                self.prop_wrp_target)
            target_output = Output2Field(
                slm_res, target_coord, num_ch_output=num_ch_output)
            target_cnns = [target_input, target_cnn, target_output]
            self.target_cnn = nn.Sequential(*target_cnns)
        else:
            self.target_cnn = None

    def forward(self, field):
        if self.slm_cnn is not None:
            slm_field = self.slm_cnn(field)  # Applying CNN at SLM plane.
        else:
            slm_field = field
        if self.prop_slm_wrp is not None:
            # Propagation from SLM to Intermediate plane.
            wrp_field = self.prop_slm_wrp(slm_field)
        if self.prop_wrp_target is not None:
            # Propagation from Intermediate plane to Target planes.
            target_field = self.prop_wrp_target(wrp_field)
        if self.target_cnn is not None:
            # Applying CNN at Target planes.
            amp = self.target_cnn(target_field).abs()
            phase = target_field.angle()
            output_field = amp * torch.exp(1j * phase)
        else:
            output_field = target_field

        return output_field

    def epoch_end_images(self, prefix):
        """
        execute at the end of epochs

        :param prefix:
        :return:
        """
        #################
        # Reconstructions
        logger = self.logger.experiment
        recon_amp = self.recon_amp[prefix][0]
        target_amp = self.target_amp[prefix][0]
        for i in range(recon_amp.shape[0]):
            logger.add_image(
                f'amp_recon/{prefix}_{i}', recon_amp[i:i+1, ...].clip(0, 1), self.global_step)
            logger.add_image(
                f'amp_target/{prefix}_{i}', target_amp[i:i+1, ...].clip(0, 1), self.global_step)

    @property
    def plane_idx(self):
        return self._plane_idx

    @plane_idx.setter
    def plane_idx(self, idx):
        """

        """
        if idx is None:
            return
        self._plane_idx = idx
        if self.prop_wrp_target is not None and len(self.prop_wrp_target) > 1:
            self.prop_wrp_target.plane_idx = idx
        if self.target_cnn is not None and self.target_cnn[-1].num_ch_output > 1:
            self.target_cnn[-1].num_ch_output = 1
