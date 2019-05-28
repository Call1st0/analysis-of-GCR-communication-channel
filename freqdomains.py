import skimage
import skimage.measure as msr
import skimage.color as color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
from scipy import fftpack


class FreqDomains:
    """ Class with methods to transform image to Fourier or Cosine domain"""

    def __init__(self):
        pass

    @staticmethod
    def compare_images(originalImg, modifiedImg):
        """ Helper method for displaying comparison of images """
        fig, axes = plt.subplots(nrows=1, ncols=2, sharex='all', sharey='all',dpi=144)
        # ax = axes.ravel()

        psnr_orig = msr.compare_psnr(originalImg, originalImg)
        ssim_orig = msr.compare_ssim(
            originalImg, originalImg, multichannel=True)

        psnr_mod = msr.compare_psnr(originalImg, modifiedImg)
        ssim_mod = msr.compare_ssim(
            originalImg, modifiedImg, multichannel=True)

        label = 'PSNR: {:.2f}, SSIM: {:.2f}'

        axes[0].imshow(originalImg, cmap=plt.cm.gray)
        axes[0].set_xlabel(label.format(psnr_orig, ssim_orig))
        axes[0].set_title('Original image')

        axes[1].imshow(modifiedImg, cmap=plt.cm.gray)
        axes[1].set_xlabel(label.format(psnr_mod, ssim_mod))
        axes[1].set_title('Modified image')

        plt.show()

    @staticmethod
    def imshow_four(img1, img2, img3, img4):
        """ Helper method for displaying arbitrary number of images side by side"""
        fig, axes = plt.subplots(
            nrows=2, ncols=2, sharex='all', sharey='all', figsize=(7, 8), dpi=300)
        ax = axes.ravel()

        ax[0].imshow(img1)
        pcm = ax[0].imshow(img1)
        ax[0].set_title('Cyan')
        ax[0].axis('off')

        ax[1].imshow(img2)
        pcm = ax[1].imshow(img2)
        ax[1].set_title('Magenta')
        ax[1].axis('off')

        ax[2].imshow(img3)
        pcm = ax[2].imshow(img3)
        ax[2].set_title('Yellow')
        ax[2].axis('off')

        ax[3].imshow(img4)
        pcm = ax[3].imshow(img4)
        ax[3].set_title('Black')
        ax[3].axis('off')

        fig.colorbar(pcm, orientation='horizontal', ax=ax.tolist())

        plt.show()

    @staticmethod
    def imshow_components(img, img_title):
        """Helper method for displaying image and its components"""
        fig, axes = plt.subplots(
            nrows=2, ncols=2, sharex='all', sharey='all', dpi=300)
        ax = axes.ravel()
        try:
            if img.shape[2] == 3:  # Image has 3 channels
                ax[0].imshow(img)
                ax[0].set_title('Original image')
                ax[0].axis('off')

                ax[1].imshow(img[:, :, 0], cmap=plt.cm.gray)
                ax[1].set_title('R channel')
                ax[1].axis('off')

                ax[2].imshow(img[:, :, 1], cmap=plt.cm.gray)
                ax[2].set_title('G channel')
                ax[2].axis('off')

                ax[3].imshow(img[:, :, 2], cmap=plt.cm.gray)
                ax[3].set_title('B channel')
                ax[3].axis('off')

            elif img.shape[2] == 4:  # Image has 4 channels
                ax[0].imshow(img[:, :, 0], cmap=plt.cm.gray)
                ax[0].set_title(
                    f'C channel - AC {FreqDomains.average_tac(img[:, :, 0]):2.1f}')
                ax[0].axis('off')

                ax[1].imshow(img[:, :, 1], cmap=plt.cm.gray)
                ax[1].set_title(
                    f'M channel - AC {FreqDomains.average_tac(img[:, :, 1]):2.1f}')
                ax[1].axis('off')

                ax[2].imshow(img[:, :, 2], cmap=plt.cm.gray)
                ax[2].set_title(
                    f'Y channel - AC {FreqDomains.average_tac(img[:, :, 2]):2.1f}')
                ax[2].axis('off')

                ax[3].imshow(img[:, :, 3], cmap=plt.cm.gray)
                ax[3].set_title(
                    f'K channel - AC {FreqDomains.average_tac(img[:, :, 3]):2.1f}')
                ax[3].axis('off')
        except IndexError:
            print(f"Image has wrong number of channels. Can't create plot! ")
            pass
        plt.tight_layout
        fig.suptitle(img_title)
        plt.savefig(img_title, format='eps', dpi=300)
        plt.show()

    @staticmethod
    def average_tac(img):
        """ Method that takes an image and returns total area coverage """
        return img.sum()/img.size

    @staticmethod
    def spatial2dft(img):
        """ Takes image and returns its magnitude and phase in fourier domain"""
        # convert to float64 to keep precision
        img_f = skimage.img_as_float32(img)
        fft2 = fftpack.fft2(img_f)
        # shift lowest frequency to the center
        magnitude = fftpack.fftshift(np.absolute(fft2))
        phase = np.angle(fft2)
        return magnitude, phase

    @staticmethod
    def dft2spatial(magnitude, phase):
        """ Takes magnitude and phase of the image and returns img in spatial domain"""
        img = fftpack.ifft2(np.multiply(
            fftpack.ifftshift(magnitude), np.exp(1j * phase)))
        img = np.real(img)  # Image still is complex so take only real part
        # TODO image has to be converted to uint8!
        return skimage.img_as_ubyte(img)

    @staticmethod
    def spatial2dct(img):
        """ Takes image and returns its coeffs in cosine domain"""
        # convert to float64 to keep precision
        img_f = skimage.img_as_float32(img)
        return fftpack.dctn(img_f, norm='ortho')

    @staticmethod
    def dct2spatial(dct2):
        """ Takes cosine domain coeffs and returns an image """
        img = fftpack.idctn(dct2, norm='ortho')
        return skimage.img_as_ubyte(img)

    @staticmethod
    def imread(imgName):
        """ Read the image """
        return imageio.imread(imgName)

    @staticmethod
    def block_process(a, blocksize, filt):
        """ function for block processing. 
        To call a filter filter1(a, filtsize) use block_process(a, blocksize, filter1, ) """
        block = np.empty(a.shape)
        for row in range(0, a.shape[0], blocksize):
            for col in range(0, a.shape[1], blocksize):
                block[row:row + blocksize, col:col + blocksize] = (
                    filt(a[row:row + blocksize, col:col + blocksize]))
        return block

    @staticmethod
    def block_average(orig, modified, blocksize):
        """ function to get the differences of all dct blocks. The difference between blocks
        is stacked in third dimension of the array """
        block = np.empty(
            (np.reshape(orig, (blocksize, blocksize, -1)).copy()).shape)
        layer = 0
        for row in range(0, orig.shape[0], blocksize):
            for col in range(0, orig.shape[1], blocksize):
                block[:, :, layer] = np.abs(
                    orig[row:row + blocksize, col:col + blocksize]-modified[row:row + blocksize, col:col + blocksize])
            layer += 1
        return np.mean(block, axis=2)

    @staticmethod
    def abs_diff(img1, img2):
        # convert to float32 to keep precision
        img_f = np.ravel(skimage.img_as_float32(img1))
        img2_f = np.ravel(skimage.img_as_float32(img2))
        return np.mean(np.abs(img_f - img2_f))
