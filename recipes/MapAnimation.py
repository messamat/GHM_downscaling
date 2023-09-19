import glob

import matplotlib.pyplot as plt
import imageio
import tempfile
import rasterio as rio
import numpy as np


class FlowAnimation:
    def __init__(self, tif_path, out_path, vmin, vmax):
        self.tif_path = tif_path
        self.tempdir = tempfile.TemporaryDirectory()
        self.vminmax = (vmin, vmax)
        self.write_gif(out_path)

    def write_gif(self, outfn):
        img_list = self.generate_images()
        imageio.mimwrite(outfn, img_list)

    def generate_images(self):
        tiffiles = glob.glob(self.tif_path +'*.tif')
        img_list = []
        for tiffile in tiffiles:
            img_list.append(imageio.imread(self.get_map(tiffile)))
        return img_list



    def get_map(self, tiffile):
        with rio.open(tiffile) as disr:
            ar = disr.read(1)[::-1]
            fig, ax = plt.subplots()
            cmap = plt.cm.get_cmap('viridis_r')
            cmap.set_bad('white',1.)
            if np.sum(ar<0):
                print('warning {} has negative values'.format(tiffile))
            ar[ar < 0] = np.nan
            im = ax.imshow(ar, cmap=cmap, vmin=self.vminmax[0], vmax=self.vminmax[1])
            plt.colorbar(im)
            ax.set_axis_off()
            fn = tiffile.split('/')[-1].split('.')[0].split('_')
            plt.title('{}_{}'.format(fn[-2], fn[-1]))
            plt.savefig('{}/{}.png'.format(self.tempdir.name, tiffile.split('/')[-1][:-5]))
            plt.close()
            return '{}/{}.png'.format(self.tempdir.name, tiffile.split('/')[-1][:-5])

    def __del__(self):
        del self.tempdir

def main():

    a = FlowAnimation('/home/home8/dryver/rastertemp/nobackup/rhone/22dant_ref/', '../validation/rhone_ref.gif', 0, 200)
    a = FlowAnimation('/home/home8/dryver/rastertemp/nobackup/rhone/22dant_simplecorr_thres/', '../validation/rhone_simplecorr_thres.gif', 0, 200)
    a = FlowAnimation('/home/home8/dryver/rastertemp/nobackup/rhone/22dant_simplecorr_preccorr1/', '../validation/rhone_simplecorr_preccor1.gif', 0, 200)


if __name__ == '__main__':
    main()
