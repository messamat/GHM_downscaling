import os
import warnings

from osgeo import gdal
import numpy as np


class FlowAccTT:
    def __init__(self, fd, fa, pad=True):
        """ Flow accumulation class, which builts upon flow direction and a static flow accumulation.

        This class takes flow directions and a static flow accumulation (generated with ArcGIS no weights with the same
        flow directions) and uses those files to generate a preset of consequential numpy accumulations along the
        river network.

        :param fd: Flow directions
        :type fd: path or np.ndarray
        :param fa: Flow accumulations
        :type fa: path or np.ndarray
        """
        if isinstance(fd, np.ndarray):
            self.fd = fd
        elif os.path.exists(fd):
            fd = gdal.Open(fd)
            self.fd = fd.ReadAsArray()
        else:
            raise Exception

        if isinstance(fa, np.ndarray):
            self.fa = fa
            self.fanan = -9999
        elif os.path.exists(fa):
            fa = gdal.Open(fa)
            self.fa = fa.ReadAsArray().copy()
            self.fanan = fa.GetRasterBand(1).GetNoDataValue()
        else:
            raise Exception

        self.shape = self.fd.shape
        if self.fa.shape != self.shape:
            raise Exception

        self.pad = pad
        if pad:
            self.fa = np.pad(self.fa, 1, 'constant', constant_values=np.nanmax(self.fa)+1)
            self.fd = np.pad(self.fd, 1, 'constant', constant_values=0)
        self.six, self.eix = self.prepare_flowacc()

    def prepare_flattend_edges(self):
        """ Based on flow directions array it generates start and end indices of river flows.

        As indices the flattend position range(size) is used.

        Returns
        -------
        list, list
            indices of origin of flow, indices of end of flow
        """
        fdar = self.fd
        startix = np.arange(fdar.size, dtype=np.int32)
        endix = startix.copy()
        fdflat = fdar.flatten()
        for dire, xm, ym in [(1, 0, 1), (2, 1, 1), (4, 1, 0), (8, 1, -1), (16, 0, -1),
                             (32, -1, -1), (64, -1, 0), (128, -1, 1)]:

            endix[fdflat == dire] += (ym + (xm * fdar.shape[1]))
        self.fd = None
        return startix, endix

    def prepare_flowacc(self):
        """ The flow accumulation is prepared in this method.

        The idea behind this flow accumulation algorithm is to use a static extern non weighted flow accumulation as
        basis for the computation schedule. The flow accumulation must happen in the right order. For this the static
        flow accumulation is used. First the cells with 0 accumulated flows are accumulated to their downstream cell
        then those with 1 accumulated flows and so on. As the accumulation is done via numpy vectors further
        segmentation is needed as two cell cannot contribute to one cell in one vector.

        Returns
        -------
        list, list
            accumulation lists with spatial indices (six)
        """
        six, eix = self.prepare_flattend_edges()
        flowacc = self.fa
        # set non necessary values to a negative Nan value
        # flowacc[flowacc == self.fanan] = -9999
        fa = flowacc.flatten()
        # We only have to consider the flow of cells not where startcell = endcell (e.g. inland sinks)
        fan = fa[six != eix]
        del fa, flowacc
        self.fa = None
        sixn = six[six != eix]
        eixn = eix[six != eix]
        # primary order of flow acc generate based on the static flowaccumulation grid
        sortindex = np.argsort(fan)
        sixn = sixn[sortindex]
        eixn = eixn[sortindex]
        fan = fan[sortindex]
        # The arrays are split based on the number cells which flow into them
        splitar = np.where(np.diff(fan))[0]+1
        sixn = np.split(sixn, splitar)
        eixn = np.split(eixn, splitar)

        # The arrays are split further if duplicates of endcells are in one calculation array as in computation vector
        # only unique end indices are allowd to accumulate
        sixns2 = []
        eixns2 = []
        # loop through splitted array
        for i in range(len(eixn)):
            # only continue if there is more than one cell in this splitted array
            if len(sixn[i]) > 1:
                # sort the array via end indices
                sortix = np.argsort(eixn[i])
                sixs = sixn[i][sortix]
                eixs = eixn[i][sortix]
                # find duplicates
                dix = np.concatenate(([False], eixs[1:] == eixs[:-1]))
                deix = eixs[dix]
                dsix = sixs[dix]
                # this loop could theroetically happen up to 8 times (all neigbours flowing into one cell)
                while len(deix):
                    # append those indices which arent duplicated
                    sixns2.append(sixs[~dix])
                    eixns2.append(eixs[~dix])
                    # rewrite indices with array with duplicated
                    eixs = deix
                    sixs = dsix
                    # find duplicates
                    dix = np.concatenate(([False], eixs[1:] == eixs[:-1]))
                    dsix = sixs[dix]
                    deix = eixs[dix]

                # add array with no more duplicates
                sixns2.append(sixs)
                eixns2.append(eixs)
            else:
                sixns2.append(sixn[i])
                eixns2.append(eixn[i])

        return sixns2, eixns2

    def get(self, values, no_negative_accumulation=True):
        """
        Apply the flow accumulation and get the flow accumulated values.

        Parameters
        ----------
        values : np.ndarray or gdal raster
            values which should be accumulated
        Returns
        -------
        np.ndarray
            values accumulated
        """
        if isinstance(values, np.ndarray):
            values = values
        elif os.path.exists(values):
            values = gdal.Open(values).ReadAsArray()
        else:
            raise Exception
        if values.shape != self.shape:
            raise Exception
        if self.pad:
            values = np.pad(values, 1, 'constant', constant_values=1)
        disar = values
        newarflat = disar.flatten()
        for i in range(len(self.six)):
            if no_negative_accumulation:
                # make sure that negative values aren't accumulated via flow accumulation
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    newarflat[self.six[i]] = np.where(newarflat[self.six[i]] < 0, 0, newarflat[self.six[i]])
            newarflat[self.eix[i]] += newarflat[self.six[i]]
        newar = newarflat.reshape(disar.shape)
        if self.pad:
            newar = newar[1:-1, 1:-1]
        return newar
