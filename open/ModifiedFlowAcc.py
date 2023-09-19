import os
import warnings

from osgeo import gdal
import numpy as np


class FlowAccTT:
    def __init__(self, in_flowdir, in_flowacc, pad=True):
        """ Flow accumulation class, which builds upon flow direction and a static flow accumulation.

        This class takes flow directions and a static flow accumulation (generated with ArcGIS no weights with the same
        flow directions) and uses those files to generate a preset of consequential numpy accumulations along the
        river network.

        :param in_flowdir: Flow directions
        :type in_flowdir: path or np.ndarray
        :param in_flowacc: Flow accumulations
        :type in_flowacc: path or np.ndarray
        """

        #Read and format data ------------------------------------------------------------------------------------------
        if isinstance(in_flowdir, np.ndarray):
            self.flowdir = in_flowdir
        elif os.path.exists(in_flowdir):
            flowdir = gdal.Open(in_flowdir)
            self.flowdir = flowdir.ReadAsArray()
        else:
            raise Exception

        if isinstance(in_flowacc, np.ndarray):
            self.flowacc = in_flowacc
            self.flowacc_nan = -9999
        elif os.path.exists(in_flowacc):
            flowacc = gdal.Open(in_flowacc)
            self.flowacc = flowacc.ReadAsArray().copy()
            self.flowacc_nan = flowacc.GetRasterBand(1).GetNoDataValue()
        else:
            raise Exception

        self.shape = self.flowdir.shape
        if self.flowacc.shape != self.shape:
            raise Exception

        self.pad = pad
        if pad:
            self.flowacc = np.pad(self.flowacc, 1, 'constant',
                                  constant_values=np.nanmax(self.flowacc)+1)
            self.flowdir = np.pad(self.flowdir, 1, 'constant',
                                  constant_values=0)
        self.startcell_ix, self.endcell_ix = self.prepare_flowacc()

    def prepare_flattend_edges(self):
        """ Generate start and end indices of river flow for each cell based on flow directions array.

        The flattened position range(size) is used as indices .

        Returns
        -------
        list, list
            indices of origin of flow, indices of cells immediately downstream of each cell (end of flow)
        """
        flowdir_ar = self.flowdir
        startix = np.arange(flowdir_ar.size, dtype=np.int32)
        endix = startix.copy()
        flowdir_flat = flowdir_ar.flatten()
        for dire, xm, ym in [(1, 0, 1), (2, 1, 1), (4, 1, 0), (8, 1, -1), (16, 0, -1),
                             (32, -1, -1), (64, -1, 0), (128, -1, 1)]:
            endix[flowdir_flat == dire] += (ym + (xm * flowdir_ar.shape[1]))
        self.flowdir = None
        return startix, endix

    def prepare_flowacc(self):
        """ The flow accumulation is prepared in this method.

        The idea behind this flow accumulation algorithm is to use a static extern unweighted flow accumulation as
        basis for the computation schedule. The flow accumulation must happen in the right order. For this the static
        flow accumulation is used. First the cells with 0 accumulated flows are accumulated to their downstream cell
        then those with 1 accumulated flows and so on. As the accumulation is done via numpy vectors, further
        segmentation is needed as two cell cannot contribute to one cell in one vector.

        Returns
        -------
        list, list
            accumulation lists with spatial indices (startcell_ix) of sets of start and end cell indices, respectively
            (i.e., upstream and downstream cells), split by increasing flow accumulation and unique sets of end cells
            (because numpy summing cannot be performed in one iteration if two cells flow to the same end cell)
        """
        # Get ids of start and end cells
        startcell_ix, endcell_ix = self.prepare_flattend_edges()
        flowacc_copy = self.flowacc

        # set unnecessary values to a negative Nan value
        # flowacc[flowacc == self.flowacc_nan] = -9999

        #Convert flow accumulation from 2-d to 1-d
        flowacc = flowacc_copy.flatten()

        # Only consider the flow of cells where startcell != endcell (e.g. no need to route inland sinks)
        flowacc_nosink = flowacc[startcell_ix != endcell_ix]
        del flowacc, flowacc_copy
        self.flowacc = None
        startcell_ix_nosink = startcell_ix[startcell_ix != endcell_ix]
        endcell_ix_nosink = endcell_ix[startcell_ix != endcell_ix]

        # primary order of flow acc generate based on the static flowaccumulation grid
        sortindex = np.argsort(flowacc_nosink) #Get indices to sort cell indices by flow accumulation
        startcell_ix_nosink = startcell_ix_nosink[sortindex]
        endcell_ix_nosink = endcell_ix_nosink[sortindex]
        flowacc_nosink = flowacc_nosink[sortindex]

        # The arrays are split based on the number cells which flow into them
        split_ar = np.where(np.diff(flowacc_nosink))[0]+1
        startcell_ix_nosink_split = np.split(startcell_ix_nosink, split_ar)
        endcell_ix_nosink_split = np.split(endcell_ix_nosink, split_ar)

        # The arrays are split further if duplicates of endcells are in one calculation array as in computation vector
        # because only unique end indices are allowed to accumulate
        startcell_ix_nosink_split2 = []
        endcell_ix_nosink_split2 = []

        # loop through split array
        for i in range(len(endcell_ix_nosink_split)):
            # only continue if there is more than one cell in this split array
            if len(startcell_ix_nosink_split[i]) > 1:
                # sort the array via end indices
                sortix = np.argsort(endcell_ix_nosink_split[i])
                startcell_ixs = startcell_ix_nosink_split[i][sortix]
                endcell_ixs = endcell_ix_nosink_split[i][sortix]
                # find duplicates
                dix = np.concatenate(([False], endcell_ixs[1:] == endcell_ixs[:-1])) #Assign False to first instance of every index and true to all subsequent duplicates
                dendcell_ix = endcell_ixs[dix]
                dstartcell_ix = startcell_ixs[dix]

                # this loop could theroetically happen up to 8 times (all neigbours flowing into one cell)
                while len(dendcell_ix):
                    # append those indices which aren't duplicated to a newly split array
                    startcell_ix_nosink_split2.append(startcell_ixs[~dix])
                    endcell_ix_nosink_split2.append(endcell_ixs[~dix])
                    # rewrite indices with array of duplicates
                    endcell_ixs = dendcell_ix
                    startcell_ixs = dstartcell_ix
                    # find duplicates
                    dix = np.concatenate(([False], endcell_ixs[1:] == endcell_ixs[:-1]))
                    dstartcell_ix = startcell_ixs[dix]
                    dendcell_ix = endcell_ixs[dix]

                # add array with no more duplicates
                startcell_ix_nosink_split2.append(startcell_ixs)
                endcell_ix_nosink_split2.append(endcell_ixs)
            else:
                startcell_ix_nosink_split2.append(startcell_ix_nosink_split[i])
                endcell_ix_nosink_split2.append(endcell_ix_nosink_split[i])

        return startcell_ix_nosink_split2, endcell_ix_nosink_split2

    def get(self, in_valuegrid, no_negative_accumulation=True):
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
        if isinstance(in_valuegrid, np.ndarray):
            values = in_valuegrid
        elif os.path.exists(in_valuegrid):
            values = gdal.Open(in_valuegrid).ReadAsArray()
        else:
            raise Exception

        if values.shape != self.shape:
            raise Exception

        #Add a row and a column of 1 around array
        if self.pad:
            values = np.pad(values, 1, 'constant', constant_values=1)

        runoff_ar = values
        new_ar_flat = runoff_ar.flatten()
        #Iterate through every set of starting cells (split by flow accumulation values and unique end cell indices)
        for i in range(len(self.startcell_ix)):
            if no_negative_accumulation:
                # make sure that negative values aren't accumulated via flow accumulation
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    new_ar_flat[self.startcell_ix[i]] = np.where(new_ar_flat[self.startcell_ix[i]] < 0,
                                                               0,
                                                                 new_ar_flat[self.startcell_ix[i]])
            #Accumulate startcell values downstream
            new_ar_flat[self.endcell_ix[i]] += new_ar_flat[self.startcell_ix[i]]
        #Re-cast array to original shape
        newar = new_ar_flat.reshape(runoff_ar.shape)

        #Remove padding row and column
        if self.pad:
            newar = newar[1:-1, 1:-1]

        return newar
