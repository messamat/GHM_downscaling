.. _run-ts-label:

==================================
The timeseries downscaling process
==================================

Here the ts downscaling process is described with the important methods, which are used for the downscaling process.
Main method is the run_ts method, where the downscaling process is conducted.


|


.. automethod:: DryverDownscaling.DryverDownscaling.run_ts
   :noindex:

|


**Important terms, abbreviations or definitions used**

.. glossary::

    runoff
        in our context we refer to runoff as runoff generated in the vertical water balance without being routed

    discharge
        we refer to discharge as streamflow in river, which include discharge from upstream cells

    GHM
        global hydrology model (in 0.5 degree resolution)

    hr
        high resolution i.e. 15 arc seconds

    lr
        low resolution i.e. 30 arc min

    sr
        surface runoff

    gwr
        groundwater runoff also known as baseflow

