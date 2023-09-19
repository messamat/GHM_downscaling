import pandas as pd

from validation.StationComparison import getstationmetadict, StationComparison
from GSIMReader import GSIMStation
from GRDCReader import GRDCStation
from RBISReader import RBISStation


class StationHarmonization(StationComparison):
    def read_gsim_obs(self, use_daily=True):
        columns_gsimdf = pd.MultiIndex.from_product([['gsim'],
                                                     self.gsimstations,
                                                     ['obs']],
                                                    names=['stationtype',
                                                           'stationid',
                                                           'variant'])
        tempdf = pd.DataFrame(index=self.gsimdf.index, columns=columns_gsimdf)
        for stationdd in self.gsimstations:
            stationdict = {'dd_id': stationdd, **self.sm[stationdd]}
            gsimstation = GSIMStation(self.sm[stationdd]['gauge_id'], self.bp, stationdict)
            gsimstation.read_daily_data()
            # gsimstation.flag_daily_data('/home/home1/gm/projects/DRYvER/03_data/07_streamflowdata/08_qualitycontrol/')
            # gsimstation.clean_daily_data('/home/home1/gm/projects/DRYvER/03_data/07_streamflowdata/08_qualitycontrol/')
            gsimobs = gsimstation.daily_data
            if gsimobs is None:
                del self.sm[stationdd]
            else:
                gsimobs = gsimobs[str(tempdf.index.min().year):str(tempdf.index.max().year)]
                tempdf.loc[gsimobs.index, ('gsim', stationdd, 'obs')] = gsimobs
        self.gsimdf = pd.concat([self.gsimdf, tempdf], axis=1)

    def read_grdc_obs(self):
        columns_grdc = pd.MultiIndex.from_product([['grdc'],
                                                   self.grdcstations,
                                                   ['obs']],
                                                  names=['stationtype',
                                                         'stationid',
                                                         'variant'])
        tempdf = pd.DataFrame(index=self.grdcdf.index, columns=columns_grdc)
        for stationdd in self.grdcstations:
            stationdict = {'dd_id': stationdd, **self.sm[stationdd]}
            grdcstation = GRDCStation(self.sm[stationdd]['gauge_id'], self.bp, stationdict)
            grdcstation.read_daily_data()
            # grdcstation.flag_daily_data('/home/home1/gm/projects/DRYvER/03_data/07_streamflowdata/08_qualitycontrol/')
            # grdcstation.clean_daily_data('/home/home1/gm/projects/DRYvER/03_data/07_streamflowdata/08_qualitycontrol/')
            grdcobs = grdcstation.daily_data
            if grdcobs is not None:
                grdcobs = grdcobs[str(tempdf.index.min().year):str(tempdf.index.max().year)]
                tempdf.loc[grdcobs.index, ('grdc', stationdd, 'obs')] = grdcobs
            else:
                del self.sm[stationdd]
        self.grdcdf = pd.concat([self.grdcdf, tempdf], axis=1)

    def read_rbis_obs(self):
        columns_rbis = pd.MultiIndex.from_product([['rbis'],
                                                   self.rbisstations,
                                                   ['obs']],
                                                  names=['stationtype',
                                                         'stationid',
                                                         'variant'])
        tempdf = pd.DataFrame(index=self.rbisdf.index, columns=columns_rbis)
        for stationdd in self.rbisstations:
            stationdict = {'dd_id': stationdd, **self.sm[stationdd]}
            rbisstation = RBISStation('{}data_{}.txt'.format(self.bp + '05_rbis/', stationdd),
                                      station_data=stationdict)
            rbisstation.read_daily_data()
            # rbisstation.flag_daily_data('/home/home1/gm/projects/DRYvER/03_data/07_streamflowdata/08_qualitycontrol/')
            # rbisstation.clean_daily_data('/home/home1/gm/projects/DRYvER/03_data/07_streamflowdata/08_qualitycontrol/')
            rbisobs = rbisstation.daily_data
            if rbisobs is not None:
                rbisobs = rbisobs[str(tempdf.index.min().year):str(tempdf.index.max().year)]
                tempdf.loc[rbisobs.index, ('rbis', stationdd, 'obs')] = rbisobs
        self.rbisdf = pd.concat([self.rbisdf, tempdf], axis=1)


def eurasia(out_path):
        sy = 1901
        ey = 2019
        hydrosheds_path = '/home/home8/dryver/hydrosheds_dirmodified_rbisupd/'
        bp = '/home/home8/dryver/streamflow_daily/'
        metashp = '/home/home1/gm/projects/DRYvER/03_data/07_streamflowdata/04_stations/stations_europe/validation_stations_europe.shp'
        nottoeval = (
            '84d703127a7245b196aaa562b28c25e3',
            '5177c05910c94b91bed21681d6744e34',
            '9ff95c78ccf2435883f60ecca6dc1e07',
            '9dbf73007bd14a4d8d07e08e038aa8fe',
            'e349dc44230743969edec3186d7993aa',
            '9403626071fc4eb89b774f3d4f3b2357',
            '153f58c74f5e490495baf72bcb73bdbb',
            'e5415d1554fc4747baabc80d8339dc16',
            '1413c1c35b3341548df3188c21349517',
            'aded29f650b24834bf39e526404fe933',
            'c1c66c8ad43844b59bfbc3537f3b37a7',
            '582e186a3cb84bdc871f42c0a684d770',
            'e45a72f60f804f4e9fec3c2d2301a848',
            'CE2AFBED96A14AF3A89C608CAA408F2E',
            '622aca668b3b401988340f2727763a56',
            '79ef0f49359745da87b952368a24d054',
            'bf866726ce324d58b428436917b84495',
            '7DEB421D10A840B485241D887BD209A5',
            '912A627EDE2B42B6B04E17397EC31045',
            '5BAD17D26B874F13AFB90997EA99F1B8',
            '531C944FC2614F88A4F3075AFE124CFD',
            '4088EF125BCC4B28B169334AEDC59264',
            '95f58514efdc4bf090e0bb9f49afdcb7',
            'EF356DC1A7D844ABB8B56E746E8458F2',
            '7657A560F28B4657AB4B0D6A5A9495EC',
            '401d0750c0704c558c5ca1dfcca157d1',
            'd4a1682241cb4bd6bc5c15d0230a9b00',
            'ba82bfeb00c44eb4bad7fc15210b498f',
            '678186bb27e9419797110b0b84a4436b',
            '4369f45e42c14ca291a8987a448a23de',
            '9b5e0a64fa0b484b8516862c60f538a9',
            'cd409f4c7b264eb7941e3b79583e582f',
            '8a155ab294ef4bbba1a77740b9d061c6',
            'e366e4fa6a2d467b899d1a4331fad459',
            'c3723c32b0144f8c9cceada3756d4484',
            '1cb5859e33d947cfafac0a596d3702dd',
            'b79a54b3d63f465d8a9f4e07858cf9db',
            '2bcb2c6d83624fc8af56672513331ccc',
            'f05e13c11ef643dbb42383560c82cd3c',
            '4fe19f56331c41ec818697569e1b77f6',
            'af05ddfe27ac4ee294325c3711e05c6f',
            'daca68d0c8af434586b18952192c4150',
            '3d9eb3a3297740c89329c0eafe8b0876',
            '1d6e46060ea942f2b8ebec4da0c71faa',
            '4e371f04dc404d609bf5c9e2ce82d82a',
            '799cf4d1f3d14ae590434c1f9d6b3f98',
            'd3c88662cc1d481cb67fbd18c3940bd4',
            '2e797ba2a9644a2fa2bb30b1dba98364',
            'a35e9f13ace0403c871222a1bc8e0ad6',
            'fbd52e27c4d8431c8f5f31f46e917146',
            '3e7a15eca37e4b2ca375886d9236bc00',
            '5a9916f3a8d148edabb2aed6c44876cf',
            '57c63a8da12e460c93e00300b211abb2',
            'c76f7046448b425f8d7f00fced9241f5',
            'da5d76044afe49bfb7f8692dba72c86a',
            '02640f6190014c8386ab13da813682a5',
            '5ccbd9b586c54614899353828ffea033',
            '70eed40f5f5948d0bab65c67782d7a08',
            'cb1547fdab6a4450b1efe004e76abe44',
            'ce204ab5ab484937bf2923d91246b346',
            'ca9b733af6a74d6cb5a6d5d0c293f8ab',
            'baf9179d2d01476db5335b0317d24524',
            '5231d5765e794485a753982ca14c3ad6',
            '2ee422f9bf8b4c9e8a9bf2739e1ef1f8',
            '2753f2f76d3a45aa85d22e9855ff3e6a',
            '056bbd7a1f6e4252a255e5e773f815bd',
            'ff682b61bdfd40979ece731db389ef58',
            '052abe3b39a1412a9ebb3929ef2cec26',
            '294e9a13c17e4eca86190d4534111d00',
            '261fb09f4c6942e39c66583855325a79',
            '87f795c7b9d84e0ca05d27dcba1ece31',
            '2c879fa97ecb477b851d8c960594ae78',
            '1f5409811814404196f78bb980687ee2',
            'ebca2953b7d84369bd644f6c2a2f2a22',
            'd78234706d3f411ab2b551e1acdd0aa5',
            '4a2f3d871a18442da7fa1120c88ef615',
            '99eed873a060463889d5db88449aea0f',
            'e3ff236adbd14eef9a6a126b70e81215',
            '4388d4bb0bde48ec87877a5e5014dacb',
            '3d54da9fc5194dde84dfc4325cca00cf',
            'f6ed8a18b4f64c539793bc77f33da56e',
            'ee1d0aa2c7d2463b8604fede9ce25844',
            '3651c2bffe0f45e3adaefe38fcf8ce81',
            '795c836946ba4773a410e5f586293e5e',
            'd1ac4fd3396b4b34a67deae5b8ea23d8',
            '7e478110934d466eb2409c2cb9b30633',
            'b5e6eb5a02b8495bab82e69f10b9115c',
            '182e74f6e1154ccfb066a067a799f15c',
            'bd49ad735315492497bd95d72711568b',
            'ad4cdb59335442a082f5cd1504dd06f7',
            'b2f37a5a5c2649e48382e05973db0fee',
            '25155814EE844530BCBCB33E425106C8',
            '9d1c9049522b497a8a3d64ff8eb992b7',
            '9e8d964e518b412e82b407ffdedb1126',
            'BD2293B24D08456B839C9A5C58FB5AD4',
            'CDE516D645974575B567A488FFAB07AC',
            'D2AC7E6D00C3410EB5B89B4E7E550423',
            '89EA5313B65D497F80735C3CCF547496',
            'B4FFCC23638945579BCC4458A0795939',
            '1fefbbf480404fa79cf2f11a9e08461e',
            '9476853aeb464e008dd33b21fac31762',
            '10e91d30f1c541dc9617b141bd54b8c2',
            '03274d4313cd46fda770f201dbf3b3e9',
            'a5a7f59c6f304f3aa5b9379757615024',
            'e5e7ef733d774e3eb7c559aa57076ced',
            'fc619f5de898453a81db11926ea5a12f',
            '97145ba639644d268618b7076becce76',
            '496706a09c5841e58fed250b102d0d2d',
            'B3213404F1D045959E42C1578F987550',
            'D4395CEA85B84AF89936F798116DE3C4',
            '500266f3c5f2423b99ddf226ca8caa24',
            '3b92ad7eccb54acd8725d4355e1cdab6',
            '6f747e7e04ab49f6a4f8e282f1b72825',
            '592b546b69f645c2a71dfce08edf5a1f',
            'c64940372e1948c1a9fec4c289010352',
            '8bc96bdaad1a44478b9c2f00324c96cf',
            'd990d8634d594dce8e06cfd2c09ada17',
            '7e8d31b06a36491aa2a697434fd1dfe7',
            'c8bb8c8db8584d8ea3342d82f6cb9a5b',
            '0262a5a320a249679075944d664818ad',
            'adc7076baef54ac4a080e02f95976fd6',
            '433920a5344e4e14988d075a4cb22c5c',
            '62078c636dd341bfafd52e0c9a7fe2ab',
            '0e030573e41b424f8fac00ac671a64ec',
            'dca8d59b76034815a553d21564bc5e23',
            'eae6ccede52c4488be40411a3bf03ea9',
            '9cfc23582ca84c6197e30dc46dc1a17f',
            '9527277b1e614754baa52be61e57efad',
            'b07b1bc4ef414c0bbee0c98e51d3dc38',
            '049b489069c14a28b8414d113ff425b2',
            '0483d3de6c8f406bb93e5a9c83a0c6ab',
            '930e9562482f4df3ba8a76e66ca141f2',
            '1e09598933ae49d195e161999f6ae1fc',
            'e7fa749b2509408bad429cd61116b7c1',
            '98dca5220d26413aae56416551b99459',
            '527f99bedf5b4e2490a95821d501208b',
            '6445eb3349584d7b863815b785b02903',
            '1a2dca4a26814b3db2572dc84f922344',
            'b581d486facc48cdb73be0a218dca4c9',
            '5dae7c3b47894bf1b548993e0041d392',
            '0357cc58e48e4ab3a260b243742813e7',
            '8300784db82e499aa3bad826aacfbc00',
            '6c5d9d3a84d04bf0af138db3a56b79dd',
            '8673f616204b415ababa01c6f5b99de2',
            '08577c0b5c4142bfa1b7ed35d4ef4299',
            'fa57e28704db48e89891a4575dbe8984',
            '3ef401a7e68b484092d0442ae5cb0700',
            'c8f9b0eeee1f4abc833920848eeadb15',
            '3bd3f2ee6c964a6899b24afcb22a89be',
            '41f10641d6104fa290ef402551f7b666',
            'f278987a66ec4046a6bccfc4bc3cde93',
            'eeaaf0e4ba1d482fa7a9e3591c237552',
            'ae0f7c90d3d54b858c26755349d1b667',
            'f18434e0daec4dcba9c536a30895c7ac',
            '588b26fa9e30402da0a65be5b198c563',
            '9330efb4dc1d427dbdd7d5c02c323161',
            'f938ebe00e954179b4354ee021e7f45b',
            '6205316a45d84f17a4adcbbfb17cf018',
            '4cdda3b8ecb742059c1d0cbd701b71ab',
            '4286633d8841417c9f477b035876ef9e',
            'eda3d306c93d4fddaa734a72ba330303',
            'c4cd6d508dbc46d4b5198c53e95c0a50',
            '7c72e5f99e504be4b6fe59c30b90eab6',
            '3bc31551debc49f68e1bf95f00771040',
            'd5b2a476bd924b098108f76bd911d8ec',
            '2cb5cc45355046809510a5a2256d2b23',
            '758c878030ea4d35a83601583aa6acdb',
            '6d2b2ea018f141fba4156dfd6667edf4',
            'b1659107dd854f52906ebd70b15af8e8',
            'f575d895a05f4867bc4fc485f5e7f84f',
            '49b88e68d3084222a5b6ca673bf8681f',
            '5b0337ee70644f04b389511053869783',
            'd7227db8985d4d63a586021818dc0479',
            'a20382e5869d43cfb1f26fc4866dd697',
            '52a635c1d0be432b9ecdbf378bd2f653',
            'fa9c7cad3f2e4302bc09b3f6c3569932',
            'b0fd4af562314486b0b96362dcac5a41',
            '499dfe61800349448a179e13f173259a',
            '4f3ab1b789a74de4bc367aa748d21af9',
            'e3c46bfd8b514231bea2dcd0922a4256',
            'f5ae24ab3e0449cbacdae92051d34acb',
            '5666f257cdec422bb2428064020d43f3',
            'fc5c41cde9734936bd8d844edb1f4b49',
            'e99346f653134bf1b39d95ab83e554e5',
            'c38f85af529348428f943c6ad46e884a',
            '64f7674fa4f246e5a46a99cfa6940013',
            '95ddb443d481459aa701f550a82582ed',
            'f90e118e4b294be48f6382cd9f0f3198',
            '04afd1dd37d34d3fa8a2517fc0426966',
            'a682efaf27f2489ba89211087ea13911',
            '0d1325679a7b48caac79d93915063f30',
            'a3f509015f104eaa8d12c6b5341c2d94',
            '9dce0138447c483e8a93a5d46fc1215d',
            'a593c5f28f8045a486b3134bbab1887a',
            '32062b01f29e40aba91e32fc2df82e38',
            '8f38b5d2863f42009b36e3193c74fb21',
            '4a7b9ab58d9041658bdfbdfc18d31c51',
            '9e8ae6faf54b40f387161594b18ab258',
            '73859b18a5a247d98ee05e8c07fc0615',
            'afddf575f64b43598f85879454d640ec',
            '2206e6431158449bac4a1c961980a2bf',
            '8cc629ecc219452fabb8c46a9753b2f9',
            '63ed4be5bcf442df9890120a18adae1c',
            '6831816aefea43ef922cd783e02baefe',
            'a781bec5c61549238ef40c3572a50198',
            '5a4b76187a874948914f7fdb70f9633d',
            '2a80b05e6fee4681975e05254ee5ca72',
            'cf110eef02c14e36857b5649327531fa',
            'f50e731a811f42c58c0b499215bd86d9',
            '7c610ada6d4b415980cfe6e124ed5727'
        )
        sm = getstationmetadict(hydrosheds_path, metashp, 'euassi', stationsnottoevaluate=nottoeval)
        sc = StationHarmonization(sy, ey, bp, sm, 'D')
        sc.read_obs()
        sc.remove_rbis_duplicates()
        df = pd.concat([sc.grdcdf, sc.gsimdf, sc.rbisdf], axis=1)
        df.columns = [x[1] for x in df.columns]
        df.to_csv(out_path)
        pass


if __name__ == '__main__':
    eurasia('/home/home1/gm/projects/DRYvER/03_data/07_streamflowdata/10_harmonized/01_europe/harmonized_dataset_uncleaned.csv')
