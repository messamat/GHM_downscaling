#See ContinentInfo.csv from https://github.com/JKupzig/WaterGAPLite/blob/main/data-raw/ContinentInfo.csv for adding other continents
def get_continental_extent(continentlist):
    extentdict = {'eu': (-31.333333333, 52, 27.416666667, 80.833333333)}
                  #,)
                  #'as': (57, 152, 0, 68),
                  #'si': (58, 180, 45, 84),
                  #'sa': (-93, -32, -56, 15)}
    xmin, xmax, ymin, ymax = extentdict[continentlist[0]]
    if len(continentlist) > 1:
        for x in continentlist[1:]:
            tmpxmin, tmpxmax, tmpymin, tmpymax = extentdict[x]
            xmin = min(xmin, tmpxmin)
            xmax = max(xmax, tmpxmax)
            ymin = min(tmpymin, ymin)
            ymax = max(tmpymax, ymax)
    return xmin, xmax, ymin, ymax