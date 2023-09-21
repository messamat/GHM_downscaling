def get_continental_extent(continentlist):
    extentdict = {'eu': (-25, 70, 12, 84),
                  'as': (57, 152, 0, 68),
                  'si': (58, 180, 45, 84),
                  'sa': (-93, -32, -56, 15)}
    xmin, xmax, ymin, ymax = extentdict[continentlist[0]]
    if len(continentlist) > 1:
        for x in continentlist[1:]:
            tmpxmin, tmpxmax, tmpymin, tmpymax = extentdict[x]
            xmin = min(xmin, tmpxmin)
            xmax = max(xmax, tmpxmax)
            ymin = min(tmpymin, ymin)
            ymax = max(tmpymax, ymax)
    return xmin, xmax, ymin, ymax