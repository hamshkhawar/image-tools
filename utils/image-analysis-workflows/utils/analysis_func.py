import statistics
import numpy 
import pandas as pd
import re
import pathlib
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


def find_threshold(
        x: pd.DataFrame,
        variableName:str,
        value: float,
) -> float:
    values = list(x[variableName].values)
    false_positive_rate = value
    """ Computes a threshold value using a user-specified false positive rate.
    We assume that the `negative_values` follow a single gaussian distribution.
     We estimate the mean and standard deviation of this distribution and
     compute a threshold such that the area to the right of the threshold is
     equal to the given `false_positive_rate`.
    Args:
        values: drawn from a single gaussian distribution.
        false_positive_rate: A user-defined tuning parameter.
    Returns:
        The computed threshold value.
    """
    if not (0 < false_positive_rate < 1):
        raise ValueError(f'`false_positive_rate` mut be in the range (0, 1). Got {false_positive_rate:.2e} instead.')

    mu = float(numpy.nanmean(values))
    sigma = float(numpy.nanstd(values))

    distribution = statistics.NormalDist(mu, sigma)
    threshold = distribution.inv_cdf(1 - false_positive_rate)

    return threshold

def threshold_std(x:pd.DataFrame, variableName:str, value:int) -> int:
    mean = numpy.nanmean(list(x[variableName].values))
    std = numpy.nanstd(list(x[variableName].values))
    thr = round(mean + (std * value), 3)
    return thr


def threshold_otsu(x:pd.DataFrame, variableName:str, value:str) -> float:
    # Set total number of bins in the histogram
    values = list(x[variableName].values)
    bins_num = value
    #bins_num = BIN_COUNT

    # Get the image histogram
    hist, bin_edges = numpy.histogram(values, bins=bins_num)

    # Get normalized histogram if it is required
    # if is_normalized:
    # hist = numpy.divide(hist.ravel(), hist.max(initial=0))

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = numpy.cumsum(hist)
    weight2 = numpy.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = numpy.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (numpy.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = numpy.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]
    return threshold


def loading_csvs(path: pathlib.Path, pattern:str) -> pd.DataFrame:
    joined_files = os.path.join(path, pattern)
    df = pd.concat(map(pd.read_csv, glob.glob(joined_files)), ignore_index=True)
    return df


def loading_csv(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df





def metadata_processing(x:pd.DataFrame,
                        feat_data:pd.DataFrame, 
                        plate:str): 
    
    (x
     .rename(columns={'standard_name': 'intensity_image'},
             inplace=True)
    )
    
    flist = x.intensity_image.tolist()
    pattern = re.compile("p(?P<plate>\d+)_x(?P<column>\d+)_y(?P<row>\d+)_wx(?P<colpos>\d+)_wy(?P<rowpos>\d+)_c(?P<channel>\d+)")
    match = [pattern.match(i) for i in flist]
    row = [r.group("row") for r in match]
    col = [r.group("column") for r in match]
    rc = [r+c for r, c in zip(row, col)]

    well_assignment= {'01' : "A",'02': "B",'03':"C",'04':"D",
                      '05': "E",'06':"F",'07':"G",'08':"H",
                      '09':"I",'10':"J",'11':"K",'12':"L",
                      '13':"M",'14':"N",'15':"O",'16':"P"}

    well = [well_assignment.get(i[0:2])+i[2:] for i in rc]

    x['well_name'] = well
    ch = [re.search(".*c([0-9]{1})\.ome\.tif", f).groups() for f in flist]
    ch = list(zip(*ch))
    ch = list(ch[0])
    x['channel'] = pd.to_numeric(ch)
    pl = [re.search("p([0-9]{3}).*", f).groups() for f in flist]
    pl = list(zip(*pl))
    pl= list(pl[0])
    x['plate'] = pl
    x['plate'].astype(str)
    x = x[x['channel'] == 2]
    x = x[x['plate'] == str(plate)]
    xdf = x.merge(feat_data, how='outer', on='intensity_image')
    xdf.loc[(xdf.virus_negative == 1), 'Perturbation'] = 'virus_negative'
    xdf.loc[(xdf.virus_neutral == 1), 'Perturbation'] = 'virus_neutral'
    
    return xdf




def plot_cellsize_threshold(x, thr:int, outlier_p:int, plate:str, figDir:pathlib.Path):
    fig = plt.figure()
    ax = plt.subplot(111)
    x = x['AREA_PIXELS_COUNT']
    ax.hist(x, bins=200,range = [0,3000],align='left', color='skyblue', edgecolor='skyblue',
                rwidth=0.75)
    plt.xlabel('AREA_PIXELS_COUNT')
    plt.ylabel('Frequency')
    plt.title(f'Frequency Distribution of AREA Pixels Count \n Plate {plate}')
    plt.vlines(x=[thr], ymin=0, ymax=len(x)/8, colors='black', ls='--', lw=1, label='higher threshold: MEAN + 4SD') 
    plt.vlines(x=20, ymin=0, ymax=len(x)/8, colors='red', ls='--', lw=1, label='lower threshold: 20')  
    plt.text(1500, x.shape[0] // 15, f'Total cell Count: {x.shape[0]} \n outliers: {outlier_p} %', fontsize = 12)
    plt.xlim([0, 4000])
    # plt.ylim([0, 100000])
    ax.legend()
    plt.show()
    fig.savefig(pathlib.Path(figDir, f'histogram_area_pixel_count_p{plate}'),dpi=300, bbox_inches = 'tight')
    return


def plotting_control_distribution(func, 
                                  x:pd.DataFrame,
                                  variableName:str,
                                  value:int, 
                                  threshType:str, 
                                  plate:str,
                                  figDir:pathlib.Path, 
                                  otsu=False):
    
    controlname = ['virus_negative', 'virus_neutral']

    if otsu:
        controls = x.query('Perturbation in @ controlname')
    else:
        controls = x.query('Perturbation == "virus_negative"')

    thr = float("{:.2f}".format(func(controls, variableName, value)))
              
    fig = plt.figure()

    for cont in controlname:
        seltmp = x.loc[x['Perturbation'] == cont]
        sns.distplot(seltmp['MEAN'], hist=False, kde=True, kde_kws = {'shade':True, 'linewidth': .5}, label=cont)

    # # Plot formatting
    plt.legend(prop={'size': 10}, title = 'Controls')
    plt.vlines(x=[thr], ymin=0, ymax=0.008, colors='black', ls='--', lw=1, label=threshType)
    plt.text(1400, .005, f'{threshType}: {thr}', fontsize = 12)
    plt.title(f'Distribution of Cell MEAN Intensity \n Plate:{plate}')
    plt.xlabel('MEAN INTENSITY')
    plt.ylabel('Density')
    plt.xlim([0, 5000])
    plt.show()
    threshType=''.join(threshType.split())
    fig.savefig(pathlib.Path(figDir, f'plate{plate}_Distribution_of_single_cell_MEAN_{threshType}'),dpi=300, bbox_inches='tight')

    return

def thresholds_distributions_plots(x:pd.DataFrame,
                                  thr_methods,
                                  values,
                                  otsus,
                                  names,
                                  variableName:str,
                                  figDir:pathlib.Path, 
                                  plate:str
                                     ): 
    
    for thr_method, value, otsu, name in zip(thr_methods, values, otsus, names):

        plotting_control_distribution(thr_method, 
                                        x,
                                        variableName=variableName, 
                                        value=value,
                                        threshType=name,
                                        plate=plate,
                                        figDir=figDir,
                                        otsu=otsu)
    
    return

def threshold_based_cell_assignment(func, 
                                    x:pd.DataFrame, 
                                    variableName:str,
                                    value:int, 
                                    otsu=False) -> pd.DataFrame:        

        if otsu:
            controlname = ['virus_negative', 'virus_neutral']
            controls = x.query('Perturbation in @ controlname')
        else:
            controls = x.query('Perturbation == "virus_negative"')


        thr = func(controls, variableName, value)

        x['Covid_Exp'] = (x[variableName]
                             .apply(lambda x: 'Pos' if x > thr else 'Neg')
                                )
        Total_count = (x
                            .groupby(['well_name'])
                            .count().reset_index()
                            .iloc[:, :2]
                            .rename(columns={'raw_directory': 'Total'})
                            )
        exp_cells = (x.groupby(['well_name','Covid_Exp'])
                .count()
                .reset_index()
                .iloc[:, :3]
                .rename(columns={'raw_directory': 'CellCount'})
                )

        merged_df = Total_count.merge(exp_cells,  how='outer', on=['well_name'])
        wellname = pd.DataFrame(merged_df.well_name.unique(), columns=['well_name'])
        merged_df = merged_df.assign(Nproportion = lambda x: (x['CellCount'] / x['Total']) * 255)
        #         merged_df = merged_df.assign(Nproportion = lambda x: (x['CellCount'] / x['Total']) * 100)
        merged_df  = merged_df.query('Covid_Exp == "Pos"')[['well_name', 'Total', 'Nproportion']]
        merged_df.rename(columns={'Nproportion': f'{value}-Nproportion', 
                                  'Total': 'CellCount'}, inplace=True)
        merged_df = wellname.merge(merged_df, how='outer', on=['well_name'])
        merged_df.replace(numpy.nan, 0.0, inplace=True)
        return merged_df


def threshold_calculations(x:pd.DataFrame(),
                          variableName:str,
                          thr_methods,
                          values,
                          otsus:List[bool]):
    
    prf = pd.DataFrame()

    for thr_method, value, otsu in zip(thr_methods, values, otsus):

        mdf = threshold_based_cell_assignment(func=thr_method, 
                                        x=x, 
                                        variableName=variableName,
                                        value=value, 
                                        otsu=otsu)

        prf = pd.concat([prf, mdf], axis=1)   
    prf = prf.loc[:, ~prf.columns.duplicated()]
        
    prf.rename(columns={'0.001-Nproportion': 'FPR-0.001', 
                 '1e-05-Nproportion': 'FPR-0.00001', 
                 '512-Nproportion': 'OTSU',
                 '4-Nproportion': 'Mean+4STD',
                  }, inplace=True)


    
    return prf



def heatmap_visualization(x:pd.DataFrame, 
                          variableName:str,
                          title:str,
                          plate:str,
                          figDir:pathlib.Path,
                          annot=False
                         ) -> pd.DataFrame:
    # x = x.drop_duplicates(subset ="well_name",
    #                      keep='first', inplace = False)
    x = x[variableName].values.reshape(16,24)
    fig, ax = plt.subplots(figsize=(15,8))
    yticks_labels = ['A', 'B', 'C', 'D', 'E', 'F',
                'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    l = list(numpy.arange(1,25))
    xticks_labels = l
    ax = sns.heatmap(x , linewidth = 0.5 , cmap = 'coolwarm', square=True, annot=annot, fmt="d")
    ax.set_yticklabels(yticks_labels, rotation=0)
    ax.set_xticklabels(xticks_labels)
    plt.title(f'{title} \n Plate:{plate}', fontsize=18)
    plt.show()
    title = '_'.join(title.split())
    outname = f'plate{plate}_Heatmap_{title}.png'
    path = pathlib.Path(figDir)
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(pathlib.Path(figDir, outname))
