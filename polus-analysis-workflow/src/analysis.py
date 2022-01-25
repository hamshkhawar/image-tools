import os
import glob
from pathlib import Path
import pandas as pd
from re import search
import re
import plotnine as gg
from plotnine import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as numpy
import typing
from sklearn.preprocessing import StandardScaler

path = Path('../outputs')


def loading_csv(path: Path) -> str:
    joined_files = os.path.join(path, '*.csv')
    df = pd.concat(map(pd.read_csv, glob.glob(joined_files)), ignore_index=True)
    return df

def combining_csvs(func, path: Path):
    compartments = ['nuclei', 'cell', 'cytoplasm']
    df= pd.DataFrame()
    for i in compartments:
        comp_path = Path(path, i)
        csv = func(comp_path)
        csv.columns = [i+'_'+col for col in csv.columns]
        df = pd.concat([df, csv], axis=1)
        (df
        .rename(columns={'nuclei_mask_image': 'mask_image',
                'nuclei_intensity_image': 'intensity_image',
                'nuclei_label':'label'}, inplace=True))
        df.drop(df.columns[df.columns.str.contains("image|label")].tolist()[3:], axis=1, inplace=True)
        
    return df



def metadata_extraction(x):

    wellname = x['mask_image'].tolist()
    pattern = re.compile("x(?P<column>\d+)_y(?P<row>\d+)_wx(?P<colpos>\d+)_wy(?P<rowpos>\d+)_c(?P<channelnumber>\d+)")

    match = [pattern.match(i) for i in wellname]
    row = [r.group("row") for r in match]
    col = [r.group("column") for r in match]
    rc = [r+c for r, c in zip(row, col)]

    well_assignment= {'01' : "A",'02': "B",'03':"C",'04':"D",
                      '05': "E",'06':"F",'07':"G",'08':"H",
                      '09':"I",'10':"J",'11':"K",'12':"L",
                      '13':"M",'14':"N",'15':"O",'16':"P"}

    well = [well_assignment.get(i[0:2])+i[2:] for i in rc]
    x['Metadata_Well'] = well
    x['Metadata_Column'] = col
    x['Metadata_Row'] = row

    control_dict = {
        '01': 'Negcontrol', 
        '02': 'Poscontrol'
    } 
    x['Metadata_Perturbation_type'] = (x['Metadata_Column']
                    .map(control_dict)
                    .fillna('Treatments')
                    )

    return x

def znormalization(x):
       scaler = StandardScaler()
       controls = x.query('Controls == "Negcontrol"')
       metalist = ['mask_image', 'intensity_image', 'label','Metadata_Well', 'Metadata_Column','Metadata_Row',
              'Metadata_Perturbation_type']
       controls = controls[controls.columns[~controls.columns.isin(metalist)]]
       nor_data = (scaler.fit(controls)
                     .transform(pf[pf.columns[~pf.columns.isin(metalist)]])
       ) 

       meta = x[x.columns[x.columns.isin(metalist)]]

       varlist = [col for col in x.columns if not col in metalist]

       df_nor = pd.DataFrame(nor_data, columns=varlist)

       normalized_x = pd.concat([meta, df_nor], axis=1)

       return normalized_x

pdf = combining_csvs(loading_csv, path)
prf = metadata_extraction(pdf)


def threshold_std(data:Any, VARIABLE_NAME:str, value:int):
    mean = numpy.mean(data[VARIABLE_NAME].values)
    std = numpy.std(data[VARIABLE_NAME].values)
    thr = round(mean + (std * value), 3)
    return thr

def threshold_nthPercentile(data:Any, VARIABLE_NAME:str, value:str):
    thr = round(numpy.percentile(data[VARIABLE_NAME], value), 3)
    return thr


def threshold_otsu(values: numpy.ndarray) -> float:
    # Set total number of bins in the histogram
    bins_num = 512
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



def threshold_based_cell_assignment(thresholding_func, 
                                    data:pd.DataFrame, 
                                    VARIABLE_NAME:str,
                                    value:int, 
                                    otsu=False):
        if otsu:
            controlname = ['Negcontrol', 'Poscontrol']
            controls = data.query('Metadata_Perturbation_type in @ controlname')
        else:
            controls = data.query('Metadata_Perturbation_type == "Negcontrol"')

        thr = thresholding_func(controls, VARIABLE_NAME, value)

        data['Covid_Exp'] = (data[VARIABLE_NAME]
                             .apply(lambda x: 'Pos' if x > thr else 'Neg')
                                )
        Total_count = (data
                            .groupby(['Metadata_Well'])
                            .count().reset_index()
                            .iloc[:, :2]
                            .rename(columns={'mask_image': 'Total'})
                            )
        exp_cells = (data.groupby(['Metadata_Well','Covid_Exp'])
                .count()
                .reset_index()
                .iloc[:, :3]
                .rename(columns={'mask_image': 'CellCount'})
                )

        merged_df = Total_count.merge(exp_cells,  how='outer', on=['Metadata_Well'])
        merged_df = merged_df.assign(percentage = lambda x: numpy.round((x['CellCount'] / x['Total']) * 100, 5))
        merged_df  = merged_df.query('Covid_Exp == "Pos"')[['Metadata_Well', 'percentage']]
        merged_df.replace(numpy.nan, 0.0, inplace=True)



        return merged_df

def heatmap_visualization(data:pd.DataFrame, 
                         VARIABLE_NAME:str,
                         figpath:Path):
    data = data['percentage'].values.reshape(16,24)
    fig, ax = plt.subplots(figsize=(15,8))
    yticks_labels = ['A', 'B', 'C', 'D', 'E', 'F',
                'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    l1 = ['Neg', 'Pos']
    l2 = list(numpy.arange(3,25))
    xticks_labels = l1 + l2
    ax = sns.heatmap(data , linewidth = 0.5 , cmap = 'coolwarm', square=True)
    ax.set_yticklabels(yticks_labels, rotation=0)
    ax.set_xticklabels(xticks_labels)
    plt.title("Percent Positive cells for Covid Expression", fontsize=18)
    plt.show()
    outname = f'Heatmap_Percent_positive_cells_CovidExpresssion_{VARIABLE_NAME}'
    path = Path(figpath)
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(os.path.join(path, outname))

heatmap_visualization(merged_df, VARIABLE_NAME, figpath)


def plotting_boxplots(data:pd.DataFrame, figpath:Path):

    figpath = Path(figpath)
    figpath.mkdir(parents=True, exist_ok=True)

    controlist = ['Negcontrol', 'Poscontrol']
    plotcol = data.columns[3:-6]
    data = data.query('Metadata_Perturbation_type in @ controlist')
    for i in plotcol:
        g = (ggplot(data=data,
                mapping=aes(x='Controls',
                                y=i, fill = 'Controls'))
            + geom_boxplot(width=0.2)
            + geom_jitter(width=0.1)
            + ylab('Mean Well ' + str(i))
            + xlab(' ')
            + theme_classic()
        )
        ggsave(filename= "boxplot_"+ str(i) + '.png', plot = g, path = figpath)