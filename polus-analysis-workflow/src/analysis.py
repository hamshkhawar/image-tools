import os, glob, re, typing, pathlib, logging
import pandas as pd
from re import search
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger('analysis')
logger.setLevel(logging.INFO)

def loading_csv(path: pathlib.Path, pattern:str) -> str:
    joined_files = os.path.join(path, pattern)
    df = pd.concat(map(pd.read_csv, glob.glob(joined_files)), ignore_index=True)
    return df

def metadata_extraction(x:pd.DataFrame, plate:str) -> pd.DataFrame:

    wellname = x['mask_image'].tolist()
    pattern = re.compile("p(?P<plate>\d+)_x(?P<column>\d+)_y(?P<row>\d+)_wx(?P<colpos>\d+)_wy(?P<rowpos>\d+)_c(?P<channelnumber>\d+)")

    match = [pattern.match(i) for i in wellname]
    plate=[r.group('plate') for r in match]
    row = [r.group("row") for r in match]
    col = [r.group("column") for r in match]
    rc = [r+c for r, c in zip(row, col)]

    well_assignment= {'01' : "A",'02': "B",'03':"C",'04':"D",
                      '05': "E",'06':"F",'07':"G",'08':"H",
                      '09':"I",'10':"J",'11':"K",'12':"L",
                      '13':"M",'14':"N",'15':"O",'16':"P"}

    well = [well_assignment.get(i[0:2])+i[2:] for i in rc]
    x['Metadata_Plate'] = plate
    x['Metadata_Well'] = well
    x['Metadata_Column'] = col
    x['Metadata_Row'] = row

    if plate=='00':
        control_dict = {
            '01': 'Negcontrol', 
            '02': 'Poscontrol'
        } 
    else:
        control_dict = {
            '01': 'Negcontrol', 
            '21': 'Poscontrol',
            '22': 'Poscontrol',
            '23': 'Poscontrol',
        }

    x['Metadata_Perturbation_type'] = (x['Metadata_Column']
                    .map(control_dict)
                    .fillna('Treatments')
                    )

    return x

def threshold_std(x:pd.DataFrame, variableName:str, value:int) -> int:
    mean = np.mean(x[variableName].values)
    std = np.std(x[variableName].values)
    thr = round(mean + (std * value), 3)
    return thr

def threshold_otsu(x:pd.DataFrame, variableName:str, value:str) -> float:
    # Set total number of bins in the histogram
    values = x[variableName].values
    bins_num = value
    #bins_num = BIN_COUNT

    # Get the image histogram
    hist, bin_edges = np.histogram(values, bins=bins_num)

    # Get normalized histogram if it is required
    # if is_normalized:
    # hist = np.divide(hist.ravel(), hist.max(initial=0))

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]
    return threshold

def plot_cellsize_threshold(x, thr:int, outlier_p:int, plate:str, figDir:pathlib.Path):
    fig = plt.figure()
    ax = plt.subplot(111)
    x = x['AREA_PIXELS_COUNT']
    ax.hist(x, bins=200,range = [0,3000],align='left', color='skyblue', edgecolor='skyblue',
                rwidth=0.75)
    plt.xlabel('AREA_PIXELS_COUNT')
    plt.ylabel('Frequency')
    plt.title(f'Frequency Distribution of AREA Pixels Count \n Plate {plate}')
    plt.vlines(x=[thr], ymin=0, ymax=len(x)/10, colors='black', ls='--', lw=1, label='Threshold: MEAN + SD')    
    plt.text(1500, x.shape[0] // 15, f'Total cell Count: {x.shape[0]} \n outliers: {outlier_p} %', fontsize = 12)
    plt.xlim([0, 4000])
    # plt.ylim([0, 100000])
    ax.legend()
    plt.show()
    fig.savefig(pathlib.Path(figDir, f'histogram_area_pixel_count_p{plate}'),dpi=300, bbox_inches = 'tight')
    return


def plotting_control_distribution(func, x:pd.DataFrame,variableName:str, value:int, plate:str, figDir:pathlib.Path, otsu=False):
    controlname = ['Negcontrol', 'Poscontrol']
    controls = x.query('Metadata_Perturbation_type in @ controlname')
    if otsu:  
        thr = float("{:.2f}".format(func(controls, variableName, value)))

    for cont in controlname:
        seltmp = x.loc[x['Metadata_Perturbation_type'] == cont]
        sns.distplot(seltmp['MEAN'], hist=False, kde=True, kde_kws = {'shade':True, 'linewidth': .5}, label=cont)


    # # Plot formatting
    plt.legend(prop={'size': 10}, title = 'Controls')
    plt.vlines(x=[thr], ymin=0, ymax=0.008, colors='black', ls='--', lw=1, label='OTSU Threshold')
    plt.text(1400, .005, f'otsu threshold: {thr}', fontsize = 12)
    plt.title(f'Distribution of Cell MEAN Intensity \n Plate:{plate}')
    plt.xlabel('MEAN INTENSITY')
    plt.ylabel('Density')
    plt.xlim([0, 5000])
    plt.show()
    plt.savefig(pathlib.Path(figDir, f'Distribution_of_single_cell_MEAN_p{plate}'),dpi=300, bbox_inches='tight')

    return 

def threshold_based_cell_assignment(func, 
                                    x:pd.DataFrame, 
                                    variableName:str,
                                    value:int, 
                                    otsu=False) -> pd.DataFrame:
        if otsu:
            controlname = ['Negcontrol', 'Poscontrol']
            controls = x.query('Metadata_Perturbation_type in @ controlname')
        else:
            controls = x.query('Metadata_Perturbation_type == "Negcontrol"')

        thr = func(controls, variableName, value)

        x['Covid_Exp'] = (x[variableName]
                             .apply(lambda x: 'Pos' if x > thr else 'Neg')
                                )
        Total_count = (x
                            .groupby(['Metadata_Well'])
                            .count().reset_index()
                            .iloc[:, :2]
                            .rename(columns={'mask_image': 'Total'})
                            )
        exp_cells = (x.groupby(['Metadata_Well','Covid_Exp'])
                .count()
                .reset_index()
                .iloc[:, :3]
                .rename(columns={'mask_image': 'CellCount'})
                )

        merged_df = Total_count.merge(exp_cells,  how='outer', on=['Metadata_Well'])
        wellname = pd.DataFrame(merged_df.Metadata_Well.unique(), columns=['Metadata_Well'])
        merged_df = merged_df.assign(percentage = lambda x: np.round((x['CellCount'] / x['Total']) * 100, 5))
        merged_df  = merged_df.query('Covid_Exp == "Pos"')[['Metadata_Well', 'percentage']]
        merged_df = wellname.merge(merged_df, how='outer', on=['Metadata_Well'])
        merged_df.replace(np.nan, 0.0, inplace=True)
        return merged_df

def heatmap_visualization(x:pd.DataFrame, 
                          variableName:str,
                          title:str,
                          plate:str,
                          figDir:pathlib.Path
                         ) -> pd.DataFrame:
    x = x[variableName].values.reshape(16,24)
    fig, ax = plt.subplots(figsize=(15,8))
    yticks_labels = ['A', 'B', 'C', 'D', 'E', 'F',
                'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']

    if plate=="00":
        ax = sns.heatmap(x , linewidth = 0.5 , cmap = 'coolwarm', square=True)
        l1 = ['Neg', 'Pos']
        l2 = list(np.arange(3,25))
        xticks_labels = l1 + l2
    else:
        ax = sns.heatmap(x , linewidth = 0.5 , cmap = 'coolwarm', square=True)
        l1 = ['Neg']
        l2 = list(np.arange(2,21))
        l3 =['Pos', 'Pos', 'Pos']
        l4 = [24]
        xticks_labels = l1 + l2 + l3 + l4
    ax.set_yticklabels(yticks_labels, rotation=0)
    ax.set_xticklabels(xticks_labels)
    plt.title(f'{title} \n Plate:{plate}', fontsize=18)
    plt.show()
    outname = f'Heatmap_{title}_plate{plate}_{variableName}'
    path = pathlib.Path(figDir)
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(pathlib.Path(figDir, outname))


def analysis_worflow(inpDir:pathlib.Path, plate:str, outDir:pathlib.Path):
    figDir = pathlib.Path(outDir, 'Figures')
    figDir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Creating a figure directory:{figDir}")
    logger.info("Analysis workflow is running")
    pattern = f'*p{plate}*ome.tif.csv'
    logger.info(f'Step 1: loading CSVs with pattern: {pattern}')
    df = loading_csv(inpDir, pattern)
    logger.info("CSVs are loaded")
    logger.info(f'Step 2: Extracting Metadata')
    prf = metadata_extraction(df, plate)
    logger.info(f'Step 2: Finished assigning Metadata')
    logger.info(f'Step 3: Threshold estimation for cell size')
    outlier_thr = threshold_std(prf, variableName='AREA_PIXELS_COUNT', value=1) 
    logger.info(f'Step 3: Finished threshold estimation')
    logger.info(f'Step 5: Plotting Figure 1: cellsize thresholding')
    outliers_removed = [x for x in prf['AREA_PIXELS_COUNT'] if x > outlier_thr]   
    outlier_p = float("{:.2f}".format(len(outliers_removed)/prf.shape[0] * 100))
    plot_cellsize_threshold(prf, outlier_thr, outlier_p, plate, figDir)
    logger.info(f'Step 5: Finished Plotting Figure 1: cellsize thresholding')
    logger.info(f'Step 6: Data cleaning')
    dfclean =prf.query('AREA_PIXELS_COUNT < @outlier_thr')
    logger.info(f'Step 6: Finished Data cleaning')
    logger.info(f'Step 7: Calculating Mean Well Intensity & Cell Count')
    well_meanI = (dfclean
           .groupby(by=['Metadata_Well'])['MEAN'].mean()
           .reset_index()
           .round(decimals=2)
    )
    mean_cell_count = (dfclean
           .groupby(by=['Metadata_Well'])['MEAN'].count()
           .reset_index()
           .round(decimals=2)
    )
    logger.info(f'Step 7: Finished Calculating Mean Well Intensity & Cell Count')
    logger.info(f'Step 8: Plotting Figure 2: Thesholding_otsu')
    plotting_control_distribution(threshold_otsu, 
                                  dfclean,
                                  variableName='MEAN', 
                                  value=512, 
                                  plate=plate, 
                                  figDir=figDir,
                                  otsu=True)
    logger.info(f'Step 8: Finished Plotting Figure 2: Thesholding_otsu')
    logger.info(f'Step 9: Plotting Figure 3: Heatmap of Theshold Based Cell assignments')
    merged = threshold_based_cell_assignment(threshold_otsu, 
                                        dfclean, 
                                        variableName='MEAN',
                                        value=512, 
                                        otsu=True)
    heatmap_visualization(merged,
                          variableName='percentage',
                          title='Percent Positive Cells for Covid Expression', 
                          plate=plate,
                          figDir=figDir)
    logger.info(f'Step 9: Finished Plotting Figure 3: Theshold Based Cell assignments')
    logger.info(f'Step 10: Plotting Figure 4: Heatmap of Mean cell count')
    heatmap_visualization(mean_cell_count,
                          variableName='MEAN',
                          title='Cell count per well',
                          plate=plate, 
                          figDir=figDir)
    logger.info(f'Step 9: Finished Plotting Figure 4: Heatmap of Mean Cell Count')
    logger.info(f'Step 10: Plotting Figure 5: Heatmap of Mean well intensity')
    heatmap_visualization(well_meanI, 
                          variableName='MEAN',
                          title='Mean Well Covid Reporter Expression',
                          plate=plate,
                          figDir=figDir)
    logger.info(f'Step 10: Finished Plotting Figure 5: Heatmap of Mean well intensity')
    logger.info(f'Analyis pipeline is completed!!!')      
    return