from src import config as cf
from src import preprocess as pre
from src import utils as ut

def main():
    eval = cf.EVAL

    for path in cf.DATA_PATHS:

        #Load Data Sets
        raw, sensor, df = pre.load_sensor_data(path)
        if eval == True:
            ut.count_issues(df)

        #Frequency Reindexing
        rd = pre.reindex(raw)
        if eval == True:
            ut.reindex_report(rd, raw)
  
        # Hampel Filter
        hampel_df = pre.hampel_filter(rd)
        if eval == True:
            ut.filter_report(rd, hampel_df) 
        
        # Interpolation of Segments
        fill_seg = pre.interpolate(hampel_df)
        if eval == True:
            ut.interpolate_report(rd, fill_seg)

        # Segments valid timeseries
        segments, report = pre.segment_gaps(fill_seg)
        processed_segments = [pre.rolling(seg) for seg in segments]
        if eval:
            ut.seg_to_csv(processed_segments)

        #Thresholding 
        
        
        # Interactive Plotting
        states = {
            "raw": raw,
            "hampel": hampel_df
        }
        ut.interactive_plots(states, sensor_cols=cf.SENSOR_COLUMNS, datetime_col=cf.DATETIME_COL)
        
if __name__ == "__main__":
    main()