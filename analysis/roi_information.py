#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:33:26 2019

@author: saskiad

Creates a dataframe of ROI mask information, unique id, valid calls for an imaging session
"""

import os
import json

import pandas as pd


def get_roi_information(storage_directory):
    """Parse ROI information from JSON file into a DataFrame.

    Input:
        storage_directory (str)
            -- Path to folder in which to look for ROI info JSON file.

    Returns:
        DataFrame with ROI mask information with the following columns:
            id     -- ROI number???
            x      -- x coordinate of ROI center
            y      -- y coordinate of ROI center
            width  -- width of ROI
            height -- height of ROI
            valid  -- ???
            mask   -- Boolean mask for pixels included in ROI

    """
    exp_path_head = storage_directory

    #reformat path for mac with local access
    #TODO: might need to adapt this when data is shared via DropBox
#    temp = exp_path_head.split('/')
#    temp[1] = 'Volumes'
#    exp_path_head = '/'.join(temp)

    # Find experiment dir in storage_directory
    exp_path_files = os.listdir(exp_path_head)
    exp_folder_list = [i for i in exp_path_files if 'ophys_experiment' in i]
    if len(exp_folder_list) > 1:
        raise Exception('Multiple experiment folders in '+exp_path_head)
    else:
        exp_folder = exp_folder_list[0]

    # Find file by suffix
    processed_path = os.path.join(exp_path_head, exp_folder)
    for fname in os.listdir(processed_path):
        if fname.endswith('output_cell_roi_creation.json'):
            jsonpath = os.path.join(processed_path, fname)
            with open(jsonpath, 'r') as f:
                jin = json.load(f)
                f.close()
            break

    # Assemble DataFrame.
#    roi_locations = pd.DataFrame.from_records(
#        data = jin['rois'],
##        columns = ['id', 'x', 'y', 'width', 'height', 'valid', 'mask']#for input_cell_extract...
##        columns=['valid','width','height','x','y','mask','exclusion_labels','exclude_code','mask_page']
#        columns=jin['rois'][jin['rois'].keys()[0]].keys()
#    )
    roi_locations = pd.DataFrame.from_dict(jin['rois'], orient='index')
    session_id = int(exp_path_head.split('/')[-1].split('_')[-1])
    roi_locations['session_id'] = session_id
    roi_locations.reset_index(drop=True, inplace=True)
    return roi_locations, session_id

if __name__ == "__main__":
    datapath = r'/Volumes/My Passport/Openscope Motion'

    for f in os.listdir(datapath):
        if f.startswith('ophys_session'):
            try:
                newdatapath = os.path.join(datapath, f)
                roi_locations, session_id = get_roi_information(newdatapath)
                roi_locations.to_csv(os.path.join(r'/Users/saskiad/Dropbox/Openscope Motion/roi_tables', str(session_id)+'_rois.csv'))
            except:
                print(newdatapath)