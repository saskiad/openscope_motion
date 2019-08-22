#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:01:11 2018

@author: saskiad
"""

import pg8000
import pandas as pd

def make_targeted_manifest(savepath=r'/Volumes/programs/braintv/workgroups/nc-ophys/VisualCoding/targeted_manifest.csv'):
    conn = pg8000.connect(user="limsreader", host="limsdb2", database="lims2", password="limsro", port=5432)
    cursor = conn.cursor()
    
    cursor.execute("select ophys_sessions.date_of_acquisition, ophys_sessions.specimen_id, ophys_sessions.stimulus_name, ophys_sessions.storage_directory, ophys_sessions.id as ophys_session_id, projects.code as project_code, structures.acronym as targeted_structure, donors.name as donor_name, genotypes.name as genotype_name, imaging_depths.depth as imaging_depth from ophys_sessions join projects on ophys_sessions.project_id = projects.id join structures on ophys_sessions.targeted_structure_id = structures.id join specimens on ophys_sessions.specimen_id = specimens.id join donors on specimens.donor_id = donors.id join donors_genotypes on donors.id = donors_genotypes.donor_id join genotypes on donors_genotypes.genotype_id = genotypes.id join imaging_depths on ophys_sessions.imaging_depth_id = imaging_depths.id where projects.code ilike 'OpenscopeMotionPilot' and genotypes.name ilike '%cre%'")
    
    columns = [ d[0] for d in cursor.description ]
    table = [ dict(zip(columns, c)) for c in cursor.fetchall() ]
    
    conn.close()
    
    targeted_expts = pd.DataFrame(table)
    targeted_expts.to_csv(savepath)
    return targeted_expts

targeted_expts = make_targeted_manifest()

###
#cursor.execute("select ophys_sessions.date_of_acquisition, ophys_sessions.specimen_id, ophys_sessions.stimulus_name, ophys_sessions.storage_directory, ophys_sessions.id as ophys_session_id, 
#               projects.code as project_code, structures.acronym as targeted_structure, donors.name as donor_name, genotypes.name as genotype_name, imaging_depths.depth as imaging_depth 
#               from ophys_sessions 
#               join projects on ophys_sessions.project_id = projects.id 
#               join structures on ophys_sessions.targeted_structure_id = structures.id 
#               join specimens on ophys_sessions.specimen_id = specimens.id 
#               join donors on specimens.donor_id = donors.id 
#               join donors_genotypes on donors.id = donors_genotypes.donor_id 
#               join genotypes on donors_genotypes.genotype_id = genotypes.id 
#               join imaging_depths on ophys_sessions.imaging_depth_id = imaging_depths.id 
#               where projects.code ilike 'VisCodingTargeted%' and genotypes.name ilike '%cre%'")
###

