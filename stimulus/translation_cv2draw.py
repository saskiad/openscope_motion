#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 23:13:10 2019

@author: corbennett
"""

import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

viewingDistance = 100
radius = 10
speed = 4#1
dotNum = 20
timePoints = 500#1000
pixelsPerDegree = 10 
screenWidth = 1960
screenHeight = 1200
centerPosDegrees = np.array([-60,-20])

#Function to generate random dot positions at specified distance range from center
def getNewDotPositions(dotNum, minDist=5, maxDist=90):
    minDist *= pixelsPerDegree
    maxDist *= pixelsPerDegree
    dists = np.random.uniform(minDist, maxDist, size=dotNum)
    angles = np.pi * np.random.uniform(0, 2, size=dotNum)
    x = dists * np.cos(angles)
    y = dists * np.sin(angles)
    return np.stack([x,y]).T

#Function to repopulate dots that are past 90 degrees visual angle
def repop(repopInds, distToCenter, newPos, unitVectors, newTheta, virtualTimePoint):
    newPos[repopInds] = getNewDotPositions(repopInds.size) + centerPos
    distToCenter[repopInds] = np.sqrt(np.sum((newPos[repopInds] - centerPos)**2, axis=1))
    unitVectors[repopInds] = (newPos[repopInds] - centerPos)/distToCenter[repopInds, None]
    newTheta[repopInds] = np.deg2rad(distToCenter[repopInds]/pixelsPerDegree)
    virtualTimePoint[repopInds] = (viewingDistance - (radius/np.tan(newTheta[repopInds])))/speed
    
    return distToCenter, newPos, unitVectors, newTheta, virtualTimePoint

#Function to redraw dots for the next time point
def redraw(distToCenter, newPos, unitVectors, newTheta, virtualTimePoint):
    #increment "time point" for each dot
    virtualTimePoint += 1
    
    #find new visual angle at that time point
    newTheta = np.arctan(radius/(viewingDistance - speed*(virtualTimePoint+1)))
    
    #convert angle to pixels and update dot positions
    newThetaPix = np.rad2deg(newTheta)*pixelsPerDegree
    newPos = unitVectors*newThetaPix[:, None] + centerPos
    
    #find dots past 90 and reorient them towards the perspective point at 180 degrees
    past90 = np.rad2deg(newTheta)<0
    if np.sum(past90)>0:
        unitVectors_past90 = np.copy(unitVectors[past90])
        unitVectors_past90 = unitVectors_past90 * [-1,1] #reflect about vertical
        newTheta[past90]  = np.abs(newTheta[past90])
        newThetaPix[past90] = np.rad2deg(newTheta[past90])*pixelsPerDegree
        newPos[past90] = unitVectors_past90*newThetaPix[past90][:,None] + centerPos + [180*pixelsPerDegree,0]
    
    onscreen = np.array([(0<p[0]<screenWidth) & (0<p[1]<screenHeight) for p in newPos])
    #repop if necessary
    #if not all(onscreen):
    #if any(past90 & (np.rad2deg(newTheta)<30)):
        #repopInds = np.where(~onscreen)[0]
#        repopInds = np.where(past90 & (np.rad2deg(newTheta)<30))[0]
#        distToCenter, newPos, unitVectors, newTheta, virtualTimePoint = repop(repopInds, distToCenter, newPos, unitVectors, newTheta, virtualTimePoint)
    repopInds = np.where(~onscreen)[0]
    if len(repopInds)>0:
        distToCenter, newPos, unitVectors, newTheta, virtualTimePoint = repop(repopInds, distToCenter, newPos, unitVectors, newTheta, virtualTimePoint)
    
    return distToCenter, newPos, unitVectors, newTheta, virtualTimePoint


#pick dots at random screen positions with respect to center point
centerPos= centerPosDegrees*pixelsPerDegree + np.array([screenWidth/2, screenHeight/2])
pos = getNewDotPositions(dotNum, 5, 10) + centerPos

#find how far they are from center and find unit vectors along which to move them
distToCenter = np.sqrt(np.sum((pos - centerPos)**2, axis=1))
unitVectors = (pos - centerPos)/distToCenter[:, None]

#find the visual angle at which each dot lives
theta = np.deg2rad(distToCenter/pixelsPerDegree)

#find the "time point" when each dot would reach its visual angle if it had started at the original viewing distance
virtualTimePoint = (viewingDistance - (radius/np.tan(theta)))/speed

#probably unnecessary
newTheta = np.copy(theta)
newPos = np.copy(pos)


pos_array = []
for i in np.arange(timePoints):
    distToCenter, newPos, unitVectors, newTheta, virtualTimePoint = redraw(distToCenter, newPos, unitVectors, newTheta, virtualTimePoint)
    pos_array.append(newPos)
    
    
    
scaleFactor = 1/20.
im_array = []
for points in pos_array:
    im = np.ones([screenHeight, screenWidth])*127
    for p in points:
        dist = ((p[0] - centerPos[0])**2 + (p[1]-centerPos[1])**2)**0.5
        dist180 = ((p[0] - centerPos[0] - 180*pixelsPerDegree)**2 + (p[1]-centerPos[1])**2)**0.5
        radius = np.min([dist, dist180])*scaleFactor
        cv2.circle(im, tuple(np.round(p).astype(np.int)), int(radius), 255, -1)
    im_array.append(im)
    
    
for i,im in enumerate(im_array[:200]):
    cv2.imwrite('im_'+str(i)+'_onscreenrepop.jpg', im)





