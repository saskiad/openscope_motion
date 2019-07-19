#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 22:47:36 2019

@author: corbennett
"""
from __future__ import division
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

class dotTranslation():
    def __init__(self):
        self.viewingDistance = 100
        self.radius = 10
        self.speed = 4
        self.dotNum = 20
        self.timePoints = 120
        self.pixelsPerDegree = 10/2 
        self.screenWidth = int(1960/2)
        self.screenHeight = int(1200/2)
        self.centerPosDegrees = np.array([-60,-20])
        self.scaleFactor = 1/20.
        self.dotColors = [0, 254]
        self.heading = 'forward'
        self.backgroundColor = 127
        self.saveDir = "/Volumes/LC/motionscope_stimuli"
        
        
    #Function to generate random dot positions at specified distance range from center
    def getNewDotPositions(self, dotNum, minDist=5, maxDist=90):
        minDist *= self.pixelsPerDegree
        maxDist *= self.pixelsPerDegree
        dists = np.random.uniform(minDist, maxDist, size=dotNum)
        angles = np.pi * np.random.uniform(0, 2, size=dotNum)
        x = dists * np.cos(angles)
        y = dists * np.sin(angles)
        return np.stack([x,y]).T

    #Function to repopulate dots that are past 90 degrees visual angle
    def repop(self, repopInds):
        self.pos[repopInds] = self.getNewDotPositions(repopInds.size) + self.centerPos
        self.distToCenter[repopInds] = np.sqrt(np.sum((self.pos[repopInds] - self.centerPos)**2, axis=1))
        self.unitVectors[repopInds] = (self.pos[repopInds] - self.centerPos)/self.distToCenter[repopInds, None]
        self.theta[repopInds] = np.deg2rad(self.distToCenter[repopInds]/self.pixelsPerDegree)
        self.virtualTimePoint[repopInds] = (self.viewingDistance - (self.radius/np.tan(self.theta[repopInds])))/self.speed
        

    #Function to redraw dots for the next time point
    def redraw(self):
        
        #find new visual angle at that time point
        self.theta = np.arctan(self.radius/(self.viewingDistance - self.speed*(self.virtualTimePoint)))
        
        #convert angle to pixels and update dot positions
        newThetaPix = np.rad2deg(self.theta)*self.pixelsPerDegree
        self.pos = self.unitVectors*newThetaPix[:, None] + self.centerPos
        
        #find dots past 90 and reorient them towards the perspective point at 180 degrees
        past90 = np.rad2deg(self.theta)<0
        if np.sum(past90)>0:
            unitVectors_past90 = np.copy(self.unitVectors[past90])
            unitVectors_past90 = unitVectors_past90 * [-1,1] #reflect about vertical
            self.theta[past90]  = np.abs(self.theta[past90])
            newThetaPix[past90] = np.rad2deg(self.theta[past90])*self.pixelsPerDegree
            self.pos[past90] = unitVectors_past90*newThetaPix[past90][:,None] + self.centerPos + [180*self.pixelsPerDegree,0]
        
        onscreen = np.array([(0<p[0]<self.screenWidth) & (0<p[1]<self.screenHeight) for p in self.pos])
       
        repopInds = np.where(~onscreen)[0]
        if len(repopInds)>0:
            self.repop(repopInds)
        
        #increment "time point" for each dot
        self.virtualTimePoint += 1
        
    def findCenterDotMetrics(self):
        #find some metrics describing dot size and speed around where RFs are expected to be (center of screen)
        degToCenter = np.sum(self.centerPosDegrees**2)**0.5
        self.dotRadiusInCenter = degToCenter * self.scaleFactor
        theta = np.rad2deg([np.arctan(self.radius/(self.viewingDistance - self.speed*t)) for t in np.arange(0, int(round(self.viewingDistance/self.speed)))])
        diffTheta = np.diff(theta)
        thetaIndNearCenter = np.where(theta<=degToCenter)[0][-1]
        self.dotSpeedInCenter = diffTheta[thetaIndNearCenter] * 60 #multiply by 60 to make speed per second rather than per frame
        
    def makeStimulusArray(self, save=False, fileName='', compressed=True, offset=0):
        #INITIALIZE DOTS
        
        
        #pick dots at random screen positions with respect to center point
        self.centerPos = self.centerPosDegrees*self.pixelsPerDegree + np.array([self.screenWidth/2, self.screenHeight/2])
        self.pos = self.getNewDotPositions(self.dotNum, 5, 10) + self.centerPos
        
        #find how far they are from center and find unit vectors along which to move them
        self.distToCenter = np.sqrt(np.sum((self.pos - self.centerPos)**2, axis=1))
        self.unitVectors = (self.pos - self.centerPos)/self.distToCenter[:, None]
        
        #find the visual angle at which each dot lives
        self.theta = np.deg2rad(self.distToCenter/self.pixelsPerDegree)
        
        #find the "time point" when each dot would reach its visual angle if it had started at the original viewing distance
        self.virtualTimePoint = (self.viewingDistance - (self.radius/np.tan(self.theta)))/self.speed
        
        self.dotColorAssignments = np.random.choice(self.dotColors, self.dotNum)
        #RUN DOTS
        self.pos_array = []
        for i in np.arange(self.timePoints):
            self.redraw()
            self.pos_array.append(self.pos)
            
        self.im_array = []
        for points in self.pos_array[offset:]:
            im = np.ones([self.screenHeight, self.screenWidth])*self.backgroundColor
            for p,c in zip(points, self.dotColorAssignments):
                dist = ((p[0] - self.centerPos[0])**2 + (p[1]-self.centerPos[1])**2)**0.5
                dist180 = ((p[0] - self.centerPos[0] - 180*self.pixelsPerDegree)**2 + (p[1]-self.centerPos[1])**2)**0.5
                radius = np.min([dist, dist180])*self.scaleFactor
                cv2.circle(im, tuple(np.round(p).astype(np.int)), int(radius), int(c), -1)
            self.im_array.append(im.astype(np.uint8))
        
        if self.heading == 'backward':
            self.im_array = self.im_array[::-1]
        
        if save and compressed:
            np.savez_compressed(os.path.join(self.saveDir, fileName), im_array=self.im_array)
        elif save and not compressed:
            np.save(os.path.join(self.saveDir, fileName), self.im_array)
            
    def playMovie(self, offset=0):
        plt.figure()
        mov = plt.imshow(self.im_array[offset], cmap='gray')
        for im in self.im_array[offset:]:
            mov.set_array(im)
            plt.pause(0.001)
                    
                    
            
            
            
            
            
            
            
            
            
            
        