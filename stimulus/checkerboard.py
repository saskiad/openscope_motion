# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 12:39:29 2016
@author: samg
"""

import copy, os, itertools, math, random
import numpy as np
import matplotlib.pyplot as plt
import fileIO


class MovingCheckerboard():

    def __init__(self):
        self.savePath = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\motionscope\checkerboard"
        self.frameRate = 60.0
        self.imagePosition = (0,0)
        self.imageSize = (960*2,600)
        self.pixelsPerDeg = 5
        self.moveLikeOpticFlow = True # left/right bckgnd move oppositely for horizontal motion
        self.squareSize = 1 # degrees
        self.bckgndSpeed = [0,20,80] # degrees/s
        self.bckgndDir = [0,180] # list containing any of [0,90,180,270]
        self.patchSize = [15] # degrees; even multiple of squareSize
        self.patchSpeed = [0,20,80] # degrees/s
        self.patchDir = [0,180] # list containing any of [0,90,180,270]
        self.patchElevation = [-15, 0, 15] # for horizontal motion trials
        self.patchAzimuth = [50] # for vertical motion trials
          
    def checkParameterValues(self):
        for param in ('bckgndSpeed','bckgndDir','patchSize','patchSpeed','patchDir','patchElevation','patchAzimuth'):
            if isinstance(getattr(self,param),(int,float)):
                setattr(self,param,[getattr(self,param)])
        if min(self.patchSize)<self.squareSize:
            raise ValueError('patch size must be larger than square size')
    
    def run(self):
        self.checkParameterValues()
        
        self.numpyRandomSeed = random.randint(0,2**32)
        self._numpyRandom = np.random.RandomState(self.numpyRandomSeed)
        
        # create checkerboard background
        self._squareSizePix = round(self.squareSize*self.pixelsPerDeg)
        leftOffset = rightOffset = topOffset = bottomOffset = self._squareSizePix
        bckgnd = self.makeCheckerboard((math.ceil(self.imageSize[1]/self._squareSizePix),math.ceil(self.imageSize[0]/self._squareSizePix)))
        if self.moveLikeOpticFlow:
            centerOffset = (self.imageSize[0]/2)%self._squareSizePix
            if centerOffset>0:
                self.shiftBckgnd(bckgnd[:,::-1],centerOffset,centerOffset)
                rightOffset = centerOffset
        checkerboardImage = np.copy(bckgnd[:self.imageSize[1],:self.imageSize[0]])
        
        # get trialTypes
        trialTypes = list(itertools.product(self.bckgndSpeed,[dir for dir in self.bckgndDir if dir in (0,180)],self.patchSize,self.patchSpeed,[dir for dir in self.patchDir if dir in (0,180)],self.patchElevation))
        trialTypes.extend(itertools.product(self.bckgndSpeed,[dir for dir in self.bckgndDir if dir in (90,270)],self.patchSize,self.patchSpeed,[dir for dir in self.patchDir if dir in (90,270)],self.patchAzimuth))
        for params in trialTypes[:]:
            # don't need all bckgnd directions for bckgndSpeed=0
            # or all patch sizes, directions, and positions for patchSpeed=0
            # or patch speed and direction = bckgnd speed and direction when both speeds > 0
            if ((params[0]==0 and params[1]!=self.bckgndDir[0])
               or (params[3]==0 and (params[2]!=self.patchSize[0] or params[3]!=self.patchSpeed[0] or params[4]!=self.patchDir[0] or (params[5]!=self.patchElevation[0] if params[4] in (0,180) else params[5]!=self.patchAzimuth[0])))
               or (params[0]>0 and params[3]>0 and params[0]==params[3] and params[1]==params[4])):
                   if params in trialTypes:
                       trialTypes.remove(params)
        
        # run
        frame = 0
        trial = -1
        trialFrame = -1
        self.trialStartFrame = []
        self.trialNumFrames = []
        self.trialBckgndSpeed = []
        self.trialBckgndDir = []
        self.trialPatchSize = []
        self.trialPatchSpeed = []
        self.trialPatchDir = []
        self.trialPatchPos = []
        while trial<=len(trialTypes)-1: #This used to be just <, which I think omits one of the stimulus conditions...
            try:
                if trialFrame==-1:
                    trial += 1
                    print('starting trial: ' + str(trial))
                    trialFrame = 0
                    self.trialBckgndSpeed.append(trialTypes[trial][0])
                    self.trialBckgndDir.append(trialTypes[trial][1])
                    self.trialPatchSize.append(trialTypes[trial][2])
                    self.trialPatchSpeed.append(trialTypes[trial][3])
                    self.trialPatchDir.append(trialTypes[trial][4])
                    self.trialPatchPos.append(trialTypes[trial][5])
                    if self.trialBckgndDir[-1]==0:
                        bckgndOffset = leftOffset
                    elif self.trialBckgndDir[-1]==180:
                        bckgndOffset = rightOffset
                    elif self.trialBckgndDir[-1]==90:
                        bckgndOffset = bottomOffset
                    else:
                        bckgndOffset = topOffset
                    bckgndMovPerFrame = int(round(self.trialBckgndSpeed[-1]*self.pixelsPerDeg/self.frameRate))
                    patchMovPerFrame = int(round(self.trialPatchSpeed[-1]*self.pixelsPerDeg/self.frameRate))
                    patchSizePix = int(round(self.trialPatchSize[-1]*self.pixelsPerDeg))
                    if patchMovPerFrame>0:
                        patchSquares = round(self.trialPatchSize[-1]/self.squareSize)
                        patch = self.makeCheckerboard((patchSquares,)*2)
                        if self.trialPatchDir[-1] in [0,180]:
                            y = self._squareSizePix*round((self.imageSize[1]/2-self.imagePosition[1]+self.trialPatchPos[-1]*self.pixelsPerDeg)/self._squareSizePix-patchSquares/2)
                            if topOffset<self._squareSizePix:
                                y -= topOffset
                            if self.trialPatchDir[-1]==0:
                                patchPos = [self.imageSize[0]/2-patchSizePix,y]
                            else:
                                patchPos = [self.imageSize[0],y]
                            self.trialNumFrames.append(int(round((self.imageSize[0]/2+patchSizePix)/patchMovPerFrame)))
                        else:
                            x = self._squareSizePix*round((self.imageSize[0]/2-self.imagePosition[0]+self.trialPatchPos[-1]*self.pixelsPerDeg)/self._squareSizePix)-int(patchSizePix/2)
                            if leftOffset<self._squareSizePix:
                                x -= leftOffset
                            if self.trialPatchDir[-1]==90:
                                patchPos = [x,self.imageSize[1]]
                            else:
                                patchPos = [x,-patchSizePix]
                            self.trialNumFrames.append(int(round((self.imageSize[1]+patchSizePix)/patchMovPerFrame)))  
                    else:
                        if bckgndMovPerFrame>0:
                            if self.trialBckgndDir[-1] in [0,180]:
                                self.trialNumFrames.append(int(round(self.imageSize[0]/2/bckgndMovPerFrame)))
                            else:
                                self.trialNumFrames.append(int(round(self.imageSize[1]/bckgndMovPerFrame)))
                        else:
                            self.trialNumFrames.append(int(round(2*self.frameRate)))
                    trialMovie = []
                else:     
                    if trialFrame==0:
                        self.trialStartFrame.append(frame)
                        print('starting movie for trial: ' + str(trial))
                    if bckgndMovPerFrame>0:
                        if bckgndOffset==0:
                            bckgndOffset = self._squareSizePix
                        bckgndOffset += bckgndMovPerFrame
                        if bckgndOffset>self._squareSizePix:
                            newSqOffset = int(bckgndOffset-self._squareSizePix)
                            bckgndOffset %= self._squareSizePix
                        else:
                            newSqOffset = 0
                        if self.trialBckgndDir[-1]==0:
                            if self.moveLikeOpticFlow:
                                self.shiftBckgnd(bckgnd[:,self.imageSize[0]/2+1:],bckgndMovPerFrame,newSqOffset)
                                self.shiftBckgnd(bckgnd[:,self.imageSize[0]/2::-1],bckgndMovPerFrame,newSqOffset)
                            else:
                                self.shiftBckgnd(bckgnd,bckgndMovPerFrame,newSqOffset)
                        elif self.trialBckgndDir[-1]==180:
                            if self.moveLikeOpticFlow:
                                self.shiftBckgnd(bckgnd[:,-1:self.imageSize[0]/2:-1],bckgndMovPerFrame,newSqOffset)
                                self.shiftBckgnd(bckgnd[:,:self.imageSize[0]/2+1],bckgndMovPerFrame,newSqOffset)
                            else:
                                self.shiftBckgnd(bckgnd[:,::-1],bckgndMovPerFrame,newSqOffset)
                        elif self.trialBckgndDir[-1]==90:
                            self.shiftBckgnd(bckgnd[:,::-1].T,bckgndMovPerFrame,newSqOffset)
                        else:
                            self.shiftBckgnd(bckgnd.T,bckgndMovPerFrame,newSqOffset)
                    checkerboardImage = np.copy(bckgnd[:self.imageSize[1],:self.imageSize[0]])
                    if patchMovPerFrame>0:
                        if self.trialPatchDir[-1]==0:
                            patchPos[0] += patchMovPerFrame
                        elif self.trialPatchDir[-1]==180:
                            patchPos[0] -= patchMovPerFrame
                        elif self.trialPatchDir[-1]==90:
                            patchPos[1] -= patchMovPerFrame
                        else:
                            patchPos[1] += patchMovPerFrame
                        if patchPos[0]<=self.imageSize[0] and patchPos[1]<=self.imageSize[1]:
                            patchImagePos = copy.copy(patchPos)
                            patchImage = patch
                            if patchPos[0]<self.imageSize[0]/2:
                                patchImage = patch[:,int(self.imageSize[0]/2-patchPos[0]):]
                                patchImagePos[0] = self.imageSize[0]/2
                            if patchPos[1]<0:
                                patchImage = patch[-patchPos[1]:,:]
                                patchImagePos[1] = 0
                            if patchPos[0]+patch.shape[1]>self.imageSize[0]:
                                patchImage = patch[:,:int(self.imageSize[0]-patchPos[0])]                  
                            if patchPos[1]+patch.shape[0]>self.imageSize[1]:
                                patchImage = patch[:self.imageSize[1]-patchPos[1],:]
                            checkerboardImage[int(patchImagePos[1]):int(patchImagePos[1]+patchImage.shape[0]),int(patchImagePos[0]):int(patchImagePos[0]+patchImage.shape[1])] = patchImage
                    if trialFrame==self.trialNumFrames[-1]-1:
                        if self.trialBckgndDir[-1]==0:
                            leftOffset = bckgndOffset
                            rightOffset = self._squareSizePix-bckgndOffset
                            if self.moveLikeOpticFlow:
                                rightOffset += centerOffset
                                if rightOffset>self._squareSizePix:
                                    rightOffset -= self._squareSizePix
                        elif self.trialBckgndDir[-1]==180:
                            rightOffset = bckgndOffset
                            leftOffset = self._squareSizePix-bckgndOffset
                            if self.moveLikeOpticFlow:
                                leftOffset -= centerOffset
                                if leftOffset<0:
                                    leftOffset += self._squareSizePix
                        elif self.trialBckgndDir[-1]==90:
                            bottomOffset = bckgndOffset
                            topOffset = self._squareSizePix-bckgndOffset
                        else:
                            topOffset = bckgndOffset
                            bottomOffset = self._squareSizePix-bckgndOffset
                    trialMovie.append(checkerboardImage[:,self.imageSize[0]/2:])
                    frame += 1
                    trialFrame += 1
                    if trialFrame==self.trialNumFrames[-1]:
                        print('saving trial:' + str(trial))
                        np.save(os.path.join(self.savePath,'trial'+str(trial)+ '_duration_' + str(self.trialNumFrames[-1]/self.frameRate) + '.npy'),np.stack(trialMovie))
                        trialFrame = -1
            except:
                print('Failed to render trial: ' + str(trial))
                continue
        fileIO.objToHDF5(self,os.path.join(self.savePath,'params.hdf5'))
                
        
    def makeCheckerboard(self,shape):
        # shape is nSquares height x nSquares width
        shape = [int(s) for s in shape]
        checkerboard = self._numpyRandom.randint(0,2,shape).astype(np.uint8)*255
        checkerboard = np.repeat(checkerboard,self._squareSizePix,0)
        checkerboard = np.repeat(checkerboard,self._squareSizePix,1)
        return checkerboard
        
    def shiftBckgnd(self,img,mov,newSqOffset):
        mov = int(mov)
        newSqOffset = int(newSqOffset)
        # shift right
        img[:,mov:] = img[:,:-mov]
        # fill in lagging edge
        img[:,newSqOffset:mov] = img[:,mov][:,None]
        # add new squares
        if newSqOffset>0:
            newSquares = self.makeCheckerboard((math.ceil(img.shape[0]/self._squareSizePix),newSqOffset//self._squareSizePix+1))
            img[:,:newSqOffset] = newSquares[0:img.shape[0],-newSqOffset:]
            
    def playMovie(self):
        filePath = fileIO.getFile()
        if len(filePath)>0:
            frames = np.load(filePath)
            plt.figure()
            mov = plt.imshow(frames[0],cmap='gray')
            for img in frames[1:]:
                mov.set_array(img)
                plt.pause(0.001)


if __name__=="__main__":
    pass