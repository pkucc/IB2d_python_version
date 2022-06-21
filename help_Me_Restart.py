#-------------------------------------------------------------------------------------------------------------------#
#
# IB2d is an Immersed Boundary Code (IB) for solving fully coupled non-linear
# 	fluid-structure interaction models. This version of the code is based off of
#	Peskin's Immersed Boundary Method Paper in Acta Numerica, 2002.
#
# Author: Nicholas A. Battista
# Email:  nickabattista@gmail.com
# Date Created: October 8th, 2018
# Institution: UNC-CH
#
# This code is capable of creating Lagrangian Structures using:
# 	1. Springs
# 	2. Beams (*torsional springs / non-invariant beams *)
# 	3. Target Points
#	4. Muscle-Model (combined Force-Length-Velocity model, "HIll+(Length-Tension)")
#
# One is able to update those Lagrangian Structure parameters, e.g., spring constants, resting lengths, etc
#
# There are a number of built in Examples, mostly used for teaching purposes.
#
# If you would like us #to add a specific muscle model, please let Nick (nickabattista@gmail.com) know.
#
#--------------------------------------------------------------------------------------------------------------------#

######################################################################################
#
# FUNCTION: helps input previous .vtk information for restarting a
#           simulation that has ended because of power failure, etc.
#
#      NOTE: for restart protocol, need to have .vtk data for:
#                       1. lagPts (Lagrangian positions)
#                       2. u (velocity field)
#
######################################################################################
import numpy as np
import os
def help_Me_Restart(ctsave):
# function [current_time,cter,ctsave,U,V,xLag,yLag,xLag_P,yLag_P,path_to_data] = help_Me_Restart(dt)

# NEEDS TO BE HARDCODED PER SIMULATION BASIS
    #ctsave = 1
# Last time-step of data saved (# at end of .vtk file);
    #print_dump = 800           # Print_dump interval as given in input2d
    ctsave = int(ctsave)
    return pass_Back_Data_For_Restart(ctsave-1)

def pass_Back_Data_For_Restart(ctsave):
    #cter = ctsave * print_dump # Total  of time-steps thus far up to and included last data point saved
    #current_time = cter * dt # Current time in simulation when last time-step was saved

    # Read in LAGRANGIAN data for last and second to last timepoint for U, V, mVelocity, F_Poro, xLag, yLag, xLag_P, yLag_P
    xLag, yLag, xLag_P, yLag_P = please_Give_Saved_Lag_Point_Info(ctsave)

    # Read in EULERIAN data for last timepoint for U, V
    U, V = please_Give_Saved_Eulerian_Info(ctsave)

    # Update for next iteration(*pretends data from last timepoint was just saved * )
    #ctsave = ctsave + 1 # Update for next ctsave number (always increases by 1 after data is saved)
    #current_time = current_time + dt # Update for next time - step
    #cter = cter + 1 # Update for next time - step

    return U, V, xLag, yLag, xLag_P, yLag_P


def please_Give_Saved_Lag_Point_Info(ctsave):
    iP = ctsave - 1
    numSim = str(ctsave)

    while len(numSim) < 4:
        numSim = '0' + numSim

    numSim_Prev = str(iP)
    while len(numSim_Prev) < 4:
        numSim_Prev = '0' + numSim_Prev

    xLag, yLag = give_Lag_Positions(numSim)
    xLag_P, yLag_P = give_Lag_Positions(numSim_Prev)

    return xLag, yLag, xLag_P, yLag_P


def give_Lag_Positions(numSim):
    filename = 'lagsPts.' + numSim + '.vtk'
    #with open('viz_IB2d\\' + filename, 'r') as f:
    #syspath = os.path.join(os.getcwd(),'viz_IB2d')
    with open(os.path.join(os.getcwd(), filename), "r") as f:
        for i in range(5):f.readline()
        n = int(f.readline().split()[1])
        xLag=[]
        yLag=[]
        for i in range(n):
            tmp=f.readline().split()
            xLag.append(float(tmp[0]))
            yLag.append(float(tmp[1]))



    return np.array(xLag), np.array(yLag)


def please_Give_Saved_Eulerian_Info(ctsave):
    numSim = str(ctsave)
    while len(numSim)<4:
        numSim = '0' + numSim
    filename = 'uX.' + numSim + '.vtk'
    with open(os.path.join(os.getcwd(), filename), "r") as f:
        U = np.loadtxt(f, unpack=True, skiprows=14)
    filename = 'uY.' + numSim + '.vtk'
    with open(os.path.join(os.getcwd(), filename), "r") as f:
        V = np.loadtxt(f, unpack=True, skiprows=14)
    return U.T, V.T

