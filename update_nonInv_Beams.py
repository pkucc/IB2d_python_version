import pandas as pd
import numpy as np
import math
import os
def update_nonInv_Beams(dt,current_time,beams_info,Restart_Flag):

# beams_info:   col 1: 1ST PT.
#               col 2: MIDDLE PT. (where force is exerted)
#               col 3: 3RD PT.
#               col 4: beam stiffness
#               col 5: x-curvature
#               col 6: y-curvature

#IDs = beams_info(:,1);   % Gives Middle Pt.

# Coefficients for Polynomial Phase-Interpolation
    a = 2.739726027397260     # y1(t) = at^2
    b = 2.739726027397260     # y3(t) = -b(t-1)^2+1
    c = -2.029426686960933    # y2(t) = ct^3 + dt^2 + gt + h
    d = 3.044140030441400
    g = -0.015220700152207
    h = 0.000253678335870

# Period Info
    tP1 = 0.25                       # Down-stroke
    tP2 = 0.25                       # Up-stroke
    period = tP1+tP2                  # Period
#t = rem(current_time,period)      # Current time in simulation ( 'modular arithmetic to get time in period')
    t=current_time
    while t>=period:
       t -= period
# Read In y_Pts for two Phases!
    xP1,yP1,yP2 = read_File_In('swimmer.phases',Restart_Flag)  # NOTE xP1 = xP2
    xP2 = xP1
#
# FIRST WE COMPUTE THE INTERPOLATE GEOMETRY BETWEEN BOTH PHASES
#
    #PHASE 1 --> PHASE 2
    if (t <= tP1):
        t1 = 0.1*tP1  
        t2 = 0.9*tP1
        if (t<t1): 							#For Polynomial Phase Interp.
            g1 = a*math.pow((t/tP1),2)
        elif t>=t1 and t<t2:
            g1 = c*math.pow((t/tP1),3) + d*math.pow((t/tP1),2) + g*(t/tP1) + h
        elif (t>=t2):
            g1 = -b*math.pow(((t/tP1) - 1),2) + 1
        xPts = xP1 + g1*( xP2 - xP1 )	
        yPts = yP1 + g1*( yP2 - yP1 )	
		
    #PHASE 2 --> PHASE 1
    elif t>tP1 and t<=period:
        tprev = tP1
        t1 = 0.1*tP2 + tP1
        t2 = 0.9*tP2 + tP1
        if (t<t1):
            g2 = a*math.pow( ( (t-tprev)/tP2) ,2)
        elif t>=t1 and t<t2: 
            g2 = c*math.pow( ( (t-tprev)/tP2) ,3) + d*math.pow( ((t-tprev)/tP2) ,2) + g*( (t-tprev)/tP2) + h
        elif t>=t2: 
            g2 = -b*math.pow( (( (t-tprev)/tP2) - 1) ,2) + 1
        xPts = xP2 + g2*( xP1 - xP2 )
        yPts = yP2 + g2*( yP1 - yP2 )
    

#
# NOW WE UPDATE THE CURAVTURES APPROPRIATELY
#
#beams_info(:,4) = xPts(1:end-2)+xPts(3:end)-2*xPts(2:end-1);
#beams_info(:,5) = yPts(1:end-2)+yPts(3:end)-2*yPts(2:end-1);
    n=len(xPts)
    #if n>1000:
    xPts = np.array(xPts)
    yPts = np.array(yPts)
    beams_info = np.mat(beams_info)
    #test_info = beams_info[:,4]
    beams_info[:,4] = (xPts[:n-2] + xPts[2:] - 2 * xPts[1:n-1]).reshape(n-2,1)

    beams_info[:,5] = (yPts[:n-2] + yPts[2:] - 2 * yPts[1:n-1]).reshape(n-2,1)


    return beams_info

def read_File_In(file_name,Restart_Flag):
    #path = os.getcwd()
    #if Restart_Flag == 0:
    try:
        with open(file_name, 'r') as f:
            x1,y1,y2 = np.loadtxt(f,unpack=True)
    #else:
    except:
        path = os.path.abspath(os.pardir)

        with open(os.path.join(path,file_name), 'r') as f:
            x1,y1,y2 = np.loadtxt(f,unpack=True)

    return np.array(x1), np.array(y1), np.array(y2)

#filename = file_name;  %Name of file to read in
       
#fileID = fopen(filename);

    # Read in the file, use 'CollectOutput' to gather all similar data together
    # and 'CommentStyle' to to end and be able to skip lines in file.
    #C = textscan(fileID,'%f %f %f','CollectOutput',1);

#fclose(fileID);        %Close the data file.

#mat_info = C{1};   %Stores all read in data

#Store all elements in matrix
#mat = mat_info(1:end,1:end);

#x1 =  mat(:,1); %store xVals1/2
#y1 =  mat(:,2); %store yVals1 
#y2 =  mat(:,3); %store yVals2