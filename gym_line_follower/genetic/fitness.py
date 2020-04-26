#!/usr/bin/env python
import numpy as np
from gym_line_follower.trackutils import collision_dect,get_curve,get_rect
    
def curves_fitness(curves,track_obj):
    """
    Fitness for the n curves to add to the track

    Params:
    curves - Solution to test [ds1,da1,ds2,da2,ds3,da3]
    track_obj - state of the track, dictionary with points, start point and end point
    """
    start=track_obj["start"] #(x1,y1,ang1)
    end=track_obj["end"] ##(x2,y2,ang2)
    curve_r=curves.reshape((-1,2))
    test_track=track_obj["track"]
    n=curve_r.shape[0]-1
    x=start[0]
    y=start[1]
    cAng=start[2]
    # Calculate end points
    fit=0
    for i,c in enumerate(curve_r):
        ds,cur = c
        da=cur*ds if np.abs(cur) >= 0.00025 else 0
        if da == 0:#rect
            #Collision
            rect=get_rect(x,y,cAng,ds,pd=50)
            if i == n:
                if collision_dect(rect[:-1],test_track,th=90):
                    fit+= 10000
            else:
                if collision_dect(rect[:-1],test_track,th=100):
                    fit+= 10000
            x=rect[-1][0]
            y=rect[-1][1]
            test_track=np.concatenate((test_track, rect), axis=0)
        else:
            curve=get_curve(x,y,cAng,da,ds,pd=50)
            if i == n:
                if collision_dect(curve[:-1],test_track,th=90):
                    fit+= 10000
            else:
                if collision_dect(curve,test_track,th=100):
                    fit+= 10000
            x=curve[-1][0]
            y=curve[-1][1]
            test_track=np.concatenate((test_track, curve), axis=0)
            cAng+=da
            cAng=np.arctan2(np.sin(cAng),np.cos(cAng))
    #fitness
    pos_err=np.sqrt((end[0]-x)**2+(end[1]-y)**2)
    ang_err=np.abs(cAng-end[2])
    fit += pos_err+100*ang_err
    if np.isnan(fit):#TODO: fix invalid curve
        return 40000
    return fit
