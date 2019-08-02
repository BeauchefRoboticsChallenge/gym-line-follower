#!/usr/bin/env python
import numpy as np
import random
from gym_line_follower.trackcpp import collision_dect
from gym_line_follower.trackcpp import rect_p,get_rect,curve_p,get_curve
from gym_line_follower.genetic.de import diff_evolution
from gym_line_follower.genetic.fitness import curves_fitness,curves_fitness_log
import pickle

def curve_fun(x0,y0,cAng,da,ds):
    def _curve(pd=5):
        c=get_curve(x0,y0,cAng,da,ds,pd=pd)
        return c.tolist()
    return _curve

def rect_fun(x0,y0,cAng,ds):
    def _rect(pd=50):
        r=get_rect(x0,y0,cAng,ds,pd=pd)
        return r.tolist()
    return _rect

class Track_Generator(object):
    """docstring for Track_Generator"""
    def __init__(self, ctrX, ctrY, aveRadius, numVerts,bar=False):
        super(Track_Generator, self).__init__()
        self.segments = []
        self.start_finish_D=1000
        self.start = (ctrX + self.start_finish_D/2, ctrY + int(-aveRadius),0)
        self.bar=bar
        self.generate_track(ctrX,ctrY,aveRadius, numVerts)
        

    def get_points(self,pd):
        points=[]
        for seg in self.segments:
            points += seg(pd)
        return points

    def add_rect(self,x0,y0,cAng,ds):
        self.segments.append(rect_fun(x0,y0,cAng,ds))
        return np.append(rect_p(x0,y0,cAng,ds),cAng)

    def add_curve(self,x0,y0,cAng,da,ds):
        self.segments.append(curve_fun(x0,y0,cAng,da,ds))
        r=ds/da
        p=curve_p(x0,y0,cAng,da,r)
        cAng+=da
        cAng=np.arctan2(np.sin(cAng),np.cos(cAng))
        return np.append(p,cAng)

    def join_points(self, p1, p2, extraSegments=[]):
        track=self.get_points(pd=50)
        extra=[]
        for s in extraSegments:
            extra+=s(pd=50)
        #Close curve
        ob = {"track":np.array(extra[1:-1]+track),
            "start":p1,
            "end":p2}
        bounds=[ (100,3500), (-np.pi, np.pi), (100,3500), (-np.pi, np.pi), (100,3500), (-np.pi, np.pi)] 
        closure,fit = diff_evolution(curves_fitness,ob,bounds,
            mut=0.2,crossp=0.9,popsize=100,its=2000,stopf=0.1,bar=self.bar)


        cX,cY,cAng=p1
        for ds,da in closure.reshape((3,2)):
            da=da if np.abs(da) >= 0.01 else 0
            if da==0:
                cX,cY,cAng=self.add_rect(cX,cY,cAng,ds)
            else:
                cX,cY,cAng=self.add_curve(cX,cY,cAng,da,ds)

        pos_err=np.sqrt((cX-p2[0])**2+(cY-p2[1])**2)
        ang_err=np.abs(cAng-p2[2])
        print("=======================")
        print("Pos error: {}".format(pos_err))
        print("Ang error: {}".format(ang_err))
        print("=======================")

     

    def generate_track(self,ctrX,ctrY,aveRadius, numVerts):
        cX,cY,cAng=self.add_rect(self.start[0],self.start[1],self.start[2],
                                    random.randint(250,aveRadius-500))

        finish = (ctrX - self.start_finish_D/2, ctrY + int(-aveRadius))
        #post_start=(ctrX + 500 + random.randint(250,aveRadius-500), ctrY + int(-aveRadius))
        pre_finish= (ctrX - self.start_finish_D/2 - random.randint(250,aveRadius-500), ctrY + int(-aveRadius))
        
        #End section
        endline=[rect_fun(pre_finish[0],pre_finish[1],0,finish[0]-pre_finish[0])]
        endline.append(rect_fun(finish[0],finish[1],0,self.start_finish_D))
        #points.append(post_start)

        #Primera curva
        r=100+random.random()*500
        da=np.clip((100+random.random()*500)/r,0,np.pi)
        ds=r*da
        cX,cY,cAng=self.add_curve(cX,cY,cAng,da,ds)

        #for i in range(numVerts-1):

        #Close curve
        self.join_points((cX,cY,cAng),pre_finish+(0,),endline)

        self.segments+=endline
        
    def get_vect(self):
        points=self.get_points(pd=5)
        c=np.array(points[:-1])
        x,y=c.T
        return x,y

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    print("Test")
    rad=1000*3.0/2
    track=Track_Generator(0,0,rad,20,bar=True)
    x,y=track.get_vect()
    plt.plot(x,y)
    plt.gca().axis('equal')
    plt.show()
