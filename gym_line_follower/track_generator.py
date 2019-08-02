#!/usr/bin/env python
import numpy as np
import random
from gym_line_follower.trackcpp import collision_dect
from gym_line_follower.trackcpp import rect_p,get_rect,curve_p,get_curve
from gym_line_follower.genetic.de import diff_evolution
from gym_line_follower.genetic.fitness import curves_fitness,curves_fitness_log
import pickle

class Segment(object):
    """docstring for Segment"""
    def __init__(self, p1, p2):
        super(Segment, self).__init__()
        self.start = p1
        self.end = p2

    def get_points(self, arg):
        return []

    def points_list(self,pd):
        return self.get_points(pd).tolist()
        
class Curve(Segment):
    """docstring for Curve"""
    def __init__(self, x0,y0,cAng,da,ds):
        cAng=np.arctan2(np.sin(cAng),np.cos(cAng))
        p1 = (x0,y0,cAng)
        r=ds/da
        x,y=curve_p(x0,y0,cAng,da,r)
        ang=cAng+da
        ang=np.arctan2(np.sin(ang),np.cos(ang))
        p2 = (x,y,ang)
        super(Curve, self).__init__(p1,p2)
        self.args = [x0,y0,cAng,da,ds]
        
    def get_points(self, pd):
        return get_curve(*self.args,pd=pd)

    @classmethod
    def random_curve(cls,x0,y0,cAng,rmax,dsmax,damax,direction=None):
        rc=100+random.random()*rmax
        dac=np.clip((100+random.random()*dsmax)/rc,0,damax)
        if direction is not None:
            dac*=direction
        else:
            dac*=random.choice([-1,1])
        dsc=abs(rc*dac)
        return cls(x0,y0,cAng,dac,dsc)

class Rect(Segment):
    """docstring for Rect"""
    def __init__(self, x0,y0,cAng,ds):
        cAng=np.arctan2(np.sin(cAng),np.cos(cAng))
        p1 = (x0,y0,cAng)
        p2 = tuple(np.append(rect_p(x0,y0,cAng,ds),cAng))
        super(Rect, self).__init__(p1,p2)
        self.args = [x0,y0,cAng,ds]

    def get_points(self,pd):
        return get_rect(*self.args,pd)

class Track_Generator(object):
    """docstring for Track_Generator"""
    def __init__(self, ctrX, ctrY, aveRadius, numVerts,bar=False):
        super(Track_Generator, self).__init__()
        self.segments = []
        self.endline = []
        self.start_finish_D=1000
        self.start = (ctrX + self.start_finish_D/2, ctrY + int(-aveRadius),0)
        self.bar=bar
        self.generate_track(ctrX,ctrY,aveRadius, numVerts)
        
    def gen_points(self, segments,pd):
        points=[]
        for seg in segments:
            points += seg.points_list(pd)
        return points

    def get_points(self,pd):
        return self.gen_points(self.segments,pd)

    def add_rect(self,x0,y0,cAng,ds):
        r=Rect(x0,y0,cAng,ds)
        self.segments.append(r)
        return r.end

    def add_curve(self,x0,y0,cAng,da,ds):
        c=Curve(x0,y0,cAng,da,ds)
        return  c.end
    
    def random_curve(self,x0,y0,cAng,rmax,dsmax,damax,direction=None):
        rc=100+random.random()*rmax
        dac=np.clip((100+random.random()*dsmax)/rc,0,damax)
        if direction is not None:
            dac*=direction
        else:
            dac*=random.choice([-1,1])
        dsc=abs(rc*dac)
        return self.curve(x0,y0,cAng,dac,dsc)

    def add_curve_seq(self,x0,y0,cAng,r,num,sec=0):
        da=[-np.pi,np.pi]
        ds=r*np.pi/2
        curves=[]

        curves.append(Curve(x0,y0,cAng,np.pi/2,ds))
        cX,cY,cAng=curves[-1].end

        ds=r*np.pi
        for _ in range(num):
            if sec!=0:
                curves.append(Rect(cX,cY,cAng,sec))
                cX,cY,cAng=curves[-1].end

            curves.append(Curve(cX,cY,cAng,da[0],ds))
            cX,cY,cAng=curves[-1].end

            if sec!=0:
                curves.append(Rect(cX,cY,cAng,sec))
                cX,cY,cAng=curves[-1].end

            curves.append(Curve(cX,cY,cAng,da[1],ds))
            cX,cY,cAng=curves[-1].end

        if self.check_seg(curves):
            self.segments+=curves
            return (cX,cY,cAng)
        else:
            return None

    def add_cross(self,x0,y0,cAng,ds1,cs1,ds2,cs2,direction=False):
        da=-np.pi/2 if direction else np.pi/2
        rect1=Rect(x0,y0,cAng,ds1)
        p1=rect1.end
        pm=rect_p(x0,y0,cAng,cs1)
        p2=np.append(rect_p(pm[0],pm[1],cAng-da,cs2),cAng+da)
        rect2=Rect(p2[0],p2[1],p2[2],ds2)
        #Random curve
        cur=Curve.random_curve(p1[0],p1[1],cAng,200,200,np.pi/2)
        pc=cur.end

        sol=self.join_points(pc,p2,[rect2]+self.endline,[rect1,cur])
        p3=rect2.end
        self.segments.append(rect1)
        self.segments.append(cur)
        self.segments+=sol
        self.segments.append(rect2)
        cAng=np.arctan2(np.sin(cAng+da),np.cos(cAng+da))

        return p3

    def join_points(self, p1, p2, preSegments=[],postSegments=[]):
        track=self.get_points(pd=50)
        pre=self.gen_points(preSegments,pd=50)
        post=self.gen_points(postSegments,pd=50)

        #Close curve
        ob = {"track":np.array(pre[1:-1]+track+post),
            "start":p1,
            "end":p2}
        bounds=[ (100*np.pi,3500), (-np.pi, np.pi), (100*np.pi,3500), (-np.pi, np.pi), (100*np.pi,3500), (-np.pi, np.pi)] 
        closure,fit = diff_evolution(curves_fitness,ob,bounds,
            mut=0.2,crossp=0.9,popsize=100,its=2000,stopf=0.2,bar=self.bar)
        if fit>=10000:
            print(curves_fitness_log(closure,ob))
        sol=[]
        cX,cY,cAng=p1
        for ds,da in closure.reshape((3,2)):
            da=da if np.abs(da) >= 0.01 else 0
            if da==0:
                sol.append(Rect(cX,cY,cAng,ds))
                cX,cY,cAng=sol[-1].end
            else:
                sol.append(Curve(cX,cY,cAng,da,ds))
                cX,cY,cAng=sol[-1].end

        pos_err=np.sqrt((cX-p2[0])**2+(cY-p2[1])**2)
        ang_err=np.abs(cAng-p2[2])
        print("=======================")
        print("Fit1: {}".format(fit))
        print("Pos error: {}".format(pos_err))
        print("Ang error: {}".format(ang_err))
        print("=======================")
        return sol

    def check_seg(self, seg_list, extraSegments=[]):
        seg=self.gen_points(seg_list,pd=50)
        track=self.get_points(pd=50)
        extra=self.gen_points(extraSegments,pd=50)
        endline=self.gen_points(self.endline, pd=50)
        seg_points=np.array(seg)
        track_points=np.array(extra+endline+track)
        return not collision_dect(seg_points,track_points,th=100)

    def generate_track(self,ctrX,ctrY,aveRadius, numVerts):
        cX,cY,cAng=self.add_rect(self.start[0],self.start[1],self.start[2],
                                    random.randint(250,aveRadius-500))

        finish = (ctrX - self.start_finish_D/2, ctrY + int(-aveRadius))
        #post_start=(ctrX + 500 + random.randint(250,aveRadius-500), ctrY + int(-aveRadius))
        pre_finish= (ctrX - self.start_finish_D/2 - random.randint(250,aveRadius-500), ctrY + int(-aveRadius))
        
        #End section
        self.endline=[Rect(pre_finish[0],pre_finish[1],0,finish[0]-pre_finish[0])]
        self.endline.append(Rect(finish[0],finish[1],0,self.start_finish_D))
        #points.append(post_start)

        #Primera curva
        self.segments.append(Curve.random_curve(cX,cY,cAng,500,500,np.pi,1))
        cX,cY,cAng = self.segments[-1].end


        p=self.add_curve_seq(cX,cY,cAng,100,2,100)
        if p is not None:
            cX,cY,cAng = p
        else:
            print("ERROR")
        cX,cY,cAng = self.add_cross(cX,cY,cAng,1000,500,300,150,direction=True)
        # #for i in range(numVerts-1):

        #Close curve
        closure=self.join_points((cX,cY,cAng),pre_finish+(0,),self.endline)
        self.segments+=closure
        self.segments+=self.endline
        
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
