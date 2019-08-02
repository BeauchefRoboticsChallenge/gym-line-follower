#include "collision.hpp"

int collision_dect(py::array_t<double> seg,
                   py::array_t<double> track,
                   double th){
    auto rtrack = track.unchecked<2>();
    auto rseg = seg.unchecked<2>();
    int contiguous=1;
    for (int i = rtrack.shape(0)-1; i >= 0; --i){
        if (contiguous){
            double dx = abs(rseg(0,0)-rtrack(i,0));
            double dy = abs(rseg(0,1)-rtrack(i,1));
            if (dx > th || dy >th)
                contiguous = 0;
        }
        else {
            for (int j = 0; j < rseg.shape(0); ++j){
                if (abs(rseg(j,0)-rtrack(i,0)) < th){
                    if (abs(rseg(j,1)-rtrack(i,1)) < th){
                        //py::print("detected");
                        return 1;
                    }
                }
            }
            
        }
    }
    return 0;
}


int collision_dect2(py::array_t<double> seg,
                   py::array_t<double> track,
                   double th){
    auto rtrack = track.unchecked<2>();
    auto rseg = seg.unchecked<2>();
    double dmin=th*th;
    //Search window
    double xmin =INFINITY;double ymin=INFINITY;
    double xmax= -INFINITY;double ymax=-INFINITY;
    for (ssize_t i = 0; i < rseg.shape(0); i++)
    {
        if (rseg(i,0) > xmax){
            xmax=rseg(i,0);
        }
        if (rseg(i,0) < xmin){
            xmin=rseg(i,0);
        }
        if (rseg(i,1) > ymax){
            ymax=rseg(i,1);
        }
        if (rseg(i,1) < ymin){
            ymin=rseg(i,1);
        }
    }
    int contiguous=1;
    for (int i = rtrack.shape(0)-1; i >= 0; --i)
    {
        if (contiguous)
        {
            double d = pow(rseg(0,0)-rtrack(i,0),2)+pow(rseg(0,1)-rtrack(i,1),2);
            if (d > dmin)
                contiguous = 0;
        }
        else
        {
            if ((rtrack(i,0) <= xmax && rtrack(i,0) >= xmin) && (rtrack(i,1) <= ymax && rtrack(i,1) >= ymin))
            {
                for (int j = 0; j < rseg.shape(0); ++j)
                {
                    double d = pow(rseg(j,0)-rtrack(i,0),2)+pow(rseg(j,1)-rtrack(i,1),2);
                    if (d < dmin)
                    {
                        py::print("detected");
                        return 1;
                    }
                }
            }
        }
    }
    return 0;
}

