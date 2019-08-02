#include "primitives.hpp"



py::array_t<double> rect_p(double x0, double y0,
                             double cAng, double ds){
    py::array_t<double> res(2);
    py::buffer_info buf = res.request();
    double *ptr = (double *) buf.ptr;
    ptr[0]=x0+ds*cos(cAng);
    ptr[1]=y0+ds*sin(cAng);
    return res;
}

py::array_t<double> get_rect(double x0, double y0,
                             double cAng, double ds, int pd){
    int nPoints = int(ceil(ds/pd));
    double dx=pd*cos(cAng);
    double dy=pd*sin(cAng);
    py::array_t<double> arr({ nPoints, 2 });
    py::buffer_info buf = arr.request();
    double *ptr = (double *) buf.ptr;
    double x = x0;
    double y = y0;
    for (int i = 0; i < (nPoints-1); ++i)
    {
        x+=dx;
        ptr[i*2] = x;
        y+=dy;
        ptr[i*2+1] = y;
    }
    ptr[2*(nPoints-1)]=x0+ds*cos(cAng);
    ptr[2*nPoints-1]=y0+ds*sin(cAng);
    return arr;
}

void calc_curve_p(double res[2], double x0, double y0,
                  double cAng, double da, double r){

    double seg=r*sqrt(2*(1-cos(da)));
    double b=M_PI/2-cAng+asin(r*sin(da)/seg);

    res[0]=x0-seg*cos(b);
    res[1]=y0+seg*sin(b);
}

py::array_t<double> curve_p(double x0, double y0,
                             double cAng, double da, double r){
    py::array_t<double> res(2);
    py::buffer_info buf = res.request();
    double *ptr = (double *) buf.ptr;
    calc_curve_p(ptr,x0,y0,cAng,da,r);
    return res;
}

py::array_t<double> get_curve(double x0, double y0,
                              double cAng, double da, double ds,
                              int pd){
    int nPoints=int(round(ds/pd));
    py::array_t<double> arr({ nPoints, 2 });
    py::buffer_info buf = arr.request();
    double *ptr = (double *) buf.ptr;
    double temp[2];

    double r=ds/da;
    for (int i = 0; i < nPoints; ++i)
    {
        calc_curve_p(temp,x0,y0,cAng,(double(i+1)/nPoints)*da,r);
        ptr[i*2] = temp[0];
        ptr[i*2+1] = temp[1];
    }
    return arr;
}