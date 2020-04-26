#include "irsensor.hpp"
#include <cmath>

/*! Transform from degrees to radians
    \param deg  angle in degrees
*/
double d2r(double deg){
    return M_PI*deg/180.0;
}

//from https://stackoverflow.com/questions/9323903/most-efficient-elegant-way-to-clip-a-number
/*! Clamp an int between min/max
    \param x  Number to clamp
    \param min  Minimum
    \param max  Maximum
*/
double clamp(double x, double min, double max)
{
    if (x < min) x = min;
    if (x > max) x = max;
    return x;
}

IrSensor::IrSensor(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img,
                   int track_ppm, double ds, double photo_heigth,
                   int array_size, double photo_sep, double photo_fov, double base_noise)
    :m_gen((std::random_device())()), m_dis(0,base_noise){
    m_ds = ds;
    m_array_size = array_size;
    m_radius = int(photo_heigth*tan(d2r(photo_fov)/2)*track_ppm);
    m_photo_sep = photo_sep;
    m_track_ppm = track_ppm;
    //Copy track
    py::buffer_info buf_info = img.request();
    if (buf_info.ndim != 2)
        throw std::runtime_error("Incompatible image dimension!");
    uint8_t *buf_ptr = (uint8_t *) buf_info.ptr;
    m_track_heigth=buf_info.shape[0];
    m_track_width=buf_info.shape[1];
    track = std::make_unique<uint8_t[]>(buf_info.size);
    for (int i = 0; i < buf_info.size; ++i) {
        track[i] = buf_ptr[i];
    }
    photo_pos = std::make_unique<ipoint[]>(array_size);
    srand (static_cast <unsigned> (time(0)));
}

ipoint IrSensor::to_pixel(dpoint p){
    int px=m_track_width/2 + int(std::round(p.x*m_track_ppm));
    int py=m_track_heigth/2 - int(std::round(p.y*m_track_ppm));
    return ipoint(px,py);
}

void IrSensor::update(double x, double y, double yaw){
    dpoint pos = dpoint(x+m_ds*cos(yaw),y+m_ds*sin(yaw)) +
            dpoint(-m_photo_sep*sin(yaw),m_photo_sep*cos(yaw))*((m_array_size-1)/2.0);
    dpoint dif = dpoint(m_photo_sep*sin(yaw),-m_photo_sep*cos(yaw));
    for (int i = 0; i < m_array_size; ++i) {
        ipoint sen_pos = to_pixel(pos + dif*double(i));
        photo_pos[i].x=sen_pos.x;
        photo_pos[i].y=sen_pos.y;
    }
}

py::array_t<double> IrSensor::read(){
    py::array_t<double> sen(m_array_size);
    py::buffer_info buf = sen.request();
    double *ptr = (double *) buf.ptr;
    for (int i = 0; i < m_array_size; ++i) {
        double sum = 0;
        int n = 0;
        for (int y_offset = -m_radius; y_offset <= m_radius; ++y_offset) {
            for (int x_offset = -m_radius; x_offset <= m_radius; ++x_offset) {
                if ((x_offset*x_offset + y_offset*y_offset)<(m_radius*m_radius)) {
                    int x=photo_pos[i].x+x_offset;
                    int y=photo_pos[i].y+y_offset;
                    if (x<m_track_width && y<m_track_heigth) {
                        sum+=double(track[x+m_track_width*y]);
                    }
                    else { //White color in out of bounds area
                        sum+=double(0);
                    }
                    n++;
                }
            }
        }
        ptr[i] = clamp(std::round((1.0/255.0)*sum/n)+m_dis(m_gen),0.0,1.0);
    }
    return sen;
}

py::array_t<int> IrSensor::get_photo_pos(){
    py::array_t<int> pos({m_array_size,2});
    py::buffer_info buf = pos.request();
    int *ptr = (int *) buf.ptr;
    for (int i = 0; i < m_array_size; ++i) {
        ptr[2*i] = photo_pos[i].x;
        ptr[2*i+1] = photo_pos[i].y;
    }
    return pos;
}

int IrSensor::get_sen_radius(){
    return m_radius;
}

