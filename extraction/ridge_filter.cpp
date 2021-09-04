#include <cmath>
#include <iostream>
#include <algorithm>

using namespace std;

extern "C"
{

///
// Filter fingerprint image (img) with filter (kernel) according to orientation field (ori)
// the output is out
// Memory of img, ori and out should be allocated outside and have the same size
// kernel length (len) should be an odd number
void ridge_filter(double* img, double* ori, double* out, int width, int height, double *kernel, int len)
{
    for(int i=0;i<height;i++)
    {
        for(int j=0;j<width;j++)
        {
            double th = ori[i*width+j];
            out[i*width+j] = 0.;
            for(int k=-len/2;k<len/2;k++)
            {
                int x = j + k*cos(th) + 0.5;
                int y = i + k*sin(th) + 0.5;
                if(x>=0&&y>=0&&x<width&&y<height)
                {
                    out[i*width+j] += img[y*width+x]*kernel[k+len/2];
                }
                else
                {
                    continue;
                    //int x = j - k*cos(th) + 0.5;
                    //int y = i - k*sin(th) + 0.5;
                    //if(x>=0&&y>=0&&x<width&&y<height)
                    //{
                    //    out[i*width+j] += img[y*width+x]*kernel[-k+len/2];
                    //}
                }
            }
        }
    }
}

///
// Filter fingerprint image (img) with filter (kernel) according to orientation field (ori)
// the output is out
// Memory of img, ori and out should be allocated outside and have the same size
// kernel length (len) should be an odd number
void ridge_gabor_filter(double* img, double* ori, double* period, double* out, int width, int height,
        double *gabor, int len, int num_gabor, double min_period, double max_period)
{
    double step = (max_period-min_period)/num_gabor;

    for(int i=0;i<height;i++)
    {
        for(int j=0;j<width;j++)
        {
            double th = ori[i*width+j];
            double p = period[i*width+j];
            int index = (p-min_period)/step + 0.5;
            index = index>=num_gabor?num_gabor-1:index;
            index = index<0?0:index;
            double* kernel = gabor + len*index;

//            out[i*width+j] = 0.;
            for(int k=-len/2;k<len/2;k++)
            {
                int x = j + k*cos(th) + 0.5;
                int y = i + k*sin(th) + 0.5;
                if(x>=0&&y>=0&&x<width&&y<height)
                {
                    out[i*width+j] += img[y*width+x]*kernel[k+len/2];
                }
                else
                {
                    int x = j - k*cos(th) + 0.5;
                    int y = i - k*sin(th) + 0.5;
                    if(x>=0&&y>=0&&x<width&&y<height)
                    {
                        out[i*width+j] += img[y*width+x]*kernel[-k+len/2];
                    }
                }
            }
        }
    }
}


///
// Filter fingerprint image (img) with median filter (kernel) according to orientation field (ori)
// the output is out
// Memory of img, ori and out should be allocated outside and have the same size
// kernel length (len) should be an odd number
void ridge_median_filter(double* img, double* ori, double* out, int width, int height, int len)
{

    double *data = new double[len];
    for(int i=0;i<height;i++)
    {
        for(int j=0;j<width;j++)
        {
            double th = ori[i*width+j];
            out[i*width+j] = 0;
            for(int k=-len/2;k<len/2;k++)
            {
                int x = j + k*cos(th) + 0.5;
                int y = i + k*sin(th) + 0.5;
                if(x>=0&&y>=0&&x<width&&y<height)
                {
                    data[k+len/2] = img[y*width+x];
                }
                else
                {
                    int x = j - k*cos(th) + 0.5;
                    int y = i - k*sin(th) + 0.5;
                    if(x>=0&&y>=0&&x<width&&y<height)
                        data[k+len/2] += img[y*width+x];
                }
            }
            sort(data, data+len-1);
            out[i*width+j] = data[len/2];
        }
    }
    delete [] data;
}




// this function fill a threshold map
// in thr, the valid threshold value should be greater than zero
// for those threshold equals zeros, their values are filled with
// values of neighborhood pixel iteratively until no invalide threshold
void fill_threshold(unsigned char* thr, int width, int height)
{

    while(1)
    {
        int flag = 0;
        for(int i=1;i<height-1;i++)
        {
            for(int j=1;j<width-1;j++)
            {
                if(thr[i*width+j]!=0) continue;
                if(thr[(i-1)*width+j]!=0) thr[i*width+j] = thr[(i-1)*width+j];
                else if(thr[(i+1)*width+j]!=0) thr[i*width+j] = thr[(i+1)*width+j];
                else if(thr[i*width+j-1]!=0) thr[i*width+j] = thr[i*width+j-1];
                else if(thr[i*width+j+1]!=0) thr[i*width+j] = thr[i*width+j+1];
                flag = 1;
            }
        }
        if(!flag)break;
    }
}



// this function compute the pore map for a given fingerprint image
void pore_map(unsigned char* im, double* out, int width, int height, int R, double sigma)
{
    for(int y=0;y<height;y++)
    {
        for(int x=0;x<width;x++)
        {
            double sum = 0.;
            for(int i=y-R/2;i<y+R/2;i++)
            {
                if(i<0||i>=height)continue;
                for(int j=x-R/2;j<x+R/2;j++)
                {
                    if(j<0||j>=width)continue;
                    double dis = sqrt((i-y)*(i-y)+(j-x)*(j-x));
                    if (dis>R) continue;
                    double dif = abs(im[i*width+j] - im[y*width+x]);
                    //sum+= exp(dif/(2*sigma))/(dis+1);
                    sum+=dif;///(dis+1);
                }
            }
            out[y*width+x]=sum;
        }
    }
}

// this function compute the pore map for a given fingerprint image
void pore_map2(unsigned char* im, double* out, int width, int height, int R)
{
    for(int y=0;y<height;y++)
    {
        for(int x=0;x<width;x++)
        {
            double sum = 0.;
            int count = 0;
            double sum_dif = 0.;
            for(int i=y-R/2;i<y+R/2;i++)
            {
                if(i<=0||i>=height)continue;
                for(int j=x-R/2;j<x+R/2;j++)
                {
                    if(j<=0||j>=width)continue;
                    double dis = sqrt((i-y)*(i-y)+(j-x)*(j-x));
                    if (dis>R) continue;

                    // sum diff
                    sum_dif+= im[y*width+x]-im[i*width+j];

                    // get gradient
                    double dx,dy;
                    dx = double(im[i*width+j])-im[i*width+j-1];
                    dy = double(im[i*width+j])-im[(i-1)*width+j];
                    // normalize dx,dy
                    double d = (sqrt(dx*dx+dy*dy));
                    if(d>0)
                    {
                        dx = dx/d;
                        dy = dy/d;
                    }

                    double cx,cy;
                    cx = x-j;
                    cy = y-i;
                    // normalize cx,cy
                    d = sqrt(cx*cx+cy*cy);
                    if(d>0)
                    {
                        cx = cx/d;
                        cy = cy/d;
                    }

                    sum+=dx*cx+dy*cy;
                    count++;
                }
            }
            if(sum_dif>0)
                out[y*width+x]=sum/count;
        }
    }
}


void find_peaks(unsigned char* im, unsigned char* peakmap, int width, int height, int thr)
{
    for(int i=1;i<height-1;i++)
    {
        for(int j=1;j<width;j++)
        {
            if(
            im[i*width+j]>=im[i*width+j-1]&&
            im[i*width+j]>=im[i*width+j+1]&&
            im[i*width+j]>=im[(i-1)*width+j]&&
            im[i*width+j]>=im[(i+1)*width+j]&&
            im[i*width+j]>thr
            )
            peakmap[i*width+j] = 255;
        }
    }
}




}
