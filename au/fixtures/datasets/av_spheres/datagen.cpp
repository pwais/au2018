/*
datagen.cpp - Generates camera images, point clouds, ego pose, and cuboids
from a synthetic sphere-based scene.  

Based upon smallpt.cpp, a path tracer in 99 lines of C++
https://www.kevinbeason.com/smallpt/

Includes jo_jpeg.cpp, a concise jpeg encoder
https://www.jonolick.com/uploads/7/9/2/1/7921194/jo_jpeg.cpp

Modifications:
 * Added Ray() default ctor
 * Added len() to Vec
 * Added const-ref cross to Vec
 * Removed front sphere from scene
 * Added data generating functions at end as noted

Usage:
 * g++ -O3 -fopenmp datagen.cpp -o datagen && ./datagen 1000

*/



// Portions smallpt.cpp
// LICENSE
// Copyright (c) 2006-2008 Kevin Beason (kevin.beason@gmail.com)
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <math.h>   // smallpt, a Path Tracer by Kevin Beason, 2008 
#include <stdlib.h> // Make : g++ -O3 -fopenmp smallpt.cpp -o smallpt 
#include <stdio.h>  //        Remove "-fopenmp" for g++ version < 4.2 
struct Vec {        // Usage: time ./smallpt 5000 && xv image.ppm 
  double x, y, z;                  // position, also color (r,g,b) 
  Vec(double x_=0, double y_=0, double z_=0){ x=x_; y=y_; z=z_; } 
  Vec operator+(const Vec &b) const { return Vec(x+b.x,y+b.y,z+b.z); } 
  Vec operator-(const Vec &b) const { return Vec(x-b.x,y-b.y,z-b.z); } 
  Vec operator*(double b) const { return Vec(x*b,y*b,z*b); } 
  Vec mult(const Vec &b) const { return Vec(x*b.x,y*b.y,z*b.z); } 
  Vec& norm(){ return *this = *this * (1/sqrt(x*x+y*y+z*z)); }
	double len() const { return sqrt(x*x+y*y+z*z); }
  double dot(const Vec &b) const { return x*b.x+y*b.y+z*b.z; } // cross: 
	Vec operator%(Vec&b){return Vec(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x);} 
	Vec cross(const Vec &b) const { Vec bb(b); return Vec(*this) % bb;}
}; 
struct Ray { Vec o, d; Ray(Vec o_=Vec(),Vec d_=Vec()) : o(o_), d(d_){}}; 
enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance() 
struct Sphere { 
  double rad;       // radius 
  Vec p, e, c;      // position, emission, color 
  Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive) 
  Sphere(double rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_): 
    rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {} 
  double intersect(const Ray &r) const { // returns distance, 0 if nohit 
    Vec op = p-r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 
    double t, eps=1e-4, b=op.dot(r.d), det=b*b-op.dot(op)+rad*rad; 
    if (det<0) return 0; else det=sqrt(det); 
    return (t=b-det)>eps ? t : ((t=b+det)>eps ? t : 0);
  } 
}; 
Sphere spheres[] = {//Scene: radius, position, emission, color, material 
  Sphere(1e5, Vec( 1e5+1,40.8,81.6), Vec(),Vec(.75,.25,.25),DIFF),//Left 
  Sphere(1e5, Vec(-1e5+99,40.8,81.6),Vec(),Vec(.25,.25,.75),DIFF),//Rght 
  Sphere(1e5, Vec(50,40.8, 1e5),     Vec(),Vec(.75,.25,.75),DIFF),//Back 
// Sphere(1e5, Vec(50,40.8,-1e5+170), Vec(),Vec(),         DIFF),//Frnt 
  Sphere(1e5, Vec(50, 1e5, 81.6),    Vec(),Vec(.75,.75,.75),DIFF),//Botm 
  Sphere(1e5, Vec(50,-1e5+81.6,81.6),Vec(),Vec(.75,.75,.75),DIFF),//Top 
  Sphere(16.5,Vec(27,16.5,47),       Vec(),Vec(1,1,1)*.999, SPEC),//Mirr 
  Sphere(16.5,Vec(73,16.5,78),       Vec(),Vec(1,1,1)*.999, REFR),//Glas 
  Sphere(600, Vec(50,681.6-.27,81.6),Vec(12,12,12),  Vec(), DIFF) //Lite 
}; 
inline double clamp(double x){ return x<0 ? 0 : x>1 ? 1 : x; } 
inline int toInt(double x){ return int(pow(clamp(x),1/2.2)*255+.5); } 
inline bool intersect(const Ray &r, double &t, int &id){ 
  double n=sizeof(spheres)/sizeof(Sphere), d, inf=t=1e20; 
  for(int i=int(n);i--;) if((d=spheres[i].intersect(r))&&d<t){t=d;id=i;} 
  return t<inf; 
} 
Vec radiance(const Ray &r, int depth, unsigned short *Xi){ 
  double t;                               // distance to intersection 
  int id=0;                               // id of intersected object 
  if (!intersect(r, t, id)) return Vec(); // if miss, return black 
  const Sphere &obj = spheres[id];        // the hit object 
  Vec x=r.o+r.d*t, n=(x-obj.p).norm(), nl=n.dot(r.d)<0?n:n*-1, f=obj.c; 
  double p = f.x>f.y && f.x>f.z ? f.x : f.y>f.z ? f.y : f.z; // max refl 
  if (++depth>5) if (erand48(Xi)<p) f=f*(1/p); else return obj.e; //R.R. 
  if (obj.refl == DIFF){                  // Ideal DIFFUSE reflection 
    double r1=2*M_PI*erand48(Xi), r2=erand48(Xi), r2s=sqrt(r2); 
    Vec w=nl, u=((fabs(w.x)>.1?Vec(0,1):Vec(1))%w).norm(), v=w%u; 
    Vec d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2)).norm(); 
    return obj.e + f.mult(radiance(Ray(x,d),depth,Xi)); 
  } else if (obj.refl == SPEC)            // Ideal SPECULAR reflection 
    return obj.e + f.mult(radiance(Ray(x,r.d-n*2*n.dot(r.d)),depth,Xi)); 
  Ray reflRay(x, r.d-n*2*n.dot(r.d));     // Ideal dielectric REFRACTION 
  bool into = n.dot(nl)>0;                // Ray from outside going in? 
  double nc=1, nt=1.5, nnt=into?nc/nt:nt/nc, ddn=r.d.dot(nl), cos2t; 
  if ((cos2t=1-nnt*nnt*(1-ddn*ddn))<0)    // Total internal reflection 
    return obj.e + f.mult(radiance(reflRay,depth,Xi)); 
  Vec tdir = (r.d*nnt - n*((into?1:-1)*(ddn*nnt+sqrt(cos2t)))).norm(); 
  double a=nt-nc, b=nt+nc, R0=a*a/(b*b), c = 1-(into?-ddn:tdir.dot(n)); 
  double Re=R0+(1-R0)*c*c*c*c*c,Tr=1-Re,P=.25+.5*Re,RP=Re/P,TP=Tr/(1-P); 
  return obj.e + f.mult(depth>2 ? (erand48(Xi)<P ?   // Russian roulette 
    radiance(reflRay,depth,Xi)*RP:radiance(Ray(x,tdir),depth,Xi)*TP) : 
    radiance(reflRay,depth,Xi)*Re+radiance(Ray(x,tdir),depth,Xi)*Tr); 
} 



/* public domain Simple, Minimalistic JPEG writer - http://jonolick.com
 *
 * Quick Notes:
 * 	Based on a javascript jpeg writer
 * 	JPEG baseline (no JPEG progressive)
 * 	Supports 1, 3 or 4 component input. (luminance, RGB or RGBX)
 *
 * Latest revisions:
 *	1.52 (2012-22-11) Added support for specifying Luminance, RGB, or RGBA via comp(onents) argument (1, 3 and 4 respectively). 
 *	1.51 (2012-19-11) Fixed some warnings
 *	1.50 (2012-18-11) MT safe. Simplified. Optimized. Reduced memory requirements. Zero allocations. No namespace polution. Approx 340 lines code.
 *	1.10 (2012-16-11) compile fixes, added docs,
 *		changed from .h to .cpp (simpler to bootstrap), etc
 * 	1.00 (2012-02-02) initial release
 *
 * Basic usage:
 *	char *foo = new char[128*128*4]; // 4 component. RGBX format, where X is unused 
 *	jo_write_jpg("foo.jpg", foo, 128, 128, 4, 90); // comp can be 1, 3, or 4. Lum, RGB, or RGBX respectively.
 * 	
 * */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static const unsigned char s_jo_ZigZag[] = { 0,1,5,6,14,15,27,28,2,4,7,13,16,26,29,42,3,8,12,17,25,30,41,43,9,11,18,24,31,40,44,53,10,19,23,32,39,45,52,54,20,22,33,38,46,51,55,60,21,34,37,47,50,56,59,61,35,36,48,49,57,58,62,63 };

static void jo_writeBits(FILE *fp, int &bitBuf, int &bitCnt, const unsigned short *bs) {
	bitCnt += bs[1];
	bitBuf |= bs[0] << (24 - bitCnt);
	while(bitCnt >= 8) {
		unsigned char c = (bitBuf >> 16) & 255;
		putc(c, fp);
		if(c == 255) {
			putc(0, fp);
		}
		bitBuf <<= 8;
		bitCnt -= 8;
	}
}

static void jo_DCT(float &d0, float &d1, float &d2, float &d3, float &d4, float &d5, float &d6, float &d7) {
	float tmp0 = d0 + d7;
	float tmp7 = d0 - d7;
	float tmp1 = d1 + d6;
	float tmp6 = d1 - d6;
	float tmp2 = d2 + d5;
	float tmp5 = d2 - d5;
	float tmp3 = d3 + d4;
	float tmp4 = d3 - d4;

	// Even part
	float tmp10 = tmp0 + tmp3;	// phase 2
	float tmp13 = tmp0 - tmp3;
	float tmp11 = tmp1 + tmp2;
	float tmp12 = tmp1 - tmp2;

	d0 = tmp10 + tmp11; 		// phase 3
	d4 = tmp10 - tmp11;

	float z1 = (tmp12 + tmp13) * 0.707106781f; // c4
	d2 = tmp13 + z1; 		// phase 5
	d6 = tmp13 - z1;

	// Odd part
	tmp10 = tmp4 + tmp5; 		// phase 2
	tmp11 = tmp5 + tmp6;
	tmp12 = tmp6 + tmp7;

	// The rotator is modified from fig 4-8 to avoid extra negations.
	float z5 = (tmp10 - tmp12) * 0.382683433f; // c6
	float z2 = tmp10 * 0.541196100f + z5; // c2-c6
	float z4 = tmp12 * 1.306562965f + z5; // c2+c6
	float z3 = tmp11 * 0.707106781f; // c4

	float z11 = tmp7 + z3;		// phase 5
	float z13 = tmp7 - z3;

	d5 = z13 + z2;			// phase 6
	d3 = z13 - z2;
	d1 = z11 + z4;
	d7 = z11 - z4;
} 

static void jo_calcBits(int val, unsigned short bits[2]) {
	int tmp1 = val < 0 ? -val : val;
	val = val < 0 ? val-1 : val;
	bits[1] = 1;
	while(tmp1 >>= 1) {
		++bits[1];
	}
	bits[0] = val & ((1<<bits[1])-1);
}

static int jo_processDU(FILE *fp, int &bitBuf, int &bitCnt, float *CDU, float *fdtbl, int DC, const unsigned short HTDC[256][2], const unsigned short HTAC[256][2]) {
	const unsigned short EOB[2] = { HTAC[0x00][0], HTAC[0x00][1] };
	const unsigned short M16zeroes[2] = { HTAC[0xF0][0], HTAC[0xF0][1] };

	// DCT rows
	for(int dataOff=0; dataOff<64; dataOff+=8) {
		jo_DCT(CDU[dataOff], CDU[dataOff+1], CDU[dataOff+2], CDU[dataOff+3], CDU[dataOff+4], CDU[dataOff+5], CDU[dataOff+6], CDU[dataOff+7]);
	}
	// DCT columns
	for(int dataOff=0; dataOff<8; ++dataOff) {
		jo_DCT(CDU[dataOff], CDU[dataOff+8], CDU[dataOff+16], CDU[dataOff+24], CDU[dataOff+32], CDU[dataOff+40], CDU[dataOff+48], CDU[dataOff+56]);
	}
	// Quantize/descale/zigzag the coefficients
	int DU[64];
	for(int i=0; i<64; ++i) {
		float v = CDU[i]*fdtbl[i];
		DU[s_jo_ZigZag[i]] = (int)(v < 0 ? ceilf(v - 0.5f) : floorf(v + 0.5f));
	}

	// Encode DC
	int diff = DU[0] - DC; 
	if (diff == 0) {
		jo_writeBits(fp, bitBuf, bitCnt, HTDC[0]);
	} else {
		unsigned short bits[2];
		jo_calcBits(diff, bits);
		jo_writeBits(fp, bitBuf, bitCnt, HTDC[bits[1]]);
		jo_writeBits(fp, bitBuf, bitCnt, bits);
	}
	// Encode ACs
	int end0pos = 63;
	for(; (end0pos>0)&&(DU[end0pos]==0); --end0pos) {
	}
	// end0pos = first element in reverse order !=0
	if(end0pos == 0) {
		jo_writeBits(fp, bitBuf, bitCnt, EOB);
		return DU[0];
	}
	for(int i = 1; i <= end0pos; ++i) {
		int startpos = i;
		for (; DU[i]==0 && i<=end0pos; ++i) {
		}
		int nrzeroes = i-startpos;
		if ( nrzeroes >= 16 ) {
			int lng = nrzeroes>>4;
			for (int nrmarker=1; nrmarker <= lng; ++nrmarker)
				jo_writeBits(fp, bitBuf, bitCnt, M16zeroes);
			nrzeroes &= 15;
		}
		unsigned short bits[2];
		jo_calcBits(DU[i], bits);
		jo_writeBits(fp, bitBuf, bitCnt, HTAC[(nrzeroes<<4)+bits[1]]);
		jo_writeBits(fp, bitBuf, bitCnt, bits);
	}
	if(end0pos != 63) {
		jo_writeBits(fp, bitBuf, bitCnt, EOB);
	}
	return DU[0];
}

bool jo_write_jpg(const char *filename, const void *data, int width, int height, int comp, int quality) {
	// Constants that don't pollute global namespace
	static const unsigned char std_dc_luminance_nrcodes[] = {0,0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0};
	static const unsigned char std_dc_luminance_values[] = {0,1,2,3,4,5,6,7,8,9,10,11};
	static const unsigned char std_ac_luminance_nrcodes[] = {0,0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,0x7d};
	static const unsigned char std_ac_luminance_values[] = {
		0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,0x07,0x22,0x71,0x14,0x32,0x81,0x91,0xa1,0x08,
		0x23,0x42,0xb1,0xc1,0x15,0x52,0xd1,0xf0,0x24,0x33,0x62,0x72,0x82,0x09,0x0a,0x16,0x17,0x18,0x19,0x1a,0x25,0x26,0x27,0x28,
		0x29,0x2a,0x34,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,
		0x5a,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x83,0x84,0x85,0x86,0x87,0x88,0x89,
		0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,
		0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe1,0xe2,
		0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,0xf9,0xfa
	};
	static const unsigned char std_dc_chrominance_nrcodes[] = {0,0,3,1,1,1,1,1,1,1,1,1,0,0,0,0,0};
	static const unsigned char std_dc_chrominance_values[] = {0,1,2,3,4,5,6,7,8,9,10,11};
	static const unsigned char std_ac_chrominance_nrcodes[] = {0,0,2,1,2,4,4,3,4,7,5,4,4,0,1,2,0x77};
	static const unsigned char std_ac_chrominance_values[] = {
		0x00,0x01,0x02,0x03,0x11,0x04,0x05,0x21,0x31,0x06,0x12,0x41,0x51,0x07,0x61,0x71,0x13,0x22,0x32,0x81,0x08,0x14,0x42,0x91,
		0xa1,0xb1,0xc1,0x09,0x23,0x33,0x52,0xf0,0x15,0x62,0x72,0xd1,0x0a,0x16,0x24,0x34,0xe1,0x25,0xf1,0x17,0x18,0x19,0x1a,0x26,
		0x27,0x28,0x29,0x2a,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,
		0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x82,0x83,0x84,0x85,0x86,0x87,
		0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,
		0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,
		0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,0xf9,0xfa
	};
	// Huffman tables
	static const unsigned short YDC_HT[256][2] = { {0,2},{2,3},{3,3},{4,3},{5,3},{6,3},{14,4},{30,5},{62,6},{126,7},{254,8},{510,9}};
	static const unsigned short UVDC_HT[256][2] = { {0,2},{1,2},{2,2},{6,3},{14,4},{30,5},{62,6},{126,7},{254,8},{510,9},{1022,10},{2046,11}};
	static const unsigned short YAC_HT[256][2] = { 
		{10,4},{0,2},{1,2},{4,3},{11,4},{26,5},{120,7},{248,8},{1014,10},{65410,16},{65411,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{12,4},{27,5},{121,7},{502,9},{2038,11},{65412,16},{65413,16},{65414,16},{65415,16},{65416,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{28,5},{249,8},{1015,10},{4084,12},{65417,16},{65418,16},{65419,16},{65420,16},{65421,16},{65422,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{58,6},{503,9},{4085,12},{65423,16},{65424,16},{65425,16},{65426,16},{65427,16},{65428,16},{65429,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{59,6},{1016,10},{65430,16},{65431,16},{65432,16},{65433,16},{65434,16},{65435,16},{65436,16},{65437,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{122,7},{2039,11},{65438,16},{65439,16},{65440,16},{65441,16},{65442,16},{65443,16},{65444,16},{65445,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{123,7},{4086,12},{65446,16},{65447,16},{65448,16},{65449,16},{65450,16},{65451,16},{65452,16},{65453,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{250,8},{4087,12},{65454,16},{65455,16},{65456,16},{65457,16},{65458,16},{65459,16},{65460,16},{65461,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{504,9},{32704,15},{65462,16},{65463,16},{65464,16},{65465,16},{65466,16},{65467,16},{65468,16},{65469,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{505,9},{65470,16},{65471,16},{65472,16},{65473,16},{65474,16},{65475,16},{65476,16},{65477,16},{65478,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{506,9},{65479,16},{65480,16},{65481,16},{65482,16},{65483,16},{65484,16},{65485,16},{65486,16},{65487,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{1017,10},{65488,16},{65489,16},{65490,16},{65491,16},{65492,16},{65493,16},{65494,16},{65495,16},{65496,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{1018,10},{65497,16},{65498,16},{65499,16},{65500,16},{65501,16},{65502,16},{65503,16},{65504,16},{65505,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{2040,11},{65506,16},{65507,16},{65508,16},{65509,16},{65510,16},{65511,16},{65512,16},{65513,16},{65514,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{65515,16},{65516,16},{65517,16},{65518,16},{65519,16},{65520,16},{65521,16},{65522,16},{65523,16},{65524,16},{0,0},{0,0},{0,0},{0,0},{0,0},
		{2041,11},{65525,16},{65526,16},{65527,16},{65528,16},{65529,16},{65530,16},{65531,16},{65532,16},{65533,16},{65534,16},{0,0},{0,0},{0,0},{0,0},{0,0}
	};
	static const unsigned short UVAC_HT[256][2] = { 
		{0,2},{1,2},{4,3},{10,4},{24,5},{25,5},{56,6},{120,7},{500,9},{1014,10},{4084,12},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{11,4},{57,6},{246,8},{501,9},{2038,11},{4085,12},{65416,16},{65417,16},{65418,16},{65419,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{26,5},{247,8},{1015,10},{4086,12},{32706,15},{65420,16},{65421,16},{65422,16},{65423,16},{65424,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{27,5},{248,8},{1016,10},{4087,12},{65425,16},{65426,16},{65427,16},{65428,16},{65429,16},{65430,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{58,6},{502,9},{65431,16},{65432,16},{65433,16},{65434,16},{65435,16},{65436,16},{65437,16},{65438,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{59,6},{1017,10},{65439,16},{65440,16},{65441,16},{65442,16},{65443,16},{65444,16},{65445,16},{65446,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{121,7},{2039,11},{65447,16},{65448,16},{65449,16},{65450,16},{65451,16},{65452,16},{65453,16},{65454,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{122,7},{2040,11},{65455,16},{65456,16},{65457,16},{65458,16},{65459,16},{65460,16},{65461,16},{65462,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{249,8},{65463,16},{65464,16},{65465,16},{65466,16},{65467,16},{65468,16},{65469,16},{65470,16},{65471,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{503,9},{65472,16},{65473,16},{65474,16},{65475,16},{65476,16},{65477,16},{65478,16},{65479,16},{65480,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{504,9},{65481,16},{65482,16},{65483,16},{65484,16},{65485,16},{65486,16},{65487,16},{65488,16},{65489,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{505,9},{65490,16},{65491,16},{65492,16},{65493,16},{65494,16},{65495,16},{65496,16},{65497,16},{65498,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{506,9},{65499,16},{65500,16},{65501,16},{65502,16},{65503,16},{65504,16},{65505,16},{65506,16},{65507,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{2041,11},{65508,16},{65509,16},{65510,16},{65511,16},{65512,16},{65513,16},{65514,16},{65515,16},{65516,16},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},
		{16352,14},{65517,16},{65518,16},{65519,16},{65520,16},{65521,16},{65522,16},{65523,16},{65524,16},{65525,16},{0,0},{0,0},{0,0},{0,0},{0,0},
		{1018,10},{32707,15},{65526,16},{65527,16},{65528,16},{65529,16},{65530,16},{65531,16},{65532,16},{65533,16},{65534,16},{0,0},{0,0},{0,0},{0,0},{0,0}
	};
	static const int YQT[] = {16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,14,17,22,29,51,87,80,62,18,22,37,56,68,109,103,77,24,35,55,64,81,104,113,92,49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99};
	static const int UVQT[] = {17,18,24,47,99,99,99,99,18,21,26,66,99,99,99,99,24,26,56,99,99,99,99,99,47,66,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99};
	static const float aasf[] = { 1.0f * 2.828427125f, 1.387039845f * 2.828427125f, 1.306562965f * 2.828427125f, 1.175875602f * 2.828427125f, 1.0f * 2.828427125f, 0.785694958f * 2.828427125f, 0.541196100f * 2.828427125f, 0.275899379f * 2.828427125f };

	if(!data || !filename || !width || !height || comp > 4 || comp < 1 || comp == 2) {
		return false;
	}

	FILE *fp = fopen(filename, "wb");
	if(!fp) {
		return false;
	}

	quality = quality ? quality : 90;
	quality = quality < 1 ? 1 : quality > 100 ? 100 : quality;
	quality = quality < 50 ? 5000 / quality : 200 - quality * 2;

	unsigned char YTable[64], UVTable[64];
	for(int i = 0; i < 64; ++i) {
		int yti = (YQT[i]*quality+50)/100;
		YTable[s_jo_ZigZag[i]] = yti < 1 ? 1 : yti > 255 ? 255 : yti;
		int uvti  = (UVQT[i]*quality+50)/100;
		UVTable[s_jo_ZigZag[i]] = uvti < 1 ? 1 : uvti > 255 ? 255 : uvti;
	}

	float fdtbl_Y[64], fdtbl_UV[64];
	for(int row = 0, k = 0; row < 8; ++row) {
		for(int col = 0; col < 8; ++col, ++k) {
			fdtbl_Y[k]  = 1 / (YTable [s_jo_ZigZag[k]] * aasf[row] * aasf[col]);
			fdtbl_UV[k] = 1 / (UVTable[s_jo_ZigZag[k]] * aasf[row] * aasf[col]);
		}
	}

	// Write Headers
	static const unsigned char head0[] = { 0xFF,0xD8,0xFF,0xE0,0,0x10,'J','F','I','F',0,1,1,0,0,1,0,1,0,0,0xFF,0xDB,0,0x84,0 };
	fwrite(head0, sizeof(head0), 1, fp);
	fwrite(YTable, sizeof(YTable), 1, fp);
	putc(1, fp);
	fwrite(UVTable, sizeof(UVTable), 1, fp);
	const unsigned char head1[] = { 0xFF,0xC0,0,0x11,8,height>>8,height&0xFF,width>>8,width&0xFF,3,1,0x11,0,2,0x11,1,3,0x11,1,0xFF,0xC4,0x01,0xA2,0 };
	fwrite(head1, sizeof(head1), 1, fp);
	fwrite(std_dc_luminance_nrcodes+1, sizeof(std_dc_luminance_nrcodes)-1, 1, fp);
	fwrite(std_dc_luminance_values, sizeof(std_dc_luminance_values), 1, fp);
	putc(0x10, fp); // HTYACinfo
	fwrite(std_ac_luminance_nrcodes+1, sizeof(std_ac_luminance_nrcodes)-1, 1, fp);
	fwrite(std_ac_luminance_values, sizeof(std_ac_luminance_values), 1, fp);
	putc(1, fp); // HTUDCinfo
	fwrite(std_dc_chrominance_nrcodes+1, sizeof(std_dc_chrominance_nrcodes)-1, 1, fp);
	fwrite(std_dc_chrominance_values, sizeof(std_dc_chrominance_values), 1, fp);
	putc(0x11, fp); // HTUACinfo
	fwrite(std_ac_chrominance_nrcodes+1, sizeof(std_ac_chrominance_nrcodes)-1, 1, fp);
	fwrite(std_ac_chrominance_values, sizeof(std_ac_chrominance_values), 1, fp);
	static const unsigned char head2[] = { 0xFF,0xDA,0,0xC,3,1,0,2,0x11,3,0x11,0,0x3F,0 };
	fwrite(head2, sizeof(head2), 1, fp);

	// Encode 8x8 macroblocks
	const unsigned char *imageData = (const unsigned char *)data;
	int DCY=0, DCU=0, DCV=0;
	int bitBuf=0, bitCnt=0;
	int ofsG = comp > 1 ? 1 : 0, ofsB = comp > 1 ? 2 : 0;
	for(int y = 0; y < height; y += 8) {
		for(int x = 0; x < width; x += 8) {
			float YDU[64], UDU[64], VDU[64];
			for(int row = y, pos = 0; row < y+8; ++row) {
				for(int col = x; col < x+8; ++col, ++pos) {
					int p = row*width*comp + col*comp;
					if(row >= height) {
						p -= width*comp*(row+1 - height);
					}
					if(col >= width) {
						p -= comp*(col+1 - width);
					}

					float r = imageData[p+0], g = imageData[p+ofsG], b = imageData[p+ofsB];
					YDU[pos]=+0.29900f*r+0.58700f*g+0.11400f*b-128;
					UDU[pos]=-0.16874f*r-0.33126f*g+0.50000f*b;
					VDU[pos]=+0.50000f*r-0.41869f*g-0.08131f*b;
				}
			}

			DCY = jo_processDU(fp, bitBuf, bitCnt, YDU, fdtbl_Y, DCY, YDC_HT, YAC_HT);
			DCU = jo_processDU(fp, bitBuf, bitCnt, UDU, fdtbl_UV, DCU, UVDC_HT, UVAC_HT);
			DCV = jo_processDU(fp, bitBuf, bitCnt, VDU, fdtbl_UV, DCV, UVDC_HT, UVAC_HT);
		}
	}
	
	// Do the bit alignment of the EOI marker
	static const unsigned short fillBits[] = {0x7F, 7};
	jo_writeBits(fp, bitBuf, bitCnt, fillBits);

	// EOI
	putc(0xFF, fp);
	putc(0xD9, fp);

	fclose(fp);
	return true;
}








// #include <vector>

// struct tensor {
// 	std::vector<double> data;
// 	std::vector<int> shape;
// 	void printJSON(FILE * f) const {

// 	}
// };

// struct transform {
// 	double rotation[9] = {
// 		1, 0, 0,
// 		0, 1, 0,
// 		0, 0, 1,
// 	};
// 	Vec translation;
// };

struct Matrix3x3 {
	double data[9];	// row-major

	double operator[](int i) const { return data[i]; }
	double &operator[](int i) { return data[i]; }

	Matrix3x3 operator+(const Matrix3x3 &b) const { 
		Matrix3x3 ret; for (int i=0; i<9; ++i) { ret[i]=data[i]+b[i];} return ret;
	}
	Matrix3x3 operator-(const Matrix3x3 &b) const { 
		Matrix3x3 ret; for (int i=0; i<9; ++i) { ret[i]=data[i]-b[i];} return ret;
	}
	Matrix3x3 operator+(double v) const { 
		Matrix3x3 ret; for (int i=0; i<9; ++i) { ret[i]=data[i]+v;} return ret;
	}
	Matrix3x3 operator-(double v) const { 
		Matrix3x3 ret; for (int i=0; i<9; ++i) { ret[i]=data[i]-v;} return ret;
	}
	Matrix3x3 operator*(double v) const { 
		Matrix3x3 ret; for (int i=0; i<9; ++i) { ret[i]=data[i]*v;} return ret;
	}

	Vec row(int r) const { return Vec(data[r*3], data[r*3+1], data[r*3+2]); }
	Vec col(int c) const { return Vec(data[c], data[c+3], data[c+6]); }
	Vec operator*(const Vec &v) const { 
		return Vec(row(0).dot(v), row(1).dot(v), row(2).dot(v));
	}
	Matrix3x3 operator*(const Matrix3x3 &m) const {
		Matrix3x3 ret;
		for (int r=0; r<3; r++) { for (int c=0; c<3; c++) {
				ret[r*3+c] = row(r).dot(m.col(c));
		}}
		return ret;
	}

	static Matrix3x3 asCross(const Vec &u) {
		// u x v = [u]x * v.  Below is [u]x, the matrix form of the cross product;
		// also called the skew of `u`.
		return {.data={
			   0,  -u.z,   u.y,
			 u.z,     0,  -u.x,
			-u.y,   u.x,     0,
		}};
	}

	static Matrix3x3 I() {
		return {.data={
			   1, 0, 0,
				 0, 1, 0,
				 0, 0, 1,
		}};
	}

	static Matrix3x3 vvT(const Vec &v) {
		// v * v.T
		return {.data={
			v.x*v.x, v.x*v.y, v.x*v.z,
			v.y*v.x, v.y*v.y, v.y*v.z,
			v.z*v.x, v.z*v.y, v.z*v.z,
		}};
	}

	// In the rendering frame, -z is forward, so this is roll
	static Matrix3x3 rotMatrixAboutZ(double theta) {
		return {.data={
			cos(theta), -sin(theta),  0,
			sin(theta),  cos(theta),  0,
						  0,           0,  1,
		}};
	}

	// In the rendering frame, +x is to the right, so this is pitch
	static Matrix3x3 rotMatrixAboutX(double theta) {
		return {.data={
			1,          0,           0,
			0, cos(theta), -sin(theta),
			0, sin(theta),  cos(theta),
		}};
	}

	// In the rendering frame, -y is up, so this is yaw
	static Matrix3x3 rotMatrixAboutY(double theta) {
		return {.data={
			cos(theta), 	0, sin(theta),
			0, 						1, 					0,
			-sin(theta), 	0, cos(theta),
		}};
	}



	static Matrix3x3 rotMatrixFromAzimuth(double phi) {
		return {.data={
			 cos(phi), -sin(phi),  0,
			 sin(phi),  cos(phi),  0,
							0,         0,  1,
		}};
	}

	static Matrix3x3 rotMatrixFromInclination(double theta) {
		return {.data={
			1,           0,          0,
			0,  cos(theta), sin(theta),
			0, -sin(theta), cos(theta),
		}};
	}

	static Matrix3x3 rotMatrixFromEuler(double y, double p, double r) {
		// return {.data={
		// 	cos(y)*cos(p), cos(y)*sin(p)*sin(r) - sin(y)*cos(r), cos(y)*sin(p)*cos(r) + sin(y)*sin(r),
		// 	sin(y)*cos(p), sin(y)*sin(p)*sin(r) + cos(y)*cos(r), sin(y)*sin(p)*cos(r) - cos(y)*sin(y),
		// 	-sin(p),       cos(p)*sin(r),                        cos(p)*cos(r),
		// }};

		float ci ( cos(y)); 
		float cj ( cos(p)); 
		float ch ( cos(r)); 
		float si ( sin(y)); 
		float sj ( sin(p)); 
		float sh ( sin(r)); 
		float cc = ci * ch; 
		float cs = ci * sh; 
		float sc = si * ch; 
		float ss = si * sh;

		return {.data={
		  cj * ch, 		sj * sc - cs, 		sj * cc + ss,
			cj * sh, 		sj * ss + cc, 		sj * cs - sc, 
			-sj,      	cj * si,      		cj * ci,
		}};
	}

	static Matrix3x3 rotMatrixFromVectors(const Vec &a, const Vec &b) {
		// First, find the rotation axis and amount (theta)
		const Vec u = a.cross(b);
		const double cosTheta = a.dot(b)/(a.len()*b.len());
		const double sinTheta = u.len()/(a.len()*b.len());

		// Use Rodrigues' formula to compute the rotation matrix between
		// two vectors.  Good reference: https://www2.cs.duke.edu/courses/fall13/compsci527/notes/rodrigues.pdf
		// Let the rotation vector be r = theta u, where u is the axis of 
		// rotation and theta is the amount.  Then the rotation matrix R is:
		//   R = I*cos(theta) + (1 - cos(theta))*u*u.T + [u]x*sin(theta)
		Matrix3x3 R = I()*cosTheta + vvT(u)*(1 - cosTheta) + asCross(u)*sinTheta;
		return R;
	}

	void print(FILE *f) const {
		fprintf(f,"\n\n");
		fprintf(f,"%5.2f %5.2f %5.2f\n", data[0], data[1], data[2]);
		fprintf(f,"%5.2f %5.2f %5.2f\n", data[3], data[4], data[5]);
		fprintf(f,"%5.2f %5.2f %5.2f\n", data[6], data[7], data[8]);
		fprintf(f,"\n\n");
	}
};


struct lidar {
	struct config {
		double inclination_min, inclination_max;
		int n_beams;
		double azimuth_step;
		Ray initial_pose;
	};

	config conf;
	Ray pose;

	lidar(config c) : conf(c) {
		pose=conf.initial_pose;
	}

	unsigned int pointsPerScan() const {
		return conf.n_beams * floor(2*M_PI/conf.azimuth_step);
	}
};


struct camera {
	struct config {
		int w, h; 							// width, height (pixels)
		double FoV_x; 					// field of view (radians)
		Ray initial_pose;				// initial position and orientation
	};

	config conf;
	double fx, fy, cx, cy;		// intrinsics
	Ray pose;		// current pose

	camera(config c) : conf(c) {
		// FoV_x = 2 arctan(w / f_x) =>
		//   f_x = w / ( 2 tan( FoV_x / 2))
		fx = conf.w / (2.*tan(conf.FoV_x/2));
		fy = fx;
		cx = .5*conf.w; cy = .5*conf.h;
		pose = conf.initial_pose;
	}

	// Matrix3x3 R() const {
	// 	return Matrix3x3::rotMatrixFromVectors(
	// 		Vec(pose.d).norm(), Vec(pose.d).norm());
	// }

	Vec T() const { return pose.o; }

	Ray pixelToRay(double x, double y) const {
		const Vec ray_in_cam = Vec((x-cx)/fx, (y-cy)/fy, -1);
		// fprintf(stderr, "x %5.2f y %5.2f\n", ray_in_cam.x, ray_in_cam.y);
		static const Vec zHat = {0, 0, -1};
		const Matrix3x3 Rinv = 
			Matrix3x3::rotMatrixFromVectors(pose.d, zHat);
		// Rinv.print(stderr);
		return Ray(pose.o , Vec( Rinv * ray_in_cam).norm());

		/*

     X  = (x - 1 * cx) / fx



		*/
		


		// s = 2*s*cx; t = 2*t*cy; 

		// return Ray(pose.o, Vec((s - cx - pose.o.x)/ fx, (t - cy - pose.o.y)/ fy, 1) );
		// return Ray(pose.o, pose.d + Vec(s * cx , t * cy ));
	}

// 		cv::Point3d ray;
//    ray.x = (uv_rect.x - cx() - Tx()) / fx();
// 00265   ray.y = (uv_rect.y - cy() - Ty()) / fy();
// 00266   ray.z = 1.0;
// 		return Ray(origin, lower_left_corner + s * horizontal + t * vertical - origin); 
// 	}

// Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, float vfov, float aspect){// vfov is top to bottom in degrees, field of view on the vertical axis
// 			Vec3 w, u, v;
// 			float theta = vfov * M_PI/180;	// convert to radiants
// 			float half_height = tan(theta/2);
// 			float half_width = aspect * half_height;
// 			origin = lookfrom;
// 			w = unit_vector(lookfrom - lookat);
// 			u = unit_vector(cross(vup, w));
// 			v = cross(w, u);
// 			lower_left_corner = origin - u*half_width - v*half_height - w;
// 			horizontal = 2*half_width*u;
// 			vertical = 2*half_height*v;
// 		}


	// int w=640, h=480; // width, height
	// double FoV_x = 45. * (M_PI / 180.); // width, height, FoV



	// // FoV_x = 2 arctan(w / f_x)
	// fov_h = 2. * math.atan(.5 * self.width / f_x)
	// fov_h / 2 = atan(.5 * w / f_x)
	// tan (fov_h / 2) = .5 * w / f_x
	// f_x = w / 2 tan(fov_h / 2) 

	// double fx, fy;
	// Ray view;

	// transform extrinsic = {
	// 	// -pi/2 yaw, -pi/2 pitch
	// 	// from scipy.spatial.transform import Rotation as R
	// 	// R.from_euler('zyx', [-pi/2, -pi/2, 0]).as_dcm()
	// 	.rotation = {
	// 		 0,  0,  -1,
	// 		-1,  0,   0,
	// 		 0,  1,   0,
	// 	}
	// };

};

// struct world {
//   Ray cam;
//   Ray ego;
//   Ray laser;
// };

// https://github.com/RichieSams/lantern/blob/669e6c9ccd33f455961a96c5cfe0a5bc3f2af59c/source/scene/camera.cpp#L86



int main(int argc, char *argv[]){ 
  int w=640, h=480, samps = argc==2 ? atoi(argv[1])/4 : 1; // # samples 
  // Ray cam(Vec(50,52,295.6), Vec(0,-0.042612,-1).norm()); // cam pos, dir
	Ray cam(Vec(50,52,295.6), Vec(0,0,-1)); // cam pos, dir 
  Vec cx=Vec(w*.5135/h), cy=(cx%cam.d).norm()*.5135, r, *c=new Vec[w*h]; 

// 	camera camera = {
// 		.conf={.w=w, .h=h, .FoV_x=48*M_PI/180, .initial_pose=cam}};

// #pragma omp parallel for schedule(dynamic, 1) private(r)       // OpenMP 
//   for (int y=0; y<h; y++){                       // Loop over image rows 
//     fprintf(stderr,"\rRendering (%d spp) %5.2f%%",samps*4,100.*y/(h-1)); 
//     for (unsigned short x=0, Xi[3]={0,0,y*y*y}; x<w; x++)   // Loop cols 
//       for (int sy=0, i=(h-y-1)*w+x; sy<2; sy++)     // 2x2 subpixel rows 
//         for (int sx=0; sx<2; sx++, r=Vec()){        // 2x2 subpixel cols 
//           for (int s=0; s<samps; s++){ 
//             double r1=2*erand48(Xi), dx=r1<1 ? sqrt(r1)-1: 1-sqrt(2-r1); 
//             double r2=2*erand48(Xi), dy=r2<1 ? sqrt(r2)-1: 1-sqrt(2-r2); 

//             // Vec d = cx*( ( (sx+.5 + dx)/2 + x)/w - .5) + 
//             //         cy*( ( (sy+.5 + dy)/2 + y)/h - .5) + cam.d; 
//             // r = r + radiance(Ray(cam.o+d*140,d.norm()),0,Xi)*(1./samps); 
// 						// // Camera rays are pushed ^^^^^ forward to start in interior 

// 						// // if (((x*w + y) % 1000) == 0) {
// 						// // Vec d = cx*( ( (sx+.5 + dx)/2 + x)/w - .5) + 
//             // //         cy*( ( (sy+.5 + dy)/2 + y)/h - .5) + cam.d; 
// 						// // fprintf(stderr, "dx %5.2f dy %5.2f\n", d.x, d.y);}

// 						Ray rr = camera.pixelToRay(
// 												x + (sx+.5 + dx)/2,
// 												y + (sy+.5 + dy)/2);
// 							// ( ( (sx+.5 + dx)/2 + x)/w - .5),
// 							// ( ( (sy+.5 + dy)/2 + y)/h - .5));
// 						// r = r + radiance(Ray(rr.o+rr.d*140,rr.d.norm()),0,Xi)*(1./samps); 
// 						r = r + radiance(rr,0,Xi)*(1./samps); 
// 						// Camera rays are pushed ^^^^^ forward to start in interior 

//           } 
//           c[i] = c[i] + Vec(clamp(r.x),clamp(r.y),clamp(r.z))*.25; 
//         } 
//   }


//   char *jimg = new char[w*h*4];

//   FILE *f = fopen("image.ppm", "w");         // Write image to PPM file.
//   fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
//   for (int i=0; i<w*h; i++) {
//     int r = toInt(c[i].x), g = toInt(c[i].y), b = toInt(c[i].z);
//     fprintf(f,"%d %d %d ", r, g, b);
//     int p=i*4; jimg[p] = r; jimg[p+1] = g; jimg[p+2] = b;
//   }

//   jo_write_jpg("foo.jpg", jimg, w, h, 4, 90);
//   delete[] jimg;


	

  // // Point Cloud
	// int n_ptc = w*h;
  // Vec *ptc = new Vec[n_ptc];
  // for (int y=0; y<h; y+= 4){
  //   for (int x=0; x<w; x+=4) {
  //   int i = (h-y-1)*w+x; ptc[i] = Vec();
  //   Vec d = cx*( double(x)/w - .5) +
  //           cy*( double(y)/h - .5) + cam.d;
  //   double t; int id;
  //   Ray r(cam.o+d*140,d.norm());
  //   if (intersect(r, t, id)) {
  //     ptc[i] = r.o + r.d * t;
  //   }
  // 	}
  // }
	// int lw = w; int lh = h;
  
	// Point Cloud
	lidar l = {.conf={
		.inclination_min=-30*M_PI/180,
		.inclination_max= 30*M_PI/180,
		.n_beams=32,
		.azimuth_step=M_PI/120,
		.initial_pose=Ray(Vec(50,52,295.6 -160), Vec(0,0,-1)),
	}};
	const int n_ptc = l.pointsPerScan();
	Vec *ptc = new Vec[n_ptc]; {
		int p=0;
		
			const double incl_step =
				(l.conf.inclination_max - l.conf.inclination_min) / l.conf.n_beams;
			for (double inclination = l.conf.inclination_min;
			         inclination < l.conf.inclination_max; inclination += incl_step) {

		for (double azimuth=-M_PI; azimuth<M_PI; azimuth+=l.conf.azimuth_step) {
				// fprintf(stderr, "az %5.2f \n", azimuth);

				auto R = Matrix3x3::rotMatrixAboutY(azimuth) * 
				Matrix3x3::rotMatrixAboutX(inclination) ;



				// works
				// auto R = Matrix3x3::rotMatrixFromEuler(inclination, azimuth,  0);

				// auto R = Matrix3x3::rotMatrixFromEuler(azimuth, inclination, 0);  xxx

			/*
			dont forget extrinsic!
			from math import pi
			from scipy.spatial.transform import Rotation as R
			axes_transformation = R.from_euler('zyx', [-pi/2, -pi/2, 0]).as_dcm()
			xyz = axes_transformation.dot(xyz.T).T
			*/

				// Matrix3x3 R = 
				// 	Matrix3x3::rotMatrixFromAzimuth(inclination ) * 
				// 	Matrix3x3::rotMatrixFromInclination( azimuth);
				
				// Matrix3x3 extrinsic = 
					 
				// 	Matrix3x3::rotMatrixFromInclination(-M_PI/2) *
				// 	Matrix3x3::rotMatrixFromAzimuth(-M_PI/2);
				
				Matrix3x3 extrinsic = Matrix3x3::rotMatrixFromEuler(-M_PI/2, -M_PI/2, 0);

				// auto Raz = Matrix3x3::rotMatrixFromAzimuth(azimuth);
				// auto Rinc = Matrix3x3::rotMatrixFromInclination(inclination);

					// Matrix3x3::rotMatrixFromAzimuth(azimuth).print(stderr);
				
				Vec zHat = {0, 0, -1};
// auto yyy = (extrinsic * zHat);
// 				fprintf(stderr, "ex * zhat %5.2f %5.2f %5.2f \n", yyy.x, yyy.y, yyy.z);

				Ray beam(l.pose.o, (  R * zHat).norm());
				// Ray beam(l.pose.o, (Raz * Rinc * zHat).norm());
				double t; int id;
				if (intersect(beam, t, id) && t < 450) {
					ptc[p] = beam.o + beam.d * t;
				} else {
					ptc[p] = Vec();
				}
				++p;
			}
		}
	}
	int lw = int(floor(2*M_PI/l.conf.azimuth_step));
	int lh = l.conf.n_beams;

{ 
  FILE *f = fopen("ptc.ppm", "w");         // Write image to PPM file. 
  fprintf(f, "P3\n%d %d\n%d\n", lw, lh, 255); 
  for (int i=0; i<n_ptc; i++) { 
    double l = sqrt(ptc[i].x * ptc[i].x + ptc[i].y * ptc[i].y + ptc[i].z * ptc[i].z);
    fprintf(f,"%d %d %d ", int(l), int(l), int(l)); 
  }
}
{
  FILE *f = fopen("ptc.ply", "w");
  fprintf(f, "ply\nformat ascii 1.0\nelement vertex %d\n", n_ptc);
  fprintf(f, "property float32 x\n");
  fprintf(f, "property float32 y\n");
  fprintf(f, "property float32 z\n");
  fprintf(f, "end_header\n");
  for (int i=0; i<n_ptc; i++)
    fprintf(f,"%5.2f %5.2f %5.2f \n", ptc[i].x, ptc[i].y, ptc[i].z);
}
}

// import numpy as np
// lines = open('/opt/au/au/fixtures/datasets/av_spheres/ptc.ply', 'r').readlines()
// lines = lines[7:]
// import numpy
// def to_v(l):
//   x, y, z = l.split()
//   return float(x), float(y), float(z)
// xyz = np.array([to_v(l) for l in lines if (to_v(l) != (0., 0., 0.))])

// import plotly.graph_objects as go
// import pandas as pd

// df_tmp = pd.DataFrame(xyz, columns=["x", "y", "z"])
// df_tmp["norm"] = np.sqrt(np.power(df_tmp[["x", "y", "z"]].values, 2).sum(axis=1))
// scatter = go.Scatter3d(
//             x=df_tmp["x"],
//             y=df_tmp["y"],
//             z=df_tmp["z"],
//             mode="markers",
//             marker=dict(size=1, color=df_tmp["norm"], opacity=0.8),)
// fig = go.Figure(data=[scatter])
// fig.update_layout(scene_aspectmode="data")
// fig.show()

