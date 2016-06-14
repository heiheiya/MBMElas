
#ifndef __MBM_ELAS_H__
#define __MBM_ELAS_H__

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include "../filter/precomp.hpp"
#include "guidedFilter.h"
#include <math.h>
#include "triangle.h"
#include "matrix.h"

using namespace std;
using namespace cv;
using namespace ximgproc;

// define fixed-width datatypes for Visual Studio projects
#ifndef _MSC_VER
#include <stdint.h>
#else
typedef __int8            int8_t;
typedef __int16           int16_t;
typedef __int32           int32_t;
typedef __int64           int64_t;
typedef unsigned __int8   uint8_t;
typedef unsigned __int16  uint16_t;
typedef unsigned __int32  uint32_t;
typedef unsigned __int64  uint64_t;
#endif

class MBMElas {

public:

	// parameter settings
	struct parameters {

		int32_t mncc_block;            // mncc window
		int32_t horizontal_block;    // horizontal window aggregation
		int32_t vertical_block;         // vertical window aggregation
		int32_t small_block;            //  small window aggregation
		int32_t large_block;           //  large window aggregation
		float     mncc_threshold;    // minimum MNCC
		int32_t disp_min;               // minimum disparity
		int32_t disp_max;               // maximum disparity

		int32_t candidate_stepsize;     // step size of regular grid on which support points are matched
		int32_t incon_window_size;      // window size of inconsistent support point check
		int32_t incon_threshold;        // disparity similarity threshold for support point to be considered consistent
		int32_t incon_min_support;      // minimum number of consistent support points
		
		int32_t grid_size;              // size of neighborhood for additional support point extrapolation

		int32_t lr_threshold;           // disparity threshold for left/right consistency check
		float     speckle_sim_threshold;  // similarity threshold for speckle segmentation
		int32_t speckle_size;           // maximal size of a speckle (small speckles get removed)
		int32_t ipol_gap_width;         // interpolate small gaps (left<->right, top<->bottom)

		// constructor
		parameters () {

			mncc_block = 3;
			horizontal_block = 61;   
			vertical_block = 1;
			small_block = 3;
			large_block = 11;
			mncc_threshold = 0.00001;

			disp_min              = 0;
			disp_max              = 31;

			candidate_stepsize    = 10;
			incon_window_size     = 10;  
			incon_threshold       = 3;    
			incon_min_support     = 50;

			grid_size             = 20;

			lr_threshold          = 3;
			speckle_sim_threshold =3;
			speckle_size          = 2000;
			ipol_gap_width        = 3;
		}
	};

	// constructor, input: parameters  
	MBMElas (parameters param) : param(param) {}

	// deconstructor
	~MBMElas () {}

	// get disparity function
	void getDisparity(Mat & left, Mat & right, Mat leftDisparity, Mat rightDisparity, Mat maxMncc, Mat lImg, Mat rImg);
	void getLeftDisparity(Mat & left, Mat & right, Mat leftDisparity, Mat maxMncc, Mat lImg);
	void getRightDisparity(Mat & left, Mat & right, Mat rightDisparity, Mat rImg);
	Mat GuidedFilter(Mat & guidedIm, Mat & pIm, int r1, int r2);

	// disparity refine
	void refineLeftDisparity (float* D1,float* D2, float* D, const int32_t* dims, Mat left, Mat maxMncc, Mat leftDisparity);

private:

	struct support_pt {
		int32_t u;
		int32_t v;
		int32_t d;
		support_pt(int32_t u,int32_t v,int32_t d):u(u),v(v),d(d){}
	};

	struct triangle {
		int32_t c1,c2,c3;
		float   t1a,t1b,t1c;
		float   t2a,t2b,t2c;
		triangle(int32_t c1,int32_t c2,int32_t c3):c1(c1),c2(c2),c3(c3){}
	};

	inline uint32_t getAddressOffsetImage (const int32_t& u,const int32_t& v,const int32_t& width) {
		return v*width+u;
	}

	inline uint32_t getAddressOffsetGrid (const int32_t& x,const int32_t& y,const int32_t& d,const int32_t& width,const int32_t& disp_num) {
		return (y*width+x)*disp_num+d;
	}

	// support point functions
	void removeInconsistentSupportPoints (int16_t* D_can,int32_t D_can_width,int32_t D_can_height);
	void removeRedundantSupportPoints (int16_t* D_can,int32_t D_can_width,int32_t D_can_height,
		int32_t redun_max_dist, int32_t redun_threshold, bool vertical);
	void addCornerSupportPoints (std::vector<support_pt> &p_support);
	std::vector<support_pt> computeSupportMatches (Mat initDisparity, Mat left);

	// triangulation & grid
	std::vector<triangle> computeDelaunayTriangulation (std::vector<support_pt> p_support);
	void computeDisparityPlanes (std::vector<support_pt> p_support,std::vector<triangle> &tri);
	void createGrid (std::vector<support_pt> p_support,int32_t* disparity_grid,int32_t* grid_dims);

	// matching
	inline void findMatch (int32_t &u,int32_t &v,float &plane_a,float &plane_b,float &plane_c,
		float* D, Mat initDisparity, Mat maxMncc, Mat leftDisparity);
	void computeDisparity (std::vector<support_pt> p_support,std::vector<triangle> tri, float* D, Mat initDisparity, Mat maxMncc, Mat leftDisparity);

	// postprocessing
	void leftRightConsistencyCheck (float* D1,float* D2);   // L/R consistency check
	void removeSmallSegments (float* D);
	void gapInterpolation (float* D);

	// parameter set
	parameters param;

	// memory aligned input images + dimensions
	int32_t width,height;

};

#endif
