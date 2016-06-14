#ifndef GUIDEDFILTER_H
#define GUIDEDFILTER_H

#include "opencv2/opencv.hpp"
using namespace cv;

typedef unsigned short u16;
typedef short s16;
typedef unsigned char u8;
typedef char s8;
typedef unsigned int u32;
typedef int s32;
typedef double s64;
typedef float f32;

//#define ARM_NEON

#define TYPENxN 0
#define TYPE1xN 1
#define TYPENx1 2

class guideFilter
{
public:
	  guideFilter() : Eps(0.0001f  ), Tp(TYPENxN), R_s(3), R_b(30) {}
	 ~guideFilter() {}

public:
	//test
	void gfAccMultiple(Mat& Img_s, Mat& Img_b, Mat& iNCC, Mat& oNCC);
	void gfAccMultiplef(Mat& Img_s, Mat& Img_b, Mat& iNCC, Mat& oNCC);
	void gfAccMultiple(const u32* pI, const u32* pG, u32* pO1, u32* pO2, u32* pO3, u16 n_rows, u16 n_cols);

	void setParams(f32 Eps_, u32 Tp_, u16 R_s_, u16 R_b_);
	
private:
	void Interpolation(const u32* pI, u16 i_rows, u16 i_cols, u32* pO, u16 o_rows, u16 o_cols);
	//test
	void gfBoxFilterEx(const u8* pI, const f32* tbl, const u8* ind, const u8* indX, const u8* indY, f32* pO1, f32* pO2, f32* pO3, u16 n_rows, u16 n_cols);
	void gfBoxFilterEx(const f32* pI, const f32* tbl, const u8* ind, const u8* indX, const u8* indY, f32* pO1, f32* pO2, f32* pO3, u16 n_rows, u16 n_cols);
	void gfBoxFilterEx(const u32* pI, const f32* tbl, const u8* ind, const u8* indX, const u8* indY, f32* pO1, f32* pO2, f32* pO3, u16 n_rows, u16 n_cols);
	void gfBoxFilterEx(const f32* pI, const f32* tbl, const u8* ind, f32* pO, u16 n_rows, u16 n_cols, u16 R);
	void gfScanLineXEx(const f32* pI, f32* pO, u16 n_rows, u16 n_cols, u16 R);
	void gfScanLineXEx(const f32* pI, const f32* tbl, const u8* ind, f32* pO, u16 n_rows, u16 n_cols, u16 R);
	void gfScanLineYEx(const f32* pI, const f32* tbl, const u8* ind, f32* pO, u16 n_rows, u16 n_cols, u16 R);
	void gfBoxFilter(const u8* pI, u8* pO, u16 n_rows, u16 n_cols, u16 R);
	void gfScanLineX(const u8* pI, u8* pO, u16 n_rows, u16 n_cols, u16 R);
	void gfScanLineY(const u8* pI, u8* pO, u16 n_rows, u16 n_cols, u16 R);
	void scanLineXEx(const u32* pI, const f32* tbl, const u8* indX, u32* pO1, f32* pO2, u16 n_rows, u16 n_cols);
	void scanLineYEx(const u32* pI1, const u32* pI2, const f32* tbl, const u8* ind, const u8* indY, f32* pO1, f32* pO2, u16 n_rows, u16 n_cols);
	void scanLineXEx(const f32* pI, const f32* tbl, const u8* indX, f32* pO1, f32* pO2, u16 n_rows, u16 n_cols);
	void scanLineYEx(const f32* pI1, const f32* pI2, const f32* tbl, const u8* ind, const u8* indY, f32* pO1, f32* pO2, u16 n_rows, u16 n_cols);
	void divid(const f32* covIP, const f32* varI, f32 *pO, u16 n_rows, u16 n_cols);
	void multiply(const f32* pI1, const f32* pI2, const f32* pI3, f32* pO, u16 n_rows, u16 n_cols);
	void multiply(const u8* pI1, const u8* pI2, const u8* pI3, u8* pO, u16 n_rows, u16 n_cols);
	void multiply(const f32* p01, const f32* p02, f32* pO, u16 n_rows, u16 n_cols);
	//test
	void multiply(const u8* p01, const u8* p02, u32* pO, u16 n_rows, u16 n_cols);
	void multiply(const u32* p01, const u32* p02, u32* pO, u16 n_rows, u16 n_cols);
	void msltiply(const f32* A, const f32* B, const f32* C, f32* D, u16 n_rows, u16 n_cols);
	//test
	void maltiply(const f32* A, const u8* B, const f32* C, u8* D, u16 n_rows, u16 n_cols);
	void maltiply(const f32* A, const u32* B, const f32* C, u32* D, u16 n_rows, u16 n_cols);
	void maltiply(const f32* A, const f32* B, const f32* C, f32* D, u16 n_rows, u16 n_cols);
	void gfSumAreaTbl(const f32* pI, f32* pO1, f32* pO2, u16 n_rows, u16 n_cols);
	void gfSumAreaTbl(const u32* pI, u32* pO1, u32* pO2, u16 n_rows, u16 n_cols);
	void gfSumAreaTbl(const f32* pI, f32* pO, u16 n_rows, u16 n_cols);
	void gfSumAreaTbl(const u32* pI, u32* pO, u16 n_rows, u16 n_cols);
	//test
	void gfSumAreaTbl(const u8* pI, u32* pO1, u32* pO2, u16 n_rows, u16 n_cols);

private:
	f32 Eps;
	u32 Tp;
	u16 R_s;
	u16 R_b;
};

#endif
