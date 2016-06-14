#include "guidedFilter.h"
#include <cstdlib>
#include <cstring>

#ifdef ARM_NEON
#include <arm_neon.h>
#endif

#undef  LOG_TAG
#define LOG_TAG "dcs_depth"

#define RANGE 255
#define IMAX  10000
#define IMIN -10000
#define ITEM 62
#define CLAMP(x) (x < 0 ? 0 : (x > RANGE ? RANGE : x))

void guideFilter::gfAccMultiple(Mat& Img_s, Mat& Img_b, Mat& iNCC, Mat& oNCC)
{
	f32 *meanIFY, *meanPFY, *meanIPFY, *meanIIFY, *covIPY, *varIY, *AY, *BY, *meanAFY, *meanBFY;
	f32 *meanIFX, *meanPFX, *meanIPFX, *meanIIFX, *covIPX, *varIX, *AX, *BX, *meanAFX, *meanBFX;
	f32 *meanIF, *meanPF, *meanIPF, *meanIIF, *covIP, *varI, *A, *B, *meanAF, *meanBF;
	u32 *T1;
	u8 *N, *NX, *NY, *pU;
	
	Mat F10(Img_s.size(), CV_32FC1);
	Mat F11(Img_s.size(), CV_32FC1);
	Mat F0(Img_s.size(), CV_32FC1);
	Mat F1(Img_s.size(), CV_32FC1);
	Mat F2(Img_s.size(), CV_32FC1);
	Mat F3(Img_s.size(), CV_32FC1);
	Mat F4(Img_s.size(), CV_32FC1);
	Mat F5(Img_s.size(), CV_32FC1);
	Mat F6(Img_s.size(), CV_32FC1);
	Mat F7(Img_s.size(), CV_32FC1);
	Mat F8(Img_s.size(), CV_32FC1);
	Mat F9(Img_s.size(), CV_32FC1);
	

	Mat iNCC_s;
	resize(iNCC, iNCC_s, Img_s.size());



	const u8 *pI = Img_s.data, *pG = iNCC_s.data;

	u16 n_rows = Img_s.rows;
	u16 n_cols = Img_s.cols;

	f32 lookUp[ITEM] = { 0 };

	u32 i, size = n_rows * n_cols;
	u16 iMAX = 0, iMIN = IMAX;
	NX = (u8*)malloc(size * sizeof(u8));
	NY = (u8*)malloc(size * sizeof(u8));
	N = (u8*)malloc(size * sizeof(u8));
	T1 = (u32*)malloc(size * sizeof(u32));

	pU = N;

#ifdef ARM_NEON
	uint8x16_t v16a = vdupq_n_u8(1);
	for (i = 0; i < size - 16; i += 16)
	{
		vst1q_u8(pU, v16a);
		pU += 16;
	}
	for (; i < size; ++i) *pU++ = 1;
#else
	for (i = 0; i < size; ++i) *pU++ = 1;
#endif

	gfScanLineX(N, NX, n_rows, n_cols, R_b);
	gfScanLineY(N, NY, n_rows, n_cols, R_b);
	gfBoxFilter(N, N, n_rows, n_cols, R_s);
	
	lookUp[0] = 1;
	for (i = 1; i < ITEM; ++i) lookUp[i] = 1.f / i;

	meanIFY = (f32*)F2.data;
	meanIFX = (f32*)F1.data;
	meanIF = (f32*)F0.data;
	gfBoxFilterEx(pI, lookUp, N, NX, NY, meanIF, meanIFX, meanIFY, n_rows, n_cols);

	meanPFY = (f32*)F5.data;
	meanPFX = (f32*)F4.data;
	meanPF = (f32*)F3.data;
	gfBoxFilterEx(pG, lookUp, N, NX, NY, meanPF, meanPFX, meanPFY, n_rows, n_cols);

	multiply(pI, pG, T1, n_rows, n_cols);

	meanIPFY = (f32*)F8.data;
	meanIPFX = (f32*)F7.data;
	meanIPF = (f32*)F6.data;
	gfBoxFilterEx(T1, lookUp, N, NX, NY, meanIPF, meanIPFX, meanIPFY, n_rows, n_cols);

	covIPY = meanIPFY;
	covIPX = meanIPFX;
	covIP = meanIPF;
	msltiply(meanIPFY, meanIFY, meanPFY, meanIPFY, n_rows, n_cols);
	msltiply(meanIPFX, meanIFX, meanPFX, meanIPFX, n_rows, n_cols);
	msltiply(meanIPF, meanIF, meanPF, meanIPF, n_rows, n_cols);

	multiply(pI, pI, T1, n_rows, n_cols);

	meanIIFY = (f32*)F11.data;
	meanIIFX = (f32*)F10.data;
	meanIIF = (f32*)F9.data;
	gfBoxFilterEx(T1, lookUp, N, NX, NY, meanIIF, meanIIFX, meanIIFY, n_rows, n_cols);

	varIY = meanIIFY;
	varIX = meanIIFX;
	varI = meanIIF;
	msltiply(meanIIFY, meanIFY, meanIFY, meanIIFY, n_rows, n_cols);
	msltiply(meanIIFX, meanIFX, meanIFX, meanIIFX, n_rows, n_cols);
	msltiply(meanIIF, meanIF, meanIF, meanIIF, n_rows, n_cols);

	AY = covIPY;
	AX = covIPX;
	A = covIP;
	divid(covIPY, varIY, AY, n_rows, n_cols);
	divid(covIPX, varIX, AX, n_rows, n_cols);
	divid(covIP, varI, A, n_rows, n_cols);

	BY = varIY;
	BX = varIX;
	B = varI;
	msltiply(meanPFY, AY, meanIFY, BY, n_rows, n_cols);
	msltiply(meanPFX, AX, meanIFX, BX, n_rows, n_cols);
	msltiply(meanPF, A, meanIF, B, n_rows, n_cols);

	meanAFY = (f32*)F2.data;
	meanAFX = (f32*)F1.data;
	meanAF = (f32*)F0.data;
	gfScanLineYEx(AY, lookUp, NY, meanAFY, n_rows, n_cols, R_b);
	gfScanLineXEx(AX, lookUp, NX, meanAFX, n_rows, n_cols, R_b);
	gfBoxFilterEx(A, lookUp, N, meanAF, n_rows, n_cols, R_s);

	meanBFY = (f32*)F5.data;
	meanBFX = (f32*)F4.data;
	meanBF = (f32*)F3.data;
	gfScanLineYEx(BY, lookUp, NY, meanBFY, n_rows, n_cols, R_b);
	gfScanLineXEx(BX, lookUp, NX, meanBFX, n_rows, n_cols, R_b);
	gfBoxFilterEx(B, lookUp, N, meanBF, n_rows, n_cols, R_s);
	
	Mat iNCC_b1(Img_b.size(), CV_8UC1);
	Mat iNCC_b2(Img_b.size(), CV_8UC1);
	Mat iNCC_b3(Img_b.size(), CV_8UC1);

	n_rows = Img_b.rows;
	n_cols = Img_b.cols;



	Mat mAY, mBY;
	resize(F2, mAY, Img_b.size());
	resize(F5, mBY, Img_b.size());
	maltiply((f32*)mBY.data, Img_b.data, (f32*)mAY.data, iNCC_b1.data, n_rows, n_cols);
	mAY.release();
	mBY.release();

	Mat mAX, mBX;
	resize(F1, mAX, Img_b.size());
	resize(F4, mBX, Img_b.size());
	maltiply((f32*)mBX.data, Img_b.data, (f32*)mAX.data, iNCC_b2.data, n_rows, n_cols);
	mAX.release();
	mBX.release();

	Mat mA, mB;
	resize(F0, mA, Img_b.size());
	resize(F3, mB, Img_b.size());
	maltiply((f32*)mB.data, Img_b.data, (f32*)mA.data, iNCC_b3.data, n_rows, n_cols);
	mA.release();
	mB.release();
	
	multiply(iNCC_b1.data, iNCC_b2.data, iNCC_b3.data, oNCC.data, iNCC.rows, iNCC.cols);
     

	iNCC_b3.copyTo(oNCC);


	iNCC_b1.release();
	iNCC_b2.release();
	iNCC_b3.release();

	free(NX);
	free(NY);
	free(N);
	free(T1);
	F10.release();
	F11.release();
	F0.release();
	F1.release();
	F2.release();
	F3.release();
	F4.release();
	F5.release(); 
	F6.release();
	F7.release();
	F8.release();
	F9.release();
}

void guideFilter::gfAccMultiplef(Mat& Img_s, Mat& Img_b, Mat& iNCC, Mat& oNCC)
{
	f32 *meanIFY, *meanPFY, *meanIPFY, *meanIIFY, *covIPY, *varIY, *AY, *BY, *meanAFY, *meanBFY;
	f32 *meanIFX, *meanPFX, *meanIPFX, *meanIIFX, *covIPX, *varIX, *AX, *BX, *meanAFX, *meanBFX;
	f32 *meanIF, *meanPF, *meanIPF, *meanIIF, *covIP, *varI, *A, *B, *meanAF, *meanBF;
	f32 *T1;
	u8 *N, *NX, *NY, *pU;

	Mat F10(Img_s.size(), CV_32FC1);
	Mat F11(Img_s.size(), CV_32FC1);
	Mat F0(Img_s.size(), CV_32FC1);
	Mat F1(Img_s.size(), CV_32FC1);
	Mat F2(Img_s.size(), CV_32FC1);
	Mat F3(Img_s.size(), CV_32FC1);
	Mat F4(Img_s.size(), CV_32FC1);
	Mat F5(Img_s.size(), CV_32FC1);
	Mat F6(Img_s.size(), CV_32FC1);
	Mat F7(Img_s.size(), CV_32FC1);
	Mat F8(Img_s.size(), CV_32FC1);
	Mat F9(Img_s.size(), CV_32FC1);

	Mat iNCC_s;
	resize(iNCC, iNCC_s, Img_s.size());

	const f32 *pI = (f32*)Img_s.data, *pG = (f32*)iNCC_s.data;

	u16 n_rows = Img_s.rows;
	u16 n_cols = Img_s.cols;

	f32 lookUp[ITEM] = { 0 };

	u32 i, size = n_rows * n_cols;
	u16 iMAX = 0, iMIN = IMAX;

	NX = (u8*)malloc(size * sizeof(u8));
	NY = (u8*)malloc(size * sizeof(u8));
	N = (u8*)malloc(size * sizeof(u8));
	T1 = (f32*)malloc(size * sizeof(f32));

	pU = N;

#ifdef ARM_NEON
	uint8x16_t v16a = vdupq_n_u8(1);
	for (i = 0; i < size - 16; i += 16)
	{
		vst1q_u8(pU, v16a);
		pU += 16;
	}
	for (; i < size; ++i) *pU++ = 1;
#else
	for (i = 0; i < size; ++i) *pU++ = 1;
#endif
	gfScanLineX(N, NX, n_rows, n_cols, R_b);
	gfScanLineY(N, NY, n_rows, n_cols, R_b);
	gfBoxFilter(N, N, n_rows, n_cols, R_s);

	lookUp[0] = 1;
	for (i = 1; i <  ITEM; ++i) lookUp[i] = 1.f / i;

	meanIFY = (f32*)F2.data;
	meanIFX = (f32*)F1.data;
	meanIF = (f32*)F0.data;
	gfBoxFilterEx(pI, lookUp, N, NX, NY, meanIF, meanIFX, meanIFY, n_rows, n_cols);

	meanPFY = (f32*)F5.data;
	meanPFX = (f32*)F4.data;
	meanPF = (f32*)F3.data;
	gfBoxFilterEx(pG, lookUp, N, NX, NY, meanPF, meanPFX, meanPFY, n_rows, n_cols);

	multiply(pI, pG, T1, n_rows, n_cols);

	meanIPFY = (f32*)F8.data;
	meanIPFX = (f32*)F7.data;
	meanIPF = (f32*)F6.data;
	gfBoxFilterEx(T1, lookUp, N, NX, NY, meanIPF, meanIPFX, meanIPFY, n_rows, n_cols);

	covIPY = meanIPFY;
	covIPX = meanIPFX;
	covIP = meanIPF;
	msltiply(meanIPFY, meanIFY, meanPFY, meanIPFY, n_rows, n_cols);
	msltiply(meanIPFX, meanIFX, meanPFX, meanIPFX, n_rows, n_cols);
	msltiply(meanIPF, meanIF, meanPF, meanIPF, n_rows, n_cols);

	multiply(pI, pI, T1, n_rows, n_cols);

	meanIIFY = (f32*)F11.data;
	meanIIFX = (f32*)F10.data;
	meanIIF = (f32*)F9.data;
	gfBoxFilterEx(T1, lookUp, N, NX, NY, meanIIF, meanIIFX, meanIIFY, n_rows, n_cols);

	varIY = meanIIFY;
	varIX = meanIIFX;
	varI = meanIIF;
	msltiply(meanIIFY, meanIFY, meanIFY, meanIIFY, n_rows, n_cols);
	msltiply(meanIIFX, meanIFX, meanIFX, meanIIFX, n_rows, n_cols);
	msltiply(meanIIF, meanIF, meanIF, meanIIF, n_rows, n_cols);

	AY = covIPY;
	AX = covIPX;
	A = covIP;
	divid(covIPY, varIY, AY, n_rows, n_cols);
	divid(covIPX, varIX, AX, n_rows, n_cols);
	divid(covIP, varI, A, n_rows, n_cols);

	BY = varIY;
	BX = varIX;
	B = varI;
	msltiply(meanPFY, AY, meanIFY, BY, n_rows, n_cols);
	msltiply(meanPFX, AX, meanIFX, BX, n_rows, n_cols);
	msltiply(meanPF, A, meanIF, B, n_rows, n_cols);

	meanAFY = (f32*)F2.data;
	meanAFX = (f32*)F1.data;
	meanAF = (f32*)F0.data;
	gfScanLineYEx(AY, lookUp, NY, meanAFY, n_rows, n_cols, R_b);
	gfScanLineXEx(AX, lookUp, NX, meanAFX, n_rows, n_cols, R_b);
	gfBoxFilterEx(A, lookUp, N, meanAF, n_rows, n_cols, R_s);

	meanBFY = (f32*)F5.data;
	meanBFX = (f32*)F4.data;
	meanBF = (f32*)F3.data;
	gfScanLineYEx(BY, lookUp, NY, meanBFY, n_rows, n_cols, R_b);
	gfScanLineXEx(BX, lookUp, NX, meanBFX, n_rows, n_cols, R_b);
	gfBoxFilterEx(B, lookUp, N, meanBF, n_rows, n_cols, R_s);

	n_rows = Img_b.rows;
	n_cols = Img_b.cols;

	Mat iNCC_b1(Img_b.size(), CV_32FC1);
	Mat iNCC_b2(Img_b.size(), CV_32FC1);
	Mat iNCC_b3(Img_b.size(), CV_32FC1);

	Mat mAY, mBY;
	resize(F2, mAY, Img_b.size());
	resize(F5, mBY, Img_b.size());
	maltiply((f32*)mBY.data, (f32*)Img_b.data, (f32*)mAY.data, (f32*)iNCC_b1.data, n_rows, n_cols);
	mAY.release();
	mBY.release();

	Mat mAX, mBX;
	resize(F1, mAX, Img_b.size());
	resize(F4, mBX, Img_b.size());
	maltiply((f32*)mBX.data, (f32*)Img_b.data, (f32*)mAX.data, (f32*)iNCC_b2.data, n_rows, n_cols);
	mAX.release();
	mBX.release();

	Mat mA, mB;
	resize(F0, mA, Img_b.size());
	resize(F3, mB, Img_b.size());
	maltiply((f32*)mB.data, (f32*)Img_b.data, (f32*)mA.data, (f32*)iNCC_b3.data, n_rows, n_cols);
	mA.release();
	mB.release();

	multiply((f32*)iNCC_b1.data, (f32*)iNCC_b2.data, (f32*)iNCC_b3.data, (f32*)oNCC.data, iNCC.rows, iNCC.cols);
	
	//iNCC_b3.copyTo(oNCC);

	iNCC_b1.release();
	iNCC_b2.release();
	iNCC_b3.release();

	free(NX);
	free(NY);
	free(N);
	free(T1);
	F10.release();
	F11.release();
	F0.release();
	F1.release();
	F2.release();
	F3.release();
	F4.release();
	F5.release();
	F6.release();
	F7.release();
	F8.release();
	F9.release();
}

void guideFilter::gfAccMultiple(const u32* pI, const u32* pG, u32* pO1, u32* pO2, u32* pO3, u16 n_rows, u16 n_cols)
{
	f32 *meanIFY, *meanPFY, *meanIPFY, *meanIIFY, *covIPY, *varIY, *AY, *BY, *meanAFY, *meanBFY;
	f32 *meanIFX, *meanPFX, *meanIPFX, *meanIIFX, *covIPX, *varIX, *AX, *BX, *meanAFX, *meanBFX;
	f32 *meanIF, *meanPF, *meanIPF, *meanIIF, *covIP, *varI, *A, *B, *meanAF, *meanBF;
	f32 *F0, *F1, *F2, *F3, *F4, *F5, *F6, *F7, *F8, *F9, *F10, *F11;
	u32 *T1;
	u8 *N, *NX, *NY, *pU;

	f32 lookUp[ITEM] = { 0 };

	u32 i, size = n_rows * n_cols;
	u16 iMAX = 0, iMIN = IMAX;

	NX = (u8*)malloc(size * sizeof(u8));
	NY = (u8*)malloc(size * sizeof(u8));
	N = (u8*)malloc(size * sizeof(u8));
	T1 = (u32*)malloc(size * sizeof(u32));
	F0 = (f32*)malloc(size * sizeof(f32));
	F1 = (f32*)malloc(size * sizeof(f32));
	F2 = (f32*)malloc(size * sizeof(f32));
	F3 = (f32*)malloc(size * sizeof(f32));
	F4 = (f32*)malloc(size * sizeof(f32));
	F5 = (f32*)malloc(size * sizeof(f32));
	F6 = (f32*)malloc(size * sizeof(f32));
	F7 = (f32*)malloc(size * sizeof(f32));
	F8 = (f32*)malloc(size * sizeof(f32));
	F9 = (f32*)malloc(size * sizeof(f32));
	F10 = (f32*)malloc(size * sizeof(f32));
	F11 = (f32*)malloc(size * sizeof(f32));

	pU = N;

#ifdef ARM_NEON
	uint8x16_t v16a = vdupq_n_u8(1);
	for (i = 0; i < size - 16; i += 16)
	{
		vst1q_u8(pU, v16a);
		pU += 16;
	}
	for (; i < size; ++i) *pU++ = 1;
#else
	for (i = 0; i < size; ++i) *pU++ = 1;
#endif
	gfScanLineX(N, NX, n_rows, n_cols, R_b);
	gfScanLineY(N, NY, n_rows, n_cols, R_b);
	gfBoxFilter(N, N, n_rows, n_cols, R_s);
	
	lookUp[0] = 1;
	for (i = 1; i <= iMAX; ++i) lookUp[i] = 1.f / i;

	meanIFY = F2;
	meanIFX = F1;
	meanIF = F0;
	gfBoxFilterEx(pI, lookUp, N, NX, NY, meanIF, meanIFX, meanIFY, n_rows, n_cols);

	meanPFY = F5;
	meanPFX = F4;
	meanPF = F3;
	gfBoxFilterEx(pG, lookUp, N, NX, NY, meanPF, meanPFX, meanPFY, n_rows, n_cols);

	multiply(pI, pG, T1, n_rows, n_cols);

	meanIPFY = F8;
	meanIPFX = F7;
	meanIPF = F6;
	gfBoxFilterEx(T1, lookUp, N, NX, NY, meanIPF, meanIPFX, meanIPFY, n_rows, n_cols);

	covIPY = meanIPFY;
	covIPX = meanIPFX;
	covIP = meanIPF;
	msltiply(meanIPFY, meanIFY, meanPFY, meanIPFY, n_rows, n_cols);
	msltiply(meanIPFX, meanIFX, meanPFX, meanIPFX, n_rows, n_cols);
	msltiply(meanIPF, meanIF, meanPF, meanIPF, n_rows, n_cols);

	multiply(pI, pI, T1, n_rows, n_cols);

	meanIIFY = F11;
	meanIIFX = F10;
	meanIIF = F9;
	gfBoxFilterEx(T1, lookUp, N, NX, NY, meanIIF, meanIIFX, meanIIFY, n_rows, n_cols);

	varIY = meanIIFY;
	varIX = meanIIFX;
	varI = meanIIF;
	msltiply(meanIIFY, meanIFY, meanIFY, meanIIFY, n_rows, n_cols);
	msltiply(meanIIFX, meanIFX, meanIFX, meanIIFX, n_rows, n_cols);
	msltiply(meanIIF, meanIF, meanIF, meanIIF, n_rows, n_cols);

	AY = covIPY;
	AX = covIPX;
	A = covIP;
	divid(covIPY, varIY, AY, n_rows, n_cols);
	divid(covIPX, varIX, AX, n_rows, n_cols);
	divid(covIP, varI, A, n_rows, n_cols);

	BY = varIY;
	BX = varIX;
	B = varI;
	msltiply(meanPFY, AY, meanIFY, BY, n_rows, n_cols);
	msltiply(meanPFX, AX, meanIFX, BX, n_rows, n_cols);
	msltiply(meanPF, A, meanIF, B, n_rows, n_cols);

	meanAFY = F0;
	meanAFX = F1;
	meanAF = F2;
	gfScanLineYEx(AY, lookUp, NY, meanAFY, n_rows, n_cols, R_b);
	gfScanLineXEx(AX, lookUp, NX, meanAFX, n_rows, n_cols, R_b);
	gfBoxFilterEx(A, lookUp, N, meanAF, n_rows, n_cols, R_s);

	meanBFY = F5;
	meanBFX = F4;
	meanBF = F3;
	gfScanLineYEx(BY, lookUp, NY, meanBFY, n_rows, n_cols, R_b);
	gfScanLineXEx(BX, lookUp, NX, meanBFX, n_rows, n_cols, R_b);
	gfBoxFilterEx(B, lookUp, N, meanBF, n_rows, n_cols, R_s);

	maltiply(meanBFY, pI, meanAFY, pO1, n_rows, n_cols);
	maltiply(meanBFX, pI, meanAFX, pO2, n_rows, n_cols);
	maltiply(meanBF, pI, meanAF, pO3, n_rows, n_cols);

	free(NX);
	free(NY);
	free(N);
	free(T1);
	free(F0);
	free(F1);
	free(F2);
	free(F3);
	free(F4);
	free(F5);
	free(F6);
	free(F7);
	free(F8);
	free(F9);
	free(F10);
	free(F11);
}

void guideFilter::setParams(f32 Eps_, u32 Tp_, u16 R_s_, u16 R_b_)
{
	Eps = Eps_;
	Tp = Tp_;
	R_s = R_s_;
	R_b = R_b_;
}

void guideFilter::Interpolation(const u32* pI, u16 i_rows, u16 i_cols, u32* pO, u16 o_rows, u16 o_cols)
{

}

void guideFilter::gfSumAreaTbl(const f32* pI, f32* pO1, f32* pO2, u16 n_rows, u16 n_cols)
{
	f32 *pSum, *pDataX, *pDataY, *plhs01 = pO1, *plhs02 = pO2;
	const f32 *pDataU, *plhs = pI;

	f32* tbl = (f32*)malloc(n_cols * sizeof(f32));
	memset(tbl, 0, n_cols * sizeof(f32));

	u16 x, y;

	for (y = 0; y < n_rows; ++y)
	{
		pDataX = plhs01;
		pDataY = plhs02;
		pDataU = plhs;
		pSum = tbl;

		f32 sum = 0, tmp;
		for (x = 0; x < n_cols; ++x)
		{
			tmp = *pDataU++;
			sum += tmp;
			*pSum += tmp;
			*pDataY++ = *pSum++;
			*pDataX++ = sum;
		}

		plhs01 += n_cols;
		plhs02 += n_cols;
		plhs += n_cols;
	}

	free(tbl);
}

void guideFilter::gfSumAreaTbl(const u8* pI, u32* pO1, u32* pO2, u16 n_rows, u16 n_cols)
{
	u32 *pSum, *pDataX, *pDataY, *plhs01 = pO1, *plhs02 = pO2;
	const u8 *pDataU, *plhs = pI;

	u32* tbl = (u32*)malloc(n_cols * sizeof(u32));
	memset(tbl, 0, n_cols * sizeof(u32));

	u16 x, y;

	for (y = 0; y < n_rows; ++y)
	{
		pDataX = plhs01;
		pDataY = plhs02;
		pDataU = plhs;
		pSum = tbl;

		u32 sum = 0, tmp;
		for (x = 0; x < n_cols; ++x)
		{
			tmp = *pDataU++;
			sum += tmp;
			*pSum += tmp;
			*pDataY++ = *pSum++;
			*pDataX++ = sum;
		}

		plhs01 += n_cols;
		plhs02 += n_cols;
		plhs += n_cols;
	}

	free(tbl);
}

void guideFilter::gfSumAreaTbl(const f32* pI, f32* pO, u16 n_rows, u16 n_cols)
{
	f32 *pSum, *pDataY, *plhs02 = pO;
	const f32 *pDataU, *plhs01 = pI;
	f32* tbl = (f32*)malloc(n_cols * sizeof(f32));
	memset(tbl, 0, n_cols * sizeof(f32));
	u16 x, y;
	for (y = 0; y < n_rows; ++y)
	{
		pDataY = plhs02;
		pDataU = plhs01;
		pSum = tbl;
		for (x = 0; x < n_cols; ++x)
		{
			*pSum += *pDataU++;
			*pDataY++ = *pSum++;
		}

		plhs01 += n_cols;
		plhs02 += n_cols;
	}
	free(tbl);
}

void guideFilter::gfSumAreaTbl(const u32* pI, u32* pO, u16 n_rows, u16 n_cols)
{
	u32 *pSum, *pDataY, *plhs02 = pO;
	const u32 *pDataU, *plhs01 = pI;
	u32* tbl = (u32*)malloc(n_cols * sizeof(u32));
	memset(tbl, 0, n_cols * sizeof(u32));
	u16 x, y;
	for (y = 0; y < n_rows; ++y)
	{
		pDataY = plhs02;
		pDataU = plhs01;
		pSum = tbl;
		for (x = 0; x < n_cols; ++x)
		{
			*pSum += *pDataU++;
			*pDataY++ = *pSum++;
		}

		plhs01 += n_cols;
		plhs02 += n_cols;
	}
	free(tbl);
}

void guideFilter::gfSumAreaTbl(const u32* pI, u32* pO1, u32* pO2, u16 n_rows, u16 n_cols)
{
	u32 *pSum, *pDataX, *pDataY, *plhs01 = pO1, *plhs02 = pO2;
	const u32 *pDataU, *plhs = pI;

	u32* tbl = (u32*)malloc(n_cols * sizeof(u32));
	memset(tbl, 0, n_cols * sizeof(u32));

	u16 x, y;

	for (y = 0; y < n_rows; ++y)
	{
		pDataX = plhs01;
		pDataY = plhs02;
		pDataU = plhs;
		pSum = tbl;

		u32 sum = 0, tmp;
		for (x = 0; x < n_cols; ++x)
		{
			tmp = *pDataU++;
			sum += tmp;
			*pSum += tmp;
			*pDataY++ = *pSum++;
			*pDataX++ = sum;
		}

		plhs01 += n_cols;
		plhs02 += n_cols;
		plhs += n_cols;
	}

	free(tbl);
}

void guideFilter::gfBoxFilterEx(const u8* pI, const f32* mtbl, const u8* ind, const u8* indX, const u8* indY, f32* pO1, f32* pO2, f32* pO3, u16 n_rows, u16 n_cols)
{
	u32 size = n_rows * n_cols;
	u32* pX = (u32*)malloc(size * sizeof(u32));
	u32* pY = (u32*)malloc(size * sizeof(u32));
	u32* pU = (u32*)malloc(size * sizeof(u32));
	gfSumAreaTbl(pI, pX, pY, n_rows, n_cols);
	scanLineXEx(pX, mtbl, indX, pU, pO2, n_rows, n_cols);
	gfSumAreaTbl(pU, pX, n_rows, n_cols);
	scanLineYEx(pY, pX, mtbl, ind, indY, pO1, pO3, n_rows, n_cols);
	free(pX);
	free(pY);
	free(pU);
}

void guideFilter::gfBoxFilterEx(const f32* pI, const f32* mtbl, const u8* ind, const u8* indX, const u8* indY, f32* pO1, f32* pO2, f32* pO3, u16 n_rows, u16 n_cols)
{
	u32 size = n_rows * n_cols * sizeof(f32);
	f32* pX = (f32*)malloc(size);
	f32* pY = (f32*)malloc(size);
	f32* pF = (f32*)malloc(size);
	gfSumAreaTbl(pI, pX, pY, n_rows, n_cols);
	scanLineXEx(pX, mtbl, indX, pF, pO2, n_rows, n_cols);
	gfSumAreaTbl(pF, pX, n_rows, n_cols);
	scanLineYEx(pY, pX, mtbl, ind, indY, pO1, pO3, n_rows, n_cols);
	free(pX);
	free(pY);
	free(pF);
}

void guideFilter::gfBoxFilterEx(const u32* pI, const f32* mtbl, const u8* ind, const u8* indX, const u8* indY, f32* pO1, f32* pO2, f32* pO3, u16 n_rows, u16 n_cols)
{
	u32 size = n_rows * n_cols;
	u32* pX = (u32*)malloc(size * sizeof(u32));
	u32* pY = (u32*)malloc(size * sizeof(u32));
	u32* pU = (u32*)malloc(size * sizeof(u32));
	gfSumAreaTbl(pI, pX, pY, n_rows, n_cols);
	scanLineXEx(pX, mtbl, indX, pU, pO2, n_rows, n_cols);
	gfSumAreaTbl(pU, pX, n_rows, n_cols);
	scanLineYEx(pY, pX, mtbl, ind, indY, pO1, pO3, n_rows, n_cols);
	free(pX);
	free(pY);
	free(pU);
}

void guideFilter::gfBoxFilterEx(const f32* pI, const f32* tbl, const u8* ind, f32* pO, u16 n_rows, u16 n_cols, u16 R)
{
	u32 size = n_rows * n_cols * sizeof(f32);
	f32* pT = (f32*)malloc(size);
	gfScanLineXEx(pI, pT, n_rows, n_cols, R);
	gfScanLineYEx(pT, tbl, ind, pO, n_rows, n_cols, R);
	free(pT);
}

void guideFilter::gfScanLineXEx(const f32* pI, f32* pO, u16 n_rows, u16 n_cols, u16 R)
{
	const f32 *pL, *pR;
	f32 *pF, sum;
	u16 x, y;

	const f32 *plhs = pI;
	f32* prhs = pO;

	for (y = 0; y < n_rows; ++y)
	{
		pR = pL = plhs;
		pF = prhs;

		sum = 0;
		for (x = 0; x < R; ++x)
		{
			sum += *pR++;
		}

		for (x = 0; x <= R; ++x)
		{
			sum += *pR++;
			*pF++ = sum;
		}

		for (; x < n_cols - R; ++x)
		{
			sum += *pR++;
			sum -= *pL++;
			*pF++ = sum;
		}

		for (; x < n_cols; ++x)
		{
			sum -= *pL++;
			*pF++ = sum;
		}

		plhs += n_cols;
		prhs += n_cols;
	}
}

void guideFilter::gfScanLineXEx(const f32* pI, const f32* mtbl, const u8* ind, f32* pO, u16 n_rows, u16 n_cols, u16 R)
{
	const f32 *pL, *pR;
	f32 *pF, sum;
	u16 x, y;

	const u8* pInd, *Index = ind;
	const f32 *plhs = pI;
	f32* prhs = pO;

	for (y = 0; y < n_rows; ++y)
	{
		pR = pL = plhs;
		pF = prhs;
		pInd = Index;

		sum = 0;
		for (x = 0; x < R; ++x)
		{
			sum += *pR++;
		}

		for (x = 0; x <= R; ++x)
		{
			sum += *pR++;
			*pF++ = sum * mtbl[*pInd++];
		}

		for (; x < n_cols - R; ++x)
		{
			sum += *pR++;
			sum -= *pL++;
			*pF++ = sum * mtbl[*pInd++];
		}

		for (; x < n_cols; ++x)
		{
			sum -= *pL++;
			*pF++ = sum * mtbl[*pInd++];
		}

		Index += n_cols;
		plhs += n_cols;
		prhs += n_cols;
	}
}

void guideFilter::gfScanLineYEx(const f32* pI, const f32* mtbl, const u8* ind, f32* pO, u16 n_rows, u16 n_cols, u16 R)
{
	f32* tbl = (f32*)malloc(n_cols * sizeof(f32));
	memset(tbl, 0, n_cols * sizeof(f32));

	const u8* pInd = ind;
	const f32 *pL, *pR;
	f32 *pF;
	u16 x, y;

	pR = pL = pI;
	pF = pO;

	for (y = 0; y < R; ++y)
	{
		for (x = 0; x < n_cols; ++x)
		{
			tbl[x] += *pR++;
		}
	}

	for (y = 0; y <= R; ++y)
	{
		for (x = 0; x < n_cols; ++x)
		{
			tbl[x] += *pR++;
			*pF++ = tbl[x] * mtbl[*pInd++];
		}

	}

	for (; y < n_rows - R; ++y)
	{
		for (x = 0; x < n_cols; ++x)
		{
			tbl[x] += *pR++;
			tbl[x] -= *pL++;
			*pF++ = tbl[x] * mtbl[*pInd++];
		}
	}

	for (; y < n_rows; ++y)
	{
		for (x = 0; x < n_cols; ++x)
		{
			tbl[x] -= *pL++;
			*pF++ = tbl[x] * mtbl[*pInd++];
		}
	}

	free(tbl);
}

void guideFilter::gfBoxFilter(const u8* pI, u8* pO, u16 n_rows, u16 n_cols, u16 R)
{
	u32 size = n_rows * n_cols * sizeof(u8);
	u8* pT = (u8*)malloc(size);
	gfScanLineX(pI, pT, n_rows, n_cols, R);
	gfScanLineY(pT, pO, n_rows, n_cols, R);
	free(pT);
}

void guideFilter::gfScanLineX(const u8* pI, u8* pO, u16 n_rows, u16 n_cols, u16 R)
{
	const u8 *pL, *pR;
	u8 *pU, sum;
	u16 x, y;

	const u8 *plhs = pI;
	u8* prhs = pO;

	for (y = 0; y < n_rows; ++y)
	{
		pR = pL = plhs;
		pU = prhs;

		sum = 0;
		for (x = 0; x < R; ++x)
		{
			sum += *pR++;
		}

		for (x = 0; x <= R; ++x)
		{
			sum += *pR++;
			*pU++ = sum;
		}

		for (; x < n_cols - R; ++x)
		{
			sum += *pR++;
			sum -= *pL++;
			*pU++ = sum;
		}

		for (; x < n_cols; ++x)
		{
			sum -= *pL++;
			*pU++ = sum;
		}

		plhs += n_cols;
		prhs += n_cols;
	}
}

void guideFilter::gfScanLineY(const u8* pI, u8* pO, u16 n_rows, u16 n_cols, u16 R)
{
	u8* tbl = (u8*)malloc(n_cols * sizeof(u8));
	memset(tbl, 0, n_cols * sizeof(u8));

	const u8 *pL, *pR;
	u8 *pU;
	u16 x, y;

	pR = pL = pI;
	pU = pO;

	for (y = 0; y < R; ++y)
	{
		for (x = 0; x < n_cols; ++x)
		{
			tbl[x] += *pR++;
		}
	}

	for (y = 0; y <= R; ++y)
	{
		for (x = 0; x < n_cols; ++x)
		{
			tbl[x] += *pR++;
			*pU++ = tbl[x];
		}

	}

	for (; y < n_rows - R; ++y)
	{
		for (x = 0; x < n_cols; ++x)
		{
			tbl[x] += *pR++;
			tbl[x] -= *pL++;
			*pU++ = tbl[x];
		}
	}

	for (; y < n_rows; ++y)
	{
		for (x = 0; x < n_cols; ++x)
		{
			tbl[x] -= *pL++;
			*pU++ = tbl[x];
		}
	}

	free(tbl);
}

void guideFilter::scanLineXEx(const u32* pI, const f32* mtbl, const u8* indX, u32* pO1, f32* pO2, u16 n_rows, u16 n_cols)
{
	u32 *plhs01 = pO1;
	f32 *plhs02 = pO2;
	u32 *pDataX;
	f32 *pDataY;
	const u32 *pDataU, *plhs = pI;
	const u8 *pInd, *Index = indX;

	u16 x, y;

	plhs01 = pO1, plhs02 = pO2, plhs = pI;

	for (y = 0; y < n_rows; ++y)
	{
		pDataX = plhs01;
		pDataY = plhs02;
		pDataU = plhs;
		pInd = Index;
		
		for (x = 0; x <= 3; ++x)
		{
			*pDataX++ = pDataU[x+3];
			*pDataY++ = pDataU[x+30] * mtbl[*pInd++];
		}

		for (x = 3 + 1; x <= 30; ++x)
		{
			*pDataX++ = pDataU[x+3] - pDataU[x-3-1];
			*pDataY++ = pDataU[x+30] * mtbl[*pInd++];
		}

		for (x = 30 + 1; x < n_cols - 30; ++x)
		{
			*pDataX++ = pDataU[x + 3] - pDataU[x - 3 - 1];
			*pDataY++ = (pDataU[x + 30] - pDataU[x - 30 - 1]) * mtbl[*pInd++];
		}

		for (x = n_cols - 30; x < n_cols - 3; ++x)
		{
			*pDataX++ = pDataU[x + 3] - pDataU[x - 3 - 1];
			*pDataY++ = (pDataU[n_cols - 1] - pDataU[x - 30 - 1]) * mtbl[*pInd++];
		}

		for (x = n_cols - 3; x < n_cols; ++x)
		{
			*pDataX++ = pDataU[n_cols - 1] - pDataU[x - 3 - 1];
			*pDataY++ = (pDataU[n_cols - 1] - pDataU[x - 30 - 1]) * mtbl[*pInd++];
		}

		Index += n_cols;
		plhs01 += n_cols;
		plhs02 += n_cols;
		plhs += n_cols;
	}
}

void guideFilter::scanLineYEx(const u32* pI1, const u32* pI2, const f32* mtbl, const u8* ind, const u8* indY, f32* pO1, f32* pO2, u16 n_rows, u16 n_cols)
{
	const u32 *pDataU1Y1, *pDataU1Y2, *pDataU2Y1, *pDataU2Y2;
	f32 *plhs01 = pO1, *plhs02 = pO2;
	f32 *pDataX, *pDataY;
	const u8 *pInd, *pIndY, *Index = ind, *IndexY = indY;
	u16 x, y;

	for (y = 0; y <= 3; ++y)
	{
		pDataX = plhs01;
		pDataY = plhs02;
		pIndY = IndexY;
		pInd = Index;

		pDataU1Y1 = pI2 + (y + 3)*n_cols;
		pDataU2Y1 = pI1 + (y + 30)*n_cols;

		for (x = 0; x < n_cols; ++x)
		{
			*pDataX++ = pDataU1Y1[x] * mtbl[*pInd++];
			*pDataY++ = pDataU2Y1[x] * mtbl[*pIndY++];
		}

		IndexY += n_cols;
		Index += n_cols;
		plhs01 += n_cols;
		plhs02 += n_cols;
	}
	for (y = 4; y <= 30; ++y)
	{
		pDataX = plhs01;
		pDataY = plhs02;
		pIndY = IndexY;
		pInd = Index;

		pDataU1Y1 = pI2 + (y + 3)*n_cols;
		pDataU1Y2 = pI2 + (y - 3 - 1)*n_cols;
		pDataU2Y1 = pI1 + (y + 30)*n_cols;

		for (x = 0; x < n_cols; ++x)
		{
			*pDataX++ = (pDataU1Y1[x] - pDataU1Y2[x]) * mtbl[*pInd++];
			*pDataY++ = pDataU2Y1[x] * mtbl[*pIndY++];
		}

		IndexY += n_cols;
		Index += n_cols;
		plhs01 += n_cols;
		plhs02 += n_cols;
	}

	for (y = 30 + 1; y < n_rows - 30; ++y)
	{
		pDataX = plhs01;
		pDataY = plhs02;
		pIndY = IndexY;
		pInd = Index;

		pDataU1Y1 = pI2 + (y + 3)*n_cols;
		pDataU1Y2 = pI2 + (y - 3 - 1)*n_cols;
		pDataU2Y1 = pI1 + (y + 30)*n_cols;
		pDataU2Y2 = pI1 + (y - 30 - 1)*n_cols;

		for (x = 0; x < n_cols; ++x)
		{
			*pDataX++ = (pDataU1Y1[x] - pDataU1Y2[x]) * mtbl[*pInd++];
			*pDataY++ = (pDataU2Y1[x] - pDataU2Y2[x]) * mtbl[*pIndY++];
		}

		IndexY += n_cols;
		Index += n_cols;
		plhs01 += n_cols;
		plhs02 += n_cols;
	}

	for (y = n_rows - 30; y < n_rows - 3; ++y)
	{
		pDataX = plhs01;
		pDataY = plhs02;
		pIndY = IndexY;
		pInd = Index;
		pDataU1Y1 = pI2 + (y + 3)*n_cols;
		pDataU1Y2 = pI2 + (y - 3 - 1)*n_cols;
		pDataU2Y1 = pI1 + (n_rows - 1)*n_cols;
		pDataU2Y2 = pI1 + (y - 30 - 1)*n_cols;

		for (x = 0; x < n_cols; ++x)
		{
			*pDataX++ = (pDataU1Y1[x] - pDataU1Y2[x]) * mtbl[*pInd++];
			*pDataY++ = (pDataU2Y1[x] - pDataU2Y2[x]) * mtbl[*pIndY++];
		}

		IndexY += n_cols;
		Index += n_cols;
		plhs01 += n_cols;
		plhs02 += n_cols;
	}

	for (y = n_rows - 3; y < n_rows; ++y)
	{
		pDataX = plhs01;
		pDataY = plhs02;
		pIndY = IndexY;
		pInd = Index;
		pDataU1Y1 = pI2 + (n_rows - 1)*n_cols;
		pDataU1Y2 = pI2 + (y - 3 - 1)*n_cols;
		pDataU2Y1 = pI1 + (n_rows - 1)*n_cols;
		pDataU2Y2 = pI1 + (y - 30 - 1)*n_cols;

		for (x = 0; x < n_cols; ++x)
		{
			*pDataX++ = (pDataU1Y1[x] - pDataU1Y2[x]) * mtbl[*pInd++];
			*pDataY++ = (pDataU2Y1[x] - pDataU2Y2[x]) * mtbl[*pIndY++];
		}

		IndexY += n_cols;
		Index += n_cols;
		plhs01 += n_cols;
		plhs02 += n_cols;
	}
}

void guideFilter::scanLineXEx(const f32* pI, const f32* mtbl, const u8* indX, f32* pO1, f32* pO2, u16 n_rows, u16 n_cols)
{
	f32 *plhs01 = pO1, *plhs02 = pO2;
	f32 *pDataX, *pDataY;
	const f32 *pDataU, *plhs = pI;
	const u8 *pInd, *Index = indX;
	u16 x, y;

	plhs01 = pO1, plhs02 = pO2, plhs = pI;

	for (y = 0; y < n_rows; ++y)
	{
		pDataX = plhs01;
		pDataY = plhs02;
		pDataU = plhs;
		pInd = Index;

		for (x = 0; x <= 3; ++x)
		{
			*pDataX++ = pDataU[x + 3];
			*pDataY++ = pDataU[x + 30] * mtbl[*pInd++];
		}

		for (x = 4; x <= 30; ++x)
		{
			*pDataX++ = pDataU[x + 3] - pDataU[x - 3 - 1];
			*pDataY++ = pDataU[x + 30] * mtbl[*pInd++];
		}

		for (x = 30 + 1; x < n_cols - 30; ++x)
		{
			*pDataX++ = pDataU[x + 3] - pDataU[x - 3 - 1];
			*pDataY++ = (pDataU[x + 30] - pDataU[x - 30 - 1]) * mtbl[*pInd++];
		}

		for (x = n_cols - 30; x < n_cols - 3; ++x)
		{
			*pDataX++ = pDataU[x + 3] - pDataU[x - 3 - 1];
			*pDataY++ = (pDataU[n_cols - 1] - pDataU[x - 30 - 1]) * mtbl[*pInd++];
		}

		for (x = n_cols - 3; x < n_cols; ++x)
		{
			*pDataX++ = pDataU[n_cols - 1] - pDataU[x - 3 - 1];
			*pDataY++ = (pDataU[n_cols - 1] - pDataU[x - 30 - 1]) * mtbl[*pInd++];
		}

		Index += n_cols;
		plhs01 += n_cols;
		plhs02 += n_cols;
		plhs += n_cols;
	}
}

void guideFilter::scanLineYEx(const f32* pI1, const f32* pI2, const f32* mtbl, const u8* ind, const u8* indY, f32* pO1, f32* pO2, u16 n_rows, u16 n_cols)
{
	const f32 *pDataU1Y1, *pDataU1Y2, *pDataU2Y1, *pDataU2Y2;
	f32 *plhs01 = pO1, *plhs02 = pO2;
	f32 *pDataX, *pDataY;
	const u8 *pInd, *pIndY, *Index = ind, *IndexY = indY;
	u16 x, y;

	for (y = 0; y <= 3; ++y)
	{
		pDataX = plhs01;
		pDataY = plhs02;
		pIndY = IndexY;
		pInd = Index;
		pDataU1Y1 = pI2 + (y + 3)*n_cols;
		pDataU2Y1 = pI1 + (y + 30)*n_cols;

		for (x = 0; x < n_cols; ++x)
		{
			*pDataX++ = pDataU1Y1[x] * mtbl[*pInd++];
			*pDataY++ = pDataU2Y1[x] * mtbl[*pIndY++];
		}

		IndexY += n_cols;
		Index += n_cols;
		plhs01 += n_cols;
		plhs02 += n_cols;
	}
	for (y = 4; y <= 30; ++y)
	{
		pDataX = plhs01;
		pDataY = plhs02;
		pIndY = IndexY;
		pInd = Index;
		pDataU1Y1 = pI2 + (y + 3)*n_cols;
		pDataU1Y2 = pI2 + (y - 3 - 1)*n_cols;
		pDataU2Y1 = pI1 + (y + 30)*n_cols;

		for (x = 0; x < n_cols; ++x)
		{
			*pDataX++ = (pDataU1Y1[x] - pDataU1Y2[x]) * mtbl[*pInd++];
			*pDataY++ = pDataU2Y1[x] * mtbl[*pIndY++];
		}

		IndexY += n_cols;
		Index += n_cols;
		plhs01 += n_cols;
		plhs02 += n_cols;
	}

	for (y = 30 + 1; y < n_rows - 30; ++y)
	{
		pDataX = plhs01;
		pDataY = plhs02;
		pIndY = IndexY;
		pInd = Index;
		pDataU1Y1 = pI2 + (y + 3)*n_cols;
		pDataU1Y2 = pI2 + (y - 3 - 1)*n_cols;
		pDataU2Y1 = pI1 + (y + 30)*n_cols;
		pDataU2Y2 = pI1 + (y - 30 - 1)*n_cols;

		for (x = 0; x < n_cols; ++x)
		{
			*pDataX++ = (pDataU1Y1[x] - pDataU1Y2[x]) * mtbl[*pInd++];
			*pDataY++ = (pDataU2Y1[x] - pDataU2Y2[x]) * mtbl[*pIndY++];
		}

		IndexY += n_cols;
		Index += n_cols;
		plhs01 += n_cols;
		plhs02 += n_cols;
	}

	for (y = n_rows - 30; y < n_rows - 3; ++y)
	{
		pDataX = plhs01;
		pDataY = plhs02;
		pIndY = IndexY;
		pInd = Index;
		pDataU1Y1 = pI2 + (y + 3)*n_cols;
		pDataU1Y2 = pI2 + (y - 3 - 1)*n_cols;
		pDataU2Y1 = pI1 + (n_rows - 1)*n_cols;
		pDataU2Y2 = pI1 + (y - 30 - 1)*n_cols;

		for (x = 0; x < n_cols; ++x)
		{
			*pDataX++ = (pDataU1Y1[x] - pDataU1Y2[x]) * mtbl[*pInd++];
			*pDataY++ = (pDataU2Y1[x] - pDataU2Y2[x]) * mtbl[*pIndY++];
		}

		IndexY += n_cols;
		Index += n_cols;
		plhs01 += n_cols;
		plhs02 += n_cols;
	}

	for (y = n_rows - 3; y < n_rows; ++y)
	{
		pDataX = plhs01;
		pDataY = plhs02;
		pIndY = IndexY;
		pInd = Index;
		pDataU1Y1 = pI2 + (n_rows - 1)*n_cols;
		pDataU1Y2 = pI2 + (y - 3 - 1)*n_cols;
		pDataU2Y1 = pI1 + (n_rows - 1)*n_cols;
		pDataU2Y2 = pI1 + (y - 30 - 1)*n_cols;

		for (x = 0; x < n_cols; ++x)
		{
			*pDataX++ = (pDataU1Y1[x] - pDataU1Y2[x]) * mtbl[*pInd++];
			*pDataY++ = (pDataU2Y1[x] - pDataU2Y2[x]) * mtbl[*pIndY++];
		}

		IndexY += n_cols;
		Index += n_cols;
		plhs01 += n_cols;
		plhs02 += n_cols;
	}
}

void guideFilter::divid(const f32* covIP, const f32* varI, f32 *pO, u16 n_rows, u16 n_cols)
{
	const f32 *pDataYF = varI, *pDataZF = covIP;
	f32 *pDataXF = pO;
	u32 i, size = n_rows * n_cols;

	if (covIP == pO)
	{
		for (i = 0; i < size - 4; i += 4)
		{
			*pDataXF++ /= (*pDataYF++ + Eps);
			*pDataXF++ /= (*pDataYF++ + Eps);
			*pDataXF++ /= (*pDataYF++ + Eps);
			*pDataXF++ /= (*pDataYF++ + Eps);
		}
		for (; i < size; ++i)
		{
			*pDataXF++ /= (*pDataYF++ + Eps);
		}
	}
	else
	{
		for (i = 0; i < size - 4; i += 4)
		{
			*pDataXF++ = *pDataZF++ / (*pDataYF++ + Eps);
			*pDataXF++ = *pDataZF++ / (*pDataYF++ + Eps);
			*pDataXF++ = *pDataZF++ / (*pDataYF++ + Eps);
			*pDataXF++ = *pDataZF++ / (*pDataYF++ + Eps);
		}
		for (; i < size; ++i)
		{
			*pDataXF++ = *pDataZF++ / (*pDataYF++ + Eps);
		}
	}
}

void guideFilter::multiply(const u8* pI1, const u8* pI2, const u8* pI3, u8* pO, u16 n_rows, u16 n_cols)
{
	const u8 *pDataF32X = pI1, *pDataF32Y = pI2, *pDataF32Z = pI3;
	u32 i, size = n_rows * n_cols;
	u8 *pDataF32O = pO;

#ifdef ARM_NEON
	float32x4_t vf32a, vf32b, vf32c, vf32d, vf32e;
	for (i = 0; i < size - 4; i += 4)
	{
		vf32a = vld1q_f32(pDataF32X);
		vf32b = vld1q_f32(pDataF32Y);
		vf32c = vld1q_f32(pDataF32Z);

		vf32d = vmulq_f32(vf32a, vf32b);
		vf32e = vmulq_f32(vf32c, vf32d);

		vst1q_f32(pDataF32O, vf32e);

		pDataF32X += 4;
		pDataF32Y += 4;
		pDataF32Z += 4;
		pDataF32O += 4;
	}
	for (; i < size; ++i)
	{
		*pDataF32O++ = *pDataF32X++ * *pDataF32Y++ * *pDataF32Z++;
	}
#else
	f32 s0 = 1.f / 255 / 255;
	for (i = 0; i < size - 4; i += 4)
	{
		*pDataF32O++ = *pDataF32X++ * *pDataF32Y++ * *pDataF32Z++ * s0;
		*pDataF32O++ = *pDataF32X++ * *pDataF32Y++ * *pDataF32Z++ * s0;
		*pDataF32O++ = *pDataF32X++ * *pDataF32Y++ * *pDataF32Z++ * s0;
		*pDataF32O++ = *pDataF32X++ * *pDataF32Y++ * *pDataF32Z++ * s0;
	}
	for (; i < size; ++i)
	{
		*pDataF32O++ = *pDataF32X++ * *pDataF32Y++ * *pDataF32Z++ * s0;
	}
#endif
}

void guideFilter::multiply(const f32* pI1, const f32* pI2, const f32* pI3, f32* pO, u16 n_rows, u16 n_cols)
{
	const f32 *pDataF32X = pI1, *pDataF32Y = pI2, *pDataF32Z = pI3;
	u32 i, size = n_rows * n_cols;
	f32 *pDataF32O = pO;

#ifdef ARM_NEON
	float32x4_t vf32a, vf32b, vf32c, vf32d, vf32e;
	for (i = 0; i < size - 4; i += 4)
	{
		vf32a = vld1q_f32(pDataF32X);
		vf32b = vld1q_f32(pDataF32Y);
		vf32c = vld1q_f32(pDataF32Z);

		vf32d = vmulq_f32(vf32a, vf32b);
		vf32e = vmulq_f32(vf32c, vf32d);

		vst1q_f32(pDataF32O, vf32e);

		pDataF32X += 4;
		pDataF32Y += 4;
		pDataF32Z += 4;
		pDataF32O += 4;
	}
	for (; i < size; ++i)
	{
		*pDataF32O++ = *pDataF32X++ * *pDataF32Y++ * *pDataF32Z++;
	}
#else
	for (i = 0; i < size - 4; i += 4)
	{
		*pDataF32O++ = *pDataF32X++ * *pDataF32Y++ * *pDataF32Z++;
		*pDataF32O++ = *pDataF32X++ * *pDataF32Y++ * *pDataF32Z++;
		*pDataF32O++ = *pDataF32X++ * *pDataF32Y++ * *pDataF32Z++;
		*pDataF32O++ = *pDataF32X++ * *pDataF32Y++ * *pDataF32Z++;
	}
	for (; i < size; ++i)
	{
		*pDataF32O++ = *pDataF32X++ * *pDataF32Y++ * *pDataF32Z++;
	}
#endif
}

void guideFilter::multiply(const f32* p01, const f32* p02, f32* pO, u16 n_rows, u16 n_cols)
{
	u32 i, size = n_rows * n_cols;
	f32 *pDataU32X;
	const f32 *pDataI, *pDataG;

	pDataU32X = pO;
	pDataI = p01;
	pDataG = p02;

	if (p01 == p02)
	{
#ifdef ARM_NEON
		float32x4_t vf32a, vf32b;
		for (i = 0; i < size - 4; i += 4)
		{
			vf32a = vsetq_lane_f32(*pDataI++, vf32a, 0);
			vf32a = vsetq_lane_f32(*pDataI++, vf32a, 1);
			vf32a = vsetq_lane_f32(*pDataI++, vf32a, 2);
			vf32a = vsetq_lane_f32(*pDataI++, vf32a, 3);

			vf32b = vmulq_u32(vf32a, vf32a);

			*pDataU32X++ = vgetq_lane_f32(vf32b, 0);
			*pDataU32X++ = vgetq_lane_f32(vf32b, 1);
			*pDataU32X++ = vgetq_lane_f32(vf32b, 2);
			*pDataU32X++ = vgetq_lane_f32(vf32b, 3);

			pDataI += 4;
		}
		for (; i < size; ++i)
		{
			f32 s0 = *pDataI++;
			*pDataU32X++ = s0 * s0;
		}
#else
		for (i = 0; i < size - 4; i += 4)
		{
			f32 s0 = *pDataI++;
			*pDataU32X++ = s0 * s0;
			s0 = *pDataI++;
			*pDataU32X++ = s0 * s0;
			s0 = *pDataI++;
			*pDataU32X++ = s0 * s0;
			s0 = *pDataI++;
			*pDataU32X++ = s0 * s0;
		}
		for (; i < size; ++i)
		{
			f32 s0 = *pDataI++;
			*pDataU32X++ = s0 * s0;
		}
#endif
	}
	else
	{
#ifdef ARM_NEON
		uint32x4_t vu32a, vu32b, vu32c;
		for (i = 0; i < size - 4; i += 4)
		{
			vu32a = vsetq_lane_u32(*pDataI++, vu32a, 0);
			vu32a = vsetq_lane_u32(*pDataI++, vu32a, 1);
			vu32a = vsetq_lane_u32(*pDataI++, vu32a, 2);
			vu32a = vsetq_lane_u32(*pDataI++, vu32a, 3);
			vu32b = vsetq_lane_u32(*pDataG++, vu32b, 0);
			vu32b = vsetq_lane_u32(*pDataG++, vu32b, 1);
			vu32b = vsetq_lane_u32(*pDataG++, vu32b, 2);
			vu32b = vsetq_lane_u32(*pDataG++, vu32b, 3);

			vu32c = vmulq_u32(vu32a, vu32b);

			*pDataU32X++ = vgetq_lane_u32(vu32c, 0);
			*pDataU32X++ = vgetq_lane_u32(vu32c, 1);
			*pDataU32X++ = vgetq_lane_u32(vu32c, 2);
			*pDataU32X++ = vgetq_lane_u32(vu32c, 3);
			pDataI += 4;
			pDataG += 4;
		}
		for (; i < size; ++i)
		{
			*pDataU32X++ = *pDataI++ * *pDataG++;
		}
#else
		for (i = 0; i < size - 4; i += 4)
		{
			*pDataU32X++ = *pDataI++ * *pDataG++;
			*pDataU32X++ = *pDataI++ * *pDataG++;
			*pDataU32X++ = *pDataI++ * *pDataG++;
			*pDataU32X++ = *pDataI++ * *pDataG++;
		}
		for (; i < size; ++i)
		{
			*pDataU32X++ = *pDataI++ * *pDataG++;
		}
#endif
	}
}

void guideFilter::multiply(const u8* p01, const u8* p02, u32* pO, u16 n_rows, u16 n_cols)
{
	u32 i, size = n_rows * n_cols;
	u32 *pDataU32X;
	const u8 *pDataI, *pDataG;

	pDataU32X = pO;
	pDataI = p01;
	pDataG = p02;

	if (p01 == p02)
	{
#ifdef ARM_NEON
		uint32x4_t vu32a, vu32b;
		for (i = 0; i < size - 4; i += 4)
		{
			vu32a = vsetq_lane_u32(*pDataI++, vu32a, 0);
			vu32a = vsetq_lane_u32(*pDataI++, vu32a, 1);
			vu32a = vsetq_lane_u32(*pDataI++, vu32a, 2);
			vu32a = vsetq_lane_u32(*pDataI++, vu32a, 3);

			vu32b = vmulq_u32(vu32a, vu32a);

			*pDataU32X++ = vgetq_lane_u32(vu32b, 0);
			*pDataU32X++ = vgetq_lane_u32(vu32b, 1);
			*pDataU32X++ = vgetq_lane_u32(vu32b, 2);
			*pDataU32X++ = vgetq_lane_u32(vu32b, 3);

			pDataI += 4;
		}
		for (; i < size; ++i)
		{
			u32 s0 = *pDataI++;
			*pDataU32X++ = s0 * s0;
		}
#else
		for (i = 0; i < size - 4; i += 4)
		{
			u32 s0 = *pDataI++;
			*pDataU32X++ = s0 * s0;
			s0 = *pDataI++;
			*pDataU32X++ = s0 * s0;
			s0 = *pDataI++;
			*pDataU32X++ = s0 * s0;
			s0 = *pDataI++;
			*pDataU32X++ = s0 * s0;
		}
		for (; i < size; ++i)
		{
			u32 s0 = *pDataI++;
			*pDataU32X++ = s0 * s0;
		}
#endif
	}
	else
	{
#ifdef ARM_NEON
		uint32x4_t vu32a, vu32b, vu32c;
		for (i = 0; i < size - 4; i += 4)
		{
			vu32a = vsetq_lane_u32(*pDataI++, vu32a, 0);
			vu32a = vsetq_lane_u32(*pDataI++, vu32a, 1);
			vu32a = vsetq_lane_u32(*pDataI++, vu32a, 2);
			vu32a = vsetq_lane_u32(*pDataI++, vu32a, 3);
			vu32b = vsetq_lane_u32(*pDataG++, vu32b, 0);
			vu32b = vsetq_lane_u32(*pDataG++, vu32b, 1);
			vu32b = vsetq_lane_u32(*pDataG++, vu32b, 2);
			vu32b = vsetq_lane_u32(*pDataG++, vu32b, 3);

			vu32c = vmulq_u32(vu32a, vu32b);

			*pDataU32X++ = vgetq_lane_u32(vu32c, 0);
			*pDataU32X++ = vgetq_lane_u32(vu32c, 1);
			*pDataU32X++ = vgetq_lane_u32(vu32c, 2);
			*pDataU32X++ = vgetq_lane_u32(vu32c, 3);
			pDataI += 4;
			pDataG += 4;
		}
		for (; i < size; ++i)
		{
			*pDataU32X++ = *pDataI++ * *pDataG++;
		}
#else
		for (i = 0; i < size - 4; i += 4)
		{
			*pDataU32X++ = *pDataI++ * *pDataG++;
			*pDataU32X++ = *pDataI++ * *pDataG++;
			*pDataU32X++ = *pDataI++ * *pDataG++;
			*pDataU32X++ = *pDataI++ * *pDataG++;
		}
		for (; i < size; ++i)
		{
			*pDataU32X++ = *pDataI++ * *pDataG++;
		}
#endif
	}
}

void guideFilter::multiply(const u32* p01, const u32* p02, u32* pO, u16 n_rows, u16 n_cols)
{
	u32 i, size = n_rows * n_cols;
	u32 *pDataU32X;
	const u32 *pDataI, *pDataG;

	pDataU32X = pO;
	pDataI = p01;
	pDataG = p02;

	if (p01 == p02)
	{
#ifdef ARM_NEON
		uint32x4_t vu32a, vu32b;
		for (i = 0; i < size - 4; i += 4)
		{
			vu32a = vsetq_lane_u32(*pDataI++, vu32a, 0);
			vu32a = vsetq_lane_u32(*pDataI++, vu32a, 1);
			vu32a = vsetq_lane_u32(*pDataI++, vu32a, 2);
			vu32a = vsetq_lane_u32(*pDataI++, vu32a, 3);

			vu32b = vmulq_u32(vu32a, vu32a);

			*pDataU32X++ = vgetq_lane_u32(vu32b, 0);
			*pDataU32X++ = vgetq_lane_u32(vu32b, 1);
			*pDataU32X++ = vgetq_lane_u32(vu32b, 2);
			*pDataU32X++ = vgetq_lane_u32(vu32b, 3);

			pDataI += 4;
		}
		for (; i < size; ++i)
		{
			u32 s0 = *pDataI++;
			*pDataU32X++ = s0 * s0;
		}
#else
		for (i = 0; i < size - 4; i += 4)
		{
			u32 s0 = *pDataI++;
			*pDataU32X++ = s0 * s0;
			s0 = *pDataI++;
			*pDataU32X++ = s0 * s0;
			s0 = *pDataI++;
			*pDataU32X++ = s0 * s0;
			s0 = *pDataI++;
			*pDataU32X++ = s0 * s0;
		}
		for (; i < size; ++i)
		{
			u32 s0 = *pDataI++;
			*pDataU32X++ = s0 * s0;
		}
#endif
	}
	else
	{
#ifdef ARM_NEON
		uint32x4_t vu32a, vu32b, vu32c;
		for (i = 0; i < size - 4; i += 4)
		{
			vu32a = vsetq_lane_u32(*pDataI++, vu32a, 0);
			vu32a = vsetq_lane_u32(*pDataI++, vu32a, 1);
			vu32a = vsetq_lane_u32(*pDataI++, vu32a, 2);
			vu32a = vsetq_lane_u32(*pDataI++, vu32a, 3);
			vu32b = vsetq_lane_u32(*pDataG++, vu32b, 0);
			vu32b = vsetq_lane_u32(*pDataG++, vu32b, 1);
			vu32b = vsetq_lane_u32(*pDataG++, vu32b, 2);
			vu32b = vsetq_lane_u32(*pDataG++, vu32b, 3);

			vu32c = vmulq_u32(vu32a, vu32b);

			*pDataU32X++ = vgetq_lane_u32(vu32c, 0);
			*pDataU32X++ = vgetq_lane_u32(vu32c, 1);
			*pDataU32X++ = vgetq_lane_u32(vu32c, 2);
			*pDataU32X++ = vgetq_lane_u32(vu32c, 3);
			pDataI += 4;
			pDataG += 4;
		}
		for (; i < size; ++i)
		{
			*pDataU32X++ = *pDataI++ * *pDataG++;
		}
#else
		for (i = 0; i < size - 4; i += 4)
		{
			*pDataU32X++ = *pDataI++ * *pDataG++;
			*pDataU32X++ = *pDataI++ * *pDataG++;
			*pDataU32X++ = *pDataI++ * *pDataG++;
			*pDataU32X++ = *pDataI++ * *pDataG++;
		}
		for (; i < size; ++i)
		{
			*pDataU32X++ = *pDataI++ * *pDataG++;
		}
#endif
	}
}

void guideFilter::msltiply(const f32* A, const f32* B, const f32* C, f32* D, u16 n_rows, u16 n_cols)
{
	const f32 *pDataXF, *pDataYF, *pDataZF;
	f32 *pDataWF;

	u32 i, size = n_rows * n_cols;

	pDataXF = A;
	pDataYF = B;
	pDataZF = C;
	pDataWF = D;

	if (A == D)
	{
#ifdef ARM_NEON
		float32x4_t vf32a, vf32b, vf32c, vf32d;
		for (i = 0; i < size - 4; i += 4)
		{
			vf32a = vld1q_f32(pDataWF);
			vf32b = vld1q_f32(pDataYF);
			vf32c = vld1q_f32(pDataZF);
			vf32d = vmlsq_f32(vf32a, vf32b, vf32c);
			vst1q_f32(pDataXF, vf32d);
			pDataWF += 4;
			pDataYF += 4;
			pDataZF += 4;
		}
		for (; i < size; ++i)
		{
			*pDataWF++ -= *pDataYF++ * *pDataZF++;
		}
#else
		for (i = 0; i < size - 4; i += 4)
		{
			*pDataWF++ -= *pDataYF++ * *pDataZF++;
			*pDataWF++ -= *pDataYF++ * *pDataZF++;
			*pDataWF++ -= *pDataYF++ * *pDataZF++;
			*pDataWF++ -= *pDataYF++ * *pDataZF++;
		}
		for (; i < size; ++i)
		{
			*pDataWF++ -= *pDataYF++ * *pDataZF++;
		}
#endif
	}
	else if (A == D && B == C)
	{
		pDataWF = D;
		pDataYF = B;

#ifdef ARM_NEON
		float32x4_t vf32a, vf32b, vf32c;
		for (i = 0; i < size - 4; i += 4)
		{
			vf32a = vld1q_f32(pDataWF);
			vf32b = vld1q_f32(pDataYF);
			vf32c = vmlsq_f32(vf32a, vf32b, vf32b);
			vst1q_f32(pDataXF, vf32c);
			pDataWF += 4;
			pDataYF += 4;
		}
		for (; i < size; ++i)
		{
			f32 s0 = *pDataYF++;
			*pDataWF++ -= s0 * s0;
		}
#else
		for (i = 0; i < size - 4; i += 4)
		{
			f32 s0 = *pDataYF++;
			*pDataWF++ -= s0 * s0;
			s0 = *pDataYF++;
			*pDataWF++ -= s0 * s0;
			s0 = *pDataYF++;
			*pDataWF++ -= s0 * s0;
			s0 = *pDataYF++;
			*pDataWF++ -= s0 * s0;
		}
		for (; i < size; ++i)
		{
			f32 s0 = *pDataYF++;
			*pDataWF++ -= s0 * s0;
		}
#endif
	}
	else
	{
#ifdef ARM_NEON
		float32x4_t vf32a, vf32b, vf32c, vf32d;
		for (i = 0; i < size - 4; i += 4)
		{
			vf32a = vld1q_f32(pDataXF);
			vf32b = vld1q_f32(pDataYF);
			vf32c = vld1q_f32(pDataZF);
			vf32d = vmlsq_f32(vf32a, vf32b, vf32c);
			vst1q_f32(pDataWF, vf32d);
			pDataXF += 4;
			pDataYF += 4;
			pDataZF += 4;
			pDataWF += 4;
		}
		for (; i < size; ++i)
		{
			*pDataWF++ = *pDataXF++ - *pDataYF++ * *pDataZF++;
		}
#else
		for (i = 0; i < size - 4; i += 4)
		{
			*pDataWF++ = *pDataXF++ - *pDataYF++ * *pDataZF++;
			*pDataWF++ = *pDataXF++ - *pDataYF++ * *pDataZF++;
			*pDataWF++ = *pDataXF++ - *pDataYF++ * *pDataZF++;
			*pDataWF++ = *pDataXF++ - *pDataYF++ * *pDataZF++;
		}
		for (; i < size; ++i)
		{
			*pDataWF++ = *pDataXF++ - *pDataYF++ * *pDataZF++;
		}
#endif
	}
}

void guideFilter::maltiply(const f32* A, const f32* B, const f32* C, f32* D, u16 n_rows, u16 n_cols)
{
	u32 i, size = n_rows * n_cols;
	const f32 *pDataXF, *pDataYF;
	const f32* pDataI;
	f32 *pDataO;

	pDataXF = A;
	pDataI = B;
	pDataYF = C;
	pDataO = D;

#ifdef ARM_NEON
	float32x4_t vf32a, vf32b, vf32c, vf32d;
	for (i = 0; i < size - 4; i += 4)
	{
		vf32a = vld1q_f32(pDataXF);
		vf32b = vsetq_lane_f32(*pDataI++, vf32b, 0);
		vf32b = vsetq_lane_f32(*pDataI++, vf32b, 1);
		vf32b = vsetq_lane_f32(*pDataI++, vf32b, 2);
		vf32b = vsetq_lane_f32(*pDataI++, vf32b, 3);
		vf32c = vld1q_f32(pDataYF);
		vf32d = vmlaq_f32(vf32a, vf32b, vf32c);
		*pDataO++ = vgetq_lane_f32(vf32d, 0);
		*pDataO++ = vgetq_lane_f32(vf32d, 1);
		*pDataO++ = vgetq_lane_f32(vf32d, 2);
		*pDataO++ = vgetq_lane_f32(vf32d, 3);
		pDataXF += 4;
		pDataYF += 4;
	}
	for (; i < size; ++i)
	{
		*pDataO++ = *pDataXF++ + *pDataI++ * *pDataZF++;
	}
#else
	for (i = 0; i < size - 4; i += 4)
	{
		*pDataO++ = *pDataXF++ + *pDataI++ * *pDataYF++;
		*pDataO++ = *pDataXF++ + *pDataI++ * *pDataYF++;
		*pDataO++ = *pDataXF++ + *pDataI++ * *pDataYF++;
		*pDataO++ = *pDataXF++ + *pDataI++ * *pDataYF++;
	}
	for (; i < size; ++i)
	{
		*pDataO++ = *pDataXF++ + *pDataI++ * *pDataYF++;
	}
#endif
}

void guideFilter::maltiply(const f32* A, const u8* B, const f32* C, u8* D, u16 n_rows, u16 n_cols)
{
	u32 i, size = n_rows * n_cols;
	const f32 *pDataXF, *pDataYF;
	const u8* pDataI;
	u8 *pDataO;

	pDataXF = A;
	pDataI = B;
	pDataYF = C;
	pDataO = D;

#ifdef ARM_NEON
	float32x4_t vf32a, vf32b, vf32c, vf32d;
	for (i = 0; i < size - 4; i += 4)
	{
		vf32a = vld1q_f32(pDataXF);
		vf32b = vsetq_lane_f32(*pDataI++, vf32b, 0);
		vf32b = vsetq_lane_f32(*pDataI++, vf32b, 1);
		vf32b = vsetq_lane_f32(*pDataI++, vf32b, 2);
		vf32b = vsetq_lane_f32(*pDataI++, vf32b, 3);
		vf32c = vld1q_f32(pDataYF);
		vf32d = vmlaq_f32(vf32a, vf32b, vf32c);
		*pDataO++ = vgetq_lane_f32(vf32d, 0);
		*pDataO++ = vgetq_lane_f32(vf32d, 1);
		*pDataO++ = vgetq_lane_f32(vf32d, 2);
		*pDataO++ = vgetq_lane_f32(vf32d, 3);
		pDataXF += 4;
		pDataYF += 4;
	}
	for (; i < size; ++i)
	{
		*pDataO++ = *pDataXF++ + *pDataI++ * *pDataZF++;
	}
#else
	for (i = 0; i < size - 4; i += 4)
	{
		f32 s0 = *pDataXF++ + *pDataI++ * *pDataYF++;
		*pDataO++ = CLAMP(s0);
		s0 = *pDataXF++ + *pDataI++ * *pDataYF++;
		*pDataO++ = CLAMP(s0);
		s0 = *pDataXF++ + *pDataI++ * *pDataYF++;
		*pDataO++ = CLAMP(s0);
		s0 = *pDataXF++ + *pDataI++ * *pDataYF++;
		*pDataO++ = CLAMP(s0);
	}
	for (; i < size; ++i)
	{
		f32 s0 = *pDataXF++ + *pDataI++ * *pDataYF++;
		*pDataO++ = CLAMP(s0);
	}
#endif
}

void guideFilter::maltiply(const f32* A, const u32* B, const f32* C, u32* D, u16 n_rows, u16 n_cols)
{
	u32 i, size = n_rows * n_cols;
	const f32 *pDataXF, *pDataYF;
	const u32* pDataI;
	u32 *pDataO;

	pDataXF = A;
	pDataI = B;
	pDataYF = C;
	pDataO = D;

#ifdef ARM_NEON
	float32x4_t vf32a, vf32b, vf32c, vf32d;
	for (i = 0; i < size - 4; i += 4)
	{
		vf32a = vld1q_f32(pDataXF);
		vf32b = vsetq_lane_f32(*pDataI++, vf32b, 0);
		vf32b = vsetq_lane_f32(*pDataI++, vf32b, 1);
		vf32b = vsetq_lane_f32(*pDataI++, vf32b, 2);
		vf32b = vsetq_lane_f32(*pDataI++, vf32b, 3);
		vf32c = vld1q_f32(pDataYF);
		vf32d = vmlaq_f32(vf32a, vf32b, vf32c);
		*pDataO++ = vgetq_lane_f32(vf32d, 0);
		*pDataO++ = vgetq_lane_f32(vf32d, 1);
		*pDataO++ = vgetq_lane_f32(vf32d, 2);
		*pDataO++ = vgetq_lane_f32(vf32d, 3);
		pDataXF += 4;
		pDataYF += 4;
	}
	for (; i < size; ++i)
	{
		*pDataO++ = *pDataXF++ + *pDataI++ * *pDataZF++;
	}
#else
	for (i = 0; i < size - 4; i += 4)
	{
		f32 s0 = *pDataXF++ + *pDataI++ * *pDataYF++;
		*pDataO++ = CLAMP(s0);
		s0 = *pDataXF++ + *pDataI++ * *pDataYF++;
		*pDataO++ = CLAMP(s0);
		s0 = *pDataXF++ + *pDataI++ * *pDataYF++;
		*pDataO++ = CLAMP(s0);
		s0 = *pDataXF++ + *pDataI++ * *pDataYF++;
		*pDataO++ = CLAMP(s0);
	}
	for (; i < size; ++i)
	{
		f32 s0 = *pDataXF++ + *pDataI++ * *pDataYF++;
		*pDataO++ = CLAMP(s0);
	}
#endif
}
