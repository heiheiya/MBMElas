
#include "mbmElas.h"


void MBMElas::getLeftDisparity(Mat & left, Mat & right, Mat leftDisparity, Mat maxMncc, Mat lImg){

	//get variance with 3x3
	Mat meanl, meanll, varl, squarel;
	squarel = left.mul(left);
	boxFilter(left, meanl, CV_32FC1, Size(param.mncc_block, param.mncc_block));
	boxFilter(squarel, meanll, CV_32FC1, Size(param.mncc_block, param.mncc_block));
	varl = meanll - meanl.mul(meanl);

	//get right image with fill zero
	//double duration = static_cast<double>(getTickCount());
	Mat rightAllImg = Mat::zeros(height, (width + param.disp_max), CV_32FC1);
	for (int i=0; i<rightAllImg.rows; i++)
		for (int j=param.disp_max; j<rightAllImg.cols; j++){
			rightAllImg.at<float>(i, j) = right.at<float>(i, j-param.disp_max);
		}
	
	/*duration = static_cast<double>(getTickCount())-duration;
	duration /= cv::getTickFrequency(); 
	cout << "duration = " <<duration * 1000 << endl;*/

	 guideFilter gf;
	 for (int d=param.disp_min; d<param.disp_max; d++){
		//get right image, O(1)
		Mat rightImg = rightAllImg(Range::all(), Range(param.disp_max - d, rightAllImg.cols - d));

		//get MNCC of right image
		Mat meanr, meanrr, varr, squarer;
		squarer = rightImg.mul(rightImg);
		boxFilter(rightImg, meanr, CV_32FC1, Size(param.mncc_block, param.mncc_block));
		boxFilter(squarer, meanrr, CV_32FC1, Size(param.mncc_block, param.mncc_block));
		varr = meanrr - meanr.mul(meanr);

		Mat covlr;
		boxFilter(left.mul(rightImg), covlr, CV_32FC1, Size(param.mncc_block, param.mncc_block));
		covlr -= meanl.mul(meanr);

		Mat Mncc, Mncc1, Mncc2, Mncc3, Mncc4;
		divide(covlr, varl+varr, Mncc, 2);

	   Mat left1;
	   resize(left, left1 , left.size()/4);

	   Mat Mnccc(left.size(), CV_32FC1);
		gf.gfAccMultiplef(left1, left, Mncc, Mnccc);

		Mncc = Mnccc;

		//double duration1 = static_cast<double>(getTickCount());
		/*boxFilter(Mncc, Mncc1, CV_32FC1, Size(param.horizontal_block, param.vertical_block));
		boxFilter(Mncc, Mncc2, CV_32FC1, Size(param.vertical_block, param.horizontal_block));
		boxFilter(Mncc, Mncc3, CV_32FC1, Size(param.large_block, param.large_block));
		boxFilter(Mncc, Mncc4, CV_32FC1, Size(param.small_block, param.small_block));

		Mncc1 = Mncc1.mul(Mncc2);
		//Mncc3 = Mncc3.mul(Mncc4);
		Mncc = Mncc1.mul(Mncc3);
		/*Mncc1 = Mncc1+1;
		Mncc1.convertTo(Mncc2, CV_8UC1, 255);

		double duration = static_cast<double>(getTickCount());

		for (int i=0; i<10; i++)
			guidedFilter(lImg, Mncc2, Mncc3, 31, 0.0001);

		duration = static_cast<double>(getTickCount())-duration;
		duration /= cv::getTickFrequency(); 
		cout << "duration = " <<duration * 100 << endl;

		imshow("Mncc3", Mncc3);
		waitKey(0);*/
	
		/*Mncc1 = GuidedFilter(left, Mncc, 61, 1);
		Mncc2 = GuidedFilter(left, Mncc, 1, 61);
		Mncc3 = GuidedFilter(left, Mncc, 7, 7);
		Mncc1 = max(Mncc1, Mncc2);
		Mncc = Mncc1.mul(Mncc3);*/
		/*Mat tem;
		lImg.convertTo(tem, CV_32FC3, 1.0/255);
		guidedFilter(tem, Mncc1, Mncc, 31, 0.0001);*/
		//boxFilter(Mncc1, Mncc, CV_32FC1, Size(param.small_block, param.small_block));

		//Mncc = Mncc1.mul(Mncc2);

		for (int i=0; i<height; i++){
			for (int j=0; j<width; j++){
				if(Mncc.at<float>(i, j)>maxMncc.at<float>(i, j) /*&& Mncc.at<float>(i, j)>param.mncc_threshold*/){
					maxMncc.at<float>(i, j) =Mncc.at<float>(i, j);
					leftDisparity.at<uchar>(i, j) = d;
				}
			}
		}
	}
}

void MBMElas::getRightDisparity(Mat & left, Mat & right, Mat rightDisparity, Mat rImg){

	//get variance with 3x3
	Mat meanr, meanrr, varr, squarer;
	boxFilter(right, meanr, CV_32FC1, Size(param.mncc_block, param.mncc_block));
	squarer = right.mul(right);
	boxFilter(squarer, meanrr, CV_32FC1, Size(param.mncc_block, param.mncc_block));
	varr = meanrr - meanr.mul(meanr);

	//get left image with fill zero
	Mat leftAllImg = Mat::zeros(height, (width + param.disp_max), CV_32FC1);
	for (int i=0; i<leftAllImg.rows; i++){
		for (int j=0; j<width; j++){
			leftAllImg.at<float>(i, j) = left.at<float>(i, j);
		}
	}

	guideFilter gf;
	//init disparity
	Mat maxMncc = Mat::ones(height, width, CV_32FC1) * (-1);
	for (int d=param.disp_min; d<param.disp_max; d++){
		//get right image, O(1)
		Mat leftImg = leftAllImg(Range::all(), Range(d, width + d));

		//get MNCC of right image
		Mat meanl, meanll, varl, squarel;
		boxFilter(leftImg, meanl, CV_32FC1, Size(param.mncc_block, param.mncc_block));
		squarel = leftImg.mul(leftImg);
		boxFilter(squarel, meanll, CV_32FC1, Size(param.mncc_block, param.mncc_block));
		varl = meanll - meanl.mul(meanl);

		Mat covlr;
		boxFilter(right.mul(leftImg), covlr, CV_32FC1, Size(param.mncc_block, param.mncc_block));
		covlr -= meanl.mul(meanr);

		Mat Mncc, Mncc1, Mncc2, Mncc3, Mncc4;
		divide(covlr, varl+varr, Mncc, 2);
		
		/*Mat tem;
		Mncc.convertTo(tem, CV_8UC1, 255);
		imshow("tem", Mncc);
		waitKey(0);*/

	    Mat left1;
		resize(left, left1 , left.size()/4);

		Mat Mnccc(left.size(), CV_32FC1);
		gf.gfAccMultiplef(left1, left, Mncc, Mnccc);

		Mncc = Mnccc;

		//boxFilter(Mncc, Mncc1, CV_32FC1, Size(param.horizontal_block, param.vertical_block));
		//boxFilter(Mncc, Mncc2, CV_32FC1, Size(param.vertical_block, param.horizontal_block));
		//boxFilter(Mncc, Mncc3, CV_32FC1, Size(param.large_block, param.large_block));
		//boxFilter(Mncc, Mncc4, CV_32FC1, Size(param.small_block, param.small_block));

		//Mncc = Mncc1.mul(Mncc2);
		//Mncc = Mncc3.mul(Mncc1);
		//Mncc = Mncc1.mul(Mncc3);
		
		/*Mncc1 = GuidedFilter(left, Mncc, 61, 1);
		Mncc2 = GuidedFilter(left, Mncc, 1, 61);
		Mncc3 = GuidedFilter(left, Mncc, 7, 7);

		Mncc1 = max(Mncc1, Mncc2);
		Mncc = Mncc1.mul(Mncc3);*/

		/*Mat tem;
		rImg.convertTo(tem, CV_32FC3, 1.0/255);
		guidedFilter(tem, Mncc1, Mncc, 31, 0.0001);*/

		//boxFilter(Mncc1, Mncc, CV_32FC1, Size(param.small_block, param.small_block));

		for (int i=0; i<height; i++){
			for (int j=0; j<width; j++){
				if(Mncc.at<float>(i, j)>maxMncc.at<float>(i, j) /*&& Mncc.at<float>(i, j)>param.mncc_threshold*/){
					maxMncc.at<float>(i, j) =Mncc.at<float>(i, j);
					rightDisparity.at<uchar>(i, j) = d;
				}
			}
		}
	}
}

void MBMElas::getDisparity(Mat & left, Mat & right, Mat leftDisparity, Mat rightDisparity, Mat maxMncc, Mat lImg, Mat rImg){
	width = left.cols;
	height = left.rows;

	Mat left1, right1;
	left.convertTo(left1, CV_32FC1);
	right.convertTo(right1, CV_32FC1);

	getLeftDisparity(left1, right1, leftDisparity, maxMncc, lImg);
	getRightDisparity(left1, right1, rightDisparity, rImg);
}

void MBMElas::refineLeftDisparity (float* D1,float* D2, float* D,const int32_t* dims, Mat left, Mat maxMncc, Mat leftDisparity){

	// get width, height and bytes per line
	width  = dims[0];
	height = dims[1];

	leftRightConsistencyCheck(D1,D2);
	removeSmallSegments(D1);
	gapInterpolation(D1);
	
	Mat initDisparity = Mat(height, width, CV_32FC1, D1);
	//Mat temp;
	//initDisparity.convertTo(temp, CV_8UC1, 8);

	//imshow("temp", temp);
	//waitKey(0);
	vector<support_pt> p_support = computeSupportMatches(initDisparity, left);
	// if not enough support points for triangulation
	if (p_support.size()<3) {
		cout << "ERROR: Need at least 3 support points!" << endl;
		return;
	}

	//Mat temp;
	//initDisparity.convertTo(temp, CV_8UC1, 8);
	//for (int i=0; i<p_support.size(); i++)
	//{
	//	Point center(cvRound(p_support[i].u ), cvRound(p_support[i].v));
	//	circle(temp,center,8,Scalar(255, 255, 255),1);
	//}
	//imshow("temp", temp);
	//waitKey(0);


	vector<triangle> tri_1 = computeDelaunayTriangulation(p_support);
	computeDisparityPlanes(p_support,tri_1);

	// allocate memory for disparity grid
	int32_t grid_width   = (int32_t)ceil((float)width/(float)param.grid_size);
	int32_t grid_height  = (int32_t)ceil((float)height/(float)param.grid_size);
	int32_t grid_dims[3] = {param.disp_max+2,grid_width,grid_height};
	int32_t* disparity_grid_1 = (int32_t*)calloc((param.disp_max+2)*grid_height*grid_width,sizeof(int32_t));

	createGrid(p_support,disparity_grid_1,grid_dims);
	computeDisparity(p_support,tri_1 ,D, initDisparity, maxMncc, leftDisparity);

	/*param.speckle_size = 500;
	removeSmallSegments(D);*/
}

void MBMElas::removeInconsistentSupportPoints (int16_t* D_can,int32_t D_can_width,int32_t D_can_height) {

	// for all valid support points do
	for (int32_t u_can=0; u_can<D_can_width; u_can++) {
		for (int32_t v_can=0; v_can<D_can_height; v_can++) {
			int16_t d_can = *(D_can+getAddressOffsetImage(u_can,v_can,D_can_width));  //D_can中某个网格视差点的值
			if (d_can>=0) {

				// compute number of other points supporting the current point
				int32_t support = 0;
				for (int32_t u_can_2=u_can-param.incon_window_size; u_can_2<=u_can+param.incon_window_size; u_can_2++) {
					for (int32_t v_can_2=v_can-param.incon_window_size; v_can_2<=v_can+param.incon_window_size; v_can_2++) {
						if (u_can_2>=0 && v_can_2>=0 && u_can_2<D_can_width && v_can_2<D_can_height) {
							int16_t d_can_2 = *(D_can+getAddressOffsetImage(u_can_2,v_can_2,D_can_width));
							if (d_can_2>=0 && abs(d_can-d_can_2)<=param.incon_threshold)
								support++;
						}
					}
				}

				// invalidate support point if number of supporting points is too low
				if (support<param.incon_min_support)
					*(D_can+getAddressOffsetImage(u_can,v_can,D_can_width)) = -1;
			}
		}
	}
}

void MBMElas::removeRedundantSupportPoints(int16_t* D_can,int32_t D_can_width,int32_t D_can_height,
	int32_t redun_max_dist, int32_t redun_threshold, bool vertical) {

		// parameters
		int32_t redun_dir_u[2] = {0,0};
		int32_t redun_dir_v[2] = {0,0};
		if (vertical) {
			redun_dir_v[0] = -1;
			redun_dir_v[1] = +1;
		} else {
			redun_dir_u[0] = -1;
			redun_dir_u[1] = +1;
		}

		// for all valid support points do
		for (int32_t u_can=0; u_can<D_can_width; u_can++) {
			for (int32_t v_can=0; v_can<D_can_height; v_can++) {
				int16_t d_can = *(D_can+getAddressOffsetImage(u_can,v_can,D_can_width));
				if (d_can>=0) {

					// check all directions for redundancy
					bool redundant = true;
					for (int32_t i=0; i<2; i++) {

						// search for support
						int32_t u_can_2 = u_can;
						int32_t v_can_2 = v_can;
						int16_t d_can_2;
						bool support = false;
						for (int32_t j=0; j<redun_max_dist; j++) {           //redun_max_dist=5, redun_threshold=1
							u_can_2 += redun_dir_u[i];                                   //此处，也是在10*10的窗口内计算支撑点是否有冗余
							v_can_2 += redun_dir_v[i];
							if (u_can_2<0 || v_can_2<0 || u_can_2>=D_can_width || v_can_2>=D_can_height)  //判断是否越界
								break;
							d_can_2 = *(D_can+getAddressOffsetImage(u_can_2,v_can_2,D_can_width));
							if (d_can_2>=0 && abs(d_can-d_can_2)<=redun_threshold) {
								support = true;
								break;
							}
						}

						// if we have no support => point is not redundant
						if (!support) {
							redundant = false;
							break;
						}
					}

					// invalidate support point if it is redundant
					if (redundant)
						*(D_can+getAddressOffsetImage(u_can,v_can,D_can_width)) = -1;
				}
			}
		}
}

void MBMElas::addCornerSupportPoints(vector<support_pt> &p_support) {

	// list of border points
	vector<support_pt> p_border;   //定义四个边界支撑点
	p_border.push_back(support_pt(0,0,0));
	p_border.push_back(support_pt(0,height-1,0));
	p_border.push_back(support_pt(width-1,0,0));
	p_border.push_back(support_pt(width-1,height-1,0));

	// find closest d
	for (int32_t i=0; i<p_border.size(); i++) {    //对于每一个边界支撑点，查找与其距离最近的一个支撑点，将该支撑点的视差值作为边界支撑点的值
		int32_t best_dist = 10000000;
		for (int32_t j=0; j<p_support.size(); j++) {
			int32_t du = p_border[i].u-p_support[j].u;
			int32_t dv = p_border[i].v-p_support[j].v;
			int32_t curr_dist = du*du+dv*dv;
			if (curr_dist<best_dist) {
				best_dist = curr_dist;
				p_border[i].d = p_support[j].d;
			}
		}
	}

	// for right image
	p_border.push_back(support_pt(p_border[2].u+p_border[2].d,p_border[2].v,p_border[2].d));
	p_border.push_back(support_pt(p_border[3].u+p_border[3].d,p_border[3].v,p_border[3].d));

	// add border points to support points
	for (int32_t i=0; i<p_border.size(); i++)
		p_support.push_back(p_border[i]);
}

vector<MBMElas::support_pt> MBMElas::computeSupportMatches (Mat initDisparity, Mat left) {

	// be sure that at half resolution we only need data
	// from every second line!
	int32_t D_candidate_stepsize = param.candidate_stepsize;

	// create matrix for saving disparity candidates
	int32_t D_can_width  = 0;
	int32_t D_can_height = 0;
	for (int32_t u=0; u<width;  u+=D_candidate_stepsize) D_can_width++;
	for (int32_t v=0; v<height; v+=D_candidate_stepsize) D_can_height++;
	int16_t* D_can = (int16_t*)calloc(D_can_width*D_can_height,sizeof(int16_t));

	// loop variables
	int32_t u,v;
	int16_t d,d2;

	// for all point candidates in image 1 do
	for (int32_t u_can=1; u_can<D_can_width; u_can++) {
		u = u_can*D_candidate_stepsize;
		for (int32_t v_can=1; v_can<D_can_height; v_can++) {
			v = v_can*D_candidate_stepsize;

			// initialize disparity candidate to invalid
			*(D_can+getAddressOffsetImage(u_can,v_can,D_can_width)) = -1;   //v_can*D_can_width+u_can

			if(initDisparity.at<float>(v, u)>=0){
				*(D_can+getAddressOffsetImage(u_can,v_can,D_can_width)) = (int)initDisparity.at<float>(v, u);
			}
		}
	}

	// remove inconsistent support points
	removeInconsistentSupportPoints(D_can,D_can_width,D_can_height);  //比较当前点与周围领域内的点的视差，如果在一个范围内则认为该点稳定，否则不稳定，

	// remove support points on straight lines, since they are redundant
	// this reduces the number of triangles a little bit and hence speeds up
	// the triangulation process
	removeRedundantSupportPoints(D_can,D_can_width,D_can_height,5,1,true);
	removeRedundantSupportPoints(D_can,D_can_width,D_can_height,5,1,false);

	// move support points from image representation into a vector representation
	vector<support_pt> p_support;
	for (int32_t u_can=1; u_can<D_can_width; u_can++)
		for (int32_t v_can=1; v_can<D_can_height; v_can++)
			if (*(D_can+getAddressOffsetImage(u_can,v_can,D_can_width))>=0)
				p_support.push_back(support_pt(u_can*D_candidate_stepsize,   //此处，将支撑点定义成坐标
				v_can*D_candidate_stepsize,
				*(D_can+getAddressOffsetImage(u_can,v_can,D_can_width))));

	// if flag is set, add support points in image corners
	// with the same disparity as the nearest neighbor support point
	addCornerSupportPoints(p_support);

	// free memory
	free(D_can);

	// return support point vector
	return p_support; 
}

vector<MBMElas::triangle> MBMElas::computeDelaunayTriangulation (vector<support_pt> p_support) {

	// input/output structure for triangulation
	struct triangulateio in, out;
	int32_t k;

	// inputs
	in.numberofpoints = p_support.size();
	in.pointlist = (float*)malloc(in.numberofpoints*2*sizeof(float));
	k=0;

	for (int32_t i=0; i<p_support.size(); i++) {
		in.pointlist[k++] = p_support[i].u;
		in.pointlist[k++] = p_support[i].v;
	}


	in.numberofpointattributes = 0;
	in.pointattributelist      = NULL;
	in.pointmarkerlist         = NULL;
	in.numberofsegments        = 0;
	in.numberofholes           = 0;
	in.numberofregions         = 0;
	in.regionlist              = NULL;

	// outputs
	out.pointlist              = NULL;
	out.pointattributelist     = NULL;
	out.pointmarkerlist        = NULL;
	out.trianglelist           = NULL;
	out.triangleattributelist  = NULL;
	out.neighborlist           = NULL;
	out.segmentlist            = NULL;
	out.segmentmarkerlist      = NULL;
	out.edgelist               = NULL;
	out.edgemarkerlist         = NULL;

	// do triangulation (z=zero-based, n=neighbors, Q=quiet, B=no boundary markers)
	char parameters[] = "zQB";
	triangulate(parameters, &in, &out, NULL);

	// put resulting triangles into vector tri
	vector<triangle> tri;
	k=0;
	for (int32_t i=0; i<out.numberoftriangles; i++) {
		tri.push_back(triangle(out.trianglelist[k],out.trianglelist[k+1],out.trianglelist[k+2]));
		k+=3;
	}

	// free memory used for triangulation
	free(in.pointlist);
	free(out.pointlist);
	free(out.trianglelist);

	// return triangles
	return tri;
}

void MBMElas::computeDisparityPlanes (vector<support_pt> p_support,vector<triangle> &tri) {

	// init matrices
	Matrix A(3,3);
	Matrix b(3,1);

	// for all triangles do
	for (int32_t i=0; i<tri.size(); i++) {

		// get triangle corner indices
		int32_t c1 = tri[i].c1;
		int32_t c2 = tri[i].c2;
		int32_t c3 = tri[i].c3;

		// compute matrix A for linear system of left triangle
		A.val[0][0] = p_support[c1].u;
		A.val[1][0] = p_support[c2].u;
		A.val[2][0] = p_support[c3].u;
		A.val[0][1] = p_support[c1].v; A.val[0][2] = 1;
		A.val[1][1] = p_support[c2].v; A.val[1][2] = 1;
		A.val[2][1] = p_support[c3].v; A.val[2][2] = 1;

		// compute vector b for linear system (containing the disparities)
		b.val[0][0] = p_support[c1].d;
		b.val[1][0] = p_support[c2].d;
		b.val[2][0] = p_support[c3].d;

		// on success of gauss jordan elimination
		if (b.solve(A)) {

			// grab results from b
			tri[i].t1a = b.val[0][0];
			tri[i].t1b = b.val[1][0];
			tri[i].t1c = b.val[2][0];

			// otherwise: invalid
		} else {
			tri[i].t1a = 0;
			tri[i].t1b = 0;
			tri[i].t1c = 0;
		}

		// compute matrix A for linear system of right triangle
		A.val[0][0] = p_support[c1].u-p_support[c1].d;
		A.val[1][0] = p_support[c2].u-p_support[c2].d;
		A.val[2][0] = p_support[c3].u-p_support[c3].d;
		A.val[0][1] = p_support[c1].v; A.val[0][2] = 1;
		A.val[1][1] = p_support[c2].v; A.val[1][2] = 1;
		A.val[2][1] = p_support[c3].v; A.val[2][2] = 1;

		// compute vector b for linear system (containing the disparities)
		b.val[0][0] = p_support[c1].d;
		b.val[1][0] = p_support[c2].d;
		b.val[2][0] = p_support[c3].d;

		// on success of gauss jordan elimination
		if (b.solve(A)) {

			// grab results from b
			tri[i].t2a = b.val[0][0];
			tri[i].t2b = b.val[1][0];
			tri[i].t2c = b.val[2][0];

			// otherwise: invalid
		} else {
			tri[i].t2a = 0;
			tri[i].t2b = 0;
			tri[i].t2c = 0;
		}
	}  
}

void MBMElas::createGrid(vector<support_pt> p_support,int32_t* disparity_grid,int32_t* grid_dims) {

	// get grid dimensions
	int32_t grid_width  = grid_dims[1];
	int32_t grid_height = grid_dims[2];

	// allocate temporary memory
	int32_t* temp1 = (int32_t*)calloc((param.disp_max+1)*grid_height*grid_width,sizeof(int32_t));
	int32_t* temp2 = (int32_t*)calloc((param.disp_max+1)*grid_height*grid_width,sizeof(int32_t));

	// for all support points do
	for (int32_t i=0; i<p_support.size(); i++) {

		// compute disparity range to fill for this support point
		int32_t x_curr = p_support[i].u;
		int32_t y_curr = p_support[i].v;
		int32_t d_curr = p_support[i].d;
		int32_t d_min  = max(d_curr-1,0);
		int32_t d_max  = min(d_curr+1,param.disp_max);

		// fill disparity grid helper
		for (int32_t d=d_min; d<=d_max; d++) {
			int32_t x;

			x = floor((float)(x_curr/param.grid_size));

			int32_t y = floor((float)y_curr/(float)param.grid_size);

			// point may potentially lay outside (corner points)
			if (x>=0 && x<grid_width &&y>=0 && y<grid_height) {
				int32_t addr = getAddressOffsetGrid(x,y,d,grid_width,param.disp_max+1);
				*(temp1+addr) = 1;
			}
		}
	}

	// diffusion pointers
	const int32_t* tl = temp1 + (0*grid_width+0)*(param.disp_max+1);
	const int32_t* tc = temp1 + (0*grid_width+1)*(param.disp_max+1);
	const int32_t* tr = temp1 + (0*grid_width+2)*(param.disp_max+1);
	const int32_t* cl = temp1 + (1*grid_width+0)*(param.disp_max+1);
	const int32_t* cc = temp1 + (1*grid_width+1)*(param.disp_max+1);
	const int32_t* cr = temp1 + (1*grid_width+2)*(param.disp_max+1);
	const int32_t* bl = temp1 + (2*grid_width+0)*(param.disp_max+1);
	const int32_t* bc = temp1 + (2*grid_width+1)*(param.disp_max+1);
	const int32_t* br = temp1 + (2*grid_width+2)*(param.disp_max+1);

	int32_t* result    = temp2 + (1*grid_width+1)*(param.disp_max+1); 
	int32_t* end_input = temp1 + grid_width*grid_height*(param.disp_max+1);

	// diffuse temporary grid
	for( ; br != end_input; tl++, tc++, tr++, cl++, cc++, cr++, bl++, bc++, br++, result++ )
		*result = *tl | *tc | *tr | *cl | *cc | *cr | *bl | *bc | *br;

	// for all grid positions create disparity grid
	for (int32_t x=0; x<grid_width; x++) {
		for (int32_t y=0; y<grid_height; y++) {

			// start with second value (first is reserved for count)
			int32_t curr_ind = 1;

			// for all disparities do
			for (int32_t d=0; d<=param.disp_max; d++) {

				// if yes => add this disparity to current cell
				if (*(temp2+getAddressOffsetGrid(x,y,d,grid_width,param.disp_max+1))>0) {
					*(disparity_grid+getAddressOffsetGrid(x,y,curr_ind,grid_width,param.disp_max+2))=d;
					curr_ind++;
				}
			}

			// finally set number of indices
			*(disparity_grid+getAddressOffsetGrid(x,y,0,grid_width,param.disp_max+2))=curr_ind-1;
		}
	}

	// release temporary memory
	free(temp1);
	free(temp2);
}

inline void MBMElas::findMatch(int32_t &u,int32_t &v,float &plane_a,float &plane_b,float &plane_c,
	float* D, Mat initDisparity, Mat maxMncc, Mat leftDisparity){

		// get image width and height
		const int32_t window_size = 2;

		// address of disparity we want to compute
		uint32_t  d_addr = getAddressOffsetImage(u,v,width);

		// check if u is ok
		if (u<window_size || u>=width-window_size)
			return;

		// compute disparity, min disparity and max disparity of plane prior
		int32_t d_plane     = (int32_t)(plane_a*(float)u+plane_b*(float)v+plane_c);   //公式(3)
		float d_init = initDisparity.at<float>(v, u);

		if (d_init>0 && d_init<param.disp_max - 1) 
		{
			*(D+d_addr) = d_init;
			return;
		}

		uchar d_left = leftDisparity.at<uchar>(v, u);
		if (abs(d_plane - d_left)<4)
		{
			*(D+d_addr) = d_left;
		}
		else// if(abs(d_plane - d_left)>param.disp_min+8)
		{
			*(D+d_addr) = MIN(d_left, d_plane);
			//*(D+d_addr) = d_left > d_plane ? d_plane : (d_left+d_plane)/2;
		}
		/*else
		*(D+d_addr) = d_plane; */
}

// TODO: %2 => more elegantly
void MBMElas::computeDisparity(vector<support_pt> p_support,vector<triangle> tri, float* D, Mat initDisparity, Mat maxMncc, Mat leftDisparity) {

	// init disparity image to -10
	for (int32_t i=0; i<width*height; i++)
		*(D+i) = -10;

	// loop variables
	int32_t c1, c2, c3;
	float plane_a,plane_b,plane_c,plane_d;

	// for all triangles do
	for (uint32_t i=0; i<tri.size(); i++) {

		// get plane parameters
		uint32_t p_i = i*3;

		plane_a = tri[i].t1a;
		plane_b = tri[i].t1b;
		plane_c = tri[i].t1c;
		plane_d = tri[i].t2a;


		// triangle corners
		c1 = tri[i].c1;
		c2 = tri[i].c2;
		c3 = tri[i].c3;

		// sort triangle corners wrt. u (ascending)    
		float tri_u[3];

		tri_u[0] = p_support[c1].u;
		tri_u[1] = p_support[c2].u;
		tri_u[2] = p_support[c3].u;

		float tri_v[3] = {p_support[c1].v,p_support[c2].v,p_support[c3].v};

		for (uint32_t j=0; j<3; j++) {
			for (uint32_t k=0; k<j; k++) {
				if (tri_u[k]>tri_u[j]) {
					float tri_u_temp = tri_u[j]; tri_u[j] = tri_u[k]; tri_u[k] = tri_u_temp;
					float tri_v_temp = tri_v[j]; tri_v[j] = tri_v[k]; tri_v[k] = tri_v_temp;
				}
			}
		}

		// rename corners
		float A_u = tri_u[0]; float A_v = tri_v[0];
		float B_u = tri_u[1]; float B_v = tri_v[1];
		float C_u = tri_u[2]; float C_v = tri_v[2];

		// compute straight lines connecting triangle corners
		float AB_a = 0; float AC_a = 0; float BC_a = 0;
		if ((int32_t)(A_u)!=(int32_t)(B_u)) AB_a = (A_v-B_v)/(A_u-B_u);
		if ((int32_t)(A_u)!=(int32_t)(C_u)) AC_a = (A_v-C_v)/(A_u-C_u);
		if ((int32_t)(B_u)!=(int32_t)(C_u)) BC_a = (B_v-C_v)/(B_u-C_u);
		float AB_b = A_v-AB_a*A_u;
		float AC_b = A_v-AC_a*A_u;
		float BC_b = B_v-BC_a*B_u;

		// first part (triangle corner A->B)
		if ((int32_t)(A_u)!=(int32_t)(B_u)) {
			for (int32_t u=max((int32_t)A_u,0); u<min((int32_t)B_u,width); u++){
				{
					int32_t v_1 = (uint32_t)(AC_a*(float)u+AC_b);
					int32_t v_2 = (uint32_t)(AB_a*(float)u+AB_b);
					for (int32_t v=min(v_1,v_2); v<max(v_1,v_2); v++)
					{
						findMatch(u,v,plane_a,plane_b,plane_c, D, initDisparity, maxMncc, leftDisparity);
					}
				}
			}
		}

		// second part (triangle corner B->C)
		if ((int32_t)(B_u)!=(int32_t)(C_u)) {
			for (int32_t u=max((int32_t)B_u,0); u<min((int32_t)C_u,width); u++){
				{
					int32_t v_1 = (uint32_t)(AC_a*(float)u+AC_b);
					int32_t v_2 = (uint32_t)(BC_a*(float)u+BC_b);
					for (int32_t v=min(v_1,v_2); v<max(v_1,v_2); v++)
					{
						findMatch(u,v,plane_a,plane_b,plane_c, D, initDisparity, maxMncc, leftDisparity);
					}
				}
			}
		}

	}
}

void MBMElas::leftRightConsistencyCheck(float* D1,float* D2) {

	// get disparity image dimensions
	int32_t D_width  = width;
	int32_t D_height = height;

	// make a copy of both images
	float* D1_copy = (float*)malloc(D_width*D_height*sizeof(float));
	float* D2_copy = (float*)malloc(D_width*D_height*sizeof(float));
	memcpy(D1_copy,D1,D_width*D_height*sizeof(float));
	memcpy(D2_copy,D2,D_width*D_height*sizeof(float));

	// loop variables
	uint32_t addr,addr_warp;
	float    u_warp_1,u_warp_2,d1,d2;

	// for all image points do
	for (int32_t u=0; u<D_width; u++) {
		for (int32_t v=0; v<D_height; v++) {

			// compute address (u,v) and disparity value
			addr     = getAddressOffsetImage(u,v,D_width);
			d1       = *(D1_copy+addr);

			u_warp_1 = (float)u-d1;

			// check if left disparity is valid
			if (d1>=0 && u_warp_1>=0 && u_warp_1<D_width) {       

				// compute warped image address
				addr_warp = getAddressOffsetImage((int32_t)u_warp_1,v,D_width);

				// if check failed
				if (fabs(*(D2_copy+addr_warp)-d1)>param.lr_threshold)
					*(D1+addr) = -10;

				// set invalid
			} else
				*(D1+addr) = -10;
		}
	}

	// release memory
	free(D1_copy);
	free(D2_copy);
}

void MBMElas::removeSmallSegments (float* D) {

	// get disparity image dimensions
	int32_t D_width        = width;
	int32_t D_height       = height;
	int32_t D_speckle_size = param.speckle_size;

	// allocate memory on heap for dynamic programming arrays
	int32_t *D_done     = (int32_t*)calloc(D_width*D_height,sizeof(int32_t));
	int32_t *seg_list_u = (int32_t*)calloc(D_width*D_height,sizeof(int32_t));
	int32_t *seg_list_v = (int32_t*)calloc(D_width*D_height,sizeof(int32_t));
	int32_t seg_list_count;
	int32_t seg_list_curr;
	int32_t u_neighbor[4];
	int32_t v_neighbor[4];
	int32_t u_seg_curr;
	int32_t v_seg_curr;

	// declare loop variables
	int32_t addr_start, addr_curr, addr_neighbor;

	// for all pixels do
	for (int32_t u=0; u<D_width; u++) {
		for (int32_t v=0; v<D_height; v++) {

			// get address of first pixel in this segment
			addr_start = getAddressOffsetImage(u,v,D_width);

			// if this pixel has not already been processed
			if (*(D_done+addr_start)==0) {

				// init segment list (add first element
				// and set it to be the next element to check)
				*(seg_list_u+0) = u;
				*(seg_list_v+0) = v;
				seg_list_count  = 1;
				seg_list_curr   = 0;

				// add neighboring segments as long as there
				// are none-processed pixels in the seg_list;
				// none-processed means: seg_list_curr<seg_list_count
				while (seg_list_curr<seg_list_count) {

					// get current position from seg_list
					u_seg_curr = *(seg_list_u+seg_list_curr);
					v_seg_curr = *(seg_list_v+seg_list_curr);

					// get address of current pixel in this segment
					addr_curr = getAddressOffsetImage(u_seg_curr,v_seg_curr,D_width);

					// fill list with neighbor positions
					u_neighbor[0] = u_seg_curr-1; v_neighbor[0] = v_seg_curr;
					u_neighbor[1] = u_seg_curr+1; v_neighbor[1] = v_seg_curr;
					u_neighbor[2] = u_seg_curr;   v_neighbor[2] = v_seg_curr-1;
					u_neighbor[3] = u_seg_curr;   v_neighbor[3] = v_seg_curr+1;

					// for all neighbors do
					for (int32_t i=0; i<4; i++) {

						// check if neighbor is inside image
						if (u_neighbor[i]>=0 && v_neighbor[i]>=0 && u_neighbor[i]<D_width && v_neighbor[i]<D_height) {

							// get neighbor pixel address
							addr_neighbor = getAddressOffsetImage(u_neighbor[i],v_neighbor[i],D_width);

							// check if neighbor has not been added yet and if it is valid
							if (*(D_done+addr_neighbor)==0 && *(D+addr_neighbor)>=0) {

								// is the neighbor similar to the current pixel
								// (=belonging to the current segment)
								if (fabs(*(D+addr_curr)-*(D+addr_neighbor))<=param.speckle_sim_threshold) {

									// add neighbor coordinates to segment list
									*(seg_list_u+seg_list_count) = u_neighbor[i];
									*(seg_list_v+seg_list_count) = v_neighbor[i];
									seg_list_count++;            

									// set neighbor pixel in I_done to "done"
									// (otherwise a pixel may be added 2 times to the list, as
									//  neighbor of one pixel and as neighbor of another pixel)
									*(D_done+addr_neighbor) = 1;
								}
							}

						} 
					}

					// set current pixel in seg_list to "done"
					seg_list_curr++;

					// set current pixel in I_done to "done"
					*(D_done+addr_curr) = 1;

				} // end: while (seg_list_curr<seg_list_count)

				// if segment NOT large enough => invalidate pixels
				if (seg_list_count<D_speckle_size) {

					// for all pixels in current segment invalidate pixels
					for (int32_t i=0; i<seg_list_count; i++) {
						addr_curr = getAddressOffsetImage(*(seg_list_u+i),*(seg_list_v+i),D_width);
						*(D+addr_curr) = -10;
					}
				}
			} // end: if (*(I_done+addr_start)==0)

		}
	}

	// free memory
	free(D_done);
	free(seg_list_u);
	free(seg_list_v);
}

void MBMElas::gapInterpolation(float* D) {

	// get disparity image dimensions
	int32_t D_width          = width;
	int32_t D_height         = height;
	int32_t D_ipol_gap_width = param.ipol_gap_width;

	// discontinuity threshold
	float discon_threshold = 3.0;

	// declare loop variables
	int32_t count,addr,v_first,v_last,u_first,u_last;
	float   d1,d2,d_ipol;

	// 1. Row-wise:
	// for each row do
	for (int32_t v=0; v<D_height; v++) {

		// init counter
		count = 0;

		// for each element of the row do
		for (int32_t u=0; u<D_width; u++) {

			// get address of this location
			addr = getAddressOffsetImage(u,v,D_width);

			// if disparity valid
			if (*(D+addr)>=0) {

				// check if speckle is small enough
				if (count>=1 && count<=D_ipol_gap_width) {

					// first and last value for interpolation
					u_first = u-count;
					u_last  = u-1;

					// if value in range
					if (u_first>0 && u_last<D_width-1) {

						// compute mean disparity
						d1 = *(D+getAddressOffsetImage(u_first-1,v,D_width));
						d2 = *(D+getAddressOffsetImage(u_last+1,v,D_width));
						if (fabs(d1-d2)<discon_threshold) d_ipol = (d1+d2)/2;
						else                              d_ipol = min(d1,d2);

						// set all values to d_ipol
						for (int32_t u_curr=u_first; u_curr<=u_last; u_curr++)
							*(D+getAddressOffsetImage(u_curr,v,D_width)) = d_ipol;
					}

				}

				// reset counter
				count = 0;

				// otherwise increment counter
			} else {
				count++;
			}
		}

		// if full size disp map requested

		// extrapolate to the left
		for (int32_t u=0; u<D_width; u++) {

			// get address of this location
			addr = getAddressOffsetImage(u,v,D_width);

			// if disparity valid
			if (*(D+addr)>=0) {
				for (int32_t u2=max(u-D_ipol_gap_width,0); u2<u; u2++)
					*(D+getAddressOffsetImage(u2,v,D_width)) = *(D+addr);
				break;
			}
		}

		// extrapolate to the right
		for (int32_t u=D_width-1; u>=0; u--) {

			// get address of this location
			addr = getAddressOffsetImage(u,v,D_width);

			// if disparity valid
			if (*(D+addr)>=0) {
				for (int32_t u2=u; u2<=min(u+D_ipol_gap_width,D_width-1); u2++)
					*(D+getAddressOffsetImage(u2,v,D_width)) = *(D+addr);
				break;
			}
		}
	}

	// 2. Column-wise:
	// for each column do
	for (int32_t u=0; u<D_width; u++) {

		// init counter
		count = 0;

		// for each element of the column do
		for (int32_t v=0; v<D_height; v++) {

			// get address of this location
			addr = getAddressOffsetImage(u,v,D_width);

			// if disparity valid
			if (*(D+addr)>=0) {

				// check if gap is small enough
				if (count>=1 && count<=D_ipol_gap_width) {

					// first and last value for interpolation
					v_first = v-count;
					v_last  = v-1;

					// if value in range
					if (v_first>0 && v_last<D_height-1) {

						// compute mean disparity
						d1 = *(D+getAddressOffsetImage(u,v_first-1,D_width));
						d2 = *(D+getAddressOffsetImage(u,v_last+1,D_width));
						if (fabs(d1-d2)<discon_threshold) d_ipol = (d1+d2)/2;
						else                              d_ipol = min(d1,d2);

						// set all values to d_ipol
						for (int32_t v_curr=v_first; v_curr<=v_last; v_curr++)
							*(D+getAddressOffsetImage(u,v_curr,D_width)) = d_ipol;
					}

				}

				// reset counter
				count = 0;

				// otherwise increment counter
			} else {
				count++;
			}
		}

		// added extrapolation to top and bottom since bottom rows sometimes stay unlabeled...
		// DS 5/12/2014

		// if full size disp map requested

		// extrapolate towards top
		for (int32_t v=0; v<D_height; v++) {

			// get address of this location
			addr = getAddressOffsetImage(u,v,D_width);

			// if disparity valid
			if (*(D+addr)>=0) {
				for (int32_t v2=max(v-D_ipol_gap_width,0); v2<v; v2++)
					*(D+getAddressOffsetImage(u,v2,D_width)) = *(D+addr);
				break;
			}
		}

		// extrapolate towards the bottom
		for (int32_t v=D_height-1; v>=0; v--) {

			// get address of this location
			addr = getAddressOffsetImage(u,v,D_width);

			// if disparity valid
			if (*(D+addr)>=0) {
				for (int32_t v2=v; v2<=min(v+D_ipol_gap_width,D_height-1); v2++)
					*(D+getAddressOffsetImage(u,v2,D_width)) = *(D+addr);
				break;
			}
		}
	}
}

Mat MBMElas::GuidedFilter(Mat & guidedIm, Mat & pIm, int r1, int r2){

	double eps = 0.00001;

	Mat I, p;
	resize(guidedIm, I, guidedIm.size()/4, 0, 0, INTER_NEAREST);
	resize(pIm, p, pIm.size()/4, 0, 0, INTER_NEAREST);
	int typep = guidedIm.depth();
	//double duration1 = static_cast<double>(getTickCount());
	Mat mean_I;
	boxFilter(I, mean_I, typep, Size(r1, r2));
	Mat mean_II;
	boxFilter(I.mul(I), mean_II, typep, Size(r1, r2));
	Mat var_I = mean_II - mean_I.mul(mean_I);

	Mat mean_p;
	boxFilter(p, mean_p, typep, Size(r1, r2));
	Mat mean_Ip;
	boxFilter(I.mul(p), mean_Ip, typep, Size(r1, r2));
	Mat cov_Ip = mean_Ip - mean_I.mul(mean_p); // this is the covariance of (I, p) in each local patch.

	Mat a = cov_Ip / (var_I + eps); // Eqn. (5) in the paper;
	Mat b = mean_p - a.mul(mean_I); // Eqn. (6) in the paper;

	Mat mean_a;
	boxFilter(a, mean_a, typep, Size(r1, r2));
	Mat mean_b;
	boxFilter(b, mean_b, typep, Size(r1, r2));

	//duration1 = static_cast<double>(getTickCount())-duration1;
	//duration1 /= cv::getTickFrequency(); 
	//cout << "duration1 = " <<duration1 * 1000 << endl;

	Mat mean_aI, mean_bI;
	resize(mean_a, mean_aI, guidedIm.size());
	resize(mean_b, mean_bI, guidedIm.size());

	return mean_aI.mul(guidedIm) + mean_bI;
}
