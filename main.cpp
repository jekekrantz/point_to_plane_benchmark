#include <stdio.h>
#include <vector>
#include <string>
#include <cmath>
#include <Eigen/Dense>
#include <sys/time.h>
#include <pmmintrin.h>
#include <immintrin.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>

#include <omp.h>

#include <thread>

# include <stdio.h>
# include <omp.h>
# include <limits>
# include <cmath>
# include <cassert>
# define NUMBER_THREADS 2 

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

const double olp = 0.1;
__m256       olp_avx =  _mm256_set1_ps(olp);

const double fast_exp_threshold = -16;
__m256       fast_exp_threshold_avx = _mm256_set1_ps(fast_exp_threshold);
__m256       zero_avx = _mm256_set1_ps(0);
__m256       one_avx = _mm256_set1_ps(1);
__m256       inv1024_avx = _mm256_set1_ps(1.0/1024.0);
const double std_div = 0.05;
const double exp_weight = -0.5/(std_div*std_div);
__m256       exp_weight_avx = _mm256_set1_ps(exp_weight);


inline __m256 ADD(const __m256 A, const __m256 B){return _mm256_add_ps(A, B);}
inline __m256 SUB(const __m256 A, const __m256 B){return _mm256_sub_ps(A, B);}
inline __m256 MUL(const __m256 A, const __m256 B){return _mm256_mul_ps(A, B);}
inline __m256 DIV(const __m256 A, const __m256 B){return _mm256_div_ps(A, B);}
inline double SUM(const __m256 A){
  float b [8];
  _mm256_storeu_ps(b, A);
  return b[0]+b[1]+b[2]+b[3]+b[4]+b[5]+b[6]+b[7];
}

inline double GetTime()
{
  struct timeval start1;
	gettimeofday(&start1, NULL);
	return double(start1.tv_sec+(start1.tv_usec/1000000.0));
}

inline void Normalize(pcl::PointXYZRGBNormal & p)
{
  double n = sqrt(p.normal_x*p.normal_x+
                  p.normal_y*p.normal_y+
                  p.normal_z*p.normal_z);
       p.normal_x /= n;
       p.normal_y /= n;
       p.normal_z /= n;
}

inline double GetWeight(double d)
{
  double inlp = exp(exp_weight*d*d);
  return inlp/(inlp+olp);
}

inline double fast_exp(double x)
{
  if(x < fast_exp_threshold)
    return 0;
  x = 1.0 + x / 1024;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x;
  return x;
}

inline double GetWeightFast(double d)
{
  double inlp = fast_exp(exp_weight*d*d);
  return inlp/(inlp+olp);
}


inline __m256 fast_exp_avx(__m256 x)
{
  x = _mm256_max_ps(x, fast_exp_threshold_avx);
  x = ADD( one_avx , MUL(x , inv1024_avx));
  x = MUL(x,x);
  x = MUL(x,x);
  x = MUL(x,x);
  x = MUL(x,x);
  x = MUL(x,x);
  x = MUL(x,x);
  x = MUL(x,x);
  x = MUL(x,x);
  x = MUL(x,x);
  x = MUL(x,x);
  return x;
}

inline __m256 GetWeightFastAVX(__m256 d)
{
  __m256 inlp = fast_exp_avx(MUL(exp_weight_avx,MUL(d,d)));
  return DIV(inlp,ADD(inlp,olp_avx));
}

std::default_random_engine generator;

struct TestData
{
 public:
  pcl::PointCloud<pcl::PointXYZRGBNormal> src;
  pcl::PointCloud<pcl::PointXYZRGBNormal> dst;
  std::vector< size_t > src_matches;
  std::vector< size_t > dst_matches;

  TestData(size_t nr_matches, size_t nr_data, double outlier_ratio = 0.5, bool randomize_order = false)
  {
    src.points.resize(nr_data);
    dst.points.resize(nr_data);
    for(size_t i = 0; i < nr_data;++i)
    {
      pcl::PointXYZRGBNormal p;
      p.x = 2.0*double(rand())/double(RAND_MAX) - 1.0;
      p.y = 2.0*double(rand())/double(RAND_MAX) - 1.0;
      p.z = 2.0*double(rand())/double(RAND_MAX) - 1.0;
      p.normal_x = 2.0*double(rand())/double(RAND_MAX) - 1.0;
      p.normal_y = 2.0*double(rand())/double(RAND_MAX) - 1.0;
      p.normal_z = 2.0*double(rand())/double(RAND_MAX) - 1.0;
      Normalize(p);
      src.points[i] = p;

      if(double(rand())/double(RAND_MAX) < outlier_ratio )
      {
        pcl::PointXYZRGBNormal p;
        p.x = 2.0*double(rand())/double(RAND_MAX) - 1.0;
        p.y = 2.0*double(rand())/double(RAND_MAX) - 1.0;
        p.z = 2.0*double(rand())/double(RAND_MAX) - 1.0;
        p.normal_x = 2.0*double(rand())/double(RAND_MAX) - 1.0;
        p.normal_y = 2.0*double(rand())/double(RAND_MAX) - 1.0;
        p.normal_z = 2.0*double(rand())/double(RAND_MAX) - 1.0;
        Normalize(p);
        dst.points[i] = p;
      }
      else
      {
        std::normal_distribution<double> pt_distribution(0.0,0.0000001);
        std::normal_distribution<double> n_distribution(0.0,0.1);
        p.x += pt_distribution(generator);
        p.y += pt_distribution(generator);
        p.z += pt_distribution(generator);
        p.normal_x += n_distribution(generator);
        p.normal_y += n_distribution(generator);
        p.normal_z += n_distribution(generator);
        Normalize(p);
        dst.points[i] = p;
      }
    }
    src_matches.resize(nr_matches);
    dst_matches.resize(nr_matches);
    
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(0.1*M_PI, Eigen::Vector3f::UnitX())
      * Eigen::AngleAxisf(0.0*M_PI, Eigen::Vector3f::UnitY())
      * Eigen::AngleAxisf(0.0*M_PI, Eigen::Vector3f::UnitZ());
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block(0,0,3,3) = m;
    pcl::PointCloud<pcl::PointXYZRGBNormal> transformed_cloud;
    pcl::transformPointCloud (dst, transformed_cloud, T);
    dst = transformed_cloud;
  
    src_matches.resize(nr_data);
    dst_matches.resize(nr_data);
    for(size_t i = 0; i < nr_data;++i)
    {
      src_matches[i] = i; 
      dst_matches[i] = i; 
    }

    if(randomize_order)
    {
      for(size_t i = 0; i < nr_data;++i)
      {
        const size_t src_rand_id = rand()%nr_data;
        const size_t src_matches_tmp = src_matches[src_rand_id];
        src_matches[src_rand_id] = src_matches[i];
        src_matches[i] = src_matches_tmp;

        const size_t dst_rand_id = rand()%nr_data;
        const size_t dst_matches_tmp = dst_matches[dst_rand_id];
        dst_matches[dst_rand_id] = dst_matches[i];
        dst_matches[i] = dst_matches_tmp;
      }

      pcl::PointCloud<pcl::PointXYZRGBNormal> tmpsrc = src;
      pcl::PointCloud<pcl::PointXYZRGBNormal> tmpdst = dst;
      for(unsigned int i=0; i < nr_matches; ++i)
      {
        src.points[src_matches[i]] = tmpsrc.points[i];
        dst.points[dst_matches[i]] = tmpdst.points[i];
      }
    }
    src_matches.resize(nr_matches);
    dst_matches.resize(nr_matches);
  }
};

float * GetCompact(const TestData & testdata)
{
  const size_t nr_matches = testdata.src_matches.size();
  float * data = new float[nr_matches*9];
 
  size_t ind = 0;
  for(size_t i = 0; i < nr_matches; i++)
  {
    const pcl::PointXYZRGBNormal & sp = testdata.src[testdata.src_matches[i]];
    const pcl::PointXYZRGBNormal & dp = testdata.dst[testdata.dst_matches[i]];
    
    data[ind++] = sp.x;
    data[ind++] = sp.y;
    data[ind++] = sp.z;
    data[ind++] = dp.x;
    data[ind++] = dp.y;
    data[ind++] = dp.z;
    data[ind++] = dp.normal_x;
    data[ind++] = dp.normal_y;
    data[ind++] = dp.normal_z;
  }
  return data;
}

struct TestDataSOA
{
 public:
  float * sx;
  float * sy;
  float * sz;
  float * dx;
  float * dy;
  float * dz;
  float * nx;
  float * ny;
  float * nz;
  size_t nr_matches;

  TestDataSOA(TestData & testdata, size_t threads = 1)
  {
    nr_matches = testdata.src_matches.size();
    sx = new float[nr_matches];
    sy = new float[nr_matches];
    sz = new float[nr_matches];
    dx = new float[nr_matches];
    dy = new float[nr_matches];
    dz = new float[nr_matches];
    nx = new float[nr_matches];
    ny = new float[nr_matches];
    nz = new float[nr_matches];

    if(threads>1)
    {
      #pragma omp parallel for
      for(size_t i = 0; i < nr_matches; i++)
      {
        const pcl::PointXYZRGBNormal & sp = testdata.src[testdata.src_matches[i]];
        const pcl::PointXYZRGBNormal & dp = testdata.dst[testdata.dst_matches[i]];
        
        sx[i] = sp.x;
        sy[i] = sp.y;
        sz[i] = sp.z;
        dx[i] = dp.x;
        dy[i] = dp.y;
        dz[i] = dp.z;
        nx[i] = dp.normal_x;
        ny[i] = dp.normal_y;
        nz[i] = dp.normal_z;
      }
    }
    else
    {
      for(size_t i = 0; i < nr_matches; i++)
      {
        const pcl::PointXYZRGBNormal & sp = testdata.src[testdata.src_matches[i]];
        const pcl::PointXYZRGBNormal & dp = testdata.dst[testdata.dst_matches[i]];
      
        sx[i] = sp.x;
        sy[i] = sp.y;
        sz[i] = sp.z;
        dx[i] = dp.x;
        dy[i] = dp.y;
        dz[i] = dp.z;
        nx[i] = dp.normal_x;
        ny[i] = dp.normal_y;
        nz[i] = dp.normal_z;
      }
    }
  }

  ~TestDataSOA()
  {
    delete sx;
    delete sy;
    delete sz;
    delete dx;
    delete dy;
    delete dz;
    delete nx;
    delete ny;
    delete nz;
  }
};


Eigen::Matrix4d constructTransformationMatrix (const double & alpha, const double & beta, const double & gamma, const double & tx,    const double & ty,   const double & tz)
{
		// Construct the transformation matrix from rotation and translation
		Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Zero ();
		transformation_matrix (0, 0) =  cos (gamma) * cos (beta);
		transformation_matrix (0, 1) = -sin (gamma) * cos (alpha) + cos (gamma) * sin (beta) * sin (alpha);
		transformation_matrix (0, 2) =  sin (gamma) * sin (alpha) + cos (gamma) * sin (beta) * cos (alpha);
		transformation_matrix (1, 0) =  sin (gamma) * cos (beta);
		transformation_matrix (1, 1) =  cos (gamma) * cos (alpha) + sin (gamma) * sin (beta) * sin (alpha);
		transformation_matrix (1, 2) = -cos (gamma) * sin (alpha) + sin (gamma) * sin (beta) * cos (alpha);
		transformation_matrix (2, 0) = -sin (beta);
		transformation_matrix (2, 1) =  cos (beta) * sin (alpha);
		transformation_matrix (2, 2) =  cos (beta) * cos (alpha);

		transformation_matrix (0, 3) = tx;
		transformation_matrix (1, 3) = ty;
		transformation_matrix (2, 3) = tz;
		transformation_matrix (3, 3) = 1;
		return transformation_matrix;
}

Matrix6d GetATA(__m256 * vATAv)
{
  size_t ind = 0;
  Matrix6d ATA;
  ATA.coeffRef (0) = SUM(vATAv[ind++]);
  ATA.coeffRef (1) = SUM(vATAv[ind++]);
  ATA.coeffRef (2) = SUM(vATAv[ind++]);
  ATA.coeffRef (3) = SUM(vATAv[ind++]);
  ATA.coeffRef (4) = SUM(vATAv[ind++]);
  ATA.coeffRef (5) = SUM(vATAv[ind++]);
  ATA.coeffRef (7) = SUM(vATAv[ind++]);
  ATA.coeffRef (8) = SUM(vATAv[ind++]);
  ATA.coeffRef (9) = SUM(vATAv[ind++]);
  ATA.coeffRef (10) = SUM(vATAv[ind++]);
  ATA.coeffRef (11) = SUM(vATAv[ind++]);
  ATA.coeffRef (14) = SUM(vATAv[ind++]);
  ATA.coeffRef (15) = SUM(vATAv[ind++]);
  ATA.coeffRef (16) = SUM(vATAv[ind++]);
  ATA.coeffRef (17) = SUM(vATAv[ind++]);
  ATA.coeffRef (21) = SUM(vATAv[ind++]);
  ATA.coeffRef (22) = SUM(vATAv[ind++]);
  ATA.coeffRef (23) = SUM(vATAv[ind++]);
  ATA.coeffRef (28) = SUM(vATAv[ind++]);
  ATA.coeffRef (29) = SUM(vATAv[ind++]);
  ATA.coeffRef (35) = SUM(vATAv[ind++]);

  ATA.coeffRef (6) = ATA.coeff (1);
  ATA.coeffRef (12) = ATA.coeff (2);
  ATA.coeffRef (13) = ATA.coeff (8);
  ATA.coeffRef (18) = ATA.coeff (3);
  ATA.coeffRef (19) = ATA.coeff (9);
  ATA.coeffRef (20) = ATA.coeff (15);
  ATA.coeffRef (24) = ATA.coeff (4);
  ATA.coeffRef (25) = ATA.coeff (10);
  ATA.coeffRef (26) = ATA.coeff (16);
  ATA.coeffRef (27) = ATA.coeff (22);
  ATA.coeffRef (30) = ATA.coeff (5);
  ATA.coeffRef (31) = ATA.coeff (11);
  ATA.coeffRef (32) = ATA.coeff (17);
  ATA.coeffRef (33) = ATA.coeff (23);
  ATA.coeffRef (34) = ATA.coeff (29);

  return ATA;
}

Matrix6d GetATA(float * ATAv)
{
  size_t ind = 0;
  Matrix6d ATA;
  ATA.coeffRef (0) = ATAv[ind++];
  ATA.coeffRef (1) = ATAv[ind++];
  ATA.coeffRef (2) = ATAv[ind++];
  ATA.coeffRef (3) = ATAv[ind++];
  ATA.coeffRef (4) = ATAv[ind++];
  ATA.coeffRef (5) = ATAv[ind++];
  ATA.coeffRef (7) = ATAv[ind++];
  ATA.coeffRef (8) = ATAv[ind++];
  ATA.coeffRef (9) = ATAv[ind++];
  ATA.coeffRef (10) = ATAv[ind++];
  ATA.coeffRef (11) = ATAv[ind++];
  ATA.coeffRef (14) = ATAv[ind++];
  ATA.coeffRef (15) = ATAv[ind++];
  ATA.coeffRef (16) = ATAv[ind++];
  ATA.coeffRef (17) = ATAv[ind++];
  ATA.coeffRef (21) = ATAv[ind++];
  ATA.coeffRef (22) = ATAv[ind++];
  ATA.coeffRef (23) = ATAv[ind++];
  ATA.coeffRef (28) = ATAv[ind++];
  ATA.coeffRef (29) = ATAv[ind++];
  ATA.coeffRef (35) = ATAv[ind++];

  ATA.coeffRef (6) = ATA.coeff (1);
  ATA.coeffRef (12) = ATA.coeff (2);
  ATA.coeffRef (13) = ATA.coeff (8);
  ATA.coeffRef (18) = ATA.coeff (3);
  ATA.coeffRef (19) = ATA.coeff (9);
  ATA.coeffRef (20) = ATA.coeff (15);
  ATA.coeffRef (24) = ATA.coeff (4);
  ATA.coeffRef (25) = ATA.coeff (10);
  ATA.coeffRef (26) = ATA.coeff (16);
  ATA.coeffRef (27) = ATA.coeff (22);
  ATA.coeffRef (30) = ATA.coeff (5);
  ATA.coeffRef (31) = ATA.coeff (11);
  ATA.coeffRef (32) = ATA.coeff (17);
  ATA.coeffRef (33) = ATA.coeff (23);
  ATA.coeffRef (34) = ATA.coeff (29);

  return ATA;
}

void RunInnerLoopAVX(TestDataSOA * data_ptr, const  Eigen::Matrix4d * Tptr, __m256 * ATAv, __m256 * ATbv, const size_t start, const size_t stop)
{
  Eigen::Matrix4d T = *Tptr;
  TestDataSOA & data = *data_ptr;

  __m256 t00 = _mm256_set1_ps(T(0,0));
  __m256 t01 = _mm256_set1_ps(T(0,1));
  __m256 t02 = _mm256_set1_ps(T(0,2));
  __m256 t03 = _mm256_set1_ps(T(0,3));
  __m256 t10 = _mm256_set1_ps(T(1,0));
  __m256 t11 = _mm256_set1_ps(T(1,1));
  __m256 t12 = _mm256_set1_ps(T(1,2));
  __m256 t13 = _mm256_set1_ps(T(1,3));
  __m256 t20 = _mm256_set1_ps(T(2,0));
  __m256 t21 = _mm256_set1_ps(T(2,1));
  __m256 t22 = _mm256_set1_ps(T(2,2));
  __m256 t23 = _mm256_set1_ps(T(2,3));

  for(size_t i = 0; i < 21; ++i)
    ATAv[i] = _mm256_set1_ps(0);

  for(size_t i = 0; i < 6; ++i)
    ATbv[i] = _mm256_set1_ps(0);

  for(size_t match = start; match < stop; match += 8)
  {
    __m256 x = _mm256_loadu_ps (&(data.sx[match]));
    __m256 y = _mm256_loadu_ps (&(data.sy[match]));
    __m256 z = _mm256_loadu_ps (&(data.sz[match]));

    __m256 dx = _mm256_loadu_ps (&(data.dx[match]));
    __m256 dy = _mm256_loadu_ps (&(data.dy[match]));
    __m256 dz = _mm256_loadu_ps (&(data.dz[match]));

    __m256 nx = _mm256_loadu_ps (&(data.nx[match]));
    __m256 ny = _mm256_loadu_ps (&(data.ny[match]));
    __m256 nz = _mm256_loadu_ps (&(data.nz[match]));

    __m256 sx = ADD( ADD( ADD( MUL(t00,x), MUL(t01,y)), MUL(t02,z)), t03);
    __m256 sy = ADD( ADD( ADD( MUL(t10,x), MUL(t11,y)), MUL(t12,z)), t13);
    __m256 sz = ADD( ADD( ADD( MUL(t20,x), MUL(t21,y)), MUL(t22,z)), t23);


    __m256 d = SUB(SUB(SUB(ADD(ADD(MUL(nx,dx),MUL(ny,dy)),MUL(nz,dz)),MUL(nx,sx)),MUL(ny,sy)),MUL(nz,sz));

    __m256 weight = GetWeightFastAVX(d);

    __m256 a = SUB( MUL(nz,sy), MUL(ny,sz));
    __m256 b = SUB( MUL(nx,sz), MUL(nz,sx));
    __m256 c = SUB( MUL(ny,sx), MUL(nx,sy));

    __m256 weighta = MUL(weight,a);
    __m256 weightb = MUL(weight,b);
    __m256 weightc = MUL(weight,c);
    __m256 weightnx = MUL(weight,nx);
    __m256 weightny = MUL(weight,ny);
    __m256 weightnz = MUL(weight,nz);

    ATAv[0]  = ADD(ATAv[0], MUL(weighta,a));
    ATAv[1]  = ADD(ATAv[1], MUL(weighta,b));
    ATAv[2]  = ADD(ATAv[2], MUL(weighta,c));
    ATAv[3]  = ADD(ATAv[3], MUL(weighta,nx));
    ATAv[4]  = ADD(ATAv[4], MUL(weighta,ny));
    ATAv[5]  = ADD(ATAv[5], MUL(weighta,nz));

    ATAv[6]  = ADD(ATAv[6], MUL(weightb,b));
    ATAv[7]  = ADD(ATAv[7], MUL(weightb,c));
    ATAv[8]  = ADD(ATAv[8], MUL(weightb,nx));
    ATAv[9]  = ADD(ATAv[9], MUL(weightb,ny));
    ATAv[10] = ADD(ATAv[10],MUL(weightb,nz));

    ATAv[11] = ADD(ATAv[11], MUL(weightc,c));
    ATAv[12] = ADD(ATAv[12], MUL(weightc,nx));
    ATAv[13] = ADD(ATAv[13], MUL(weightc,ny));
    ATAv[14] = ADD(ATAv[14], MUL(weightc,nz));

    ATAv[15] = ADD(ATAv[15], MUL(weightnx,nx));
    ATAv[16] = ADD(ATAv[16], MUL(weightnx,ny));
    ATAv[17] = ADD(ATAv[17], MUL(weightnx,nz));

    ATAv[18] = ADD(ATAv[18], MUL(weightny,ny));
    ATAv[19] = ADD(ATAv[19], MUL(weightny,nz));

    ATAv[20] = ADD(ATAv[20], MUL(weightnz,nz));
    
    ATbv[0] = ADD(ATbv[0], MUL(weighta, d));
    ATbv[1] = ADD(ATbv[1], MUL(weightb, d));
    ATbv[2] = ADD(ATbv[2], MUL(weightc, d));
    ATbv[3] = ADD(ATbv[3], MUL(weightnx, d));
    ATbv[4] = ADD(ATbv[4], MUL(weightny, d));
    ATbv[5] = ADD(ATbv[5], MUL(weightnz, d));
  }
}

inline double TestConversion(std::vector<TestData> & test_data, size_t num_threads)
{  
  double time_start = GetTime();
  for(size_t i = 0; i < test_data.size();++i)
    TestDataSOA inp (test_data[i], num_threads);
  double time_stop = GetTime();
  return time_stop-time_start;
}

///////////////////////////////////////////
//////// Multithread alignment
//////// Multithread data construction
//////// float array instead of eigen matrix for accumulation
//////// SOA Representation
//////// Smart convergence
//////// Reverse Order and single loop transformation
//////// Fast Weight 
//////// NO NAN CHECK, 
//////// MANUAL MULTIPLICATION (USE SYMETRIC PROPERTY)
///////////////////////////////////////////

inline void Method11( TestDataSOA & data, size_t num_threads = 1)
{
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  __m256 arrATAv [12][21];
  __m256 arrATbv [12][6];
  const size_t nr_matches = data.nr_matches;

  for(size_t it = 0; it < 30; ++it)
  {
    size_t step = nr_matches / num_threads;
    size_t offset = step%8;
    step -= offset;
    step += 8;

    TestDataSOA * data_ptr = &data;

    #pragma omp prallel for
    for(size_t tid = 0; tid < num_threads; tid++)
      RunInnerLoopAVX(data_ptr, &T, arrATAv[tid],arrATbv[tid], tid*step, std::min(nr_matches-8,(tid+1)*step));
//    std::vector< std::thread > threads;
//    for(size_t tid = 0; tid < num_threads; tid++)
//      threads.push_back(std::thread(RunInnerLoopAVX,data_ptr, &T, arrATAv[tid],arrATbv[tid], tid*step, std::min(nr_matches-8,(tid+1)*step)));

//    for(size_t i = 0; i < threads.size(); ++i)
//      threads[i].join();

    Matrix6d ATA;
    Vector6d ATb;
    ATA.setZero();
    ATb.setZero();

    for(size_t tid = 0; tid < num_threads; tid++)
    {
      ATA += GetATA(arrATAv[tid]);
      for(size_t i = 0; i < 6; ++i)
        ATb[i] += SUM(arrATbv[tid][i]);
    }

    Vector6d x = static_cast<Vector6d> (ATA.inverse () * ATb);

    T = constructTransformationMatrix(x(0,0),x(1,0),x(2,0),x(3,0),x(4,0),x(5,0)) * T;
    
    if(x.array().abs().maxCoeff() < 1e-6)
      break;
  }
}

inline double Test11(std::vector<TestData> & test_data)
{  
  double time_start = GetTime();
  
  for(size_t i = 0; i < test_data.size();++i)
  {
    TestDataSOA inp (test_data[i],true);
    Method11(inp);
  }
  
  double time_stop = GetTime();
  printf("test11 time used: %10.10fs\n",time_stop-time_start);
  return time_stop-time_start;
}

///////////////////////////////////////////
//////// Multithread alignment
//////// Multithread data construction
//////// float array instead of eigen matrix for accumulation
//////// SOA Representation
//////// Smart convergence
//////// Reverse Order and single loop transformation
//////// Fast Weight 
//////// NO NAN CHECK, 
//////// MANUAL MULTIPLICATION (USE SYMETRIC PROPERTY)
///////////////////////////////////////////

inline void Method10(const TestDataSOA & data, const size_t num_threads=6)
{
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  __m256 arrATAv [12][21];
  __m256 arrATbv [12][6];
  const size_t nr_matches = data.nr_matches-data.nr_matches%8;
  omp_set_num_threads(num_threads);

  for(size_t it = 0; it < 30; ++it)
  {
     __m256 t00 = _mm256_set1_ps(T(0,0));
     __m256 t01 = _mm256_set1_ps(T(0,1));
     __m256 t02 = _mm256_set1_ps(T(0,2));
     __m256 t03 = _mm256_set1_ps(T(0,3));
     __m256 t10 = _mm256_set1_ps(T(1,0));
     __m256 t11 = _mm256_set1_ps(T(1,1));
     __m256 t12 = _mm256_set1_ps(T(1,2));
     __m256 t13 = _mm256_set1_ps(T(1,3));
     __m256 t20 = _mm256_set1_ps(T(2,0));
     __m256 t21 = _mm256_set1_ps(T(2,1));
     __m256 t22 = _mm256_set1_ps(T(2,2));
     __m256 t23 = _mm256_set1_ps(T(2,3));
 
    for(size_t tid = 0; tid < 12; ++tid)
    {
      __m256 * ATAv = arrATAv[tid];
      for(size_t i = 0; i < 21; ++i)
        ATAv[i] = _mm256_set1_ps(0);

      __m256 * ATbv = arrATbv[tid];
      for(size_t i = 0; i < 6; ++i)
        ATbv[i] = _mm256_set1_ps(0);
    }

    //TODO add second loop/padd data to deal with last 7 possible floats
    size_t match;
    #pragma omp parallel for
    for(match = 0; match < nr_matches; match += 8)
    {
      size_t tid = omp_get_thread_num();
      __m256 * ATAv = arrATAv[tid];
      __m256 * ATbv = arrATbv[tid];

      __m256 x = _mm256_loadu_ps (&(data.sx[match]));
      __m256 y = _mm256_loadu_ps (&(data.sy[match]));
      __m256 z = _mm256_loadu_ps (&(data.sz[match]));

      __m256 dx = _mm256_loadu_ps (&(data.dx[match]));
      __m256 dy = _mm256_loadu_ps (&(data.dy[match]));
      __m256 dz = _mm256_loadu_ps (&(data.dz[match]));

      __m256 nx = _mm256_loadu_ps (&(data.nx[match]));
      __m256 ny = _mm256_loadu_ps (&(data.ny[match]));
      __m256 nz = _mm256_loadu_ps (&(data.nz[match]));

      __m256 sx = ADD( ADD( ADD( MUL(t00,x), MUL(t01,y)), MUL(t02,z)), t03);
      __m256 sy = ADD( ADD( ADD( MUL(t10,x), MUL(t11,y)), MUL(t12,z)), t13);
      __m256 sz = ADD( ADD( ADD( MUL(t20,x), MUL(t21,y)), MUL(t22,z)), t23);


      __m256 d = SUB(SUB(SUB(ADD(ADD(MUL(nx,dx),MUL(ny,dy)),MUL(nz,dz)),MUL(nx,sx)),MUL(ny,sy)),MUL(nz,sz));

      __m256 weight = GetWeightFastAVX(d);

      __m256 a = SUB( MUL(nz,sy), MUL(ny,sz));
      __m256 b = SUB( MUL(nx,sz), MUL(nz,sx));
      __m256 c = SUB( MUL(ny,sx), MUL(nx,sy));

      __m256 weighta = MUL(weight,a);
      __m256 weightb = MUL(weight,b);
      __m256 weightc = MUL(weight,c);
      __m256 weightnx = MUL(weight,nx);
      __m256 weightny = MUL(weight,ny);
      __m256 weightnz = MUL(weight,nz);

      ATAv[0]  = ADD(ATAv[0], MUL(weighta,a));
      ATAv[1]  = ADD(ATAv[1], MUL(weighta,b));
      ATAv[2]  = ADD(ATAv[2], MUL(weighta,c));
      ATAv[3]  = ADD(ATAv[3], MUL(weighta,nx));
      ATAv[4]  = ADD(ATAv[4], MUL(weighta,ny));
      ATAv[5]  = ADD(ATAv[5], MUL(weighta,nz));

      ATAv[6]  = ADD(ATAv[6], MUL(weightb,b));
      ATAv[7]  = ADD(ATAv[7], MUL(weightb,c));
      ATAv[8]  = ADD(ATAv[8], MUL(weightb,nx));
      ATAv[9]  = ADD(ATAv[9], MUL(weightb,ny));
      ATAv[10] = ADD(ATAv[10],MUL(weightb,nz));

      ATAv[11] = ADD(ATAv[11], MUL(weightc,c));
      ATAv[12] = ADD(ATAv[12], MUL(weightc,nx));
      ATAv[13] = ADD(ATAv[13], MUL(weightc,ny));
      ATAv[14] = ADD(ATAv[14], MUL(weightc,nz));

      ATAv[15] = ADD(ATAv[15], MUL(weightnx,nx));
      ATAv[16] = ADD(ATAv[16], MUL(weightnx,ny));
      ATAv[17] = ADD(ATAv[17], MUL(weightnx,nz));

      ATAv[18] = ADD(ATAv[18], MUL(weightny,ny));
      ATAv[19] = ADD(ATAv[19], MUL(weightny,nz));

      ATAv[20] = ADD(ATAv[20], MUL(weightnz,nz));
      
      ATbv[0] = ADD(ATbv[0], MUL(weighta, d));
      ATbv[1] = ADD(ATbv[1], MUL(weightb, d));
      ATbv[2] = ADD(ATbv[2], MUL(weightc, d));
      ATbv[3] = ADD(ATbv[3], MUL(weightnx, d));
      ATbv[4] = ADD(ATbv[4], MUL(weightny, d));
      ATbv[5] = ADD(ATbv[5], MUL(weightnz, d));
    }
    Matrix6d ATA;
    Vector6d ATb;
    ATA.setZero();
    ATb.setZero();

    for(size_t tid = 0; tid < 12; ++tid)
    {
      ATA += GetATA(arrATAv[tid]);
      for(size_t i = 0; i < 6; ++i)
        ATb[i] += SUM(arrATbv[tid][i]);
    }

    Vector6d x = static_cast<Vector6d> (ATA.inverse () * ATb);
    T = constructTransformationMatrix(x(0,0),x(1,0),x(2,0),x(3,0),x(4,0),x(5,0)) * T;

    if(x.array().abs().maxCoeff() < 1e-6)
      break;
  }
}

std::vector<double> Test10(std::vector<TestData> & test_data, const size_t num_threads_compute = 1, const size_t num_threads_load = 1)
{
  std::vector<double> ret;
  ret.push_back(0);
  ret.push_back(0);
  omp_set_dynamic(0);
  for(size_t i = 0; i < test_data.size();++i)
  {
    double time0 = GetTime();
    omp_set_num_threads(num_threads_load);
    TestDataSOA inp (test_data[i], num_threads_load);
    double time1 = GetTime();
    omp_set_num_threads(num_threads_compute);
    Method10(inp, num_threads_compute);
    double time2 = GetTime();
    ret[0] += time1-time0;
    ret[1] += time2-time1;
  }

  //printf("%3i %3i -> %5.5f %5.5f\n",num_threads_compute, num_threads_load,ret[0],ret[1]);
/*
  double time_start = GetTime();
  std::vector<TestDataSOA> data;

  for(size_t i = 0; i < test_data.size();++i)
    data.push_back;

  ret.push_back(GetTime()-time_start);

  time_start = GetTime();
  if(num_threads_compute > 0)
    for(size_t i = 0; i < test_data.size();++i)
      Method10(data[i], num_threads_compute);
  
  ret.push_back(GetTime()-time_start);
  */
  return ret;
}
 

///////////////////////////////////////////
//////// float array instead of eigen matrix for accumulation
//////// SOA Representation
//////// Smart convergence
//////// Reverse Order and single loop transformation
//////// Fast Weight 
//////// NO NAN CHECK, 
//////// MANUAL MULTIPLICATION (USE SYMETRIC PROPERTY)
///////////////////////////////////////////

inline void Method9(const TestDataSOA & data)
{

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  __m256 ATAv [21];
  __m256 ATbv [6];
  const size_t nr_matches = data.nr_matches;

  for(size_t it = 0; it < 30; ++it)
  {
    
     __m256 t00 = _mm256_set1_ps(T(0,0));
     __m256 t01 = _mm256_set1_ps(T(0,1));
     __m256 t02 = _mm256_set1_ps(T(0,2));
     __m256 t03 = _mm256_set1_ps(T(0,3));
     __m256 t10 = _mm256_set1_ps(T(1,0));
     __m256 t11 = _mm256_set1_ps(T(1,1));
     __m256 t12 = _mm256_set1_ps(T(1,2));
     __m256 t13 = _mm256_set1_ps(T(1,3));
     __m256 t20 = _mm256_set1_ps(T(2,0));
     __m256 t21 = _mm256_set1_ps(T(2,1));
     __m256 t22 = _mm256_set1_ps(T(2,2));
     __m256 t23 = _mm256_set1_ps(T(2,3));

    for(size_t i = 0; i < 21; ++i)
      ATAv[i] = _mm256_set1_ps(0);

    for(size_t i = 0; i < 6; ++i)
      ATbv[i] = _mm256_set1_ps(0);
   
    //TODO add second loop/padd data to deal with last 7 possible floats
    size_t match = 0; 
    for(match = 0; match < nr_matches; match += 8)
    {
      __m256 x = _mm256_loadu_ps (&(data.sx[match]));
      __m256 y = _mm256_loadu_ps (&(data.sy[match]));
      __m256 z = _mm256_loadu_ps (&(data.sz[match]));

      __m256 dx = _mm256_loadu_ps (&(data.dx[match]));
      __m256 dy = _mm256_loadu_ps (&(data.dy[match]));
      __m256 dz = _mm256_loadu_ps (&(data.dz[match]));

      __m256 nx = _mm256_loadu_ps (&(data.nx[match]));
      __m256 ny = _mm256_loadu_ps (&(data.ny[match]));
      __m256 nz = _mm256_loadu_ps (&(data.nz[match]));

      __m256 sx = ADD( ADD( ADD( MUL(t00,x), MUL(t01,y)), MUL(t02,z)), t03);
      __m256 sy = ADD( ADD( ADD( MUL(t10,x), MUL(t11,y)), MUL(t12,z)), t13);
      __m256 sz = ADD( ADD( ADD( MUL(t20,x), MUL(t21,y)), MUL(t22,z)), t23);


      __m256 d = SUB(SUB(SUB(ADD(ADD(MUL(nx,dx),MUL(ny,dy)),MUL(nz,dz)),MUL(nx,sx)),MUL(ny,sy)),MUL(nz,sz));

      __m256 weight = GetWeightFastAVX(d);

      __m256 a = SUB( MUL(nz,sy), MUL(ny,sz));
      __m256 b = SUB( MUL(nx,sz), MUL(nz,sx));
      __m256 c = SUB( MUL(ny,sx), MUL(nx,sy));

      __m256 weighta = MUL(weight,a);
      __m256 weightb = MUL(weight,b);
      __m256 weightc = MUL(weight,c);
      __m256 weightnx = MUL(weight,nx);
      __m256 weightny = MUL(weight,ny);
      __m256 weightnz = MUL(weight,nz);

      ATAv[0]  = ADD(ATAv[0], MUL(weighta,a));
      ATAv[1]  = ADD(ATAv[1], MUL(weighta,b));
      ATAv[2]  = ADD(ATAv[2], MUL(weighta,c));
      ATAv[3]  = ADD(ATAv[3], MUL(weighta,nx));
      ATAv[4]  = ADD(ATAv[4], MUL(weighta,ny));
      ATAv[5]  = ADD(ATAv[5], MUL(weighta,nz));

      ATAv[6]  = ADD(ATAv[6], MUL(weightb,b));
      ATAv[7]  = ADD(ATAv[7], MUL(weightb,c));
      ATAv[8]  = ADD(ATAv[8], MUL(weightb,nx));
      ATAv[9]  = ADD(ATAv[9], MUL(weightb,ny));
      ATAv[10] = ADD(ATAv[10],MUL(weightb,nz));

      ATAv[11] = ADD(ATAv[11], MUL(weightc,c));
      ATAv[12] = ADD(ATAv[12], MUL(weightc,nx));
      ATAv[13] = ADD(ATAv[13], MUL(weightc,ny));
      ATAv[14] = ADD(ATAv[14], MUL(weightc,nz));

      ATAv[15] = ADD(ATAv[15], MUL(weightnx,nx));
      ATAv[16] = ADD(ATAv[16], MUL(weightnx,ny));
      ATAv[17] = ADD(ATAv[17], MUL(weightnx,nz));

      ATAv[18] = ADD(ATAv[18], MUL(weightny,ny));
      ATAv[19] = ADD(ATAv[19], MUL(weightny,nz));

      ATAv[20] = ADD(ATAv[20], MUL(weightnz,nz));
      
      ATbv[0] = ADD(ATbv[0], MUL(weighta, d));
      ATbv[1] = ADD(ATbv[1], MUL(weightb, d));
      ATbv[2] = ADD(ATbv[2], MUL(weightc, d));
      ATbv[3] = ADD(ATbv[3], MUL(weightnx, d));
      ATbv[4] = ADD(ATbv[4], MUL(weightny, d));
      ATbv[5] = ADD(ATbv[5], MUL(weightnz, d));
    }

    Matrix6d ATA = GetATA(ATAv);

    Vector6d ATb;
    for(size_t i = 0; i < 6; ++i)
      ATb[i] = SUM(ATbv[i]);

    Vector6d x = static_cast<Vector6d> (ATA.inverse () * ATb);
    T = constructTransformationMatrix(x(0,0),x(1,0),x(2,0),x(3,0),x(4,0),x(5,0)) * T;

    if(x.array().abs().maxCoeff() < 1e-6)
      break;
  }
}

inline double Test9(std::vector<TestData> & test_data)
{  
  double time_start = GetTime();
  
  for(size_t i = 0; i < test_data.size();++i)
    Method9(TestDataSOA(test_data[i]));
  
  double time_stop = GetTime();
  printf("test9 time used: %10.10fs\n",time_stop-time_start);
  return time_stop-time_start;
}
 

///////////////////////////////////////////
//////// float array instead of eigen matrix for accumulation
//////// SOA Representation
//////// Smart convergence
//////// Reverse Order and single loop transformation
//////// Fast Weight 
//////// NO NAN CHECK, 
//////// MANUAL MULTIPLICATION (USE SYMETRIC PROPERTY)
///////////////////////////////////////////

inline void Method8(const TestDataSOA & data)
{

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  float ATAv [21];
  float ATbv [6];
  const size_t nr_matches = data.nr_matches;

  for(size_t it = 0; it < 30; ++it)
  {
    float t00 = T(0,0); float t01 = T(0,1); float t02 = T(0,2); float t03 = T(0,3);
    float t10 = T(1,0); float t11 = T(1,1); float t12 = T(1,2); float t13 = T(1,3);
    float t20 = T(2,0); float t21 = T(2,1); float t22 = T(2,2); float t23 = T(2,3);

    for(size_t i = 0; i < 21; ++i)
      ATAv[i] = 0;

    for(size_t i = 0; i < 6; ++i)
      ATbv[i] = 0;

    size_t ind = 0;
    for(size_t i=0; i < nr_matches; ++i)
    {
      const float & x = data.sx[i];
      const float & y = data.sy[i];
      const float & z = data.sz[i];

      const float & sx = t00*x + t01*y+t02*z+t03;
      const float & sy = t10*x + t11*y+t12*z+t13;
      const float & sz = t20*x + t21*y+t22*z+t23;
      const float & dx = data.dx[i];
      const float & dy = data.dy[i];
      const float & dz = data.dz[i];
      const float & nx = data.nx[i];
      const float & ny = data.ny[i];
      const float & nz = data.nz[i];

      float d = nx*dx + ny*dy + nz*dz - nx*sx - ny*sy - nz*sz;
      const float weight = GetWeightFast(d);

      const float a = nz*sy - ny*sz;
      const float b = nx*sz - nz*sx;
      const float c = ny*sx - nx*sy;

      ATAv[0] += weight * a * a;
      ATAv[1] += weight * a * b;
      ATAv[2] += weight * a * c;
      ATAv[3] += weight * a * nx;
      ATAv[4] += weight * a * ny;
      ATAv[5] += weight * a * nz;
      ATAv[6] += weight * b * b;
      ATAv[7] += weight * b * c;
      ATAv[8] += weight * b * nx;
      ATAv[9] += weight * b * ny;
      ATAv[10] += weight * b * nz;
      ATAv[11] += weight * c * c;
      ATAv[12] += weight * c * nx;
      ATAv[13] += weight * c * ny;
      ATAv[14] += weight * c * nz;
      ATAv[15] += weight * nx * nx;
      ATAv[16] += weight * nx * ny;
      ATAv[17] += weight * nx * nz;
      ATAv[18] += weight * ny * ny;
      ATAv[19] += weight * ny * nz;
      ATAv[20] += weight * nz * nz;

      d *= weight;

      ATbv[0] += a * d;
      ATbv[1] += b * d;
      ATbv[2] += c * d;
      ATbv[3] += nx * d;
      ATbv[4] += ny * d;
      ATbv[5] += nz * d;
    }

    Matrix6d ATA = GetATA(ATAv);

    Vector6d ATb;
    for(size_t i = 0; i < 6; ++i)
      ATb[i] = ATbv[i];

    Vector6d x = static_cast<Vector6d> (ATA.inverse () * ATb);
    T = constructTransformationMatrix(x(0,0),x(1,0),x(2,0),x(3,0),x(4,0),x(5,0)) * T;

    if(x.array().abs().maxCoeff() < 1e-6)
      break;
  }
}

inline double Test8(std::vector<TestData> & test_data)
{  
  double time_start = GetTime();
  
  for(size_t i = 0; i < test_data.size();++i)
    Method8(TestDataSOA(test_data[i]));
  
  double time_stop = GetTime();
  printf("test8 time used: %10.10fs\n",time_stop-time_start);
  return time_stop-time_start;
}

///////////////////////////////////////////
//////// SOA Representation
//////// Smart convergence
//////// Reverse Order and single loop transformation
//////// Fast Weight 
//////// NO NAN CHECK, 
//////// MANUAL MULTIPLICATION (USE SYMETRIC PROPERTY)
///////////////////////////////////////////

inline void Method7(const TestDataSOA & data)
{
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  Matrix6d ATA;
  Vector6d ATb;
  const size_t nr_matches = data.nr_matches;

  for(size_t it = 0; it < 30; ++it)
  {
    double t00 = T(0,0); double t01 = T(0,1); double t02 = T(0,2); double t03 = T(0,3);
    double t10 = T(1,0); double t11 = T(1,1); double t12 = T(1,2); double t13 = T(1,3);
    double t20 = T(2,0); double t21 = T(2,1); double t22 = T(2,2); double t23 = T(2,3);

    ATA.setZero();
    ATb.setZero();

    size_t ind = 0;
    for(size_t i=0; i < nr_matches; ++i)
    {
      const double & x = data.sx[i];
      const double & y = data.sy[i];
      const double & z = data.sz[i];

      const double & sx = t00*x + t01*y+t02*z+t03;
      const double & sy = t10*x + t11*y+t12*z+t13;
      const double & sz = t20*x + t21*y+t22*z+t23;
      const double & dx = data.dx[i];
      const double & dy = data.dy[i];
      const double & dz = data.dz[i];
      const double & nx = data.nx[i];
      const double & ny = data.ny[i];
      const double & nz = data.nz[i];

      double d = nx*dx + ny*dy + nz*dz - nx*sx - ny*sy - nz*sz;
      const double weight = GetWeightFast(d);

      const double a = nz*sy - ny*sz;
      const double b = nx*sz - nz*sx;
      const double c = ny*sx - nx*sy;

      ATA.coeffRef (0) += weight * a * a;
      ATA.coeffRef (1) += weight * a * b;
      ATA.coeffRef (2) += weight * a * c;
      ATA.coeffRef (3) += weight * a * nx;
      ATA.coeffRef (4) += weight * a * ny;
      ATA.coeffRef (5) += weight * a * nz;
      ATA.coeffRef (7) += weight * b * b;
      ATA.coeffRef (8) += weight * b * c;
      ATA.coeffRef (9) += weight * b * nx;
      ATA.coeffRef (10) += weight * b * ny;
      ATA.coeffRef (11) += weight * b * nz;
      ATA.coeffRef (14) += weight * c * c;
      ATA.coeffRef (15) += weight * c * nx;
      ATA.coeffRef (16) += weight * c * ny;
      ATA.coeffRef (17) += weight * c * nz;
      ATA.coeffRef (21) += weight * nx * nx;
      ATA.coeffRef (22) += weight * nx * ny;
      ATA.coeffRef (23) += weight * nx * nz;
      ATA.coeffRef (28) += weight * ny * ny;
      ATA.coeffRef (29) += weight * ny * nz;
      ATA.coeffRef (35) += weight * nz * nz;

      d *= weight;

      ATb.coeffRef (0) += a * d;
      ATb.coeffRef (1) += b * d;
      ATb.coeffRef (2) += c * d;
      ATb.coeffRef (3) += nx * d;
      ATb.coeffRef (4) += ny * d;
      ATb.coeffRef (5) += nz * d;
    }

    ATA.coeffRef (6) = ATA.coeff (1);
    ATA.coeffRef (12) = ATA.coeff (2);
    ATA.coeffRef (13) = ATA.coeff (8);
    ATA.coeffRef (18) = ATA.coeff (3);
    ATA.coeffRef (19) = ATA.coeff (9);
    ATA.coeffRef (20) = ATA.coeff (15);
    ATA.coeffRef (24) = ATA.coeff (4);
    ATA.coeffRef (25) = ATA.coeff (10);
    ATA.coeffRef (26) = ATA.coeff (16);
    ATA.coeffRef (27) = ATA.coeff (22);
    ATA.coeffRef (30) = ATA.coeff (5);
    ATA.coeffRef (31) = ATA.coeff (11);
    ATA.coeffRef (32) = ATA.coeff (17);
    ATA.coeffRef (33) = ATA.coeff (23);
    ATA.coeffRef (34) = ATA.coeff (29);

    Vector6d x = static_cast<Vector6d> (ATA.inverse () * ATb);
    T = constructTransformationMatrix(x(0,0),x(1,0),x(2,0),x(3,0),x(4,0),x(5,0)) * T;

    if(x.array().abs().maxCoeff() < 1e-6)
      break;
  }
}

inline double Test7(std::vector<TestData> & test_data)
{  
  double time_start = GetTime();
  for(size_t i = 0; i < test_data.size();++i)
    Method7(TestDataSOA(test_data[i]));
  
  double time_stop = GetTime();
  printf("test7 time used: %10.10fs\n",time_stop-time_start);

  return time_stop-time_start;
}



///////////////////////////////////////////
//////// Compact Representation
//////// Smart convergence
//////// Reverse Order and single loop transformation
//////// Fast Weight 
//////// NO NAN CHECK, 
//////// MANUAL MULTIPLICATION (USE SYMETRIC PROPERTY)
///////////////////////////////////////////

inline void Method6(const float * data, const size_t nr_matches)
{

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  Matrix6d ATA;
  Vector6d ATb;

  for(size_t it = 0; it < 30; ++it)
  {
    double t00 = T(0,0); double t01 = T(0,1); double t02 = T(0,2); double t03 = T(0,3);
    double t10 = T(1,0); double t11 = T(1,1); double t12 = T(1,2); double t13 = T(1,3);
    double t20 = T(2,0); double t21 = T(2,1); double t22 = T(2,2); double t23 = T(2,3);

    ATA.setZero();
    ATb.setZero();

    size_t ind = 0;
    for(size_t i=0; i < nr_matches; ++i)
    {
      const double & x = data[ind++];
      const double & y = data[ind++];
      const double & z = data[ind++];

      const double & sx = t00*x + t01*y+t02*z+t03;
      const double & sy = t10*x + t11*y+t12*z+t13;
      const double & sz = t20*x + t21*y+t22*z+t23;
      const double & dx = data[ind++];
      const double & dy = data[ind++];
      const double & dz = data[ind++];
      const double & nx = data[ind++];
      const double & ny = data[ind++];
      const double & nz = data[ind++];

      double d = nx*dx + ny*dy + nz*dz - nx*sx - ny*sy - nz*sz;
      const double weight = GetWeightFast(d);

      const double a = nz*sy - ny*sz;
      const double b = nx*sz - nz*sx;
      const double c = ny*sx - nx*sy;

      ATA.coeffRef (0) += weight * a * a;
      ATA.coeffRef (1) += weight * a * b;
      ATA.coeffRef (2) += weight * a * c;
      ATA.coeffRef (3) += weight * a * nx;
      ATA.coeffRef (4) += weight * a * ny;
      ATA.coeffRef (5) += weight * a * nz;
      ATA.coeffRef (7) += weight * b * b;
      ATA.coeffRef (8) += weight * b * c;
      ATA.coeffRef (9) += weight * b * nx;
      ATA.coeffRef (10) += weight * b * ny;
      ATA.coeffRef (11) += weight * b * nz;
      ATA.coeffRef (14) += weight * c * c;
      ATA.coeffRef (15) += weight * c * nx;
      ATA.coeffRef (16) += weight * c * ny;
      ATA.coeffRef (17) += weight * c * nz;
      ATA.coeffRef (21) += weight * nx * nx;
      ATA.coeffRef (22) += weight * nx * ny;
      ATA.coeffRef (23) += weight * nx * nz;
      ATA.coeffRef (28) += weight * ny * ny;
      ATA.coeffRef (29) += weight * ny * nz;
      ATA.coeffRef (35) += weight * nz * nz;

      d *= weight;

      ATb.coeffRef (0) += a * d;
      ATb.coeffRef (1) += b * d;
      ATb.coeffRef (2) += c * d;
      ATb.coeffRef (3) += nx * d;
      ATb.coeffRef (4) += ny * d;
      ATb.coeffRef (5) += nz * d;
    }

    ATA.coeffRef (6) = ATA.coeff (1);
    ATA.coeffRef (12) = ATA.coeff (2);
    ATA.coeffRef (13) = ATA.coeff (8);
    ATA.coeffRef (18) = ATA.coeff (3);
    ATA.coeffRef (19) = ATA.coeff (9);
    ATA.coeffRef (20) = ATA.coeff (15);
    ATA.coeffRef (24) = ATA.coeff (4);
    ATA.coeffRef (25) = ATA.coeff (10);
    ATA.coeffRef (26) = ATA.coeff (16);
    ATA.coeffRef (27) = ATA.coeff (22);
    ATA.coeffRef (30) = ATA.coeff (5);
    ATA.coeffRef (31) = ATA.coeff (11);
    ATA.coeffRef (32) = ATA.coeff (17);
    ATA.coeffRef (33) = ATA.coeff (23);
    ATA.coeffRef (34) = ATA.coeff (29);

    Vector6d x = static_cast<Vector6d> (ATA.inverse () * ATb);
    T = constructTransformationMatrix(x(0,0),x(1,0),x(2,0),x(3,0),x(4,0),x(5,0)) * T;

    if(x.array().abs().maxCoeff() < 1e-6)
      break;
  }
}

inline double Test6(std::vector<TestData> & test_data)
{  
  double time_start = GetTime();
  const size_t nr_matches = test_data.front().src_matches.size();
  for(size_t i = 0; i < test_data.size();++i)
  {
    float * compact =  GetCompact(test_data[i]);
    Method6(compact,nr_matches);
    delete[] compact;
  }
  double time_stop = GetTime();
  printf("test6 time used: %10.10fs\n",time_stop-time_start);

  return time_stop-time_start;
}


///////////////////////////////////////////
//////// Smart convergence
//////// Reverse Order and single loop transformation
//////// Fast Weight 
//////// NO NAN CHECK, 
//////// MANUAL MULTIPLICATION (USE SYMETRIC PROPERTY)
///////////////////////////////////////////

inline void Method5(const TestData & data)
{

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  unsigned int nr_matches = data.src_matches.size();
  Matrix6d ATA;
  Vector6d ATb;

  for(size_t it = 0; it < 30; ++it)
  {
    double t00 = T(0,0); double t01 = T(0,1); double t02 = T(0,2); double t03 = T(0,3);
    double t10 = T(1,0); double t11 = T(1,1); double t12 = T(1,2); double t13 = T(1,3);
    double t20 = T(2,0); double t21 = T(2,1); double t22 = T(2,2); double t23 = T(2,3);

    pcl::PointCloud<pcl::PointXYZRGBNormal> src = data.src;
    pcl::PointCloud<pcl::PointXYZRGBNormal> dst = data.dst;

    ATA.setZero();
    ATb.setZero();
    for(unsigned int i=0; i < nr_matches; ++i)
    {
      pcl::PointXYZRGBNormal src_p = src.points[data.src_matches[i]];
      pcl::PointXYZRGBNormal dst_p = dst.points[data.dst_matches[i]];

      const double & x = src_p.x;
      const double & y = src_p.y;
      const double & z = src_p.z;

      const double & sx = t00*x + t01*y+t02*z+t03;
      const double & sy = t10*x + t11*y+t12*z+t13;
      const double & sz = t20*x + t21*y+t22*z+t23;
      const double & dx = dst_p.x;
      const double & dy = dst_p.y;
      const double & dz = dst_p.z;
      const double & nx = dst_p.normal_x;
      const double & ny = dst_p.normal_y;
      const double & nz = dst_p.normal_z;

      double d = nx*dx + ny*dy + nz*dz - nx*sx - ny*sy - nz*sz;
      const double weight = GetWeightFast(d);

      const double a = nz*sy - ny*sz;
      const double b = nx*sz - nz*sx;
      const double c = ny*sx - nx*sy;

      ATA.coeffRef (0) += weight * a * a;
      ATA.coeffRef (1) += weight * a * b;
      ATA.coeffRef (2) += weight * a * c;
      ATA.coeffRef (3) += weight * a * nx;
      ATA.coeffRef (4) += weight * a * ny;
      ATA.coeffRef (5) += weight * a * nz;
      ATA.coeffRef (7) += weight * b * b;
      ATA.coeffRef (8) += weight * b * c;
      ATA.coeffRef (9) += weight * b * nx;
      ATA.coeffRef (10) += weight * b * ny;
      ATA.coeffRef (11) += weight * b * nz;
      ATA.coeffRef (14) += weight * c * c;
      ATA.coeffRef (15) += weight * c * nx;
      ATA.coeffRef (16) += weight * c * ny;
      ATA.coeffRef (17) += weight * c * nz;
      ATA.coeffRef (21) += weight * nx * nx;
      ATA.coeffRef (22) += weight * nx * ny;
      ATA.coeffRef (23) += weight * nx * nz;
      ATA.coeffRef (28) += weight * ny * ny;
      ATA.coeffRef (29) += weight * ny * nz;
      ATA.coeffRef (35) += weight * nz * nz;

      d *= weight;

      ATb.coeffRef (0) += a * d;
      ATb.coeffRef (1) += b * d;
      ATb.coeffRef (2) += c * d;
      ATb.coeffRef (3) += nx * d;
      ATb.coeffRef (4) += ny * d;
      ATb.coeffRef (5) += nz * d;
    }

    ATA.coeffRef (6) = ATA.coeff (1);
    ATA.coeffRef (12) = ATA.coeff (2);
    ATA.coeffRef (13) = ATA.coeff (8);
    ATA.coeffRef (18) = ATA.coeff (3);
    ATA.coeffRef (19) = ATA.coeff (9);
    ATA.coeffRef (20) = ATA.coeff (15);
    ATA.coeffRef (24) = ATA.coeff (4);
    ATA.coeffRef (25) = ATA.coeff (10);
    ATA.coeffRef (26) = ATA.coeff (16);
    ATA.coeffRef (27) = ATA.coeff (22);
    ATA.coeffRef (30) = ATA.coeff (5);
    ATA.coeffRef (31) = ATA.coeff (11);
    ATA.coeffRef (32) = ATA.coeff (17);
    ATA.coeffRef (33) = ATA.coeff (23);
    ATA.coeffRef (34) = ATA.coeff (29);

    Vector6d x = static_cast<Vector6d> (ATA.inverse () * ATb);
    T = constructTransformationMatrix(x(0,0),x(1,0),x(2,0),x(3,0),x(4,0),x(5,0)) * T;

    if(x.array().abs().maxCoeff() < 1e-6)
      break;
  }
}

inline double Test5(std::vector<TestData> & test_data)
{
  double time_start = GetTime();
  for(size_t i = 0; i < test_data.size();++i)
    Method5(test_data[i]);

  double time_stop = GetTime();
  printf("test5 time used: %10.10fs\n",time_stop-time_start);
  return time_stop-time_start;
}

///////////////////////////////////////////
//////// Reverse Order and single loop transformation
//////// Fast Weight 
//////// NO NAN CHECK, 
//////// MANUAL MULTIPLICATION (USE SYMETRIC PROPERTY)
///////////////////////////////////////////

inline void Method4(const TestData & data)
{

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  unsigned int nr_matches = data.src_matches.size();
  Matrix6d ATA;
  Vector6d ATb;

  for(size_t it = 0; it < 30; ++it)
  {
    double t00 = T(0,0); double t01 = T(0,1); double t02 = T(0,2); double t03 = T(0,3);
    double t10 = T(1,0); double t11 = T(1,1); double t12 = T(1,2); double t13 = T(1,3);
    double t20 = T(2,0); double t21 = T(2,1); double t22 = T(2,2); double t23 = T(2,3);

    pcl::PointCloud<pcl::PointXYZRGBNormal> src = data.src;
    pcl::PointCloud<pcl::PointXYZRGBNormal> dst = data.dst;

    ATA.setZero();
    ATb.setZero();
    for(unsigned int i=0; i < nr_matches; ++i)
    {
      pcl::PointXYZRGBNormal src_p = src.points[data.src_matches[i]];
      pcl::PointXYZRGBNormal dst_p = dst.points[data.dst_matches[i]];

      const double & x = src_p.x;
      const double & y = src_p.y;
      const double & z = src_p.z;

      const double & sx = t00*x + t01*y+t02*z+t03;
      const double & sy = t10*x + t11*y+t12*z+t13;
      const double & sz = t20*x + t21*y+t22*z+t23;
      const double & dx = dst_p.x;
      const double & dy = dst_p.y;
      const double & dz = dst_p.z;
      const double & nx = dst_p.normal_x;
      const double & ny = dst_p.normal_y;
      const double & nz = dst_p.normal_z;

      double d = nx*dx + ny*dy + nz*dz - nx*sx - ny*sy - nz*sz;
      const double weight = GetWeightFast(d);

      const double a = nz*sy - ny*sz;
      const double b = nx*sz - nz*sx;
      const double c = ny*sx - nx*sy;

      ATA.coeffRef (0) += weight * a * a;
      ATA.coeffRef (1) += weight * a * b;
      ATA.coeffRef (2) += weight * a * c;
      ATA.coeffRef (3) += weight * a * nx;
      ATA.coeffRef (4) += weight * a * ny;
      ATA.coeffRef (5) += weight * a * nz;
      ATA.coeffRef (7) += weight * b * b;
      ATA.coeffRef (8) += weight * b * c;
      ATA.coeffRef (9) += weight * b * nx;
      ATA.coeffRef (10) += weight * b * ny;
      ATA.coeffRef (11) += weight * b * nz;
      ATA.coeffRef (14) += weight * c * c;
      ATA.coeffRef (15) += weight * c * nx;
      ATA.coeffRef (16) += weight * c * ny;
      ATA.coeffRef (17) += weight * c * nz;
      ATA.coeffRef (21) += weight * nx * nx;
      ATA.coeffRef (22) += weight * nx * ny;
      ATA.coeffRef (23) += weight * nx * nz;
      ATA.coeffRef (28) += weight * ny * ny;
      ATA.coeffRef (29) += weight * ny * nz;
      ATA.coeffRef (35) += weight * nz * nz;

      d *= weight;

      ATb.coeffRef (0) += a * d;
      ATb.coeffRef (1) += b * d;
      ATb.coeffRef (2) += c * d;
      ATb.coeffRef (3) += nx * d;
      ATb.coeffRef (4) += ny * d;
      ATb.coeffRef (5) += nz * d;
    }

    ATA.coeffRef (6) = ATA.coeff (1);
    ATA.coeffRef (12) = ATA.coeff (2);
    ATA.coeffRef (13) = ATA.coeff (8);
    ATA.coeffRef (18) = ATA.coeff (3);
    ATA.coeffRef (19) = ATA.coeff (9);
    ATA.coeffRef (20) = ATA.coeff (15);
    ATA.coeffRef (24) = ATA.coeff (4);
    ATA.coeffRef (25) = ATA.coeff (10);
    ATA.coeffRef (26) = ATA.coeff (16);
    ATA.coeffRef (27) = ATA.coeff (22);
    ATA.coeffRef (30) = ATA.coeff (5);
    ATA.coeffRef (31) = ATA.coeff (11);
    ATA.coeffRef (32) = ATA.coeff (17);
    ATA.coeffRef (33) = ATA.coeff (23);
    ATA.coeffRef (34) = ATA.coeff (29);

    Vector6d x = static_cast<Vector6d> (ATA.inverse () * ATb);
    T = constructTransformationMatrix(x(0,0),x(1,0),x(2,0),x(3,0),x(4,0),x(5,0)) * T;
  }
}

inline double Test4(std::vector<TestData> & test_data)
{
  double time_start = GetTime();
  for(size_t i = 0; i < test_data.size();++i)
    Method4(test_data[i]);

  double time_stop = GetTime();
  printf("test4 time used: %10.10fs\n",time_stop-time_start);
  return time_stop-time_start;
}


///////////////////////////////////////////
//////// Fast Weight 
//////// NO NAN CHECK, 
//////// MANUAL MULTIPLICATION (USE SYMETRIC PROPERTY)
///////////////////////////////////////////

inline void Method3(const TestData & data)
{

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  unsigned int nr_matches = data.src_matches.size();
  Matrix6d ATA;
  Vector6d ATb;

  for(size_t it = 0; it < 30; ++it)
  {
    pcl::PointCloud<pcl::PointXYZRGBNormal> dst;
    pcl::transformPointCloud (data.dst, dst, T);

    ATA.setZero();
    ATb.setZero();
    for(unsigned int i=0; i < nr_matches; ++i)
    {
      pcl::PointXYZRGBNormal src_p = data.src.points[data.src_matches[i]];
      pcl::PointXYZRGBNormal dst_p = dst.points[data.dst_matches[i]];

      const double & sx = src_p.x;
      const double & sy = src_p.y;
      const double & sz = src_p.z;
      const double & dx = dst_p.x;
      const double & dy = dst_p.y;
      const double & dz = dst_p.z;
      const double & nx = dst_p.normal_x;
      const double & ny = dst_p.normal_y;
      const double & nz = dst_p.normal_z;

      double d = nx*dx + ny*dy + nz*dz - nx*sx - ny*sy - nz*sz;
      const double weight = GetWeightFast(d);

      const double a = nz*sy - ny*sz;
      const double b = nx*sz - nz*sx;
      const double c = ny*sx - nx*sy;

      ATA.coeffRef (0) += weight * a * a;
      ATA.coeffRef (1) += weight * a * b;
      ATA.coeffRef (2) += weight * a * c;
      ATA.coeffRef (3) += weight * a * nx;
      ATA.coeffRef (4) += weight * a * ny;
      ATA.coeffRef (5) += weight * a * nz;
      ATA.coeffRef (7) += weight * b * b;
      ATA.coeffRef (8) += weight * b * c;
      ATA.coeffRef (9) += weight * b * nx;
      ATA.coeffRef (10) += weight * b * ny;
      ATA.coeffRef (11) += weight * b * nz;
      ATA.coeffRef (14) += weight * c * c;
      ATA.coeffRef (15) += weight * c * nx;
      ATA.coeffRef (16) += weight * c * ny;
      ATA.coeffRef (17) += weight * c * nz;
      ATA.coeffRef (21) += weight * nx * nx;
      ATA.coeffRef (22) += weight * nx * ny;
      ATA.coeffRef (23) += weight * nx * nz;
      ATA.coeffRef (28) += weight * ny * ny;
      ATA.coeffRef (29) += weight * ny * nz;
      ATA.coeffRef (35) += weight * nz * nz;

      d *= weight;

      ATb.coeffRef (0) += a * d;
      ATb.coeffRef (1) += b * d;
      ATb.coeffRef (2) += c * d;
      ATb.coeffRef (3) += nx * d;
      ATb.coeffRef (4) += ny * d;
      ATb.coeffRef (5) += nz * d;
    }

    ATA.coeffRef (6) = ATA.coeff (1);
    ATA.coeffRef (12) = ATA.coeff (2);
    ATA.coeffRef (13) = ATA.coeff (8);
    ATA.coeffRef (18) = ATA.coeff (3);
    ATA.coeffRef (19) = ATA.coeff (9);
    ATA.coeffRef (20) = ATA.coeff (15);
    ATA.coeffRef (24) = ATA.coeff (4);
    ATA.coeffRef (25) = ATA.coeff (10);
    ATA.coeffRef (26) = ATA.coeff (16);
    ATA.coeffRef (27) = ATA.coeff (22);
    ATA.coeffRef (30) = ATA.coeff (5);
    ATA.coeffRef (31) = ATA.coeff (11);
    ATA.coeffRef (32) = ATA.coeff (17);
    ATA.coeffRef (33) = ATA.coeff (23);
    ATA.coeffRef (34) = ATA.coeff (29);

    Vector6d x = -static_cast<Vector6d> (ATA.inverse () * ATb);
    T = constructTransformationMatrix(x(0,0),x(1,0),x(2,0),x(3,0),x(4,0),x(5,0)) * T;
  }
}

inline double Test3(std::vector<TestData> & test_data)
{
  double time_start = GetTime();
  for(size_t i = 0; i < test_data.size();++i)
    Method3(test_data[i]);

  double time_stop = GetTime();
  printf("test3 time used: %10.10fs\n",time_stop-time_start);
  return time_stop-time_start;
}

///////////////////////////////////////////
//////// Fast Weight 
//////// WITH EIGEN, NO NAN CHECK /////////
///////////////////////////////////////////

inline void Method2(const TestData & data)
{
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  unsigned int nr_matches = data.src_matches.size();
  Matrix6d ATA;
  Vector6d ATb;

  for(size_t it = 0; it < 30; ++it)
  {
    pcl::PointCloud<pcl::PointXYZRGBNormal> dst;
    pcl::transformPointCloud (data.dst, dst, T);

    ATA.setZero();
    ATb.setZero();

    for(unsigned int i=0; i < nr_matches; ++i)
    {
      pcl::PointXYZRGBNormal src_p = data.src.points[data.src_matches[i]];
      pcl::PointXYZRGBNormal dst_p = dst.points[data.dst_matches[i]];

      const double & sx = src_p.x;
      const double & sy = src_p.y;
      const double & sz = src_p.z;
      const double & dx = dst_p.x;
      const double & dy = dst_p.y;
      const double & dz = dst_p.z;
      const double & nx = dst_p.normal_x;
      const double & ny = dst_p.normal_y;
      const double & nz = dst_p.normal_z;

      const double d = nx*dx + ny*dy + nz*dz - nx*sx - ny*sy - nz*sz;
      const double weight = GetWeightFast(d);

      Vector6d J;
      J(0) = nz*sy - ny*sz;
      J(1) = nx*sz - nz*sx;
      J(2) = ny*sx - nx*sy;
      J(3) = nx;
      J(4) = ny;
      J(5) = nz;
      ATA += weight * J * J.transpose();
      ATb += weight * d * J;
    }
    Vector6d x = -static_cast<Vector6d> (ATA.inverse () * ATb);
    T = constructTransformationMatrix(x(0,0),x(1,0),x(2,0),x(3,0),x(4,0),x(5,0)) * T;
  }
}

inline double Test2(std::vector<TestData> & test_data)
{
  double time_start = GetTime();
  for(size_t i = 0; i < test_data.size();++i)
    Method2(test_data[i]);

  double time_stop = GetTime();
  printf("test2 time used: %10.10fs\n",time_stop-time_start);
  return time_stop-time_start;
}

///////////////////////////
/////// Fast Weight 
///////////////////////////
inline void Method1(const TestData & data)
{
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  unsigned int nr_matches = data.src_matches.size();
  Matrix6d ATA;
  Vector6d ATb;

  for(size_t it = 0; it < 30; ++it)
  {
    pcl::PointCloud<pcl::PointXYZRGBNormal> dst = data.dst;
    pcl::transformPointCloud (data.dst, dst, T);

    ATA.setZero();
    ATb.setZero();
    for(unsigned int i=0; i < nr_matches; ++i)
    {
      pcl::PointXYZRGBNormal src_p = data.src.points[data.src_matches[i]];
      pcl::PointXYZRGBNormal dst_p = dst.points[data.dst_matches[i]];

      const double & sx = src_p.x;
      const double & sy = src_p.y;
      const double & sz = src_p.z;
      const double & dx = dst_p.x;
      const double & dy = dst_p.y;
      const double & dz = dst_p.z;
      const double & nx = dst_p.normal_x;
      const double & ny = dst_p.normal_y;
      const double & nz = dst_p.normal_z;
      if( std::isnan(sx) ||
          std::isnan(sy) || 
          std::isnan(sz) || 
          std::isnan(dx) || 
          std::isnan(dy) || 
          std::isnan(dz) ||
          std::isnan(nx) ||
          std::isnan(ny) ||
          std::isnan(nz))
      {
        continue;
      }
      
      const double d = nx*dx + ny*dy + nz*dz - nx*sx - ny*sy - nz*sz;
      const double weight = GetWeightFast(d);

      Vector6d J;
      J(0) = nz*sy - ny*sz;
      J(1) = nx*sz - nz*sx;
      J(2) = ny*sx - nx*sy;
      J(3) = nx;
      J(4) = ny;
      J(5) = nz;
      ATA += weight * J * J.transpose();
      ATb += weight * d * J;
    }
    Vector6d x = -static_cast<Vector6d> (ATA.inverse () * ATb);
    T = constructTransformationMatrix(x(0,0),x(1,0),x(2,0),x(3,0),x(4,0),x(5,0)) * T;
  }
}

inline double Test1(std::vector<TestData> & test_data)
{
  double time_start = GetTime();
  for(size_t i = 0; i < test_data.size();++i)
    Method1(test_data[i]);

  double time_stop = GetTime();
  printf("test1 time used: %10.10fs\n",time_stop-time_start);
  return time_stop-time_start;
}


///////////////////////////
///// RAW WITH EIGEN //////
///////////////////////////
inline void Method0(const TestData & data)
{
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  unsigned int nr_matches = data.src_matches.size();
  Matrix6d ATA;
  Vector6d ATb;

  for(size_t it = 0; it < 30; ++it)
  {
    pcl::PointCloud<pcl::PointXYZRGBNormal> dst = data.dst;
    pcl::transformPointCloud (data.dst, dst, T);

    ATA.setZero();
    ATb.setZero();
    for(unsigned int i=0; i < nr_matches; ++i)
    {
      pcl::PointXYZRGBNormal src_p = data.src.points[data.src_matches[i]];
      pcl::PointXYZRGBNormal dst_p = dst.points[data.dst_matches[i]];

      const double & sx = src_p.x;
      const double & sy = src_p.y;
      const double & sz = src_p.z;
      const double & dx = dst_p.x;
      const double & dy = dst_p.y;
      const double & dz = dst_p.z;
      const double & nx = dst_p.normal_x;
      const double & ny = dst_p.normal_y;
      const double & nz = dst_p.normal_z;
      if( std::isnan(sx) ||
          std::isnan(sy) || 
          std::isnan(sz) || 
          std::isnan(dx) || 
          std::isnan(dy) || 
          std::isnan(dz) ||
          std::isnan(nx) ||
          std::isnan(ny) ||
          std::isnan(nz))
      {
        continue;
      }
      
      const double d = nx*dx + ny*dy + nz*dz - nx*sx - ny*sy - nz*sz;
      const double weight = GetWeight(d);

      Vector6d J;
      J(0) = nz*sy - ny*sz;
      J(1) = nx*sz - nz*sx;
      J(2) = ny*sx - nx*sy;
      J(3) = nx;
      J(4) = ny;
      J(5) = nz;
      ATA += weight * J * J.transpose();
      ATb += weight * d * J;
    }
    Vector6d x = -static_cast<Vector6d> (ATA.inverse () * ATb);
    T = constructTransformationMatrix(x(0,0),x(1,0),x(2,0),x(3,0),x(4,0),x(5,0)) * T;
  }
}

inline double Test0(std::vector<TestData> & test_data)
{
  double time_start = GetTime();
  for(size_t i = 0; i < test_data.size();++i)
    Method0(test_data[i]);

  double time_stop = GetTime();
  printf("test0 time used: %10.10fs\n",time_stop-time_start);
  return time_stop-time_start;
}

int main()
{
  for(size_t nrp = 1000; nrp <= 10000000; nrp*=1.25)
  {
    size_t test_count = std::max(size_t(1),40000000/nrp);
    ///printf("NUMBER OF POINTS: %i NUMBER OF TESTS: %i\n",nrp, test_count);
    
    std::vector<TestData> test_data;
    for(size_t i = 0; i < test_count; i++)
      test_data.push_back(TestData(nrp,nrp,0.5,true));
/*
    Test0(test_data);
    Test1(test_data);
    Test2(test_data);
    Test3(test_data);
    Test4(test_data);
    Test5(test_data);
    Test6(test_data);
    Test7(test_data);
    Test8(test_data);
    Test9(test_data);
    Test10(test_data);
    Test11(test_data);
    */

    size_t min_i = 1;
    size_t min_j = 1;
    std::vector<double> ret;
    ret.push_back(1000000);
    ret.push_back(1000000);
    for(size_t i = 1; i <= 12; ++i)
    {
      for(size_t j = 1; j < 12; ++j){
        std::vector<double> r = Test10(test_data,i,j);
        if(r[0]+r[1] < ret[0]+ret[1])
        {
          ret = r;
          min_i = i;
          min_j = j;
        }
      }
    }

 
    ret[0] /= nrp*test_count;
    ret[1] /= nrp*test_count;

    printf("LOAD POINTS: %10i COUNT: %5i -> LOAD(%2i): %5ld COMPUTE(%2i): %5ld FULL: %5ld RATIO: %5.5f\n", nrp, test_count, min_j,size_t(1e10*ret[0]), min_i, size_t(1e10*ret[1]), size_t(1e10*(ret[0]+ret[1])), ret[0]/(ret[0]+ret[1]));
/*
    printf("LOAD POINTS: %10i -> ",nrp);
    std::vector<double> ppp_load;
    for(size_t i = 1; i <= 12; ++i)
    {
      ppp_load.push_back(1e12*TestConversion(test_data,i)/double(nrp*test_count));
      ppp_full.push_back(1e12*Test10(test_data,0,min_load)/double(nrp*test_count));
      if(ppp_load.back() < best_load)
      {
        min_load = i;
        best_load = ppp_load.back();
      }
      printf("%9ld ",(unsigned long)ppp_load.back());
      fflush(stdout);
    }
    printf("\n");
    
    printf("FULL                    -> ",nrp);
    std::vector<double> ppp_full;
    for(size_t i = 1; i <= 12; ++i)
    {
      ppp_full.push_back(1e12*Test10(test_data,i,min_load)/double(nrp*test_count));
      printf("%9ld ",(unsigned long)ppp_load.back());
      fflush(stdout);
   }
    printf("\n");

    printf("RATIO                   -> ",nrp);
    for(size_t i = 1; i <= 12; ++i)
    {
      printf("  %5.5f ", ppp_load[i-1]/ppp_full[min_load-1]);
      fflush(stdout);
    }
    printf("\n\n\n");
    */
    //exit(0);
  }
  return 0;
}
