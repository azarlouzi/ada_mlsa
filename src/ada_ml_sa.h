#ifndef _ADA_ML_SA_
#define _ADA_ML_SA_

#include "adaptivity.h"
#include "ml_sa.h"
#include "nested_sa.h"
#include "sa.h"

class ML_Threshold {
public:
   ML_Threshold() {}
   ML_Threshold(const Step&       u,
	        const Bias&       bias,
	        long int          N,
	        double            theta,
	        double            r,
	        int               L,
	        double            scaler,
	        const Loss_Model& loss_model);
   ML_Threshold& operator=(const ML_Threshold& threshold);
   ~ML_Threshold();
   double operator()(int       k,
                     int       level,
                     long int  n) const;
private:
   double*** threshold_array = nullptr;
   long int N;
   int L;
   int K;
   void verify_threshold_access(int k, int level, long int n) const;
   void init();
   void free_up();
   void deep_copy(double*** array);
};

class ML_Adapter {
public:
   ML_Adapter(double theta);
   void adapt(IN OUT ML_Simulations&     ml_simulation,
              IN     bool                compute_sd_flag,
              IN     const Refiner&      refiner,
              IN     const ML_Threshold& threshold,
              IN     double              xi,
              IN     int                 level,
              IN     long int            n) const;
private:
   double theta;
};

void configure_adaptive_ml_sa(IN     double            beta,
                              IN     double            h_0,
                              IN     double            M,
                              IN     double            accuracy,
                              IN     double            scaler,
                              IN     const Loss_Model& loss_model,
	                      IN     double            r,
		              IN     double            theta,
			      IN     double            gamma_0,
			      IN     long int          smoothing,
			      IN     double            threshold_scaler,
                                 OUT double*           h,
                                 OUT long int*         N,
				 OUT ML_Threshold&     threshold,
                                 OUT int&              L);

double adaptive_ml_sa(IN     double              xi_0,
                      IN     double              alpha,
                      IN     int                 L,
                      IN     const double*       h,
                      IN     const long int*     N,
                      IN     const Step&         step,
                      IN     const ML_Threshold& threshold,
	              IN     const ML_Adapter&   adapter,
		      IN     const Refiner&      refiner,
		      IN     bool                compute_sd_flag,
                      IN OUT Nested_Simulator&   simulator,
                      IN OUT ML_Simulator&       ml_simulator);

int a_ml_sa_optimal_levels(double accuracy, double h_0, double theta, double M);

#endif // _ADA_ML_SA_
