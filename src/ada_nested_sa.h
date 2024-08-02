#ifndef _ADA_NESTED_SA_
#define _ADA_NESTED_SA_

#include "adaptivity.h"
#include "nested_sa.h"
#include "sa.h"

class Nested_Threshold {
public:
   Nested_Threshold() {}
   Nested_Threshold(const Step&       u,
	            const Bias&       bias,
	            long int          N,
	            double            theta,
	            double            r,
	            int               level,
	            double            scaler,
	            const Loss_Model& loss_model);
   Nested_Threshold& operator=(const Nested_Threshold& threshold);
   ~Nested_Threshold();
   double operator()(int       k,
                     long int  n) const;
private:
   double** threshold_array = nullptr;
   long int N;
   int level;
   int K;
   void verify_threshold_access(int k, long int n) const;
   void init();
   void free_up();
   void deep_copy(double** array);
};

class Nested_Adapter {
public:
   Nested_Adapter(double theta);
   void adapt(IN OUT Nested_Simulation&      nested_simulation,
              IN     bool                    compute_sd_flag,
              IN     const Refiner&          refiner,
              IN     const Nested_Threshold& threshold,
              IN     double                  xi,
              IN     int                     level,
              IN     long int                n) const;
private:
   double theta;
};

void configure_adaptive_nested_sa(IN     double            beta,
                                  IN     double            h_0,
                                  IN     double            M,
                                  IN     double            accuracy,
				  IN     const Step&       step,
                                  IN     double            scaler,
                                  IN     const Loss_Model& loss_model,
	                          IN     double            r,
		                  IN     double            theta,
			          IN     double            gamma_0,
			          IN     long int          smoothing,
			          IN     double            threshold_scaler,
				     OUT long int&         n,
			             OUT Nested_Threshold& threshold,
                                     OUT int&              level);

double adaptive_nested_sa(IN     double                  xi_0,
                          IN     double                  alpha,
			  IN     double                  h_0,
                          IN     double                  M,
                          IN     int                     level,
                          IN     long int                n,
                          IN     const Step&             step,
		          IN     bool                    compute_sd_flag,
                          IN     const Nested_Threshold& threshold,
	                  IN     const Nested_Adapter&   adapter,
		          IN     const Refiner&          refiner,
                          IN OUT Nested_Simulator&       simulator);

int a_nested_sa_optimal_level(double accuracy,
                              double h_0,
                              double M,
                              double theta);

long int a_nested_sa_optimal_steps(double            accuracy,
                                   const Loss_Model& loss_model,
		                   const Step&       step,
			           const Step&       u,
			           double            scaler);

#endif // _ADA_NESTED_SA_
