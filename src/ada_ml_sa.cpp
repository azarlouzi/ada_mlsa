#include "ada_ml_sa.h"
#include "helpers.h"
#include <cmath>
#include <cstdio>
#include <exception>

ML_Threshold::ML_Threshold(const Step&       u,
	                   const Bias&       bias,
	                   long int          N,
	                   double            theta,
	                   double            r,
	                   int               L,
	                   double            scaler,
	                   const Loss_Model& loss_model):
      N(N), L(L), K(int(std::ceil(theta*L))) {
   init();
   double threshold;
   for (int k = 0; k < K; k++) {
      for (int level = 1; level < L+1; level++) {
         for (long int n = 1L; n < N+1L; n++) {
            switch (loss_model.concentration) {
            case power_concentration:
               threshold = scaler*std::pow(u(n), -1./loss_model.p)*std::pow(bias(theta*level*(r-1)+k), 1./r);
	       break;
            default:
               threshold = scaler*std::pow(bias(theta*level*(r-1)+k), 1./r)*std::sqrt(std::log(std::pow(u(n)*std::pow(bias(level+k), 1+theta), -1./2)));
	       break;
            }
	    threshold_array[k][level-1][n-1L] = threshold;
	 }
      }
   }
}

ML_Threshold& ML_Threshold::operator=(const ML_Threshold& threshold) {
   free_up();
   N = threshold.N;
   L = threshold.L;
   K = threshold.K;
   init();
   deep_copy(threshold.threshold_array);
   return *this;
}

ML_Threshold::~ML_Threshold() {
   free_up();
}

void ML_Threshold::verify_threshold_access(int      k,
                                           int      level,
                                           long int n) const {
   if (!((0<=k) && (k<K))) {
      throw Threshold_Access_Exception(level_k, double(k), double(K));
   }
   if (!((1<=level) && (level<=L))) {
      throw Threshold_Access_Exception(level_l, double(level), double(L));
   }
   if (!((1L<=n) && (n<=N))) {
      throw Threshold_Access_Exception(level_n, double(n), double(N));
   }
}

void ML_Threshold::init() {
   if (threshold_array == nullptr) {
      threshold_array = new double**[K];
      for (int k = 0; k < K; k++) {
         threshold_array[k] = new double*[L];
         for (int level = 1; level < L+1; level++) {
	    threshold_array[k][level-1] = new double[N]();
         }
      }
   }
}

void ML_Threshold::free_up() {
   if (threshold_array != nullptr) {
      for (int k = 0; k < K; k++) {
         for (int level = 1; level < L+1; level++) {
	    delete[] threshold_array[k][level-1];
         }
         delete[] threshold_array[k];
      }
      delete[] threshold_array;
   }
   threshold_array = nullptr;
}

void ML_Threshold::deep_copy(double*** array) {
   if (array != nullptr) {
      for (int k = 0; k < K; k++) {
         for (int level = 1; level < L+1; level++) {
            for (long int n = 1L; n < N+1L; n++) {
               threshold_array[k][level-1][n-1L] = array[k][level-1][n-1L];
	    }
         }
      }
   }
}

double ML_Threshold::operator()(int       k,
			        int       level,
			        long int  n) const {
   verify_threshold_access(k, level, n);
   return threshold_array[k][level-1][n-1L];
}

ML_Adapter::ML_Adapter(double theta): theta(theta) {}

void ML_Adapter::adapt(IN OUT ML_Simulations&     ml_simulation,
		       IN     bool                compute_sd_flag,
		       IN     const Refiner&      refiner,
		       IN     const ML_Threshold& threshold,
		       IN     double              xi,
		       IN     int                 level,
		       IN     long int            n) const {
   double& X_fine = ml_simulation.fine;
   double& X_coarse = ml_simulation.coarse;
   double& ms = ml_simulation.ms;
   double Y = ml_simulation.Y;
   int eta = 0;
   int saturation = int(std::ceil(theta*level));
   double criterion;

   double X_tmp_1 = X_coarse;
   double X_tmp_2 = X_fine;
   while(true) {
      if (eta == saturation) {
         break;
      }
      criterion = threshold(eta, level, n);
      if (compute_sd_flag) {
         criterion *= compute_sd(X_fine, ms);
      }
      if (std::abs(X_fine - xi) < criterion) {
         refiner.refine(X_fine, ms, Y, level+eta);
         eta++;
         X_coarse = X_tmp_1;
         X_tmp_1 = X_tmp_2;
         X_tmp_2 = X_fine;
      } else {
         break;
      }
   }
}

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
                                 OUT int&              L) {
   verify_power_concentration(loss_model);
   verify_power_concentration_adaptivity(loss_model);

   // Configuring L
   L = a_ml_sa_optimal_levels(accuracy, h_0, theta, M);

   // Configuring biases (h_l)_{0 <= l <= L}
   configure_h(h_0, M, L, h);

   // Configuring iterations amounts (N_l)_{0 <= l <= L}
   double tmp = 0;
   for (int l = 0; l < L+1; l++) {
      switch (loss_model.concentration) {
      case power_concentration:
	 if (loss_model.delta < beta) {
            tmp += std::pow(h[l], ((3*(1+theta) - 2*loss_model.delta)*loss_model.p*loss_model.p + (2*(1+theta) + loss_model.delta*(1+3*theta))*loss_model.p + 2*loss_model.delta*(1+theta))/(2*(1 + loss_model.p)*(loss_model.delta + (1+loss_model.delta)*loss_model.p)));
	 } else {
            tmp += std::pow(h[l], -((2*beta - (1+theta))*loss_model.p + (2*beta - (1+theta)*loss_model.delta))*loss_model.p/(2*(1 + loss_model.p)*(loss_model.delta + (1+beta)*loss_model.p)));
         }
         break;
      case gaussian_concentration:
         tmp += std::pow(h[l], -(2*beta - (1+theta))/(2*(1+beta)))*std::pow(std::abs(std::log(h[l])), (1+theta)/(2*(1+beta)));
	 break;
      case lipschitz_concentration:
         tmp += std::pow(h[l], -(2*beta - (1+theta))/(2*(1+beta)));
         break;
      }
   }

   switch (loss_model.concentration) {
   case power_concentration:
      if (loss_model.delta < beta) {
         tmp = std::pow(tmp, 1./loss_model.delta);
         tmp *= std::pow(accuracy, -2./loss_model.delta);
      } else {
         tmp = std::pow(tmp, 1./beta);
         tmp *= std::pow(accuracy, -2./beta);
      }
      break;
   case gaussian_concentration:
      tmp = std::pow(tmp, 1./beta);
      tmp *= std::pow(accuracy, -2./beta);
      break;
   case lipschitz_concentration:
      tmp = std::pow(tmp, 1./beta);
      tmp *= std::pow(accuracy, -2./beta);
      break;
   }

   for (int l = 0; l < L+1; l++) {
      switch (loss_model.concentration) {
      case power_concentration:
         if (loss_model.delta < beta) {
            N[l] = (long int) std::ceil(scaler * tmp * std::pow(h[l], ((5+3*theta)*loss_model.p+4+2*theta)*loss_model.p/(2*(1+loss_model.p)*(loss_model.delta+(1+loss_model.delta)*loss_model.p))));
         } else {
            N[l] = (long int) std::ceil(scaler * tmp * std::pow(h[l], (2+(3+theta)*loss_model.p)*loss_model.p/(2*(1+loss_model.p)*(loss_model.delta+(1+beta)*loss_model.p))));
	 }
         break;
      case gaussian_concentration:
         N[l] = (long int) std::ceil(scaler * tmp * std::pow(h[l], (3+theta)/(2*(1+beta))) * std::pow(std::abs(std::log(h[l])), (1+theta)/(2*(1+beta))));
         break;
      case lipschitz_concentration:
         N[l] = (long int) std::ceil(scaler * tmp * std::pow(h[l], (3+theta)/(2*(1+beta))));
         break;
      }
   }

   // Configuring thresholds (C * psi_{k,l}^n)_{k,l,n}
   Bias bias(h_0, M); // Bias function s -> h_0 / M^s
   Gamma u_amlsa;
   if (loss_model.concentration == power_concentration) {
      u_amlsa = Gamma(gamma_0, loss_model.delta, smoothing);
   } else {
      u_amlsa = Gamma(gamma_0, beta, smoothing);
   }
   threshold = ML_Threshold(u_amlsa, bias, N[1], theta, r, L, threshold_scaler, loss_model);
}

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
                      IN OUT ML_Simulator&       ml_simulator) {
   double xi[L+1][2];
   for (int l = 0; l < L+1; l++) {
      xi[l][0] = xi_0;
      xi[l][1] = xi_0;
   }

   simulator.set_bias_parameter(h[0]);
   Nested_Simulation X;
   for (long int i = 0L; i < N[0]; i++) {
      X = simulator();
      xi[0][0] = xi[0][0] - step(i+1L)*H_1(alpha, xi[0][0], X.X);
   }

   ML_Simulations X_ml;
   for (int l = 1; l < L+1; l++) {
      ml_simulator.set_bias_parameters(h[l-1], h[l]);
      for (long int i = 0L; i < N[l]; i++) {
         X_ml = ml_simulator();
	 adapter.adapt(X_ml, compute_sd_flag, refiner, threshold, xi[l][1], l, i+1L);
         xi[l][0] = xi[l][0] - step(i+1L)*H_1(alpha, xi[l][0], X_ml.coarse);
         xi[l][1] = xi[l][1] - step(i+1L)*H_1(alpha, xi[l][1], X_ml.fine);
      }
   }

   double xi_aml = xi[0][0];
   for (int l = 1; l < L+1; l++) {
      xi_aml += xi[l][1] - xi[l][0];
   }
   return xi_aml;
}

int a_ml_sa_optimal_levels(double accuracy, double h_0, double theta, double M) {
   verify_optimal_level(accuracy, h_0);
   return int(std::ceil(std::log(h_0/accuracy)/((1+theta)*std::log(M))));
}
