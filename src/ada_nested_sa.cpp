#include "ada_nested_sa.h"
#include <cmath>

Nested_Threshold::Nested_Threshold(const Step&       u,
                                   const Bias&       bias,
                                   long int          N,
                                   double            theta,
                                   double            r,
                                   int               level,
                                   double            scaler,
                                   const Loss_Model& loss_model):
      N(N), level(level), K(int(std::ceil(theta*level))) {
   init();
   double threshold;
   for (int k = 0; k < K; k++) {
     for (long int n = 1L; n < N+1L; n++) {
        switch (loss_model.concentration) {
        case power_concentration:
           threshold = scaler*std::pow(u(n), -1./loss_model.p)*std::pow(bias(theta*level*(r-1)+k), 1./r);
           break;
        default:
           threshold = scaler*std::pow(bias(theta*level*(r-1)+k), 1./r)*std::sqrt(std::log(std::pow(u(n)*std::pow(bias(level+k), 1+theta), -1./2)));
           break;
        }
        threshold_array[k][n-1L] = threshold;
     }
  }
}

Nested_Threshold& Nested_Threshold::operator=(const Nested_Threshold& threshold) {
   free_up();
   N = threshold.N;
   level = threshold.level;
   K = threshold.K;
   init();
   deep_copy(threshold.threshold_array);
   return *this;
}

Nested_Threshold::~Nested_Threshold() {
   free_up();
}

double Nested_Threshold::operator()(int       k,
			            long int  n) const {
   verify_threshold_access(k, n);
   return threshold_array[k][n-1L];
}

void Nested_Threshold::verify_threshold_access(int      k,
                                               long int n) const {
   if (!((0<=k) && (k<K))) {
      throw Threshold_Access_Exception(level_k, double(k), double(K));
   }
   if (!((1L<=n) && (n<=N))) {
      throw Threshold_Access_Exception(level_n, double(n), double(N));
   }
}

void Nested_Threshold::init() {
   if (threshold_array == nullptr) {
      threshold_array = new double*[K];
      for (int k = 0; k < K; k++) {
         threshold_array[k] = new double[N]();
      }
   }
}

void Nested_Threshold::free_up() {
   if (threshold_array != nullptr) {
      for (int k = 0; k < K; k++) {
	 delete[] threshold_array[k];
      }
      delete[] threshold_array;
   }
   threshold_array = nullptr;
}

void Nested_Threshold::deep_copy(double** array) {
   if (array != nullptr) {
      for (int k = 0; k < K; k++) {
         for (long int n = 1L; n < N+1L; n++) {
            threshold_array[k][n-1L] = array[k][n-1L];
         }
      }
   }
}

Nested_Adapter::Nested_Adapter(double theta): theta(theta) {}

void Nested_Adapter::adapt(IN OUT Nested_Simulation&      nested_simulation,
		           IN     bool                    compute_sd_flag,
		           IN     const Refiner&          refiner,
		           IN     const Nested_Threshold& threshold,
		           IN     double                  xi,
		           IN     int                     level,
		           IN     long int                n) const {
   double& X = nested_simulation.X;
   double& ms = nested_simulation.ms;
   double Y = nested_simulation.Y;
   int eta = 0;
   int saturation = int(std::ceil(theta*level));
   double criterion;

   while(true) {
      if (eta == saturation) {
         break;
      }
      criterion = threshold(eta, n);
      if (compute_sd_flag) {
         criterion *= compute_sd(X, ms);
      }
      if (std::abs(X - xi) < criterion) {
         refiner.refine(X, ms, Y, level+eta);
         eta++;
      } else {
         break;
      }
   }
}

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
                                     OUT int&              level) {
   verify_power_concentration(loss_model);
   verify_power_concentration_adaptivity(loss_model);

   Bias bias(h_0, M); // Bias function s -> h_0 / M^s
   Gamma u;
   if (loss_model.concentration == power_concentration) {
      u = Gamma(gamma_0, loss_model.delta, smoothing);
   } else {
      u = Gamma(gamma_0, beta, smoothing);
   }

   n = a_nested_sa_optimal_steps(accuracy, loss_model, step, u, scaler);
   level = a_nested_sa_optimal_level(accuracy, h_0, M, theta);
   threshold = Nested_Threshold(u, bias, n, theta, r,
                                level, threshold_scaler, loss_model);
}

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
                          IN OUT Nested_Simulator&       simulator) {
   double h = h_0/std::pow(M, level);
   simulator.set_bias_parameter(h);
   double xi = xi_0;

   Nested_Simulation X_h;
   for (long int i = 0L; i < n; i++) {
      X_h = simulator();
      adapter.adapt(X_h, compute_sd_flag, refiner, threshold, xi, level, i+1L);
      xi = xi - step(i+1L)*H_1(alpha, xi, X_h.X);
   }

   return xi;
}

int a_nested_sa_optimal_level(double accuracy,
                              double h_0,
                              double M,
                              double theta) {
   verify_optimal_level(accuracy, h_0);
   return int(std::ceil(std::log(h_0/accuracy)/((1+theta)*std::log(M))));
}

long int a_nested_sa_optimal_steps(double            accuracy,
                                   const Loss_Model& loss_model,
		                   const Step&       step,
			           const Step&       u,
			           double            scaler) {
   double beta, delta;
   switch (loss_model.concentration) {
   case power_concentration:
      beta = step.get_exponent();
      delta = u.get_exponent();
      if (delta < (beta/2)) {
         return (long int) (std::ceil(scaler*u.inverse(accuracy)));
      } else {
         return (long int) (std::ceil(scaler*step.inverse(accuracy*accuracy)));
      }
   default:
      return (long int) (std::ceil(scaler*step.inverse(accuracy*accuracy)));
   }
}
