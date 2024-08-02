#include "ml_sa.h"
#include <cmath>
#include <cstdio>
#include <exception>

class ML_Simulator_Exception: public std::exception {
public:
   ML_Simulator_Exception(double h_coarse, double h_fine): h_coarse(h_coarse), h_fine(h_fine) {}
   const char* what() const throw() override {
      char* msg = new char[exception_message_length];
      std::sprintf(msg, "Multilevel simulation bias parameters should satisfy h_coarse > h_fine > 0. Got instead: h_coarse = %.10f, h_fine = %.10f", h_coarse, h_fine);
      return msg;
   }
private:
   double h_coarse, h_fine;
};

void ML_Simulator::set_bias_parameters(double h_coarse, double h_fine) {
   this->h_coarse = h_coarse;
   this->h_fine = h_fine;
   verify_bias_parameters();
}

void ML_Simulator::verify_bias_parameters() const {
   if (!(h_coarse > 0) || !(h_fine > 0) || !(h_coarse > h_fine)) {
      throw ML_Simulator_Exception(h_coarse, h_fine);
   }
}

void configure_h(IN     double  h_0,
                 IN     double  M,
                 IN     int     L,
                    OUT double* h) {
   h[0] = h_0;
   for (int l = 1; l < L+1; l++) {
      h[l] = h[l-1]/M;
   }
}

void configure_ml_sa(IN     double            beta,
                     IN     double            h_0,
                     IN     double            M,
                     IN     int               L,
                     IN     double            scaler,
                     IN     const Loss_Model& loss_model,
                        OUT double*           h,
                        OUT long int*         N,
                        OUT double&           accuracy) {
   verify_power_concentration(loss_model);

   configure_h(h_0, M, L, h);
   accuracy = h[L];

   double tmp;
   tmp = 0;
   for (int l = 0; l < L+1; l++) {
      switch (loss_model.concentration) {
      case power_concentration:
         tmp += std::pow(h[l], (- beta + loss_model.p/(2*(1 + loss_model.p)))/(1 + beta));
         break;
      case gaussian_concentration:
         tmp += std::pow(h[l], (1 - 2*beta)/(2*(1 + beta)))*std::pow(std::abs(std::log(h[l])), 1./(2*(1 + beta)));
         break;
      case lipschitz_concentration:
         tmp += std::pow(h[l], (1 - 2*beta)/(2*(1 + beta)));
         break;
      }
   }
   tmp = std::pow(tmp, 1./beta);
   tmp *= std::pow(h[L], -2./beta);

   for (int l = 0; l < L+1; l++) {
      switch (loss_model.concentration) {
      case power_concentration:
         N[l] = (long int) std::ceil(
            scaler * tmp * std::pow(h[l], (1 + loss_model.p/(2*(1 + loss_model.p)))/(1 + beta))
         );
         break;
      case gaussian_concentration:
         N[l] = (long int) std::ceil(
            scaler * tmp * std::pow(h[l], 3./(2*(1 + beta)))*std::pow(std::abs(std::log(h[l])), 1./(2*(1 + beta)))
         );
         break;
      case lipschitz_concentration:
         N[l] = (long int) std::ceil(scaler * tmp * std::pow(h[l], 3./(2*(1 + beta))));
         break;
      }
   }

}

double ml_sa(IN     double            xi_0,
             IN     double            alpha,
             IN     int               L,
             IN     const double*     h,
             IN     const long int*   N,
             IN     const Step&       step,
             IN OUT Nested_Simulator& simulator,
             IN OUT ML_Simulator&     ml_simulator) {
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
         xi[l][0] = xi[l][0] - step(i+1L)*H_1(alpha, xi[l][0], X_ml.coarse);
         xi[l][1] = xi[l][1] - step(i+1L)*H_1(alpha, xi[l][1], X_ml.fine);
      }
   }

   double xi_ml = xi[0][0];
   for (int l = 1; l < L+1; l++) {
      xi_ml += xi[l][1] - xi[l][0];
   }
   return xi_ml;
}

int ml_sa_optimal_layers(double accuracy, double h_0, double M) {
   verify_optimal_level(accuracy, h_0);
   return int(std::ceil(std::log(h_0/accuracy)/std::log(M)));
}
