#include "option_model.h"
#include "helpers.h"
#include <cmath>

Option_Simulator::Option_Simulator(double tau):
   Simulator(), tau(tau) {}

double Option_Simulator::operator()() const {
   double Y = gaussian();
   return tau*(std::pow(Y, 2) - 1);
}

Option_Nested_Payoff::Option_Nested_Payoff(double tau):
   tau(tau) {}

double Option_Nested_Payoff::operator()(double y, double z) const {
   return -std::pow(std::sqrt(tau)*y + std::sqrt(1-tau)*z, 2);
}

Option_Nested_Simulator::Option_Nested_Simulator(double tau):
   Nested_Simulator(), phi(tau) {}

Nested_Simulation Option_Nested_Simulator::operator()() const {
   long int K = (long int) (std::ceil(1./h));
   double X_h = -1;
   double Y = gaussian();
   double ms = 0;
   double cash_flow;

   for (long int k = 0L; k < K; k++) {
      cash_flow = phi(Y, gaussian());
      X_h -= cash_flow/double(K);
      ms += std::pow(-1-cash_flow, 2)/double(K);
   }

   return Nested_Simulation{
      .X = X_h,
      .Y = Y,
      .ms = ms,
   };
}

Option_ML_Simulator::Option_ML_Simulator(double tau):
   ML_Simulator(), phi(tau) {}

ML_Simulations Option_ML_Simulator::operator()() const {
   double Y = gaussian();
   double ms = 0;
   double cash_flow;

   long int K_coarse = (long int) std::ceil(1./h_coarse);
   long int K_fine = (long int) std::ceil(1./h_fine);

   double X_h_coarse = -1;
   for (long int k = 0L; k < K_coarse; k++) {
      cash_flow = phi(Y, gaussian());
      X_h_coarse -= cash_flow/double(K_coarse);
      ms += std::pow(-1-cash_flow, 2)/double(K_fine);
   }

   double X_h_fine = -1 + (X_h_coarse + 1)*double(K_coarse)/double(K_fine);
   for (long int k = 0L; k < (K_fine - K_coarse); k++) {
      cash_flow = phi(Y, gaussian());
      X_h_fine -= cash_flow/double(K_fine);
      ms += std::pow(-1-cash_flow, 2)/double(K_fine);
   }

   return ML_Simulations {
      .coarse = X_h_coarse,
      .fine = X_h_fine,
      .Y = Y,
      .ms = ms,
   };
}

Option_Refiner::Option_Refiner(double h_0,
		               double M,
			       double tau):
   Refiner(h_0, M), phi(tau) {}

void Option_Refiner::refine(IN OUT double& X,
                            IN OUT double& ms,
                            IN     double  Y,
                            IN     int     level) const {
   double cash_flow;
   long int K_coarse = (long int) std::ceil(std::pow(M, level)/h_0);
   long int K_fine = (long int) std::ceil(std::pow(M, level + 1)/h_0);
   X = -1 + (1 + X)/M;
   ms /= M;

   for (long int k = 0; k < (K_fine - K_coarse); k++) {
      cash_flow = phi(Y, gaussian());
      X -= cash_flow/double(K_fine);
      ms += std::pow(-1-cash_flow, 2)/double(K_fine);
   }
}
