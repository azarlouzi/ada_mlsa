#include "nested_sa.h"
#include <cmath>
#include <cstdio>
#include <exception>

class Nested_Simulator_Exception: public std::exception {
public:
   Nested_Simulator_Exception(double h): h(h) {}
   const char* what() const throw() override {
      char* msg = new char[exception_message_length];
      std::sprintf(msg, "Simulation bias parameter should satisfy: h > 0. Got instead: h = %.10f", h);
      return msg;
   }
private:
   double h;
};

void Nested_Simulator::set_bias_parameter(double h) {
   this->h = h;
   verify_bias_parameter();
}

void Nested_Simulator::verify_bias_parameter() const {
   if (!(h > 0)) {
      throw Nested_Simulator_Exception(h);
   }
}

double nested_sa(IN     double            xi_0,
                 IN     double            alpha,
                 IN     double            h,
                 IN     long int          n,
                 IN     const Step&       step,
                 IN OUT Nested_Simulator& simulator) {
   simulator.set_bias_parameter(h);
   double xi = xi_0;

   Nested_Simulation X_h;
   for (long int i = 0L; i < n; i++) {
      X_h = simulator();
      xi = xi - step(i+1L)*H_1(alpha, xi, X_h.X);
   }

   return xi;
}

double nested_sa_optimal_bias(double accuracy) {
   return accuracy;
}

long int nested_sa_optimal_steps(double accuracy, const Step& step, double scaler) {
   return sa_optimal_steps(accuracy, step, scaler);
}
