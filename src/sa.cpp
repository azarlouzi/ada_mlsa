#include "sa.h"
#include <cmath>
#include <cstdio>
#include <exception>

double Gamma::operator()(long int n) const {
   return gamma_0/(std::pow(double(smoothing + n), beta));
}

double Gamma::inverse(double s) const {
   return std::pow(s, -1./beta);
}

double H_1(double alpha, double xi, double x) {
   return 1 - heaviside(x-xi)/(1-alpha);
}

double sa(IN double           xi_0,
          IN double           alpha,
          IN long int         n,
          IN const Step&      step,
          IN const Simulator& simulator) {
   double xi = xi_0;
   double X_0;
   for (long int i = 0L; i < n; i++) {
      X_0 = simulator();
      xi = xi - step(i+1L)*H_1(alpha, xi, X_0);
   }
   return xi;
}

long int sa_optimal_steps(double accuracy, const Step& step, double scaler) {
   return (long int) (std::ceil(scaler*step.inverse(accuracy*accuracy)));
}

class Power_Concentration_Exception: public std::exception {
public:
   Power_Concentration_Exception(double p): p(p) {}
   const char* what() const throw() override {
      char* msg = new char[exception_message_length];
      std::sprintf(msg, "MLSA for the power concentration loss model expects the power p to satisfy: p > 1. Got instead: p = %.10f", p);
      return msg;
   }
private:
   double p;
};

void verify_power_concentration(const Loss_Model& loss_model) {
   if (loss_model.concentration == power_concentration && loss_model.p <= 1) {
       throw Power_Concentration_Exception(loss_model.p);
   }
}

class Power_Concentration_Adaptivity_Exception: public std::exception {
public:
   Power_Concentration_Adaptivity_Exception(double delta): delta(delta) {}
   const char* what() const throw() override {
      char* msg = new char[exception_message_length];
      std::sprintf(msg, "Adaptive MLSA for the power concentration loss model expects the exponent delta of the adaptive step size to satisfy: delta > 0. Got instead: delta = %.10f", delta);
      return msg;
   }
private:
   double delta;
};

void verify_power_concentration_adaptivity(const Loss_Model& loss_model) {
   if (loss_model.concentration == power_concentration && loss_model.delta <= 0) {
       throw Power_Concentration_Adaptivity_Exception(loss_model.delta);
   }
}

class Optimal_Level_Exception: public std::exception {
public:
   Optimal_Level_Exception(double accuracy, double h_0):
      accuracy(accuracy), h_0(h_0) {}
   const char* what() const throw() override {
      char* msg = new char[exception_message_length];
      std::sprintf(msg, "Parameters h_0 and accuracy should satisfy: accuracy < h_0. Got instead: accuracy = %.10f, h_0 = %.10f.", accuracy, h_0);
      return msg;
   }
private:
   double accuracy;
   double h_0;
};


void verify_optimal_level(double accuracy, double h_0) {
   if (!(accuracy < h_0)) {
      throw Optimal_Level_Exception(accuracy, h_0);
   }
}

