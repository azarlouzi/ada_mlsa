#ifndef _SA_
#define _SA_

#include "helpers.h"

class Simulator {
public:
   virtual double operator()() const=0;
};

class Step {
public:
   Step(double beta=1): beta(beta) {}
   virtual double operator()(long int p) const=0;
   virtual double inverse(double s) const=0;
   double get_exponent() const { return beta; }
protected:
   double beta;
};

class Gamma: public Step {
public:
   Gamma(double gamma_0 = 1, double beta = 1, long int smoothing = 0L):
      Step(beta), gamma_0(gamma_0), smoothing(smoothing) {}
   double operator()(long int p) const override;
   double inverse(double s) const override;
private:
   double gamma_0;
   long int smoothing;
};

double H_1(double alpha, double xi, double x);

double sa(IN double           xi_0,
          IN double           alpha,
          IN long int         n,
          IN const Step&      step,
          IN const Simulator& simulator);

long int sa_optimal_steps(double accuracy, const Step& step, double scaler);

enum Loss_Concentration {
   power_concentration,
   gaussian_concentration,
   lipschitz_concentration,
};

struct Loss_Model {
   Loss_Concentration concentration;
   double p = 0; // p > 1, exponent for power concentration
   double delta = 0; // delta > 0, step size exponent for power concentration adaptivity
};

void verify_power_concentration(const Loss_Model& loss_model);

void verify_power_concentration_adaptivity(const Loss_Model& loss_model);

void verify_optimal_level(double accuracy, double h_0);

#endif // _SA_
