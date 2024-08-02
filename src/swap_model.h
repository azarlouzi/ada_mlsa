#ifndef _SWAP_MODEL_
#define _SWAP_MODEL_

#include "adaptivity.h"
#include "ml_sa.h"
#include "nested_sa.h"
#include "sa.h"

class Discount {
public:
   Discount(double r = 0): r(r) {}
   double operator()(double t) const;
private:
   double r;
};

class Reset {
public:
   Reset(double Delta = 0): Delta(Delta) {}
   double operator()(int i) const;
private:
   double Delta;
};

class Swap_Simulator: public Simulator {
public:
   Swap_Simulator(double r,
                  double S_0,
                  double kappa,
                  double sigma,
                  Time   Delta,
                  Time   T,
                  Time   delta,
                  double leg_0);
   double operator()() const override;
private:
   double r;
   double S_0;
   double kappa;
   double sigma;
   double Delta;
   double T;
   double delta;
   double leg_0;
   int n;
   double nominal;
   double factor;
   Discount discount;
   Reset reset;
};

class Swap_Nested_Payoff {
public:
   Swap_Nested_Payoff() {}
   Swap_Nested_Payoff(double r,
                      double S_0,
                      double kappa,
                      double Delta,
                      double leg_0,
                      int    n);
   double operator()(double y, double* z) const;
private:
   double r;
   double S_0;
   double kappa;
   double Delta;
   double leg_0;
   int n;
   double nominal;
   Discount discount;
   Reset reset;
};

class Swap_Nested_Simulator: public Nested_Simulator {
public:
   Swap_Nested_Simulator(double r,
                         double S_0,
                         double kappa,
                         double sigma,
                         Time   Delta,
                         Time   T,
                         Time   delta,
                         double leg_0);
   Nested_Simulation operator()() const override;
private:
   double sigma;
   double Delta;
   double T;
   double delta;
   int n;
   Swap_Nested_Payoff phi;
};

class Swap_ML_Simulator: public ML_Simulator {
public:
   Swap_ML_Simulator(double r,
                     double S_0,
                     double kappa,
                     double sigma,
                     Time   Delta,
                     Time   T,
                     Time   delta,
                     double leg_0);
   ML_Simulations operator()() const override;
private:
   double sigma;
   double Delta;
   double T;
   double delta;
   int n;
   Swap_Nested_Payoff phi;
};

class Swap_Refiner: public Refiner {
public:
   Swap_Refiner() {}
   Swap_Refiner(double r,
                double S_0,
                double kappa,
                double sigma,
                Time   Delta,
                Time   T,
                Time   delta,
                double leg_0,
		double h_0 = 1,
		double M   = 2);
   void refine(IN OUT double& X,
               IN OUT double& ms,
               IN     double  Y,
               IN     int     level) const override;
private:
   double sigma;
   double Delta;
   double T;
   double delta;
   int n;
   Swap_Nested_Payoff phi;
};

#endif // _SWAP_MODEL_
