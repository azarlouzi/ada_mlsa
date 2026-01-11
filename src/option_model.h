#ifndef _OPTION_MODEL_
#define _OPTION_MODEL_

#include "adaptivity.h"
#include "ml_sa.h"
#include "nested_sa.h"
#include "sa.h"

class Option_Simulator: public Simulator {
public:
   Option_Simulator(double tau = 1);
   double operator()() const override;
private:
   double tau;
};

class Option_Nested_Payoff {
public:
   Option_Nested_Payoff(double tau);
   double operator()(double y, double z) const;
private:
   double tau;
};

class Option_Nested_Simulator: public Nested_Simulator {
public:
   Option_Nested_Simulator(double tau);
   Nested_Simulation operator()() const override;
private:
   Option_Nested_Payoff phi;
};

class Option_ML_Simulator: public ML_Simulator {
public:
   Option_ML_Simulator(double tau);
   ML_Simulations operator()() const override;
private:
   Option_Nested_Payoff phi;
};

class Option_Refiner: public Refiner {
public:
   Option_Refiner(double h_0 = 1, double M = 2, double tau = 0.5);
   void refine(IN OUT double& X,
               IN OUT double& ms,
               IN     double  Y,
               IN     int     level) const override;
private:
   Option_Nested_Payoff phi;
};

#endif // _OPTION_MODEL_
