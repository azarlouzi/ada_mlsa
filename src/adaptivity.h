#ifndef _ADAPTIVITY_
#define _ADAPTIVITY_

#include "helpers.h"
#include <exception>

class Bias {
public:
   Bias(double h_0 = 1, double M = 2);
   double operator()(double s) const;
private:
   double h_0;
   double M;
};

class Refiner {
public:
   Refiner(double h_0 = 1, double M = 2);
   virtual void refine(IN OUT double& X,
		       IN OUT double& ms,
		       IN     double  Y,
		       IN     int     level) const=0;
protected:
   double h_0;
   double M;
};

enum Threshold_Access_Level {
   level_k = 0,
   level_l = 1,
   level_n = 2,
};

class Threshold_Access_Exception: public std::exception {
public:
   Threshold_Access_Exception(Threshold_Access_Level access_level, double access, double access_bound);
   const char* what() const throw() override; 
private:
   Threshold_Access_Level access_level;
   double access;
   double access_bound;
};

#endif // _ADAPTIVITY_
