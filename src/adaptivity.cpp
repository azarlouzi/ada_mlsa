#include "adaptivity.h"
#include "helpers.h"
#include <cmath>
#include <cstdio>

Threshold_Access_Exception::Threshold_Access_Exception(Threshold_Access_Level access_level, double access, double access_bound):
      access_level(access_level),
      access(access),
      access_bound(access_bound) {}

const char* Threshold_Access_Exception::what() const throw() {
   char* msg = new char[exception_message_length];
   switch (access_level) {
   case level_k:
      std::sprintf(msg, "Adaptive simulation threshold access should satisfy 0 <= k < K. Got instead: k = %.0f, K = %.0f", access, access_bound);
      break;
   case level_l:
      std::sprintf(msg, "Adaptive simulation threshold access should satisfy 1 <= level <= L. Got instead: level = %.0f, L = %.0f", access, access_bound);
      break;
   case level_n:
      std::sprintf(msg, "Adaptive simulation threshold access should satisfy 0 <= n <= N. Got instead: n = %.0f, N = %.0f", access, access_bound);
      break;
   }
   return msg;
}

Bias::Bias(double h_0, double M): h_0(h_0), M(M) {}

double Bias::operator()(double s) const {
   return h_0/std::pow(M, s);
}

Refiner::Refiner(double h_0, double M): h_0(h_0), M(M) {}
