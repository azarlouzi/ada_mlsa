#ifndef _HELPERS_
#define _HELPERS_

#define IN
#define OUT

const int exception_message_length = 500;

void tik();
double tok();

struct Time {
   int y = 0;
   int m = 0;
   int d = 0;
};

double time_to_years(Time time);

double gaussian(double m = 0, double sd = 1);
double doleans_dade(double m = 0, double sd = 1);

double heaviside(double x);
double positive_part(double x);

double compute_sd(double mean, double mean_square);

#endif // _HELPERS_
