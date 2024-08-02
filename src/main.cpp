#include "ada_ml_sa.h"
#include "ada_nested_sa.h"
#include "helpers.h"
#include "ml_sa.h"
#include "nested_sa.h"
#include "sa.h"
#include "option_model.h"
#include "swap_model.h"
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>

struct ML_Setting {
   double h_0;
   double M;
   int L;
   double gamma_0;
   long int smoothing;
};

void run_test_case_1() {
   double alpha = 0.975; // 0.0 < alpha < 1
   double tau = 0.5; // 0 < tau < 1

   Option_Simulator        simulator        (tau);
   Option_Nested_Simulator nested_simulator (tau);
   Option_ML_Simulator     ml_simulator     (tau);

   int max_L = 20;
   double h[max_L+1];
   long int N[max_L+1];

   ML_Setting ml_settings[] {
      ML_Setting {.h_0 = 1./16, .M = 2, .L = 1, .gamma_0 = 2, .smoothing = 2500L,}, // accuracy = 1/32
      ML_Setting {.h_0 = 1./32, .M = 2, .L = 1, .gamma_0 = 2, .smoothing = 4000L,}, // accuracy = 1/64
      ML_Setting {.h_0 = 1./32, .M = 2, .L = 2, .gamma_0 = 0.75, .smoothing = 9000L,}, // accuracy = 1/128
      ML_Setting {.h_0 = 1./32, .M = 2, .L = 3, .gamma_0 = 0.25, .smoothing = 10000L,}, // accuracy = 1/256
      ML_Setting {.h_0 = 1./32, .M = 2, .L = 4, .gamma_0 = 0.09, .smoothing = 10000L,}, // accuracy = 1/512
   };

   double beta = 1.0; // 0.0 < beta <= 1
   Gamma gamma_sa;
   Gamma gamma_nsa;
   Gamma gamma_mlsa;

   gamma_sa = Gamma(1, beta, 100L);
   gamma_nsa = Gamma(1, beta, 100L);

   double scaler = 1;
   long int n; // n >> 1

   double xi_0 = 2.0;
   double time_sa, time_nsa, time_mlsa;
   double VaR_sa, VaR_nsa, VaR_mlsa;

   // Adaptive setup
   double adaptive_scaler = 700;
   double nested_adaptive_scaler = 2;
   Loss_Model model {
      .concentration = power_concentration,
      .p             = 11,
      .delta         = 0.95, // 0 < delta <= beta
   };

   double theta;
   if (model.concentration == power_concentration) {
      theta = (model.p/2-1)/(model.p/2+1);
   } else {
      theta = 1;
   }
   double r = 1 + 1./theta;
   double threshold_scaler = 12;
   double nested_threshold_scaler = 0.5;
   double threshold_confidence = 3;

   ML_Adapter       adapter(theta);        // Refinement strategy
   Nested_Adapter   nested_adapter(theta); // Refinement strategy
   Option_Refiner   option_refiner;        // Single-level refinement
   ML_Threshold     threshold;             // ML_Thresholds
   Nested_Threshold nested_threshold;      // Nested_Thresholds

   double accuracy;
   int ada_L;
   int level;

   double time_amlsa, time_amlsa_c, time_ansa, time_ansa_c;
   double VaR_amlsa,  VaR_amlsa_c,  VaR_ansa,  VaR_ansa_c;

   int n_runs = 200;
   std::printf("#,accuracy,status,time_sa,time_nsa,time_mlsa,time_ada_nsa,time_ada_nsa_sd,time_ada_mlsa,time_ada_mlsa_sd,VaR_sa,VaR_nsa,VaR_mlsa,VaR_ada_nsa,VaR_ada_nsa_sd,VaR_ada_mlsa,VaR_ada_mlsa_sd\n");
   for (ML_Setting ml_setting: ml_settings) {
      for (int i = 0; i < n_runs; i++) {
         try {
            // ML SA
            gamma_mlsa = Gamma(ml_setting.gamma_0, beta, ml_setting.smoothing);
            configure_ml_sa(beta, ml_setting.h_0, ml_setting.M, ml_setting.L, scaler, model, h, N, accuracy);

            tik();
            VaR_mlsa = ml_sa(xi_0, alpha, ml_setting.L, h, N, gamma_mlsa, nested_simulator, ml_simulator);
            time_mlsa = tok();

            // Nested SA
            n = nested_sa_optimal_steps(accuracy, gamma_nsa, scaler);

            tik();
            VaR_nsa = nested_sa(xi_0, alpha, accuracy, n, gamma_nsa, nested_simulator);
            time_nsa = tok();

            // Adaptive Nested SA
            configure_adaptive_nested_sa(beta, ml_setting.h_0, ml_setting.M, accuracy,
			                 gamma_nsa, nested_adaptive_scaler, model, r, theta,
				         ml_setting.gamma_0, ml_setting.smoothing,
			                 nested_threshold_scaler, n, nested_threshold, level);
            option_refiner = Option_Refiner(ml_setting.h_0, ml_setting.M, tau);

            tik();
	    VaR_ansa = adaptive_nested_sa(xi_0, alpha, ml_setting.h_0, ml_setting.M, level, n, gamma_nsa, false,
			                  nested_threshold, nested_adapter, option_refiner, nested_simulator);
            time_ansa = tok();

            // Adaptive Nested SA with computed threshold
            configure_adaptive_nested_sa(beta, ml_setting.h_0, ml_setting.M, accuracy,
			                 gamma_nsa, nested_adaptive_scaler, model, r, theta,
				         ml_setting.gamma_0, ml_setting.smoothing,
			                 threshold_confidence, n, nested_threshold, level);

            tik();
	    VaR_ansa_c = adaptive_nested_sa(xi_0, alpha, ml_setting.h_0, ml_setting.M, level, n, gamma_nsa, true,
			                  nested_threshold, nested_adapter, option_refiner, nested_simulator);
            time_ansa_c = tok();

            // SA
            n = sa_optimal_steps(accuracy, gamma_nsa, scaler);

            tik();
            VaR_sa = sa(xi_0, alpha, n, gamma_sa, simulator);
            time_sa = tok();

            // Adaptive ML SA
            configure_adaptive_ml_sa(beta, ml_setting.h_0, ml_setting.M, accuracy,
			             adaptive_scaler, model, r, theta,
				     ml_setting.gamma_0, ml_setting.smoothing,
				     threshold_scaler, h, N, threshold, ada_L);

            tik();
            VaR_amlsa = adaptive_ml_sa(xi_0, alpha, ada_L, h, N, gamma_mlsa,
				       threshold, adapter, option_refiner, false,
                                       nested_simulator, ml_simulator);
            time_amlsa = tok();

            // Adaptive ML SA with computed threshold
            configure_adaptive_ml_sa(beta, ml_setting.h_0, ml_setting.M, accuracy,
			             adaptive_scaler, model, r, theta,
				     ml_setting.gamma_0, ml_setting.smoothing,
				     threshold_confidence, h, N, threshold, ada_L);

            tik();
            VaR_amlsa_c = adaptive_ml_sa(xi_0, alpha, ada_L, h, N, gamma_mlsa,
					 threshold, adapter, option_refiner, true,
                                         nested_simulator, ml_simulator);
            time_amlsa_c = tok();

            std::printf("%d,%.15f,success,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
                        i+1, accuracy, time_sa, time_nsa, time_mlsa, time_ansa, time_ansa_c, time_amlsa, time_amlsa_c,
                        VaR_sa, VaR_nsa, VaR_mlsa, VaR_ansa, VaR_ansa_c, VaR_amlsa, VaR_amlsa_c);
         } catch (const std::exception& e) {
            std::printf("%d,%.15f,failure,%s\n", i+1, accuracy, e.what());
         }
      }
   }
}

void run_test_case_2() {
   double r = 0.02;
   double S_0 = 100.0; // in basis points
   double kappa = 0.12;
   double sigma = 0.2;
   Time Delta = Time {y: 0, m: 3, d: 0};
   Time T = Time {y: 1, m: 0, d: 0}; 
   Time delta = Time {y: 0, m: 0, d: 7};
   double leg_0 = 1e4; // in basis points
   double alpha = 0.85; // 0.0 < alpha < 1

   Swap_Simulator        simulator        (r, S_0, kappa, sigma, Delta, T, delta, leg_0);
   Swap_Nested_Simulator nested_simulator (r, S_0, kappa, sigma, Delta, T, delta, leg_0);
   Swap_ML_Simulator     ml_simulator     (r, S_0, kappa, sigma, Delta, T, delta, leg_0);

   int max_L = 20;
   double h[max_L+1];
   long int N[max_L+1];

   ML_Setting ml_settings[] {
      ML_Setting {.h_0 = 1./8, .M = 2, .L = 2, .gamma_0 = 6, .smoothing = 10L,}, // accuracy = 1/32
      ML_Setting {.h_0 = 1./16, .M = 2, .L = 2, .gamma_0 = 20, .smoothing = 500L,}, // accuracy = 1/64
      ML_Setting {.h_0 = 1./16, .M = 2, .L = 3, .gamma_0 = 21, .smoothing = 1000L,}, // accuracy = 1/128
      ML_Setting {.h_0 = 1./16, .M = 2, .L = 4, .gamma_0 = 20, .smoothing = 2000L,}, // accuracy = 1/256
      ML_Setting {.h_0 = 1./16, .M = 2, .L = 5, .gamma_0 = 21, .smoothing = 3000L,}, // accuracy = 1/512
   };

   double beta = 1.0; // 0.0 < beta <= 1
   Gamma gamma_sa (100, beta, 0L); // SA
   Gamma gamma_nsa (50, beta, 0L); // Nested SA
   Gamma gamma_mlsa; // Multilevel SA

   double scaler = 1;
   long int n; // n >> 1

   double xi_0 = 200;
   double time_sa, time_nsa, time_mlsa;
   double VaR_sa, VaR_nsa, VaR_mlsa;

   // Adaptive setup
   double adaptive_scaler = 80;
   double nested_adaptive_scaler = 2;
   Loss_Model model {
      .concentration = power_concentration,
      //.concentration = lipschitz_concentration,
      .p             = 8,
      .delta         = 0.95, // 0 < delta <= beta
   };

   double theta;
   if (model.concentration == power_concentration) {
      theta = (model.p/2-1)/(model.p/2+1);
   } else {
      theta = 1;
   }
   double r0 = 1 + 1./theta;
   double threshold_scaler = 100;
   double nested_threshold_scaler = 300;
   double threshold_confidence = 3;

   ML_Adapter       adapter(theta);        // Refinement strategy
   Nested_Adapter   nested_adapter(theta); // Refinement strategy
   Swap_Refiner     swap_refiner;          // Single-level refinement
   ML_Threshold     threshold;             // ML_Thresholds
   Nested_Threshold nested_threshold;      // Nested_Thresholds

   double accuracy;
   int ada_L;
   int level;

   double time_amlsa, time_amlsa_c, time_ansa, time_ansa_c;
   double VaR_amlsa,  VaR_amlsa_c,  VaR_ansa,  VaR_ansa_c;

   //int n_runs = 50;
   int n_runs = 200;
   std::printf("#,accuracy,status,time_sa,time_nsa,time_mlsa,time_ada_nsa,time_ada_nsa_sd,time_ada_mlsa,time_ada_mlsa_sd,VaR_sa,VaR_nsa,VaR_mlsa,VaR_ada_nsa,VaR_ada_nsa_sd,VaR_ada_mlsa,VaR_ada_mlsa_sd\n");
   for (ML_Setting ml_setting: ml_settings) {
      for (int i = 0; i < n_runs; i++) {
         try {
            // ML SA
            gamma_mlsa = Gamma(ml_setting.gamma_0, beta, ml_setting.smoothing);
            configure_ml_sa(beta, ml_setting.h_0, ml_setting.M, ml_setting.L, scaler, model, h, N, accuracy);
            tik();
            VaR_mlsa = ml_sa(xi_0, alpha, ml_setting.L, h, N, gamma_mlsa, nested_simulator, ml_simulator);
            time_mlsa = tok();

            // Nested SA
            n = nested_sa_optimal_steps(accuracy, gamma_nsa, scaler);
            tik();
            VaR_nsa = nested_sa(xi_0, alpha, accuracy, n, gamma_nsa, nested_simulator);
            time_nsa = tok();

            // Adaptive Nested SA
            configure_adaptive_nested_sa(beta, ml_setting.h_0, ml_setting.M, accuracy,
			                 gamma_nsa, nested_adaptive_scaler, model, r0, theta,
				         ml_setting.gamma_0, ml_setting.smoothing,
			                 nested_threshold_scaler, n, nested_threshold, level);
            swap_refiner = Swap_Refiner(r, S_0, kappa, sigma, Delta, T, delta, leg_0, ml_setting.h_0, ml_setting.M);

            tik();
	    VaR_ansa = adaptive_nested_sa(xi_0, alpha, ml_setting.h_0, ml_setting.M, level, n, gamma_nsa, false,
			                  nested_threshold, nested_adapter, swap_refiner, nested_simulator);
            time_ansa = tok();

            // Adaptive Nested SA with computed threshold
            configure_adaptive_nested_sa(beta, ml_setting.h_0, ml_setting.M, accuracy,
			                 gamma_nsa, nested_adaptive_scaler, model, r0, theta,
				         ml_setting.gamma_0, ml_setting.smoothing,
			                 threshold_confidence, n, nested_threshold, level);

            tik();
	    VaR_ansa_c = adaptive_nested_sa(xi_0, alpha, ml_setting.h_0, ml_setting.M, level, n, gamma_nsa, true,
			                  nested_threshold, nested_adapter, swap_refiner, nested_simulator);
            time_ansa_c = tok();

            // SA
            n = sa_optimal_steps(accuracy, gamma_sa, scaler);
            tik();
            VaR_sa = sa(xi_0, alpha, n, gamma_sa, simulator);
            time_sa = tok();

            // Adaptive ML SA
            configure_adaptive_ml_sa(beta, ml_setting.h_0, ml_setting.M, accuracy,
			             adaptive_scaler, model, r0, theta,
				     ml_setting.gamma_0, ml_setting.smoothing,
				     threshold_scaler, h, N, threshold, ada_L);

            tik();
            VaR_amlsa = adaptive_ml_sa(xi_0, alpha, ada_L, h, N, gamma_mlsa,
				       threshold, adapter, swap_refiner, false,
                                       nested_simulator, ml_simulator);
            time_amlsa = tok();

            // Adaptive ML SA with computed threshold
            configure_adaptive_ml_sa(beta, ml_setting.h_0, ml_setting.M, accuracy,
			             adaptive_scaler, model, r0, theta,
				     ml_setting.gamma_0, ml_setting.smoothing,
				     threshold_confidence, h, N, threshold, ada_L);

            tik();
            VaR_amlsa_c = adaptive_ml_sa(xi_0, alpha, ada_L, h, N, gamma_mlsa,
					 threshold, adapter, swap_refiner, true,
                                         nested_simulator, ml_simulator);
            time_amlsa_c = tok();

            std::printf("%d,%.15f,success,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
                        i+1, accuracy, time_sa, time_nsa, time_mlsa, time_ansa, time_ansa_c, time_amlsa, time_amlsa_c,
                        VaR_sa, VaR_nsa, VaR_mlsa, VaR_ansa, VaR_ansa_c, VaR_amlsa, VaR_amlsa_c);
         } catch (const std::exception& e) {
            std::printf("%d,%.15f,failure,%s\n", i+1, accuracy, e.what());
         }
      }
   }
}

void display_help(const char* name) {
   std::printf("Usage: %s [-h|--help|[--test_case {1,2}]\n", name);
   std::printf("Options:\n");
   std::printf("-h, --help        Display this usage documentation\n");
   std::printf("--test_case {1,2} Test set to run; default: 1\n");
}

class Parameter_Exception: public std::exception {
public:
   Parameter_Exception(const char* parameter_name, const char* expected_behavior, const char* encountered_instead):
      parameter_name(parameter_name), expected_behavior(expected_behavior), encountered_instead(encountered_instead) {}
   const char* what() const throw() override {
      char* msg = new char[exception_message_length];
      std::sprintf(msg, "Expected %s to satisfy: %s. Encountered instead: %s.", parameter_name, expected_behavior, encountered_instead);
      return msg;
   }
private:
   const char* parameter_name;
   const char* expected_behavior;
   const char* encountered_instead;
};

enum Test_Case {
   test_case_1 = 1,
   test_case_2 = 2,
};

Test_Case convert_string_to_test_case(std::string test_case_string) {
   if (test_case_string == "1") {
      return test_case_1;
   } else if (test_case_string == "2") {
      return test_case_2;
   } else {
      throw Parameter_Exception("test_case", "test_case in {1,2}", test_case_string.c_str());
   }
}

void run(Test_Case test_case) {
   switch (test_case) {
      case test_case_1:
         run_test_case_1();
         break;
      case test_case_2:
         run_test_case_2();
         break;
   }
}

int main(int argc, char* argv[]) {
    if ((argc < 2) ||
        (argc == 2 && (std::string(argv[1]) == "-h" ||
                       std::string(argv[1]) == "--help"))) {
      display_help(argv[0]);
      return 1;
   }

   Test_Case test_case = test_case_1;

   if (argc < 2) {
      run(test_case);
      return 0;
   }

   std::string test_case_string = "1";
   try {
      for (int i = 1; i < argc; ) {
         if (std::string(argv[i]) == "--test_case") {
            test_case_string = std::string(argv[i+1]);
            i = i + 2;
         } else {
            i = i + 1;
         }
      }
      test_case = convert_string_to_test_case(test_case_string);
   } catch(const std::exception& e) {
      std::printf("%s\n\n", e.what());
      display_help(argv[0]);
      return 1;
   }

   run(test_case);
   return 0;
}
