from typing import Callable, Dict, List, Tuple, Union
import numpy as np
import argparse
from collections import defaultdict
import csv
import math
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from statistics import mean

FIGSIZE = (8,5)

ML_SA_VAR_FOCUS = "VaR"

def plot_time_vs_rmse(title:str, algorithm_list:Tuple[str], accuracy_dict:Dict[int,float],
                      d:Dict[str,Dict[str,Union[str,List[float]]]], outpath:str):
   labels = [r"$\frac{1}{%d}$" %int(math.ceil(1./accuracy_dict[cohort])) for cohort in range(1, len(accuracy_dict)+1)]

   fig, ax = plt.subplots(figsize=FIGSIZE)
   #ax.invert_xaxis()
   ax.set_xscale("log")
   ax.set_yscale("log")
   #ax.grid(True, "both")

   for algorithm in algorithm_list:
      time_values, rmse_values, fit_time_values = d[algorithm]["time"], d[algorithm]["rmse"], d[algorithm]["time_fit"]

      ax.plot(rmse_values, time_values, d[algorithm]["symbol"], label=algorithm, color=d[algorithm]["color"],
              markerfacecolor="none", markeredgecolor=d[algorithm]["color"], markersize=d[algorithm].get("symbol_size"))
      pre, suf = d[algorithm]["color"].split(':')
      fit_color = ':'.join((pre, "dark " + suf))
      ax.plot(rmse_values, fit_time_values, "--", color=fit_color, lw=1)
      alignment = d[algorithm].get("alignment")
      if alignment is not None:
         xytext = (25,10) if alignment == "right" else (-25,-10)
         for i in range(len(rmse_values)):
            ax.annotate(labels[i], xy=(rmse_values[i], time_values[i]),
                        xytext=xytext, textcoords="offset points",
                        horizontalalignment=alignment, fontsize=12)

   ax.set_title(title, fontsize=16)
   ax.set_xlabel("RMSE", fontsize=14)
   ax.set_ylabel("Average execution time ($s$)", fontsize=14)
   ax.tick_params(labelsize=10)
   ax.legend(ncol=1, fontsize=10, frameon=False)

   plt.tight_layout()
   fname = title.lower().replace("-","_").replace(" ", "_") + "_comparison.pdf"
   fig.savefig(os.path.join(outpath, fname), bbox_inches="tight", pad_inches=0)

def plot_time_vs_accuracy(title:str, algorithm_list:Tuple[str], accuracy_dict:Dict[int,float],
                          d:Dict[str,Dict[str,Union[str,List[float]]]], outpath:str):
   accuracy_values = [accuracy_dict[cohort] for cohort in range(1, len(accuracy_dict)+1)]
   labels = [r"$\frac{1}{%d}$" %int(math.ceil(1./accuracy_dict[cohort])) for cohort in range(1, len(accuracy_dict)+1)]

   fig, ax = plt.subplots(figsize=FIGSIZE)
   #ax.invert_xaxis()
   ax.set_xscale("log")
   ax.set_yscale("log")
   #ax.grid(True, "both")

   for algorithm in algorithm_list:
      time_values, fit_time_values = d[algorithm]["time"], d[algorithm]["accuracy_time_fit"]

      ax.plot(accuracy_values, time_values, d[algorithm]["symbol"], label=algorithm, color=d[algorithm]["color"],
              markerfacecolor="none", markeredgecolor=d[algorithm]["color"], markersize=d[algorithm].get("symbol_size"))
      pre, suf = d[algorithm]["color"].split(':')
      fit_color = ':'.join((pre, "dark " + suf))
      ax.plot(accuracy_values, fit_time_values, "--", color=fit_color, lw=1)
      plt.xticks(accuracy_values, labels)

   ax.set_title(title, fontsize=16)
   ax.set_xlabel("Prescribed accuracy $\epsilon$", fontsize=14)
   ax.set_ylabel("Average execution time ($s$)", fontsize=14)
   ax.tick_params(labelsize=10)
   ax.legend(ncol=1, fontsize=10, frameon=False, loc='upper right')

   plt.tight_layout()
   fname = title.lower().replace("-","_").replace(" ", "_") + "_accuracy_comparison.pdf"
   fig.savefig(os.path.join(outpath, fname), bbox_inches="tight", pad_inches=0)

def fit_y_vs_x(y_values:List[float], x_values:List[float])->Tuple[float,List[float]]:
   [slope, intercept] = np.polyfit(np.log(x_values), np.log(y_values), 1)
   return slope, [np.exp(intercept)*np.power(x, slope) for x in x_values]

def rmse(estimate_values:List[float], risk_measure:float)->float:
   return math.sqrt(mean(map(lambda x: math.pow(x - risk_measure, 2), estimate_values)))

def average(values:List[float])->float:
   return mean(values)

def sub_process(filename:str, outpath:str, VaR:float)->None:
   cohort = 0

   accuracy_dict = {}
   time_sa_dict = defaultdict(list)
   time_nsa_dict = defaultdict(list)
   time_mlsa_dict = defaultdict(list)
   time_ada_nsa_dict = defaultdict(list)
   time_ada_mlsa_dict = defaultdict(list)
   time_ada_nsa_sd_dict = defaultdict(list)
   time_ada_mlsa_sd_dict = defaultdict(list)

   estimate_VaR_sa_dict = defaultdict(list)
   estimate_VaR_nsa_dict = defaultdict(list)
   estimate_VaR_mlsa_dict = defaultdict(list)
   estimate_VaR_ada_nsa_dict = defaultdict(list)
   estimate_VaR_ada_mlsa_dict = defaultdict(list)
   estimate_VaR_ada_nsa_sd_dict = defaultdict(list)
   estimate_VaR_ada_mlsa_sd_dict = defaultdict(list)

   with open(filename, "r") as instream:
      reader = csv.reader(instream)
      header = next(reader)
      for row in reader:
         if len(row) != len(header):
            continue
         d = dict(zip(header, row))
         if d.get("status") != "success":
            continue
         if d.get("#") == "1":
            cohort += 1
            accuracy = float(d["accuracy"])
            accuracy_dict[cohort] = accuracy

         time_sa_dict[cohort].append(float(d["time_sa"]))
         time_nsa_dict[cohort].append(float(d["time_nsa"]))
         time_mlsa_dict[cohort].append(float(d["time_mlsa"]))
         time_ada_nsa_dict[cohort].append(float(d["time_ada_nsa"]))
         time_ada_mlsa_dict[cohort].append(float(d["time_ada_mlsa"]))
         time_ada_nsa_sd_dict[cohort].append(float(d["time_ada_nsa_sd"]))
         time_ada_mlsa_sd_dict[cohort].append(float(d["time_ada_mlsa_sd"]))

         estimate_VaR_sa_dict[cohort].append(float(d["VaR_sa"]))
         estimate_VaR_nsa_dict[cohort].append(float(d["VaR_nsa"]))
         estimate_VaR_mlsa_dict[cohort].append(float(d["VaR_mlsa"]))
         estimate_VaR_ada_nsa_dict[cohort].append(float(d["VaR_ada_nsa"]))
         estimate_VaR_ada_mlsa_dict[cohort].append(float(d["VaR_ada_mlsa"]))
         estimate_VaR_ada_nsa_sd_dict[cohort].append(float(d["VaR_ada_nsa_sd"]))
         estimate_VaR_ada_mlsa_sd_dict[cohort].append(float(d["VaR_ada_mlsa_sd"]))

   average_time_sa = []
   average_time_nsa = []
   average_time_mlsa = []
   average_time_ada_nsa = []
   average_time_ada_mlsa = []
   average_time_ada_nsa_sd = []
   average_time_ada_mlsa_sd = []

   rmse_VaR_sa = []
   rmse_VaR_nsa = []
   rmse_VaR_mlsa = []
   rmse_VaR_ada_nsa = []
   rmse_VaR_ada_mlsa = []
   rmse_VaR_ada_nsa_sd = []
   rmse_VaR_ada_mlsa_sd = []

   #accuracies = sorted(accuracies)[::-1]
   n_cohort = cohort
   for cohort in range(1, n_cohort+1):
      average_time_sa.append(average(time_sa_dict[cohort]))
      average_time_nsa.append(average(time_nsa_dict[cohort]))
      average_time_mlsa.append(average(time_mlsa_dict[cohort]))
      average_time_ada_nsa.append(average(time_ada_nsa_dict[cohort]))
      average_time_ada_mlsa.append(average(time_ada_mlsa_dict[cohort]))
      average_time_ada_nsa_sd.append(average(time_ada_nsa_sd_dict[cohort]))
      average_time_ada_mlsa_sd.append(average(time_ada_mlsa_sd_dict[cohort]))

      rmse_VaR_sa.append(rmse(estimate_VaR_sa_dict[cohort], VaR))
      rmse_VaR_nsa.append(rmse(estimate_VaR_nsa_dict[cohort], VaR))
      rmse_VaR_mlsa.append(rmse(estimate_VaR_mlsa_dict[cohort], VaR))
      rmse_VaR_ada_nsa.append(rmse(estimate_VaR_ada_nsa_dict[cohort], VaR))
      rmse_VaR_ada_nsa_sd.append(rmse(estimate_VaR_ada_nsa_sd_dict[cohort], VaR))
      rmse_VaR_ada_mlsa.append(rmse(estimate_VaR_ada_mlsa_dict[cohort], VaR))
      rmse_VaR_ada_mlsa_sd.append(rmse(estimate_VaR_ada_mlsa_sd_dict[cohort], VaR))

   VaR_sa_slope, VaR_sa_fit = fit_y_vs_x(average_time_sa, rmse_VaR_sa)
   VaR_nsa_slope, VaR_nsa_fit = fit_y_vs_x(average_time_nsa, rmse_VaR_nsa)
   VaR_mlsa_slope, VaR_mlsa_fit = fit_y_vs_x(average_time_mlsa, rmse_VaR_mlsa)
   VaR_ada_nsa_slope, VaR_ada_nsa_fit = fit_y_vs_x(average_time_ada_nsa, rmse_VaR_ada_nsa)
   VaR_ada_nsa_sd_slope, VaR_ada_nsa_sd_fit = fit_y_vs_x(average_time_ada_nsa_sd, rmse_VaR_ada_nsa_sd)
   VaR_ada_mlsa_slope, VaR_ada_mlsa_fit = fit_y_vs_x(average_time_ada_mlsa, rmse_VaR_ada_mlsa)
   VaR_ada_mlsa_sd_slope, VaR_ada_mlsa_sd_fit = fit_y_vs_x(average_time_ada_mlsa_sd, rmse_VaR_ada_mlsa_sd)

   print("Time vs RMSE")
   print("VaR SA complexity exponent: %f" %VaR_sa_slope)
   print("VaR NSA complexity exponent: %f" %VaR_nsa_slope)
   print("VaR MLSA complexity exponent: %f" %VaR_mlsa_slope)
   print("VaR AdaNSA complexity exponent: %f" %VaR_ada_nsa_slope)
   print("VaR AdaNSA SD complexity exponent: %f" %VaR_ada_nsa_sd_slope)
   print("VaR AdaMLSA complexity exponent: %f" %VaR_ada_mlsa_slope)
   print("VaR AdaMLSA SD complexity exponent: %f" %VaR_ada_mlsa_sd_slope)
   print("\n")

   accuracies = [accuracy_dict[cohort] for cohort in range(1, len(accuracy_dict)+1)]

   VaR_sa_slope, VaR_sa_accuracy_fit = fit_y_vs_x(average_time_sa, accuracies)
   VaR_nsa_slope, VaR_nsa_accuracy_fit = fit_y_vs_x(average_time_nsa, accuracies)
   VaR_mlsa_slope, VaR_mlsa_accuracy_fit = fit_y_vs_x(average_time_mlsa, accuracies)
   VaR_ada_nsa_slope, VaR_ada_nsa_accuracy_fit = fit_y_vs_x(average_time_ada_nsa, accuracies)
   VaR_ada_nsa_sd_slope, VaR_ada_nsa_sd_accuracy_fit = fit_y_vs_x(average_time_ada_nsa_sd, accuracies)
   VaR_ada_mlsa_slope, VaR_ada_mlsa_accuracy_fit = fit_y_vs_x(average_time_ada_mlsa, accuracies)
   VaR_ada_mlsa_sd_slope, VaR_ada_mlsa_sd_accuracy_fit = fit_y_vs_x(average_time_ada_mlsa_sd, accuracies)

   print("Time vs accuracy")
   print("VaR SA complexity exponent: %f" %VaR_sa_slope)
   print("VaR NSA complexity exponent: %f" %VaR_nsa_slope)
   print("VaR MLSA complexity exponent: %f" %VaR_mlsa_slope)
   print("VaR AdaNSA complexity exponent: %f" %VaR_ada_nsa_slope)
   print("VaR AdaNSA SD complexity exponent: %f" %VaR_ada_nsa_sd_slope)
   print("VaR AdaMLSA complexity exponent: %f" %VaR_ada_mlsa_slope)
   print("VaR AdaMLSA SD complexity exponent: %f" %VaR_ada_mlsa_sd_slope)
   print("\n")

   summary = {
      "SA": {
         "time": average_time_sa,
         "rmse": rmse_VaR_sa,
         "time_fit": VaR_sa_fit,
         "accuracy_time_fit": VaR_sa_accuracy_fit,
         "symbol": "-s",
         "color": "xkcd:green",
         "alignment": "left",
      }, "NSA": {
         "time": average_time_nsa,
         "rmse": rmse_VaR_nsa,
         "time_fit": VaR_nsa_fit,
         "accuracy_time_fit": VaR_nsa_accuracy_fit,
         "symbol": "-^",
         "symbol_size": 8,
         "color": "xkcd:blue",
         "alignment": "right",
      }, "MLSA": {
         "time": average_time_mlsa,
         "rmse": rmse_VaR_mlsa,
         "time_fit": VaR_mlsa_fit,
         "accuracy_time_fit": VaR_mlsa_accuracy_fit,
         "symbol": "-o",
         "symbol_size": 10,
         "color": "xkcd:orange",
         "alignment": "left",
      }, "adNSA": {
         "time": average_time_ada_nsa,
         "rmse": rmse_VaR_ada_nsa,
         "time_fit": VaR_ada_nsa_fit,
         "accuracy_time_fit": VaR_ada_nsa_accuracy_fit,
         "symbol": "-X",
         "symbol_size": 8,
         "color": "xkcd:grey",
      }, "adMLSA": {
         "time": average_time_ada_mlsa,
         "rmse": rmse_VaR_ada_mlsa,
         "time_fit": VaR_ada_mlsa_fit,
         "accuracy_time_fit": VaR_ada_mlsa_accuracy_fit,
         "symbol": "-*",
         "symbol_size": 12,
         "color": "xkcd:violet",
      }, "$\sigma$-adNSA": {
         "time": average_time_ada_nsa_sd,
         "rmse": rmse_VaR_ada_nsa_sd,
         "time_fit": VaR_ada_nsa_sd_fit,
         "accuracy_time_fit": VaR_ada_nsa_sd_accuracy_fit,
         "symbol": "-d",
         "symbol_size": 8,
         "color": "xkcd:gold",
      }, "$\sigma$-adMLSA": {
         "time": average_time_ada_mlsa_sd,
         "rmse": rmse_VaR_ada_mlsa_sd,
         "time_fit": VaR_ada_mlsa_sd_fit,
         "accuracy_time_fit": VaR_ada_mlsa_sd_accuracy_fit,
         "symbol": "-v",
         "symbol_size": 8,
         "color": "xkcd:brown",
      },
   }

   plot_time_vs_rmse("Value-at-risk", ("NSA", "$\sigma$-adNSA", "adNSA", "MLSA", "adMLSA", "$\sigma$-adMLSA", "SA"), accuracy_dict, summary, outpath)
   plot_time_vs_accuracy("Value-at-risk", ("NSA", "$\sigma$-adNSA", "adNSA", "MLSA", "adMLSA", "$\sigma$-adMLSA", "SA"), accuracy_dict, summary, outpath)

def compute_option_model_risk_measures(tau:float, alpha:float)->Tuple[float,float]:
   """ Compute exact values of the risk measures """
   VaR = tau*(math.pow(norm.ppf((1 - alpha)/2), 2) - 1)
   x = math.sqrt(1 + VaR/tau)
   return VaR

def compute_rate_swap_model_risk_measures(
   r:float, S_0:float, kappa:float, sigma:float,
   Delta:float, T:float, delta:float, # in days
   leg_0:float, alpha:float)->Tuple[float,float]:
   """ Compute exact values of the risk measures """

   def discount(t):
      return math.exp(-r*t)
   def reset(i):
      return i*Delta

   Delta = Delta/360.0
   T = T/360.0
   delta = delta/360.0
   n = int(T/Delta)

   nominal = leg_0/(S_0*sum(discount(reset(i))*Delta*math.exp(kappa*reset(i-1)) for i in range(1,n+1)))

   VaR = nominal*S_0*sum(discount(reset(i))*Delta*math.exp(kappa*reset(i-1)) for i in range(2,n+1))* \
         (math.exp(norm.ppf(alpha)*sigma*math.sqrt(delta)-sigma**2*delta/2)-1)

   omega = S_0 + VaR/(nominal*sum(discount(reset(i))*Delta*math.exp(kappa*reset(i-1)) for i in range(2,n+1)))
   d = (math.log(omega/S_0)-sigma**2*delta/2)/(sigma*math.sqrt(delta))

   return VaR

def process_test_case_1(filename:str, outpath:str)->None:
   alpha = 0.975 # 0.0 < alpha < 1.0
   tau = 0.5 # 0.0 < tau < 1.0

   VaR = compute_option_model_risk_measures(tau, alpha)
   print(f"VaR: {VaR}")
   print("\n")

   sub_process(filename, outpath, VaR)

def process_test_case_2(filename:str, outpath:str)->None:
   r = 0.02
   S_0 = 100.0 # in basis points
   kappa = 0.12
   sigma = 0.2
   Delta = 90.0 # in days
   T = 360.0 # in days
   delta = 7.0 # in days
   leg_0 = 1e4 # in basis points
   alpha = 0.85

   VaR = compute_rate_swap_model_risk_measures(r, S_0, kappa, sigma, Delta, T, delta, leg_0, alpha)
   print(f"VaR: {VaR}")
   print("\n")

   sub_process(filename, outpath, VaR)

def processor(test_case:str)->Callable[[str,str],None]:
   PROCESSOR_MAP = {
      "1": process_test_case_1,
      "2": process_test_case_2,
   }
   _processor = PROCESSOR_MAP.get(test_case)
   if _processor is None:
      raise NotImplementedError(f"Test set {test_case} is not implemented")
   return _processor

if __name__ == "__main__":
   p = argparse.ArgumentParser(description="Process and plot SA results")
   p.add_argument("input_csv", help="csv file name")
   p.add_argument("output_path", help="Output path for figures")
   p.add_argument("--test_case", help="Test set to process", choices=["0","1","2"])
   args = p.parse_args()

   if not os.path.exists(args.input_csv) or not os.path.isfile(args.input_csv):
      raise FileNotFoundError(f"file {args.input_csv} not found")
   if not os.path.exists(args.output_path) or not os.path.isdir(args.output_path):
      raise IOError(f"directory {args.output_path} not found")

   processor(args.test_case)(args.input_csv, args.output_path)
