import time
import numpy as np
from .validation import validate_points,validate_config
from .costs import StableCosts
from .dynamic_program import build_frontier
from .penalty_path import select_partition,evaluate_path,select_monotone_partition,evaluate_monotone_path
from .fingerprint import fingerprint
from .monotone_dp import build_frontier as build_monotone_frontier,feasible_segments,TIME_BUDGET,check_deadline

def _series(points):
 return [{'sequence':p['sequence'],'recorded_at':p['recorded_at'].isoformat() if hasattr(p['recorded_at'],'isoformat') else p['recorded_at'],'temperature':p['temperature']} for p in points]

def _numeric(costs):
 return {'origin':float(costs.origin),'scale':float(costs.scale),'prefix_precision':'numpy.longdouble centered/scaled cumulative moments'}

def _common(result,batch,points,config,feasible,costs,series):
 result.update({'n':len(points),'feasible_max_segments':feasible,'requested_max_segments':config['max_segments'],'min_size':config['min_size'],'fingerprint':fingerprint(batch,points,config),'series':series,'numeric_stability':_numeric(costs),'boundary_definition':'breakpoint b is left sample count: zero-based [0,b) ends at displayed sequence b; next starts sequence b+1','interpretation':'Offline piecewise-constant mean approximation. Penalty path is sensitivity, not a confidence interval, online alarm, or causal attribution. Smooth drift can be approximated as steps; independent equal-variance segment errors are a working approximation.'})
 return result

def analyze_free(batch,points,config):
 points=validate_points(points);feasible=validate_config(config,len(points));costs=StableCosts([p['temperature'] for p in points]);frontier=build_frontier(costs,config['min_size'],config['max_segments']);selected=select_partition(frontier,costs,config['penalty']);path=evaluate_path(frontier,costs,config['penalties']+[config['penalty']])
 result={**selected,'mode':'free','mode_label':'自由分段（无均温顺序约束）','mode_constraint':'none','penalty_path':path,'rss_frontier':[{'segments_count':k,'optimal_rss':float(frontier['dp'][k,len(points)]*costs.scale*costs.scale),'feasible':True} for k in range(1,feasible+1)],'dp_builds':1,'tie_rule':'equal objective prefers fewer segments within relative longdouble tolerance; fixed-K uses earliest prior cut within relative longdouble roundoff tolerance (earliest last cut, recursively)'}
 return _common(result,batch,points,config,feasible,costs,_series(points))

def analyze_monotone(batch,points,config,time_budget=TIME_BUDGET):
 points=validate_points(points);feasible=validate_config(config,len(points));costs=StableCosts([p['temperature'] for p in points]);deadline=time.monotonic()+time_budget
 frontier=build_monotone_frontier(costs,config['min_size'],config['max_segments'],deadline=deadline)
 check_deadline(deadline)
 selected=select_monotone_partition(frontier,costs,config['penalty'])
 path=evaluate_monotone_path(frontier,costs,config['penalties']+[config['penalty']])
 check_deadline(deadline)
 frontier_rows=[]
 for k in range(1,feasible+1):
  value=frontier['best_value'][k]
  frontier_rows.append({'segments_count':k,'optimal_rss':float(value*costs.scale*costs.scale) if np.isfinite(value) else None,'feasible':bool(np.isfinite(value))})
 result={**selected,'mode':'monotone_mean','mode_label':'均温单调不降约束分段','mode_constraint':'segment means non-decreasing from first to last segment (relative longdouble tolerance)','penalty_path':path,'rss_frontier':frontier_rows,'dp_builds':1,'tie_rule':'equal objective prefers fewer segments within relative longdouble tolerance; fixed-K uses earliest last cut within relative longdouble roundoff tolerance, and among admissible prior cuts the earliest one that attains the minimum, recursively','monotone_feasible_segments':feasible_segments(frontier),'time_budget_seconds':float(time_budget)}
 return _common(result,batch,points,config,feasible,costs,_series(points))

def analyze(batch,points,config,time_budget=TIME_BUDGET):
 mode=config.get('mode','free')
 if mode=='monotone_mean':return analyze_monotone(batch,points,config,time_budget)
 if mode!='free':raise ValueError('未知分段方式')
 return analyze_free(batch,points,config)
