import numpy as np
from .dynamic_program import reconstruct

def select_partition(frontier,costs,penalty):
 ks=np.arange(1,frontier['feasible_max_segments']+1)
 rss=frontier['dp'][1:,frontier['n']]*costs.scale*costs.scale
 objectives=rss+np.longdouble(penalty)*(ks-1)
 best=np.min(objectives)
 # Relative machine-precision tie only; ascending K selects fewer segments.
 tolerance=np.finfo(np.longdouble).eps*64*np.maximum(np.abs(objectives),max(abs(best),np.finfo(np.longdouble).tiny))
 index=int(np.flatnonzero(objectives<=best+tolerance)[0]);k=int(ks[index]);parts=reconstruct(frontier,k)
 segments=[dict(position=i+1,**costs.segment(start,end)) for i,(start,end) in enumerate(parts)]
 return {'penalty':float(penalty),'segments_count':k,'breakpoints':[end for _,end in parts[:-1]],'rss':float(rss[index]),'objective':float(objectives[index]),'segments':segments,'adjacent_mean_differences':[{'breakpoint':left['end'],'left_mean':left['mean'],'right_mean':right['mean'],'difference':right['mean']-left['mean']} for left,right in zip(segments,segments[1:])]}

def evaluate_path(frontier,costs,penalties):
 return [select_partition(frontier,costs,b) for b in sorted(set(penalties))]

def _monotone_rss(frontier,costs,k):
 value=frontier['best_value'][k]
 return np.longdouble(value)*costs.scale*costs.scale if np.isfinite(value) else np.longdouble(np.inf)

def select_monotone_partition(frontier,costs,penalty):
 from .monotone_dp import reconstruct as reconstruct_monotone
 ks=np.arange(1,frontier['feasible_max_segments']+1)
 rss=np.array([_monotone_rss(frontier,costs,int(k)) for k in ks],dtype=np.longdouble)
 objectives=rss+np.longdouble(penalty)*(ks-1)
 finite=np.isfinite(objectives)
 if not finite.any():raise ValueError('没有满足均温单调不降的分段')
 best=np.min(objectives[finite])
 # Same tie rule: relative longdouble tolerance, ascending K prefers fewer segments.
 tolerance=np.finfo(np.longdouble).eps*64*np.maximum(np.abs(objectives[finite]),max(abs(best),np.finfo(np.longdouble).tiny))
 k=int(ks[finite][objectives[finite]<=best+tolerance][0])
 parts=reconstruct_monotone(frontier,k)
 segments=[dict(position=i+1,**costs.segment(start,end)) for i,(start,end) in enumerate(parts)]
 # Verify the constraint and cost in the same longdouble units the DP used;
 # the serialized floats above cannot reliably distinguish near-equal means.
 normalized_means=[(costs.s[end]-costs.s[start])/(end-start) for start,end in parts]
 if any(normalized_means[i]>normalized_means[i+1]+frontier['mean_tolerance'] for i in range(len(parts)-1)):raise ValueError('约束DP回溯结果未满足均温单调不降')
 reconstructed=sum(costs.normalized([start],end)[0] for start,end in parts)
 optimal=frontier['best_value'][k]
 if abs(reconstructed-optimal)>np.longdouble(1e-12)*max(abs(optimal),np.longdouble(1)):raise ValueError('约束DP回溯 RSS 与最优值不一致')
 rss_value=float(optimal*costs.scale*costs.scale)
 objective=rss_value+float(penalty)*(k-1)
 return {'penalty':float(penalty),'segments_count':k,'breakpoints':[end for _,end in parts[:-1]],'rss':rss_value,'objective':objective,'segments':segments,'adjacent_mean_differences':[{'breakpoint':left['end'],'left_mean':left['mean'],'right_mean':right['mean'],'difference':right['mean']-left['mean']} for left,right in zip(segments,segments[1:])]}

def evaluate_monotone_path(frontier,costs,penalties):
 return [select_monotone_partition(frontier,costs,b) for b in sorted(set(penalties))]
