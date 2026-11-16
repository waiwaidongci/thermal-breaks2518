import time
import numpy as np

# Wall-clock budget for the constrained solve. Exceeding it raises instead of
# returning a partially built or approximate frontier.
TIME_BUDGET=30.0
_TINY=np.finfo(np.longdouble).tiny
_REL_TOL=np.finfo(np.longdouble).eps*64


class MonotoneSolveTimeout(RuntimeError):
 pass


def check_deadline(deadline):
 if deadline is not None and time.monotonic()>deadline:
  raise MonotoneSolveTimeout('均温单调不降约束求解超过时间预算；未返回近似结果，请缩小题量或稍后重试')


def _matrices(costs,deadline):
 n=costs.n;index=np.arange(n+1,dtype=np.int64)
 start=index[:,None];end=index[None,:];length=(end-start).astype(np.longdouble)
 with np.errstate(divide='ignore',invalid='ignore'):
  means=(costs.s[end]-costs.s[start])/length
  rss=(costs.q[end]-costs.q[start])-(costs.s[end]-costs.s[start])**2/length
 valid=end>start
 means=np.where(valid,means,np.longdouble(0))
 rss=np.where(valid,np.maximum(rss,np.longdouble(0)),np.longdouble(np.inf))
 check_deadline(deadline)
 return rss,means


def build_frontier(costs,min_size,max_segments,deadline=None):
 """Exact DP over partitions whose segment means are non-decreasing.

 D[k][b,a] is the minimum normalized RSS of k segments covering [0,b) with
 last segment [a,b). A transition into [a,b) only admits prior states whose
 last mean is <= mean([a,b)). Ordering is enforced inside every layer, so the
 optimum is searched jointly over breakpoints and means: no ordinary
 segmentation is built first and no reverse-order segments are merged.

 Ties: an intermediate state keeps the earliest prior cut that attains the
 exact minimum cost; the final cut for each K uses the same relative
 longdouble roundoff tolerance as the free segmentation. No random cuts.
 """
 n=costs.n;feasible=min(max_segments,n//min_size)
 if feasible<1:raise ValueError('最小段长大于样本数')
 check_deadline(deadline)
 rss,means=_matrices(costs,deadline)
 mean_tolerance=_REL_TOL*max(np.longdouble(np.max(np.abs(means))),np.longdouble(1))
 back=[None]*(feasible+1)
 best_value=np.full(feasible+1,np.inf,dtype=np.longdouble)
 best_start=np.full(feasible+1,-1,dtype=np.int32)
 previous=None
 if n>=min_size:
  previous=np.full((n+1,n+1),np.inf,dtype=np.longdouble)
  previous[min_size:,0]=rss[0,min_size:]
  best_value[1]=rss[0,n];best_start[1]=0
 for k in range(2,feasible+1):
  check_deadline(deadline)
  current=np.full((n+1,n+1),np.inf,dtype=np.longdouble)
  pointer=np.full((n+1,n+1),-1,dtype=np.int16)
  for cut in range((k-1)*min_size,n-min_size+1):
   if cut%32==0:check_deadline(deadline)
   starts=np.arange((k-2)*min_size,cut-min_size+1)
   prior=previous[cut,starts]
   finite=np.isfinite(prior)
   if not finite.any():continue
   starts=starts[finite];prior=prior[finite]
   # Order prior cuts by their last-mean, then compute the running minimum
   # value. Composite rank by (value,start) gives the earliest start among
   # ties without a Python loop.
   order=np.argsort(means[starts,cut],kind='stable')
   starts=starts[order];prior=prior[order];prior_means=means[starts,cut]
   by_value=np.lexsort((starts,prior))
   rank=np.empty(len(starts),dtype=np.int64);rank[by_value]=np.arange(len(starts),dtype=np.int64)
   running_rank=np.minimum.accumulate(rank)
   chosen=starts[by_value[running_rank]]
   prefix=np.minimum.accumulate(prior)
   ends=np.arange(cut+min_size,n+1)
   admitted=np.searchsorted(prior_means,means[cut,ends]+mean_tolerance,side='right')
   keep=admitted>0
   if not keep.any():continue
   ends=ends[keep];slot=admitted[keep]-1
   current[ends,cut]=prefix[slot]+rss[cut,ends]
   pointer[ends,cut]=chosen[slot]
  back[k]=pointer
  finals=np.arange((k-1)*min_size,n-min_size+1)
  values=current[n,finals];finite=np.isfinite(values)
  if finite.any():
   target=np.min(values[finite]);tolerance=_REL_TOL*max(np.abs(target),_TINY)
   eligible=finals[finite][values[finite]<=target+tolerance]
   best_value[k]=target;best_start[k]=int(eligible[0])
  previous=current
  check_deadline(deadline)
 return {'back':back,'best_value':best_value,'best_start':best_start,'feasible_max_segments':feasible,'mean_tolerance':mean_tolerance,'n':n,'min_size':min_size,'shift':costs.origin,'scale':costs.scale}


def feasible_segments(frontier):
 return [k for k in range(1,frontier['feasible_max_segments']+1) if np.isfinite(frontier['best_value'][k])]


def reconstruct(frontier,k):
 if k<1 or k>frontier['feasible_max_segments']:raise ValueError('段数超出可行范围')
 if not np.isfinite(frontier['best_value'][k]):raise ValueError('该段数没有满足均温单调不降的分段')
 if k==1:return [(0,frontier['n'])]
 end=frontier['n'];start=int(frontier['best_start'][k]);segments=[]
 while k>=2:
  prior=int(frontier['back'][k][end,start])
  if prior<0:raise ValueError('约束DP回溯失败')
  segments.append((start,end));end=start;start=prior;k-=1
 if end<=0:raise ValueError('约束DP回溯失败')
 segments.append((0,end))
 return list(reversed(segments))
