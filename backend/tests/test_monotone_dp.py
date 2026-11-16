import time
import unittest

import numpy as np

from src.services.costs import StableCosts
from src.services.dynamic_program import build_frontier as build_free_frontier
from src.services.monotone_dp import build_frontier as build_monotone_frontier,reconstruct,MonotoneSolveTimeout
from src.services.penalty_path import select_partition,select_monotone_partition
from src.services.validation import validate_config


def brute_force(values,min_size,max_segments):
 sums=np.r_[0,np.cumsum(values,dtype=np.longdouble)]
 squares=np.r_[0,np.cumsum(values*values,dtype=np.longdouble)]
 def lengths(total,k):
  if k==1:
   if total>=min_size:yield (total,)
   return
  for first in range(min_size,total-min_size*(k-1)+1):
   for rest in lengths(total-first,k-1):yield (first,)+rest
 def rss(a,b):
  return float((squares[b]-squares[a])-(sums[b]-sums[a])**2/(b-a))
 best={}
 for k in range(1,min(max_segments,len(values)//min_size)+1):
  chosen=None
  for lens in lengths(len(values),k):
   edges=np.cumsum(lens);starts=np.r_[0,edges[:-1]]
   means=[float(sums[b]-sums[a])/(b-a) for a,b in zip(starts,edges)]
   if any(left>right for left,right in zip(means,means[1:])):continue
   value=sum(rss(a,b) for a,b in zip(starts,edges))
   if chosen is None or value<chosen:chosen=value
  best[k]=chosen
 return best


class MonotoneDPTest(unittest.TestCase):
 def test_matches_brute_force_and_stays_ordered(self):
  rng=np.random.default_rng(4242)
  for trial in range(40):
   n=int(rng.integers(4,21));min_size=int(rng.integers(1,max(2,n//3+1)));max_segments=5
   values=np.round(rng.normal(0,2,n),2) if trial%2 else rng.integers(0,4,n).astype(float)
   costs=StableCosts(values);frontier=build_monotone_frontier(costs,min_size,max_segments)
   expected=brute_force(values,min_size,max_segments)
   for k in range(1,frontier['feasible_max_segments']+1):
    if expected[k] is None:
     self.assertFalse(np.isfinite(frontier['best_value'][k]),f'trial {trial} k={k}')
     continue
    self.assertTrue(np.isfinite(frontier['best_value'][k]),f'trial {trial} k={k}')
    self.assertAlmostEqual(float(frontier['best_value'][k])*float(costs.scale)**2,expected[k],places=10)
    parts=reconstruct(frontier,k)
    means=[float(np.sum(costs.y[a:b]))/(b-a) for a,b in parts]
    self.assertTrue(all(left<=right+1e-12 for left,right in zip(means,means[1:])))
    self.assertTrue(all(b-a>=min_size for a,b in parts))
    self.assertEqual((parts[0][0],parts[-1][1]),(0,n))

 def test_free_segmentation_result_is_unchanged(self):
  rng=np.random.default_rng(224);n=300
  values=np.r_[np.full(90,165.),np.full(110,181.),np.full(100,173.)]+rng.normal(0,1.2,n)
  costs=StableCosts(values);selected=select_partition(build_free_frontier(costs,12,8),costs,80.)
  self.assertEqual(selected['breakpoints'],[90,200])
  self.assertAlmostEqual(selected['rss'],356.09341455693374,places=9)
  self.assertAlmostEqual(selected['objective'],516.0934145569337,places=8)

 def test_monotone_rejects_falling_step_instead_of_merging_afterwards(self):
  rng=np.random.default_rng(224);n=300
  values=np.r_[np.full(90,165.),np.full(110,181.),np.full(100,173.)]+rng.normal(0,1.2,n)
  costs=StableCosts(values)
  selected=select_monotone_partition(build_monotone_frontier(costs,12,8),costs,80.)
  self.assertEqual(selected['breakpoints'],[90])
  means=[s['mean'] for s in selected['segments']]
  self.assertTrue(all(left<=right for left,right in zip(means,means[1:])))
  self.assertAlmostEqual(selected['rss'],3778.1910269435507,places=6)
  self.assertAlmostEqual(selected['objective'],3858.1910269435507,places=6)

 def test_n600_k12_is_solved_exactly_and_quickly(self):
  rng=np.random.default_rng(11);n=600;t=np.arange(n)
  values=160+t*.04+3*np.sin(t/13)+rng.normal(0,1,n)
  costs=StableCosts(values);started=time.monotonic()
  frontier=build_monotone_frontier(costs,1,12,deadline=time.monotonic()+30)
  self.assertLess(time.monotonic()-started,30)
  self.assertEqual(frontier['feasible_max_segments'],12)
  for k in range(1,13):
   self.assertTrue(np.isfinite(frontier['best_value'][k]))
   parts=reconstruct(frontier,k)
   means=[float(np.sum(costs.y[a:b]))/(b-a) for a,b in parts]
   self.assertTrue(all(left<=right+1e-12 for left,right in zip(means,means[1:])))

 def test_overrun_raises_instead_of_returning_partial_frontier(self):
  costs=StableCosts(np.r_[np.arange(300.),np.arange(300.)[::-1]])
  with self.assertRaises(MonotoneSolveTimeout):
   build_monotone_frontier(costs,1,12,deadline=time.monotonic()-1)

 def test_config_mode_is_validated(self):
  base={'min_size':12,'max_segments':8,'penalty':80.,'penalties':[0.,20.]}
  self.assertEqual(validate_config({**base,'mode':'monotone_mean'},300),8)
  self.assertEqual(validate_config(base,300),8)
  with self.assertRaises(ValueError):validate_config({**base,'mode':'isotonic'},300)


if __name__=='__main__':
 unittest.main()
