# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:22:13 2020

@author: donbo
"""

import src.make_test_problems as mtp
import src.microweight as mw

# p = mtp.Problem(h=1000, s=10, k=5, xsd=.1, ssd=.5)
p = mtp.Problem(h=10000, s=20, k=15)

prob = mw.Microweight(wh=p.wh, xmat=p.xmat, geotargets=p.targets)

ans = prob.geoweight()
dir(ans)
ans.targets_opt
dir(prob)
prob.result.iter_opt


