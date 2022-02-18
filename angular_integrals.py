#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 11:39:52 2021

@author: noumanbutt
"""

# routine for checking angular matrix elements

import numpy as np
from scipy.special import spherical_jn
from scipy import integrate
from itertools import product,permutations
import time
from scipy import arange, pi, sqrt, zeros
from scipy.special import jv, jvp
from scipy.optimize import brentq
from sys import argv
#import matplotlib.pyplot as plt



import sympy

from sympy import S
from sympy.physics.wigner import wigner_3j
from sympy.physics.wigner import clebsch_gordan
from sympy.physics.wigner import gaunt

def delta(a,b):
    if(a==b):
      return 1.0
    else: 
      return 0.0


def angular_integral_xsq1p(l1p,l1,m1p,m1):
    
    val1 = np.sqrt(2*np.pi/15)*(float(gaunt(l1p,2,l1,-m1p,2,m1)) + float(gaunt(l1p,2,l1,-m1p,-2,m1)))
    
    val2 = (2./3.)*np.sqrt(np.pi)*(float(gaunt(l1p,0,l1,-m1p,0,m1)) - np.sqrt(1./5.)*float(gaunt(l1p,2,l1,-m1p,0,m1)))
    
    return (val1+val2)

def angular_integral_ysq1p(l1p,l1,m1p,m1):
    
    val1 = -1*np.sqrt(2*np.pi/15)*(float(gaunt(l1p,2,l1,-m1p,2,m1)) + float(gaunt(l1p,2,l1,-m1p,-2,m1)))
    
    val2 = (2./3.)*np.sqrt(np.pi)*(float(gaunt(l1p,0,l1,-m1p,0,m1)) - np.sqrt(1./5)*float(gaunt(l1p,2,l1,-m1p,0,m1)))
    
    return (val1+val2)

def angular_integral_zsq1p(l1p,l1,m1p,m1):
    
    return (1./3.)*( 4.*np.sqrt(np.pi/5.)*float(gaunt(l1p,2,l1,-m1p,0,m1)) + 2.*np.sqrt(np.pi)*float(gaunt(l1p,0,l1,-m1p,0,m1)))


def angular_integral_xsq2p(l1p,l1,l2p,l2,l3):
    val = 0.0
    for m1p in range(-l1p,l1p+1):
        for m2p in range(-l2p,l2p+1):
                  for m1 in range(-l1,l1+1):
                      for m2 in range(-l2,l2+1):
                          for m3 in range(-l3,l3+1):
                              
                              
                              m3p = m3
                              if((m1+m2+m3 != 0) and (m1p+m2p+m3p) != 0):
                                 continue
                              else:
                                 val += float(wigner_3j(l1p,l2p,l3,m1p,m2p,m3))*float(wigner_3j(l1,l2,l3,m1,m2,m3))*((-1)**(m1+m2))*angular_integral_xsq1p(l1p,l1,m1p,m1)*angular_integral_xsq1p(l2p,l2,m2p,m2)
                              
                              
                              
    return val

def angular_integral_xy1p(l1p,l1,m1p,m1):
    
    val1 = (float(gaunt(l1p,2,l1,-m1p,2,m1)) - float(gaunt(l1p,2,l1,-m1p,-2,m1)))
    
    
    return (np.sqrt(2.*pi/15)*val1)


def angular_integral_xysq2p(l1p,l1,l2p,l2,l3,vals_xy1p,wigner_3j_vals):
    val = 0.0
    for m1p in range(-l1p,l1p+1):
        for m2p in range(-l2p,l2p+1):
                  for m1 in range(-l1,l1+1):
                      for m2 in range(-l2,l2+1):
                          for m3 in range(-l3,l3+1):
                              m3p = m3
                              if((m1+m2+m3 != 0)and (m1p+m2p+m3p) != 0):
                                 continue
                              else:
                                 val += wigner_3j_vals[(l1p,l2p,l3,m1p,m2p,m3p)]*wigner_3j_vals[(l1,l2,l3,m1,m2,m3)]*(-1)*((-1)**(m1+m2))*vals_xy1p[(l1p,l1,m1p,m1)]*vals_xy1p[(l2p,l2,m2p,m2)]
    return val


def angular_integral_xz1p(l1p,l1,m1p,m1):
    
    val1 = (float(gaunt(l1p,2,l1,-m1p,1,m1)) - float(gaunt(l1p,2,l1,-m1p,-1,m1)))
    
    
    return (np.sqrt(2.*pi/15)*val1)

def angular_integral_xzsq2p(l1p,l1,l2p,l2,l3):
    val = 0.0
    for m1p in range(-l1p,l1p+1):
        for m2p in range(-l2p,l2p+1):
                  for m1 in range(-l1,l1+1):
                      for m2 in range(-l2,l2+1):
                          for m3 in range(-l3,l3+1):
                              
                              m3p = m3
                              if((m1+m2+m3 != 0)and (m1p+m2p+m3p) != 0):
                                 continue
                              else:
                                 val += float(wigner_3j(l1p,l2p,l3,m1p,m2p,m3p))*float(wigner_3j(l1,l2,l3,m1,m2,m3))*((-1)**(m1+m2))*angular_integral_xz1p(l1p,l1,m1p,m1)*angular_integral_xz1p(l2p,l2,m2p,m2)
                              
                              
                              
    return val



def angular_integral_yz1p(l1p,l1,m1p,m1):
    
    val1 = (float(gaunt(l1p,2,l1,-m1p,1,m1)) + float(gaunt(l1p,2,l1,-m1p,-1,m1)))
    
    
    return (np.sqrt(2.*pi/15)*val1)


def angular_integral_yzsq2p(l1p,l1,l2p,l2,l3):
    val = 0.0
    for m1p in range(-l1p,l1p+1):
        for m2p in range(-l2p,l2p+1):
                  for m1 in range(-l1,l1+1):
                      for m2 in range(-l2,l2+1):
                          for m3 in range(-l3,l3+1):
                              
                              m3p = m3
                              if((m1+m2+m3 != 0)and (m1p+m2p+m3p) != 0):
                                 continue
                              else:
                                 val += float(wigner_3j(l1p,l2p,l3,m1p,m2p,m3p))*float(wigner_3j(l1,l2,l3,m1,m2,m3))*(-1)*((-1)**(m1+m2))*angular_integral_yz1p(l1p,l1,m1p,m1)*angular_integral_yz1p(l2p,l2,m2p,m2)
                              
                              
                              
    return val



def angular_integral_ysq2p(l1p,l1,l2p,l2,l3):
    val = 0.0
    for m1p in range(-l1p,l1p+1):
        for m2p in range(-l2p,l2p+1):
                  for m1 in range(-l1,l1+1):
                      for m2 in range(-l2,l2+1):
                          for m3 in range(-l3,l3+1):
                              
                              
                              m3p = m3
                              if((m1+m2+m3 != 0)and (m1p+m2p+m3p) != 0):
                                 continue
                              else:
                                 val += float(wigner_3j(l1p,l2p,l3,m1p,m2p,m3))*float(wigner_3j(l1,l2,l3,m1,m2,m3))*((-1)**(m1+m2))*angular_integral_ysq1p(l1p,l1,m1p,m1)*angular_integral_ysq1p(l2p,l2,m2p,m2)
                              
                              
                              
    return val

def angular_integral_zsq2p(l1p,l1,l2p,l2,l3,vals_z1p,wigner_3j_vals):
    val = 0.0
    #time0= time.time()
    for m1p in range(-l1p,l1p+1):
        for m2p in range(-l2p,l2p+1):
                  for m1 in range(-l1,l1+1):
                      for m2 in range(-l2,l2+1):
                          for m3 in range(-l3,l3+1):
                              m3p = m3
                              if((m1+m2+m3 != 0)and (m1p+m2p+m3p) != 0):
                                 continue
                              else:
                                  #print("updating value")
                                  #l += 1.0
                                 val+= wigner_3j_vals[(l1p,l2p,l3,m1p,m2p,m3p)]*wigner_3j_vals[(l1,l2,l3,m1,m2,m3)]*((-1)**(m1+m2))*vals_z1p[(l1p,l1,m1p,m1)]*vals_z1p[(l2p,l2,m2p,m2)]
    #time1= time.time()
    #print("time to run the loops: ",time1-time0)
    return val

def wigner_3js(lmax):
    time0 = time.time()
    wigner_3j_vals={}
    for l1,l2,l3 in product(range(0,lmax+1),repeat=3):
           
        
         if((abs(l1-l2) <= l3 <= (l1+l2)) and (l1%2 == 0) and (l2%2 == 0) and (l3%2 == 0)):
            
            for m1 in range(-l1,l1+1):
                for m2 in range(-l2,l2+1):
                    for m3 in range(-l3,l3+1):
                        
                        if(m1+m2+m3 != 0):
                         wigner_3j_vals[(l1,l2,l3,m1,m2,m3)] = 0.0
                        else:
                         wigner_3j_vals[(l1,l2,l3,m1,m2,m3)] = float(wigner_3j(l1,l2,l3,m1,m2,m3))
         else:
             continue               
    time1 = time.time()
    print("time for computing wigner-3j values ",time1 - time0)                    
    return wigner_3j_vals                   


def angular_integrals_values_1p(lmax):
    time0 = time.time()
    vals_z1p = {}
    vals_xy1p = {}
    for l1p in range(0,lmax+1):
        if(l1p == 0):
          l1s = [0,2]
        elif(l1p == 1):
          l1s = [1,3]
        elif(2 <= l1p <= (lmax-2)):
          l1s= [l1p-2, l1p, l1p+2]
        elif(l1p > lmax-2):
          l1s= [l1p-2,l1p]
        for l1 in l1s:
              if(l1 > lmax):
                   continue
              else:
                 for m1p in range(-l1p,l1p+1):
                     for m1 in range(-l1,l1+1):
                        vals_z1p[(l1p,l1,m1p,m1)]  =   angular_integral_zsq1p(l1p, l1, m1p, m1)
                        vals_xy1p[(l1p,l1,m1p,m1)] =   angular_integral_xy1p(l1p, l1, m1p, m1)
    time1=time.time()
    print("Time for computing 1-particle angular integrals: ",time1-time0)
    return vals_z1p,vals_xy1p
def angular_integral_values_2p(Ls,lmax):
    time0 = time.time()
    vals_zsq2p={}
    vals_xysq2p={}
    vals_z1p,vals_xy1p =  angular_integrals_values_1p(lmax)
    wigner_3j_vals = wigner_3js(lmax)
    
    for lsp in Ls:
         if(lsp[0]==lsp[1]==lsp[2]):
             l1p = lsp[0]
             l2p = lsp[1]
             l3p = lsp[2]
             if(l1p == 0):
               l1s = [0,2]
             elif(l1p == 1):
               l1s = [1,3]
             elif(2 <= l1p <= (lmax-2)):
               l1s = [l1p-2, l1p, l1p+2]
             elif(l1p > lmax-2):
               l1s = [l1p-2,l1p]
             if(l2p == 0):
               l2s = [0,2]
             elif(l2p == 1):
               l2s = [1,3]
             elif(2 <= l2p <= (lmax-2)):
               l2s = [l2p-2, l2p, l2p+2]
             elif(l2p > lmax-2):
               l2s = [l2p-2,l2p]
             for l1 in l1s:
               for l2 in l2s:
                  if((l1 > lmax) or (l2 > lmax)):
                      continue
                  
                  elif(abs(l1-l2) <= l3p <= (l1+l2)):
                     vals_zsq2p[(l1p,l1,l2p,l2,l3p)] = angular_integral_zsq2p(l1p, l1, l2p, l2, l3p, vals_z1p,wigner_3j_vals)
                     vals_xysq2p[(l1p,l1,l2p,l2,l3p)] = angular_integral_xysq2p(l1p, l1, l2p, l2, l3p, vals_xy1p,wigner_3j_vals)
                  
                  else:
                      vals_zsq2p[(l1p,l1,l2p,l2,l3p)] = 0.0
                      vals_xysq2p[(l1p,l1,l2p,l2,l3p)] = 0.0
                  
             print("L values ",l1p,l2p,l3p)
         else:
          for [a,b,c] in list(permutations([0,1,2])):
           l1p = lsp[a]
           l2p = lsp[b]
           l3p = lsp[c]
           if(l1p == 0):
            l1s = [0,2]
           elif(l1p == 1):
            l1s = [1,3]
           elif(2 <= l1p <= (lmax-2)):
            l1s = [l1p-2, l1p, l1p+2]
           elif(l1p > lmax-2):
            l1s = [l1p-2,l1p]
           if(l2p == 0):
            l2s = [0,2]
           elif(l2p == 1):
            l2s = [1,3]
           elif(2 <= l2p <= (lmax-2)):
            l2s = [l2p-2, l2p, l2p+2]
           elif(l2p > lmax-2):
            l2s = [l2p-2,l2p]
           for l1 in l1s:
            for l2 in l2s:
               if((l1 > lmax) or (l2 > lmax)):
                   continue
               elif(abs(l1-l2) <= l3p <= (l1+l2)):
                   vals_zsq2p[(l1p,l1,l2p,l2,l3p)] = angular_integral_zsq2p(l1p, l1, l2p, l2, l3p, vals_z1p,wigner_3j_vals)
                   vals_xysq2p[(l1p,l1,l2p,l2,l3p)] = angular_integral_xysq2p(l1p, l1, l2p, l2, l3p, vals_xy1p,wigner_3j_vals)
               else:
                   vals_zsq2p[(l1p,l1,l2p,l2,l3p)] = 0.0
                   vals_xysq2p[(l1p,l1,l2p,l2,l3p)] = 0.0
           print("L values ",l1p,l2p,l3p)
    time1 = time.time()
    print("Time for computing XY^2 and Z^2: ",time1-time0)
    return vals_zsq2p,vals_xysq2p






