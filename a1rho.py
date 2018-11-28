import numpy as np
from particle import Particle
from math_utils import * 
import os

class A1RhoEvent(object):
    def __init__(self, data, args, debug=False):
        # [n, pi-, pi-, pi+, an, pi+, pi0]

        p = [Particle(data[:, 5 * i:5 * i + 4]) for i in range(7)]
        cols = []

        def get_tau1(p):
            p_tau1_nu = p[0]
            l_tau1_pi = p[1:4]                              # pi-, pi-, pi+
            p_tau1_a1 = sum(l_tau1_pi)

            l_tau1_rho = [None]*2
            l_tau1_rho[0] = l_tau1_pi[0] + l_tau1_pi[2]     # pi1- + pi+
            l_tau1_rho[1] = l_tau1_pi[1] + l_tau1_pi[2]     # pi2- + pi+

            return p_tau1_nu, p_tau1_a1, l_tau1_pi, l_tau1_rho

        def get_tau2(p):
            p_tau2_nu = p[4]
            l_tau2_pi = p[5:7]

            l_tau2_rho = [None]
            l_tau2_rho[0] = l_tau2_pi[0] + l_tau2_pi[1]     # pi+ + pi0

            return p_tau2_nu, l_tau2_pi, l_tau2_rho

        p_tau1_nu, p_tau1_a1, l_tau1_pi, l_tau1_rho = get_tau1(p)
        p_tau1_rho = sum(l_tau1_pi)

        p_tau2_nu, l_tau2_pi, l_tau2_rho = get_tau2(p)
        p_tau2_rho = sum(l_tau2_pi)

        p_a1_rho = sum(l_tau1_pi + l_tau2_pi)
        
        PHI, THETA = calc_angles(p_tau1_a1, p_a1_rho)
        lambda_noise = args.LAMBDA
 
        # all particles boosted & rotated
        for i, idx in enumerate([0, 1, 2, 3, 4, 5, 6]):
            part = boost_and_rotate(p[idx], PHI, THETA, p_a1_rho)
            if args.FEAT in ["Model-OnlyHad", "Model-Benchmark", "Model-1", "Model-2", "Model-3"]:
                if idx not in [0, 4]:
                    cols.append(part.vec)
            if args.FEAT == "Model-Oracle":
                if lambda_noise > 0:
                    cols.append(smear_exp(part.vec, lambda_noise))
                else:
                    cols.append(part.vec)

        # rho particles
        if args.FEAT == "Model-OnlyHad":
            for i, rho in enumerate(l_tau1_rho + l_tau2_rho):
                rho = boost_and_rotate(rho, PHI, THETA, p_a1_rho)
                cols.append(rho.vec)

        # recalculated masses
        if args.FEAT == "Model-OnlyHad":
            for i, part in enumerate(l_tau1_rho + l_tau2_rho + [p_tau1_a1]):
                cols.append(part.recalculated_mass)

        if args.FEAT == "Model-OnlyHad":
            for i in [1, 2]:
                rho = p[i] + p[3]
                other_pi = p[2 if i == 1 else 1]
                rho2 = p[5] + p[6]
                rho_rho = rho + rho2

                cols += [get_acoplanar_angle(p[i], p[3], p[5], p[6], rho_rho),
                         get_acoplanar_angle(rho, other_pi, p[5], p[6], p_a1_rho)]

                cols += [get_y(p[i], p[3], rho_rho), get_y(p[5], p[6], rho_rho),
                         get_y2(p_tau1_a1, rho, other_pi, p_a1_rho)]

        #------------------------------------------------------------

        pb_tau1_h = boost_and_rotate(p_tau1_rho, PHI, THETA, p_a1_rho)
        pb_tau2_h = boost_and_rotate(p_tau2_rho, PHI, THETA, p_a1_rho)
        pb_tau1_nu = boost_and_rotate(p_tau1_nu, PHI, THETA, p_a1_rho)
        pb_tau2_nu = boost_and_rotate(p_tau2_nu, PHI, THETA, p_a1_rho)

        #------------------------------------------------------------

        v_ETmiss_x = p_tau1_nu.x + p_tau2_nu.x
        v_ETmiss_y = p_tau1_nu.y + p_tau2_nu.y
        if args.FEAT == "Model-2":
            cols += [v_ETmiss_x, v_ETmiss_y]

        #------------------------------------------------------------

        if args.METHOD == "A":
            va_alpha1, va_alpha2 = approx_alpha_A(v_ETmiss_x, v_ETmiss_y, p_tau1_rho, p_tau2_rho)
        elif args.METHOD == "B":
            va_alpha1, va_alpha2 = approx_alpha_B(v_ETmiss_x, v_ETmiss_y, p_tau1_rho, p_tau2_rho)
        elif args.METHOD == "C":
            va_alpha1, va_alpha2 = approx_alpha_C(v_ETmiss_x, v_ETmiss_y, p_tau1_rho, p_tau2_rho)

        #------------------------------------------------------------

        va_tau1_nu_long = va_alpha1 * pb_tau1_h.z
        va_tau2_nu_long = va_alpha2 * pb_tau2_h.z
        va_tau1_nu_e = va_alpha1 * pb_tau1_h.e
        va_tau2_nu_e = va_alpha2 * pb_tau2_h.e

        #------------------------------------------------------------

        va_tau1_nu_E = approx_E_nu(pb_tau1_h, va_tau1_nu_long)
        va_tau2_nu_E = approx_E_nu(pb_tau2_h, va_tau2_nu_long)

        #------------------------------------------------------------

        va_tau1_nu_trans = np.sqrt(np.square(va_tau1_nu_E) - np.square(va_tau1_nu_long))
        va_tau2_nu_trans = np.sqrt(np.square(va_tau2_nu_E) - np.square(va_tau2_nu_long))

        #------------------------------------------------------------

        for i, elem1 in enumerate(va_tau1_nu_trans):
            if (np.isnan(elem1)):
                va_tau1_nu_trans[i] = 0

        for i, elem1 in enumerate(va_tau2_nu_trans):
            if (np.isnan(elem1)):
                va_tau2_nu_trans[i] = 0                  

        v_tau1_nu_phi = np.arctan2(pb_tau1_nu.x, pb_tau1_nu.y) #boosted
        v_tau2_nu_phi = np.arctan2(pb_tau2_nu.x, pb_tau2_nu.y)
        vn_tau1_nu_phi = smear_exp(v_tau1_nu_phi, lambda_noise)
        vn_tau2_nu_phi = smear_exp(v_tau2_nu_phi, lambda_noise)

        tau1_sin_phi = np.sin(vn_tau1_nu_phi)
        tau1_cos_phi = np.cos(vn_tau1_nu_phi)
        tau2_sin_phi = np.sin(vn_tau2_nu_phi)
        tau2_cos_phi = np.cos(vn_tau2_nu_phi)

        if args.FEAT in ["Model-1", "Model-2"]:
            cols += [va_tau1_nu_long, va_tau2_nu_long, va_tau1_nu_E, va_tau2_nu_E, va_tau1_nu_trans, va_tau2_nu_trans]
        elif args.FEAT == "Model-3":
            cols += [va_tau1_nu_long, va_tau2_nu_long, va_tau1_nu_E, va_tau2_nu_E, va_tau1_nu_trans * tau1_sin_phi, va_tau2_nu_trans * tau2_sin_phi, va_tau1_nu_trans * tau1_cos_phi, va_tau2_nu_trans * tau2_cos_phi]
        elif args.FEAT == "Model-Benchmark":
            cols += [va_tau1_nu_trans * tau1_sin_phi, va_tau2_nu_trans * tau2_sin_phi, va_tau1_nu_trans * tau1_cos_phi, va_tau2_nu_trans * tau2_cos_phi]

        # filter
        filt = (p_tau1_a1.pt >= 20)
        filt = filt & (l_tau2_rho[0].pt >= 1)
        for part in (l_tau1_pi + l_tau2_pi):
            filt = filt & (part.pt >= 1)    
        filt = filt.astype(np.float32)

        if args.FEAT in ["Model-Oracle", "Model-OnlyHad"]:
            cols += [filt]

        elif args.FEAT in ["Model-Benchmark", "Model-1", "Model-2", "Model-2"]:
            isFilter = np.full(p_a1_rho.e.shape, True, dtype=bool)
            for alpha in [va_alpha1, va_alpha2]:
                isFilter *= (alpha > 0)
            for energy in [va_tau1_nu_E, va_tau2_nu_E]:
                isFilter *= (energy > 0)
            for trans_comp in [va_tau1_nu_trans, va_tau2_nu_trans]:
                isFilter *= np.logical_not(np.isnan(trans_comp))
            cols += [filt * isFilter]

        for i in range(len(cols)):
            if len(cols[i].shape) == 1:
                cols[i] = cols[i].reshape([-1, 1])

        self.cols = np.concatenate(cols, 1)
