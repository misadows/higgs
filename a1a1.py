import numpy as np
from particle import Particle
from math_utils import * 
import os

class A1A1Event(object):
    def __init__(self, data, args, debug=False):
        # [n, pi-, pi-, pi+, an, pi+, pi+, pi-]
        p = [Particle(data[:, 5 * i:5 * i + 4]) for i in range(8)]
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
            l_tau2_pi = p[5:8]
            p_tau2_a1 = sum(l_tau2_pi)

            l_tau2_rho = [None]*2
            l_tau2_rho[0] = l_tau2_pi[0] + l_tau2_pi[2]     # pi1+ + pi-
            l_tau2_rho[1] = l_tau2_pi[1] + l_tau2_pi[2]     # pi2+ + pi-

            return p_tau2_nu, p_tau2_a1, l_tau2_pi, l_tau2_rho


        p_tau1_nu, p_tau1_a1, l_tau1_pi, l_tau1_rho = get_tau1(p)
        p_tau1_rho = sum(l_tau1_pi)

        p_tau2_nu, p_tau2_a1, l_tau2_pi, l_tau2_rho = get_tau2(p)
        p_tau2_rho = sum(l_tau2_pi)

        p_a1_a1 = sum(l_tau1_pi + l_tau2_pi)

        PHI, THETA = calc_angles(p_tau1_a1, p_a1_a1)
        lambda_noise = args.LAMBDA

        for i, idx in enumerate([0, 1, 2, 3, 4, 5, 6, 7]):
            part = boost_and_rotate(p[idx], PHI, THETA, p_a1_a1)
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
                rho = boost_and_rotate(rho, PHI, THETA, p_a1_a1)
                cols.append(rho.vec)

        # recalculated masses
        if args.FEAT == "Model-OnlyHad":
            for i, part in enumerate(l_tau1_rho + l_tau2_rho + [p_tau1_a1, p_tau2_a1]):
                part = boost_and_rotate(part, PHI, THETA, p_a1_a1)
                cols.append(part.recalculated_mass)

        if args.FEAT == "Model-OnlyHad":
            for i in [1, 2]:
                for ii in [5, 6]:
                    rho = p[i] + p[3]
                    other_pi = p[2 if i == 1 else 1]
                    rho2 = p[ii] + p[7]
                    other_pi2 = p[6 if i == 5 else 5]
                    rho_rho = rho + rho2
                    a1_rho = p_tau1_a1 + rho2
                    rho_a1 = rho + p_tau2_a1

                    cols += [get_acoplanar_angle(p[i], p[3], p[ii], p[7], rho_rho),
                             get_acoplanar_angle(rho, other_pi, p[ii], p[7], a1_rho),
                             get_acoplanar_angle(p[i], p[3], rho2, other_pi2, rho_a1),
                             get_acoplanar_angle(rho, other_pi, rho2, other_pi2, p_a1_a1)]

                    cols += [get_y(p[i], p[3], rho_rho), get_y(p[ii], p[7], rho_rho),
                             get_y2(p_tau1_a1, rho, other_pi, a1_rho), get_y(p[ii], p[7], a1_rho),
                             get_y(p[i], p[3], rho_a1), get_y2(p_tau2_a1, rho2, other_pi2, rho_a1),
                             get_y2(p_tau1_a1, rho, other_pi, p_a1_a1), get_y2(p_tau2_a1, rho2, other_pi2, p_a1_a1)]

        #------------------------------------------------------------

        pb_tau1_h = boost_and_rotate(p_tau1_rho, PHI, THETA, p_a1_a1)
        pb_tau2_h = boost_and_rotate(p_tau2_rho, PHI, THETA, p_a1_a1)
        pb_tau1_nu = boost_and_rotate(p_tau1_nu, PHI, THETA, p_a1_a1)
        pb_tau2_nu = boost_and_rotate(p_tau2_nu, PHI, THETA, p_a1_a1)

        #------------------------------------------------------------

        v_ETmiss_x = p_tau1_nu.x + p_tau2_nu.x
        v_ETmiss_y = p_tau1_nu.y + p_tau2_nu.y
        if args.FEAT == "Model-2":
            cols += [v_ETmiss_x, v_ETmiss_y]

        vr_ETmiss_x, vr_ETmiss_y = rotate_xy(v_ETmiss_x, v_ETmiss_y, PHI)

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

        #------------------------------------------------------------

        va_tau1_nu_e = va_alpha1 * pb_tau1_h.e
        va_tau2_nu_e = va_alpha2 * pb_tau2_h.e

        #------------------------------------------------------------

        va_tau1_nu_E = approx_E_nu(pb_tau1_h, va_tau1_nu_long)
        va_tau2_nu_E = approx_E_nu(pb_tau2_h, va_tau2_nu_long)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        va_tau1_nu_trans = np.sqrt(np.square(va_tau1_nu_E) - np.square(va_tau1_nu_long))
        va_tau2_nu_trans = np.sqrt(np.square(va_tau2_nu_E) - np.square(va_tau2_nu_long))

        for i, elem1 in enumerate(va_tau1_nu_trans):
            if (np.isnan(elem1)):
                va_tau1_nu_trans[i] = 0

        for i, elem1 in enumerate(va_tau2_nu_trans):
            if (np.isnan(elem1)):
                va_tau2_nu_trans[i] = 0    

        lambda_noise = args.LAMBDA
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
        filt = (p_tau1_a1.pt >= 20) & (p_tau2_a1.pt >= 20)
        for part in (l_tau1_pi + l_tau2_pi):
            filt = filt & (part.pt >= 1)
        filt = filt.astype(np.float32)

        isFilter = np.full(p_a1_a1.e.shape, True, dtype=bool)
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
